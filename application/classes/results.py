import json
import numpy as np


class Results:
    def __init__(self, flags, configs):
        """Results only needs flags and configs - no instance required!"""
        self.flags = flags
        self.configs = configs

        # Result Containers
        self.tree = {'vertices': [], 'successors': {}, 'score_rule': ''}
        self.stats = {
            'score_rule': '',
            'exp_guesses': 0, 'std_guesses': 0, 'max_guesses': 0,
            'distribution': None, 'build_runtime': 0, '#vertices': 0
        }


    def set_data(self, tree, runtime):
        """
        Ingests the raw data from the solver
        """
        self.tree = tree
        self.stats['score_rule'] = tree['score_rule']
        self.stats['build_runtime'] = runtime
        self.stats['#vertices'] = len(tree['vertices'])


    def evaluate(self):
        """
        Evaluates the tree by collecting depths from terminal vertices.
        No simulation needed - depths are pre-recorded during tree building!
        """
        # Collect depths from all terminal vertices
        D = np.array([
            depth
            for _, _, is_terminal, depth in self.tree['vertices'] 
            if is_terminal
        ])

        self.stats['exp_guesses'] = D.mean()
        self.stats['std_guesses'] = D.std()
        self.stats['max_guesses'] = D.max()
        self.stats['distribution'] = {int(d): int((D == d).sum()) for d in np.unique(D)}


    def print(self):
        """
        Prints the evaluation results to the console
        """
        if not self.flags['evaluate']:
            return

        print(
            f"\n\n"
            f"Score Rule: {self.stats['score_rule']}\n"
            f"Start guess: {self.tree['vertices'][0][1]}\n"
            f"Exp. guesses: {self.stats['exp_guesses']:.3f}\n"
            f"Std. guesses: {self.stats['std_guesses']:.3f}\n"
            f"Max. guesses: {self.stats['max_guesses']}\n"
            f"Distribution: {self.stats['distribution']}\n"
            f"Build Runtime: {self.stats['build_runtime']:.3f}s\n"
            f"#Vertices: {self.stats['#vertices']}\n"
        )


    def save(self):
        """
        Saves the tree to a JSON file in the instance-specific directory
        """
        if not self.flags['save_tree']:
            return

        # Convert tree to JSON-serializable format
        serializable_tree = {
            'score_rule': self.tree['score_rule'],
            'vertices': [(int(v), g, bool(term), int(d) if d is not None else None)
                        for v, g, term, d in self.tree['vertices']],
            'successors': {f"{k[0]}_{k[1]}": int(v) for k, v in self.tree['successors'].items()}
        }

        # Save to instance-specific directory
        instance_name = self.configs.get('data', 'wordle')
        filepath = f"data/{instance_name}/decision_tree.json"

        with open(filepath, "w") as f:
            json.dump(serializable_tree, f, indent=2)

        print(f"Tree saved in \"{filepath}\"\n")