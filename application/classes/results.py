import json
import os
import numpy as np
from classes.instance_loader import InstanceLoader


def _get_score_rule(configs):
    k, score = configs['k'], configs['score']
    if k == 1:
        strategy = "greedy"
    elif k == -1:
        strategy = "subtree-full"
    else:
        strategy = f"subtree-{k}"
    
    return f"{strategy} | {score}"


class Results:
    def __init__(self, flags, configs):
        """Results only needs flags and configs - no instance required!"""
        self.flags = flags
        self.configs = configs

        # Result Containers
        self.tree = {'vertices': [], 'successors': {}}
        self.stats = {
            'metadata': {},
            'exp_guesses': 0, 'std_guesses': 0, 'max_guesses': 0,
            'distribution': None, 'build_runtime': 0, '#vertices': 0
        }


    def set_data(self, tree, runtime, n_G, n_T):
        """
        Ingests the raw data from the solver and prepares metadata,
        merging with any existing saved tree that shares the same score_rule.
        """
        self.tree = tree

        runtime_key = 'cpu_runtime' if self.configs['explicit_cpu'] else 'gpu_runtime'
        score_rule = _get_score_rule(self.configs)
        metadata = {'score_rule': score_rule, runtime_key: round(runtime, 3)}
        metadata['n_G'] = n_G
        metadata['n_T'] = n_T

        filepath = f"data/{self.configs['data']}/decision_tree.json"
        if os.path.exists(filepath):
            existing = InstanceLoader.load_tree(filepath)
            if existing['metadata']['score_rule'] == score_rule:
                merged = existing['metadata']
                merged[runtime_key] = round(runtime, 3)
                merged['n_G'] = n_G
                merged['n_T'] = n_T
                metadata = merged

        self.stats['metadata'] = metadata
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
            f"Score Rule: {self.stats['metadata']['score_rule']}\n"
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
        Saves the tree to a JSON file in the instance-specific directory.
        Metadata (including any previously recorded runtimes) is already
        prepared by set_data.
        """
        if not self.flags['save_tree']:
            return

        filepath = f"data/{self.configs['data']}/decision_tree.json"
        metadata = self.stats['metadata']
        runtime_key = 'cpu_runtime' if self.configs['explicit_cpu'] else 'gpu_runtime'
        is_update = os.path.exists(filepath) and len(metadata) > 2  # score_rule + both keys

        self._write_tree(filepath, metadata, self.tree['vertices'], self.tree['successors'])

        if is_update:
            print(f"Tree already exists \u2014 updated {runtime_key}={metadata[runtime_key]}s in \"{filepath}\"\n")
        else:
            print(f"Tree saved in \"{filepath}\"\n")


    @staticmethod
    def _write_tree(filepath, metadata, vertices, successors):
        serializable = {
            'metadata': metadata,
            'vertices': [(int(v), g, bool(term), int(d) if d is not None else None)
                         for v, g, term, d in vertices],
            'successors': {f"{k[0]}_{k[1]}": int(v) for k, v in successors.items()}
        }
        with open(filepath, 'w') as f:
            json.dump(serializable, f, indent=2)