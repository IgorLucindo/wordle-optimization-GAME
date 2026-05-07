from classes.device_optimizer import DeviceOptimizer
from collections import deque
import numpy as np
import threading
import time
import sys


def _get_score_rule(configs):
    k, score = configs['k'], configs['score']
    if k == 1:
        strategy = "greedy"
    elif k == -1:
        strategy = "subtree-full"
    else:
        strategy = f"subtree-{k}"
    
    return f"{strategy} | {score}"


class Guess_Tree:
    def __init__(self, instance, flags, configs):
        # Optimizer (Hardware)
        self.optimizer = DeviceOptimizer(instance, flags, configs)

        # Solver State
        G, T, F, C, _, _, _ = instance
        print(f"{len(G)}     {len(T)}")
        self.xp = self.optimizer.xp
        self.G = self.xp.arange(len(G))
        self.T = self.xp.arange(len(T))
        self.F = F
        self.C = C
        self.flags = flags
        self.configs = configs

        self._constrained_guessing = configs['constrained_guessing']
        self._guesses_include_targets = configs['guesses_include_targets']

        # Tree Building State
        self.G_names = G
        self.T_names = T
        self.tree = {
            'vertices': [],
            'successors': {},
            'score_rule': _get_score_rule(configs),
        }
        self._stop_diagnosis = False
        self._diagnosis_thread = None
        self.v_curr = -1


    def build_tree(self):
        """
        Build tree iteratively using explicit queue (BFS)
        """
        self.start_diagnosis()
        start_time = time.time()

        # Queue: (T_curr, G_curr, v_parent, p_parent, depth)
        G_curr = self.G if self._constrained_guessing else None
        queue = deque([(self.T, G_curr, -1, None, 1)])
        self.v_curr = -1

        while queue:
            T_curr, G_curr, v_parent, p_parent, depth = queue.popleft()
            self.v_curr += 1

            # Ask optimizer for context
            T_curr, G_curr, xp, F, C, get_best_guess, _ = self.optimizer.get_context(T_curr, G_curr)

            # Pick best guess
            G_arg = G_curr if self._constrained_guessing else self.G
            g_star, is_target_flag = get_best_guess(T_curr, G_arg, F)

            # Append to tree with terminal flag and depth
            self._append2Tree(g_star, self.v_curr, v_parent, p_parent, depth,
                              is_terminal=is_target_flag)

            # Stop if we just guessed the last target
            if len(T_curr) == 1:
                continue

            # Partition candidates by feedback
            if is_target_flag:
                T_curr = T_curr[T_curr != g_star]
            feedbacks = F[T_curr, g_star]
            unique_feedbacks, inverse_indices = xp.unique(feedbacks, return_inverse=True)

            # Expand children
            for i, p in enumerate(unique_feedbacks):
                T_p = T_curr[inverse_indices == i]
                G_p = self._get_constrained_guesses(T_p, G_curr, p, g_star, F, C)
                queue.append((T_p, G_p, self.v_curr, p.item(), depth + 1))

        self.stop_diagnosis()

        runtime = time.time() - start_time
        return self.tree, runtime


    def build_subtree(self, g_start, g_start_in_T):
        """
        Build subtree iteratively using explicit queue (BFS)
        Used by the optimization strategy to evaluate candidates
        """
        # Queue: (T_curr, G_curr, v_parent, p_parent, depth)
        G_curr = self.G if self._constrained_guessing else None
        queue = deque([(self.T, G_curr, -1, None, 1)])
        self.v_curr = -1
        depths = []

        while queue:
            T_curr, G_curr, _, _, depth = queue.popleft()
            self.v_curr += 1

            # Ask optimizer for context
            T_curr, G_curr, xp, F, C, get_best_guess, _ = self.optimizer.get_context(T_curr, G_curr)

            # Get best guess considering starting guess
            if g_start is not None:
                g_star, is_target_flag = g_start, g_start_in_T
                g_start = None
            else:
                G_arg = G_curr if self._constrained_guessing else self.G
                g_star, is_target_flag = get_best_guess(T_curr, G_arg, F)

            if is_target_flag:
                depths.append(depth)

            # Stop if we just guessed the last target
            if len(T_curr) == 1:
                continue

            # Partition candidates by feedback
            if is_target_flag:
                T_curr = T_curr[T_curr != g_star]
            feedbacks = F[T_curr, g_star]
            unique_feedbacks, inverse_indices = xp.unique(feedbacks, return_inverse=True)

            # Expand children
            for i, p in enumerate(unique_feedbacks):
                T_p = T_curr[inverse_indices == i]
                G_p = self._get_constrained_guesses(T_p, G_curr, p, g_star, F, C)
                queue.append((T_p, G_p, self.v_curr, p.item(), depth + 1))

        return np.array(depths)


    def _append2Tree(self, g_star, v_curr, v_parent, p_parent, depth, is_terminal=False):
        """
        Append vertex and edge to tree

        Args:
            g_star: guess index
            v_curr: current vertex id
            v_parent: parent vertex id
            p_parent: feedback from parent
            is_terminal: True if this vertex identifies a target
            depth: depth of this vertex (recorded for terminal vertices)
        """
        is_target_offset = not self._guesses_include_targets and is_terminal
        names = self.T_names if is_target_offset else self.G_names
        depth_offset = int(is_target_offset)
        
        self.tree['vertices'].append((
            v_curr,
            names[g_star.item()],
            is_terminal,
            (depth - depth_offset) if is_terminal else None
        ))
        if v_curr != 0:
            self.tree['successors'][(v_parent, p_parent)] = v_curr


    def _get_constrained_guesses(self, T, G, feedback, g_star, F, C):
        """
        Vectorized constrained-guessing filtering using precomputed LUT and feedback matrix
        Returns subset of allowed guess indices
        """
        if not self._constrained_guessing or len(T) <= 2:
            return None
        
        # Feedbacks that each candidate (col) would produce w.r.t. previous guess (row)
        possible_feedbacks = F[G, g_star]

        # Mask of which feedbacks are compatible
        valid_mask = C[possible_feedbacks, feedback]

        return G[valid_mask]


    def start_diagnosis(self):
        if not self.flags['print_diagnosis']: return
        start_time = time.time()
        print("")
        def _diagnose():
            while not self._stop_diagnosis:
                elapsed = int(time.time() - start_time)
                sys.stdout.write(f"\rVertex count: {self.v_curr + 1} | Time: {elapsed}s   ")
                sys.stdout.flush()
                time.sleep(1)
        self._diagnosis_thread = threading.Thread(target=_diagnose, daemon=True)
        self._diagnosis_thread.start()


    def stop_diagnosis(self):
        self._stop_diagnosis = True
        if self._diagnosis_thread is not None:
            self._diagnosis_thread.join()