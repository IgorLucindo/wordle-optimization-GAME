from classes.device_optimizer import DeviceOptimizer
from collections import deque
import numpy as np
import threading
import time
import sys


class Guess_Tree:
    def __init__(self, instance, flags, configs):
        # Optimizer (Hardware)
        self.optimizer = DeviceOptimizer(instance, flags, configs)

        # Solver State
        G, T, F, C, _, _, _ = instance
        self.xp = self.optimizer.xp
        self.G = self.xp.arange(len(G))
        self.T = self.xp.arange(len(T))
        self.F = F
        self.C = C
        self.flags = flags
        self.configs = configs
        # Extract is_target array from configs
        self.is_target = configs.get('is_target')

        # Tree Building State
        self.tree = {'root': 0, 'vertices': [], 'successors': {}}
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
        G_curr = self.G if self.configs['constrained_guessing'] else None
        queue = deque([(self.T, G_curr, -1, None, 1)])
        self.v_curr = -1
        D = []

        while queue:
            T_curr, G_curr, v_parent, p_parent, depth = queue.popleft()
            self.v_curr += 1

            # Ask optimizer for context
            T_curr, G_curr, xp, F, C, get_best_guess, _ = self.optimizer.get_context(T_curr, G_curr)

            # Leaf case: a single remaining target
            if len(T_curr) == 1:
                target = int(T_curr[0])
                # Check if this target can self-identify
                # For Zoo, targets are NOT guesses, so is_target.any() is False
                # For Wordle/Mastermind, targets ARE guesses, and target < len(is_target)
                can_self_identify = (target < len(self.is_target) and self.is_target[target])

                if can_self_identify:
                    # Target can self-identify (Wordle/Mastermind)
                    # depth counts guesses, and this terminal guess is included
                    self.append2Tree(target, self.v_curr, v_parent, p_parent)
                    D.append(depth)
                else:
                    # Target identified by arrival (Zoo/UCI)
                    # depth counts queries only; cost is number of queries on path
                    self.append2Tree(None, self.v_curr, v_parent, p_parent)
                    D.append(max(0, depth - 1))
                continue

            G_arg = G_curr if self.configs['constrained_guessing'] else self.G
            g_star, is_target_flag = get_best_guess(T_curr, G_arg, F)

            self.append2Tree(g_star, self.v_curr, v_parent, p_parent)
            if is_target_flag:
                D.append(depth)

            # Partition candidates by feedback
            # Only strip g_star from T_curr if g_star is itself a target index
            # (guaranteed in Wordle and Mastermind; false for Zoo attribute queries).
            if is_target_flag:
                T_curr = T_curr[T_curr != g_star]
            feedbacks = F[T_curr, g_star]
            unique_feedbacks, inverse_indices = xp.unique(feedbacks, return_inverse=True)

            # Expand children
            for i, p in enumerate(unique_feedbacks):
                T_p = T_curr[inverse_indices == i]
                G_p = self.get_next_guesses_hardmode(T_p, G_curr, p, g_star, F, C)
                queue.append((T_p, G_p, self.v_curr, p.item(), depth + 1))

        self.stop_diagnosis()

        runtime = time.time() - start_time
        return self.tree, np.array(D), runtime


    def build_subtree(self, g_start, g_start_in_T):
        """
        Build subtree iteratively using explicit queue (BFS)
        Used by the optimization strategy to evaluate candidates
        """
        # Queue: (T_curr, G_curr, v_parent, p_parent, depth)
        G_curr = self.G if self.configs['constrained_guessing'] else None
        queue = deque([(self.T, G_curr, -1, None, 1)])
        self.v_curr = -1
        D = []

        while queue:
            T_curr, G_curr, v_parent, p_parent, depth = queue.popleft()
            self.v_curr += 1

            # Ask optimizer for context
            T_curr, G_curr, xp, F, C, get_best_guess, _ = self.optimizer.get_context(T_curr, G_curr)

            # Leaf case: single target without self-identification
            if len(T_curr) == 1:
                target = int(T_curr[0])
                # Check if target can self-identify
                can_self_identify = (target < len(self.is_target) and self.is_target[target])

                if not can_self_identify:
                    # Pure leaf for attribute-only games (Zoo)
                    # Cost is number of queries on path = depth - 1
                    D.append(max(0, depth - 1))
                    continue

            # Get best guess considering starting guess
            if g_start is not None:
                g_star, is_target_flag = g_start, g_start_in_T
                g_start = None
            else:
                G_arg = G_curr if self.configs['constrained_guessing'] else self.G
                g_star, is_target_flag = get_best_guess(T_curr, G_arg, F)

            if is_target_flag:
                D.append(depth)

            # Stop condition
            if len(T_curr) == 1:
                continue

            # Partition candidates by feedback
            # Only strip g_star from T_curr if g_star is itself a target index
            # (guaranteed in Wordle and Mastermind; false for Zoo attribute queries).
            if is_target_flag:
                T_curr = T_curr[T_curr != g_star]
            feedbacks = F[T_curr, g_star]
            unique_feedbacks, inverse_indices = xp.unique(feedbacks, return_inverse=True)

            # Expand children
            for i, p in enumerate(unique_feedbacks):
                T_p = T_curr[inverse_indices == i]
                G_p = self.get_next_guesses_hardmode(T_p, G_curr, p, g_star, F, C)
                queue.append((T_p, G_p, self.v_curr, p.item(), depth + 1))

        return np.array(D)


    def append2Tree(self, g_star, v_curr, v_parent, p_parent):
        """
        Append vertex and edge to tree
        """
        self.tree['vertices'].append((v_curr, g_star))
        if v_curr != 0:
            self.tree['successors'][(v_parent, p_parent)] = v_curr


    def get_next_guesses_hardmode(self, T, G, feedback, g_star, F, C):
        """
        Vectorized constrained-guessing filtering using precomputed LUT and feedback matrix
        Returns subset of allowed guess indices
        """
        if not self.configs['constrained_guessing'] or len(T) <= 2:
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