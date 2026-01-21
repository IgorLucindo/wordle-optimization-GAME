import numpy as np
import cupy as cp
import time

class DeviceOptimizer:
    def __init__(self, instance, flags, configs):
        G, T, F, C, _, best_guess_fns, best_guesses_fns = instance
        self.flags = flags
        self.configs = configs
        self.threshold = 0

        # --- Solver References ---
        # 0 = CPU, 1 = GPU
        self.solvers_cpu = (best_guess_fns[0], best_guesses_fns[0])
        self.solvers_gpu = (best_guess_fns[1], best_guesses_fns[1])
        
        # --- Data Management ---
        # Default to GPU if enabled
        self.xp = cp if configs['GPU'] else np
        self.F_gpu = F
        self.C_gpu = C
        
        # Pre-load CPU data for fast switching
        if configs['GPU']:
            self.F_cpu = cp.asnumpy(F)
            self.C_cpu = cp.asnumpy(C) if C is not None else None
            # Run Calibration immediately
            self.threshold = self._calibrate_threshold(T, G)
        else:
            self.F_cpu = F
            self.C_cpu = C
            self.threshold = float('inf') # Always CPU


    def get_context(self, T_curr, G_curr):
        """
        Decides the optimal compute context (CPU vs GPU) for the given workload
        Returns: (T, G, xp, F, C, get_best_guess, get_best_guesses)
        """
        # Fast Exit: If we are already on CPU, stay there.
        if isinstance(T_curr, np.ndarray):
            return (
                T_curr, G_curr, np, 
                self.F_cpu, self.C_cpu, 
                self.solvers_cpu[0], self.solvers_cpu[1]
            )

        # Default: Use GPU
        use_cpu = False
        
        # Measure Workload
        n_t = len(T_curr) if hasattr(T_curr, '__len__') else 0
        n_g = len(G_curr) if G_curr is not None and hasattr(G_curr, '__len__') else 0
        workload = n_t * n_g

        # Check Conditions
        # If GPU enabled but workload is tiny, switch to CPU
        if self.configs['GPU'] and self.configs['hard_mode'] and G_curr is not None:
            if workload < self.threshold:
                use_cpu = True
                # Move data to CPU
                T_curr = cp.asnumpy(T_curr)
                G_curr = cp.asnumpy(G_curr)

        # Return Context
        if use_cpu or not self.configs['GPU']:
            return (
                T_curr, G_curr, np, 
                self.F_cpu, self.C_cpu, 
                self.solvers_cpu[0], self.solvers_cpu[1]
            )
        else:
            return (
                T_curr, G_curr, cp, 
                self.F_gpu, self.C_gpu, 
                self.solvers_gpu[0], self.solvers_gpu[1]
            )


    def _calibrate_threshold(self, T_full, G_full):
        """
        Runs a race between CPU and GPU to find the exact crossover point
        """
        if self.flags['print_diagnosis']:
            print("  [Calibration] Calibrating CPU/GPU switch threshold...")

        # Warm Up (JIT Compile)
        warm_T, warm_G = np.arange(10), np.arange(100)
        self.solvers_cpu[0](warm_T, warm_G, self.F_cpu)
        self.solvers_gpu[0](cp.array(warm_T), cp.array(warm_G), self.F_gpu)
        cp.cuda.Stream.null.synchronize()

        # Race
        # Test points: (Targets, Guesses) -> Ops
        test_points = [(50, 1000), (250, 1000), (1000, 1000), (2000, 2500)]
        best_threshold = 5000000  # Default high
        found_crossover = False

        for n_t, n_g in test_points:
            # Safe slice
            n_t = min(n_t, len(T_full))
            n_g = min(n_g, len(G_full))
            workload = n_t * n_g
            
            # Generate Random Indices
            idxs_t = np.random.choice(len(T_full), n_t, replace=False)
            idxs_g = np.random.choice(len(G_full), n_g, replace=False)

            # Time CPU
            start = time.time()
            self.solvers_cpu[0](idxs_t, idxs_g, self.F_cpu)
            dur_cpu = time.time() - start

            # Time GPU (Include transfer!)
            start = time.time()
            self.solvers_gpu[0](cp.array(idxs_t), cp.array(idxs_g), self.F_gpu)
            cp.cuda.Stream.null.synchronize()
            dur_gpu = time.time() - start

            if self.flags['print_diagnosis']:
                print(f"    Workload {workload:8d} | CPU: {dur_cpu:.4f}s | GPU: {dur_gpu:.4f}s")

            if dur_gpu < dur_cpu:
                best_threshold = workload
                found_crossover = True
                break
        
        if not found_crossover:
            best_threshold = 5000000
        elif best_threshold == test_points[0][0] * test_points[0][1]:
            best_threshold = 0 

        if self.flags['print_diagnosis']:
            print(f"  [Calibration] Threshold set to {best_threshold} ops")
            
        return best_threshold