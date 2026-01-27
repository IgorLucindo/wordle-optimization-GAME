import numpy as np
import cupy as cp
import time
import json
import os


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
        
        # Run initialization
        self._initialize_data_and_calibration(T, G, F, C)


    def _initialize_data_and_calibration(self, T, G, F, C):
        """
        Pre-loads CPU data and handles calibration logic (load/save/run)
        """
        if self.configs['GPU']:
            self.F_cpu = cp.asnumpy(F)
            self.C_cpu = cp.asnumpy(C) if C is not None else None
            
            # Unique key based on metric AND the specific solver function being used
            k = f"_k{self.configs['k']}" if self.configs['metric'] == 1 else ""
            solver_name = self.solvers_cpu[0].__name__
            self.calibration_key = f"metric_{self.configs['metric']}{k}_{solver_name}"
            self.calibration_file = "application/results/calibration.json"

            cached_val = self._load_calibration()
            
            if cached_val is not None:
                self.threshold = cached_val
                if self.flags['print_diagnosis']:
                    print(f"  [Calibration] Loaded threshold for '{self.calibration_key}': {self.threshold}")
            else:
                self.threshold = self._calibrate_threshold(T, G)
                self._save_calibration(self.threshold)
        else:
            self.F_cpu = F
            self.C_cpu = C
            self.threshold = float('inf') # Always CPU


    def get_context(self, T_curr, G_curr):
        """
        Decides the optimal compute context (CPU vs GPU) for the given workload
        Returns: (T, G, xp, F, C, get_best_guess, get_best_guesses)
        """
        # Fast Exit: If we are already on CPU, stay there
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
        Runs a race between CPU and GPU
        Includes 'Blowout Rule' and 'Safety Brake' to exit early if CPU is winning
        """
        if self.flags['print_diagnosis']:
            print(f"  [Calibration] Calibrating CPU/GPU threshold for {self.calibration_key}...")

        # Warm Up (JIT Compile) with tiny workload
        warm_T, warm_G = np.arange(10), np.arange(50) 
        self.solvers_cpu[0](warm_T, warm_G, self.F_cpu)
        self.solvers_gpu[0](cp.array(warm_T), cp.array(warm_G), self.F_gpu)
        cp.cuda.Stream.null.synchronize()

        # Race Points
        test_points = [(10, 50), (50, 1000), (250, 1000), (1000, 1000), (2000, 2500)]
        best_threshold = float('inf')  # Default to CPU
        found_crossover = False

        for n_t, n_g in test_points:
            # Safe slice
            n_t = min(n_t, len(T_full))
            n_g = min(n_g, len(G_full))
            workload = n_t * n_g
            
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

            # --- Safety Brake (Timeout) ---
            # If any test takes > 0.5s, stop testing. It's too heavy
            if dur_cpu > 0.5 or dur_gpu > 0.5:
                if self.flags['print_diagnosis']:
                    print("    [Calibration] Test took too long. Stopping race.")
                if dur_gpu < (dur_cpu * 0.9):
                    best_threshold = workload
                    found_crossover = True
                break

            # --- Blowout Rule (Early Exit) ---
            # If GPU is >2x slower than CPU on a non-trivial task, stop
            # We don't need to test larger workloads to know CPU wins
            if dur_gpu > (dur_cpu * 2.0) and dur_cpu > 0.01:
                if self.flags['print_diagnosis']:
                    print("    [Calibration] CPU is >2x faster. Stopping race early.")
                break

            # --- Crossover Check ---
            # GPU must be at least 10% faster to justify the switch
            if dur_gpu < (dur_cpu * 0.9):
                best_threshold = workload
                found_crossover = True
                break
        
        # If we found a crossover at the very first (tiniest) point, GPU is always faster
        if found_crossover and best_threshold == test_points[0][0] * test_points[0][1]:
            best_threshold = 0

        if self.flags['print_diagnosis']:
            print(f"  [Calibration] Threshold set to {best_threshold} ops\n")
            
        return best_threshold


    def _load_calibration(self):
        """
        Loads the calibration value from the JSON file if it exists
        """
        if not os.path.exists(self.calibration_file):
            return None
        
        try:
            with open(self.calibration_file, 'r') as f:
                data = json.load(f)
                return data.get(self.calibration_key)
        except (json.JSONDecodeError, IOError):
            return None


    def _save_calibration(self, value):
        """
        Saves the calibration value to the JSON file
        """
        data = {}
        if os.path.exists(self.calibration_file):
            try:
                with open(self.calibration_file, 'r') as f:
                    data = json.load(f)
            except (json.JSONDecodeError, IOError):
                data = {} 

        data[self.calibration_key] = value

        os.makedirs(os.path.dirname(self.calibration_file), exist_ok=True)
        
        with open(self.calibration_file, 'w') as f:
            json.dump(data, f, indent=4)