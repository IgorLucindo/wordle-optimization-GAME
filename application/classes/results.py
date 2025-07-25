import pandas as pd
import numpy as np
import os


class Results:
    """
    class that contains methods to print and save results
    """
    def __init__(self, flags):
        self.flags = flags
        
        self.data = []
        self.data_columns = [
            'Solver Type', 'Correct Guesses (%)', 'Average Corrected Guesses',
            'Standard Deviation Corrected Guesses', 'Average Runtime (s)'
        ]
        self.row_length = len(self.data_columns)

        self.path = "application/results/"
        os.makedirs(self.path, exist_ok=True)


    def set_data(self, simulation_results):
        """
        Set data for saving results
        """
        if not self.flags['save_results']:
            return
        
        for solver_type, results in simulation_results.items():
            # Set params
            guesses_len = []
            correct_guesses = []

            for r in results['list']:
                correct_guesses.append(r['correct_guess'])
                if r['correct_guess']:
                    guesses_len.append(len(r['guesses']))

            # Calculate data
            avg_correct_guesses = sum(correct_guesses) / len(correct_guesses) * 100
            avg_guesses_len = sum(guesses_len) / len(guesses_len)
            std_guesses_len = np.std(guesses_len)
            runtime = results['runtime'] / len(correct_guesses)
        
            # Set data
            self.data.append([
                solver_type, avg_correct_guesses, avg_guesses_len,
                std_guesses_len, runtime
            ])


    def save(self):
        """
        Save solution results
        """
        if not self.flags['save_results']:
            return
        
        # Create dataframe for exporting to xlsx file
        df = pd.DataFrame(self.data, columns=self.data_columns)
        df.to_excel(self.path + "results.xlsx", index=False)