import pandas as pd
import os


class Results:
    """
    class that contains methods to print and save results
    """
    def __init__(self, flags, config):
        self.flags = flags
        self.config = config
        
        self.data = []
        self.data_columns = [
            'Average Guesses', 'Correct Guesses (%)'
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
        
        # Set params
        guesses_len = []
        correct_guesses = []
        
        for r in simulation_results:
            guesses_len.append(len(r['guesses']))
            correct_guesses.append(r['correct_guess'])


        avg_guesses_len = sum(guesses_len) / len(guesses_len)
        avg_correct_guesses = sum(correct_guesses) / len(correct_guesses) * 100
    
        # Set data
        self.data.append([
            avg_guesses_len, avg_correct_guesses
        ])

        # Set config data
        self.set_data_config()


    def fill_row(self, _list):
        return (_list + [""] * self.row_length)[:self.row_length]
    

    def set_data_config(self):
        """
        Set configuration for saving
        """
        for _ in range(3):
            self.data.append(self.fill_row([]))

        for key, value in self.config.items():
            self.data.append(self.fill_row([key, value]))


    def save(self):
        """
        Save solution results
        """
        if not self.flags['save_results']:
            return
        
        # Create dataframe for exporting to xlsx file
        df = pd.DataFrame(self.data, columns=self.data_columns)
        df.to_excel(self.path + "results.xlsx", index=False)