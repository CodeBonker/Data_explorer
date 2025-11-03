import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class DataExplorer:
    """
    Class for loading and exploring tabular data
    """

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.data = None

    def load_data(self):
        """ Loads a CSV file into a pandas dataframe"""
        try:
            self.data = pd.read_csv(self.file_path)
            print("File loaded successfully")

        except FileNotFoundError:
            print ("file not found")

        except Exception as e:
            print(f"error loading files: {e}")

    def show_info(self):
        """Displays basic info and summary of the dataset"""
        if self.data is None:
            self.load_data()
        print("\n--- Data Head ---")
        print(self.data.head())
        print("\n--- Info ---")
        print(self.data.info())
        print("\n--- Description ---")
        print(self.data.describe())

    def get_missing_values(self):
        # per column return karega missing values
        if self.data is None:
            self.load_data()
        return self.data.isnull().sum()
    
    def select_columns(self, columns: list):
        """
        Selects specific columns
        """
        if self.data is None:
            self.load_data()
        try:
            return self.data[columns]
        except KeyError as e:
            print(f"Invalid column name/s: {e}")
            return None
        
    def filter_data(self, condition):
        """ filter data based on some condition"""
        
        if self.data is None:
            self.load_data()
        return self.data.query(condition)
    
    # grouping and aggregation karenge
    def group_and_aggregate(self, group_by_col: str, agg_col: str, agg_func: str = 'mean'):
        """
        Groups the data by one column and applies an aggregation function (mean, sum, count, etc.)
        """
        if self.data is None:
            self.load_data()
        try:
            result = self.data.groupby(group_by_col)[agg_col].agg(agg_func)
            return result
        except Exception as e:
            print(f"Error during grouping: {e}")
            return None
        


    def visualize_data(self):

        if self.data is None:
            self.load_data()
    
        # Distribution of Age
        self.data["Age"].plot(kind="hist", bins=10, title="Age Distribution", color="skyblue", edgecolor="black")
        plt.xlabel("Age")
        plt.ylabel("Count")
        plt.show()

        # Average Leave Rate by City
        leave_rate = self.data.groupby("City")["LeaveOrNot"].mean()
        leave_rate.plot(kind="bar", color="coral", title="Average Leave Rate by City")
        plt.ylabel("Leave Rate (0-1)")
        plt.show()

        # Age vs Experience
        self.data.plot(kind="scatter", x="Age", y="ExperienceInCurrentDomain", color="green", title="Age vs Experience")
        plt.show()