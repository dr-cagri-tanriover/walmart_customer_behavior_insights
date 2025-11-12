from typing import Any

import pandas as pd
from pathlib import Path
from utils import printing as prt

def print_divider(text: str):
    def decorator(func_name: str):
        def wrapper(*args, **kwargs):
            print(f"\n" + "=" *80)
            print(text)
            print(f"=" *80)
            func_name(*args, **kwargs)
        return wrapper
    return decorator    



class DataInsights:

    @print_divider("INITIALIZING DATA INSIGHTS OBJECT")
    def __init__(self, path: str | Path):
        self.df = pd.DataFrame()
        
        try:
            self.df = pd.read_csv(path)
            print(f"Data loaded successfully: {len(self.df)} rows, {len(self.df.columns)} columns")
        except FileNotFoundError:
            print(f"File not found: {path}")
            raise
        except Exception as e:
            print(f"Error loading data: {e}")
            raise


    @print_divider("BASIC DATAFRAME INFORMATION")
    def basic_info(self):
        """
            Display basic information about the dataframe self.df
        """
        
        print(f"Number of rows: {len(self.df)}")
        print(f"Number of columns: {len(self.df.columns)}")
        #print(f"Data types: {self.df.dtypes}")
        print(f"Memory usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

        print(f"\nColumn Names:")
        for i, col in enumerate(self.df.columns, start=1):
            print(f"  {i}. {col}")


    @print_divider("MISSING VALUES ANALYSIS")
    def missing_values_analysis(self):
        """
            Display missing values in the dataframe self.df, if any.
        """
        
        missing_elements_per_column = self.df.isnull().sum()  # as pandas Series

        missing_record_df = pd.DataFrame({
            'Columns': missing_elements_per_column.index,
            'Missing Elements': missing_elements_per_column.values,
            'Percentage Missing': missing_elements_per_column.values.sum() *100 / self.df.size
        })
    
        # Optionally sort rows in according to "Missing Elements" in ascending order.
        missing_record_df = missing_record_df.loc[missing_record_df['Missing Elements'] > 0].sort_values('Missing Elements', ascending=True)

        if len(missing_record_df):
            # There is at least one row with missing elements
            print(f"Found missing values in dataset!")
            print(f"{self.df.to_string(index=False)}")
        else:
            print(f"There are no missing elements in the dataset !!")


    @print_divider("DATA TYPES SUMMARY")
    def data_types_summary(self):
        """
            Provide a summary of data types and their distributions.
        """

        numeric_columns = self.df.select_dtypes(include=["number"]).columns.tolist()
        categorical_columns = self.df.select_dtypes(include=["object"]).columns.tolist()
        datetime_columns = self.df.select_dtypes(include=["datetime", "datetime64"]).columns.tolist()

        if len(numeric_columns) > 0:
            print(f"Found {len(numeric_columns)} numeric columns in dataset:")
            for i, col_name in enumerate(numeric_columns):
                print(f"{i} - {col_name}")
        else:
            print("No numerical data found in dataset")


        if len(categorical_columns) > 0: 
            print(f"\nFound {len(categorical_columns)} categorical columns in dataset:")
            for i, col_name in enumerate(categorical_columns):
                print(f"{i} - {col_name}")
        else:
            print("\nNo categorical data found in dataset")


        if len(datetime_columns) > 0:
            print(f"\nFound {len(datetime_columns)} datetime columns in dataset:")
            for i, col_name in enumerate(datetime_columns):
                print(f"{i} - {col_name}")
        else:
            print("\nNo datetime data found in dataset")


    @print_divider("NUMERIC COLUMNS STATISTICS")
    def numeric_summary(self):
        """
        Generate descriptive statistics for numeric columns.
        """
        
        numeric_data_exists = lambda df: True if  len(df.select_dtypes(include=['number']).columns.tolist()) > 0 else False

        if numeric_data_exists(self.df):
            summary = self.df.describe(include='number')  # general statistics in summary data frame
            summary.loc['median'] = self.df.describe(include='number').median()  # add median for all numeric columns
            summary.loc['skew'] = self.df.describe(include='number').skew()  # add skew for all numeric columns
            summary.loc['kurtosis'] = self.df.describe(include='number').kurtosis()  # add kurtosis for all numeric columns

            #print(f"{summary.to_string()}")  # display full summary as text
            prt.print_dataframe(summary)
        else:
            print(f"No numeric data exists in dataset...")


    @print_divider("CATEGORICAL COLUMNS STATISTICS")
    def categorical_summary(self):
        """
        Analyze categorical columns including unique values and frequencies.
        """
        N_MAX = 10  # maximum number of unique items in a column allowed to be displayed in this method.

        categorical_columns_list = self.df.select_dtypes(include=['object']).columns.to_list()

        if len(categorical_columns_list):
            # There are categorical columns
            print(f"High level summary of categorical columns:")
            prt.print_dataframe(self.df.describe(include='object'))                
            #print(f"{self.df.describe(include='object').to_string()}")

            # Next display number of unique items and their occurence frequency (where manageable) for each categorical column.
            for col in categorical_columns_list:
                print(f"\n" + "~"*80)
                print(f"Categorical feature (column): {col}")
                n_unique_items = self.df[col].nunique()
                print(f"Number of unique items: {n_unique_items}")

                if n_unique_items <= N_MAX:
                    unique_dict = {item: count for item, count in self.df[col].value_counts().items()}
                    printPd = pd.Series(unique_dict).reset_index()  # Convert series to dataframe
                    printPd.columns = ['Item', 'Count']  # Assign custom column names to enw dataframe              
                    prt.print_dataframe(printPd, show_index=False)  # Not showing enumerated indices as they are not informative
                else:
                    print(f"\nNumber of unique items > {N_MAX}. Skipping item listing...")

                # What is the most frequent item in each categorical column?
                most_frequent_col_items = self.df[col].mode().to_list()
                if len(most_frequent_col_items) <= N_MAX:
                    print(f"\nHighest frequency items:")
                    itemDict = {item:self.df[col].value_counts()[item] for item in most_frequent_col_items}
                    printPd = pd.Series(itemDict).reset_index()
                    printPd.columns = ['Item', 'Count']
                    prt.print_dataframe(printPd, show_index=False)
                else:
                    print(f"\nThere are >{N_MAX} items at high frequency. Skipping item listing...")

        else:
            print(f"No categorical data exists in dataset...")


    @print_divider("CORRELATION ANALYSIS (applicable to the numeric columns only)")
    def correlation_analysis(self):
        """
        Analyze correlations between numeric variables using Pearson and Spearman methods.
        Uses only the original numeric columns from the dataset.
        """
        
        numeric_columns = self.df.select_dtypes(include=["number"]).columns.tolist()
        
        if len(numeric_columns) == 0:
            print("No numerical data found in dataset...")
            return
        
        if len(numeric_columns) < 2:
            print(f"Found only {len(numeric_columns)} numeric column(s). At least 2 numeric columns are required for correlation analysis.")
            return
        
        print(f"Found {len(numeric_columns)} numeric columns in dataset:")
        for i, col_name in enumerate(numeric_columns):
            print(f"  {i+1}. {col_name}")
        
        # Select only the original numeric columns
        numeric_df = self.df[numeric_columns]
        
        # Pearson Correlation
        print(f"\nPearson Correlation Matrix:")
        pearson_correlation_matrix = numeric_df.corr(method='pearson')
        #print(f"{pearson_correlation_matrix.to_string()}")
        prt.print_dataframe(pearson_correlation_matrix, justify_numeric="center")

        # Find strong Pearson correlations
        print(f"\nStrong Pearson Correlations Criterion: |r| > 0.5")
        strong_pearson_corrs = []
        for i in range(len(pearson_correlation_matrix.columns)):
            for j in range(i+1, len(pearson_correlation_matrix.columns)):
                corr_val = pearson_correlation_matrix.iloc[i, j]
                if abs(corr_val) > 0.5:
                    col1 = pearson_correlation_matrix.columns[i]
                    col2 = pearson_correlation_matrix.columns[j]
                    strong_pearson_corrs.append((col1, col2, corr_val))
        
        if len(strong_pearson_corrs) > 0:
            for col1, col2, corr_val in strong_pearson_corrs:
                print(f"  {col1} ↔ {col2}: {corr_val:.3f}")
        else:
            print("\tNo strong Pearson correlations found !")
        
        # Spearman Correlation
        print(f"\nSpearman Correlation Matrix:")
        spearman_correlation_matrix = numeric_df.corr(method='spearman')
        #print(f"{spearman_correlation_matrix.to_string()}")
        prt.print_dataframe(spearman_correlation_matrix, justify_numeric="center")

        # Find strong Spearman correlations
        print(f"\nStrong Spearman Correlations Criterion: |r| > 0.5:")
        strong_spearman_corrs = []
        for i in range(len(spearman_correlation_matrix.columns)):
            for j in range(i+1, len(spearman_correlation_matrix.columns)):
                corr_val = spearman_correlation_matrix.iloc[i, j]
                if abs(corr_val) > 0.5:
                    col1 = spearman_correlation_matrix.columns[i]
                    col2 = spearman_correlation_matrix.columns[j]
                    strong_spearman_corrs.append((col1, col2, corr_val))
        
        if len(strong_spearman_corrs) > 0:
            for col1, col2, corr_val in strong_spearman_corrs:
                print(f"  {col1} ↔ {col2}: {corr_val:.3f}")
        else:
            print("\tNo strong Spearman correlations found !")

