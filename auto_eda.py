import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def produce_eda(data_frame):
    # Check for missing values
    print("Missing values:")
    print(data_frame.isnull().sum())
    
    # Percentage of missing values
    print("Percentage of missing values per column:")
    print(data_frame.isnull().mean() * 100)

    # Check data types of columns
    print("\nData types of columns:")
    print(data_frame.dtypes)
    
    # Number of unique values
    print("Number of unique values per column:")
    print(data_frame.nunique())

    # Summary statistics of numerical columns
    numerical_cols = data_frame.select_dtypes(include=np.number).columns
    if numerical_cols.any():
        print(f"\nSummary statistics of numerical columns:")
        print(data_frame[numerical_cols].describe())

    # Top 5 most common values
    categorical_cols = data_frame.select_dtypes(include='object').columns
    for col in categorical_cols:
        print(f"Top 5 most common values in {col}:")
        print(data_frame[col].value_counts().head(5))


    # Calculate correlation between numerical columns
    if numerical_cols.any():
        corr = data_frame[numerical_cols].corr()
        print(f"\nCorrelation between numerical columns:\n{corr}")

    # Mean and standard deviation of numerical columns  
    numerical_cols = data_frame.select_dtypes(include=np.number).columns
    if numerical_cols.any():
        means = data_frame[numerical_cols].mean()
        stds = data_frame[numerical_cols].std()
        for col in numerical_cols:
            print(f"Mean of {col}: {means[col]:.2f}")
            print(f"Standard deviation of {col}: {stds[col]:.2f}")

    # Plot scatter plot of two numerical columns with limit
    if numerical_cols.any():
        col1 = numerical_cols[0]
        col2 = numerical_cols[-1]
        plt.figure()
        plt.scatter(data_frame[data_frame[col1] <= 1000][col1], 
                    data_frame[data_frame[col2] <= 1000][col2])
        plt.title(f"Scatter plot of {col1} vs {col2}")
        plt.xlabel(col1)
        plt.ylabel(col2)
        plt.show()
 
    # Calculate and plot a heatmap of the correlation between numerical columns
    if numerical_cols.any():
        corr = data_frame[numerical_cols].corr()
        plt.figure()
        plt.imshow(corr, cmap='BuPu')
        plt.colorbar()
        plt.xticks(range(len(corr)), corr.columns, rotation=90)
        plt.yticks(range(len(corr)), corr.columns)
        plt.title("Heatmap of correlation between numerical columns")
        plt.show()

    # Summary statistics of categorical columns
    
    categorical_cols = data_frame.select_dtypes(include='object').columns
    if categorical_cols.any():
        print("\nSummary statistics of categorical columns:")
        print(data_frame[categorical_cols].describe())

        
    # Perform t-test on two numerical columns
    if numerical_cols.size >= 2:
        col1 = numerical_cols[0]
        col2 = numerical_cols[-1]
        t_stat, p_val = stats.ttest_ind(data_frame[col1], data_frame[col2])
        print(f"\nT-test between {col1} and {col2}: t-statistic = {t_stat:.4f}, p-value = {p_val:.4f}")

    # Perform chi-squared test on two categorical columns
    categorical_cols = data_frame.select_dtypes(include='object').columns
    if len(categorical_cols) >= 2:
        col1 = categorical_cols[0]
        col2 = categorical_cols[-1]
        contingency_table = pd.crosstab(data_frame[col1], data_frame[col2])
        chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
        print(f"\nChi-squared test between {col1} and {col2}: chi-squared statistic = {chi2:.4f}, p-value = {p:.4f}")
