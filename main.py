import data_insights as di

def main():
    print("Welcome to Walmart-customer-behavior-insights!")

    dataset_filepath = "dataset/Walmart_customer_purchases.csv"
    dsObj = di.DataInsights(dataset_filepath)

    # Start basic data insights generation (applicable to to all types of input datasets)
    dsObj.basic_info()
    dsObj.missing_values_analysis()
    dsObj.data_types_summary()
    dsObj.numeric_summary()
    dsObj.numeric_distributions()
    dsObj.categorical_summary()
    dsObj.correlation_analysis()
    # End of common dataset analysis

    # Press Enter key to exit the program
    print("\nPress Enter key to exit the program...")
    input()



if __name__ == "__main__":
    main()
