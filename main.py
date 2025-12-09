import data_insights as di
from pathlib import Path

def main():
    print("Welcome to Walmart-customer-behavior-insights!")

    dataset_filepath = "dataset/Walmart_customer_purchases.csv"
    reportout_filepath = Path("walmart_customer_behavior_insights.pdf")
    pdf_report_title = "WALMART CUSTOMER BEHAVIOR INSIGHTS"

    dsObj = di.DataInsights(dataset_filepath, reportout_filepath, pdf_report_title)

    # Start basic data insights generation (applicable to to all types of input datasets)
    dsObj.basic_info()
    dsObj.missing_values_analysis()
    dsObj.data_types_summary()
    dsObj.numeric_summary()
    dsObj.numeric_distributions(save_path="plots/numeric_distributions.png")
    dsObj.categorical_summary()
    dsObj.correlation_analysis()
    # End of common dataset analysis

    dsObj.end_operation()  # Internal dsObj operations are wrapped up inside this function.


    # Press Enter key to exit the program
    print("\nPress Enter key to exit the program...")
    input()



if __name__ == "__main__":
    main()
