import pandas as pd


def process_groundwater_data(input_file, output_file):
    try:
        # Read the Excel file
        df = pd.read_excel(input_file)
        print(f"Initial data loaded:\n{df.head()}")

        # Convert the date column to datetime
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        invalid_dates_count = df['Date'].isna().sum()
        print(f"Dropped {invalid_dates_count} rows with invalid dates.")

        df = df.dropna(subset=['Date'])

        # Set the Date as index
        df = df.set_index('Date')

        # Fill NaN values using forward fill
        df['Groundwater'] = df['Groundwater'].ffill()

        # Check for NaN values after forward fill
        nan_count_after_ffill = df['Groundwater'].isna().sum()
        print(f"NaN values after forward fill: {nan_count_after_ffill}")

        # Calculate monthly averages - using 'ME' instead of 'M'
        monthly_summary = df['Groundwater'].resample('ME').agg(['mean', 'count'])

        print(f"Monthly summary (before filling):\n{monthly_summary}")

        # Fill NaN values in monthly mean without 'inplace=True'
        monthly_summary['mean'] = monthly_summary['mean'].ffill()

        print(f"Monthly summary (after filling):\n{monthly_summary}")

        # Reset index to extract Year and Month directly
        monthly_summary.reset_index(inplace=True)

        # Extract Year and Month
        monthly_summary['Year'] = monthly_summary['Date'].dt.year
        monthly_summary['Month'] = monthly_summary['Date'].dt.month

        # Drop the original Date index
        monthly_summary.drop('Date', axis=1, inplace=True)

        # Ensure uniqueness while grouping by Year and Month
        result_df = monthly_summary.groupby(['Year', 'Month']).mean().reset_index()

        # Create all possible Year-Month combinations
        all_months = pd.date_range(start=f"{result_df['Year'].min()}-01-01",
                                   end=f"{result_df['Year'].max()}-12-31",
                                   freq='ME').to_frame(index=False, name='Month')
        all_months['Year'] = all_months['Month'].dt.year
        all_months['Month'] = all_months['Month'].dt.month

        # Merge with monthly averages
        final_df = pd.merge(all_months, result_df, on=['Year', 'Month'], how='left').fillna(0)

        print(f"Final result DataFrame:\n{final_df.head(12)}")

        # Export to Excel
        with pd.ExcelWriter(output_file) as writer:
            final_df.to_excel(writer, sheet_name='Monthly Averages', index=False)
        print(f"Processed data exported to {output_file}")

    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
    except pd.errors.EmptyDataError:
        print("Error: The input file is empty.")
    except Exception as e:
        print(f"An error occurred: {e}")

    # Example usage:


input_file = 'Shawbury_TBU_dataset.xlsx'
output_file = 'Shawbury_Shider_monthly_averages01.xlsx'
process_groundwater_data(input_file, output_file)