import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def load_and_preprocess_data(new_file_path):
    """
    Read data from a CSV file and perform necessary preprocessing.

    Parameters:
    - new_file_path (str): The path to the CSV file.

    Returns:
    - df_years (pd.DataFrame): DataFrame with years as columns.
    - df_countries (pd.DataFrame): DataFrame with countries as columns.
    """

    # Read data from the CSV file, skipping the first 4 rows.
    data_frame = pd.read_csv(new_file_path, skiprows=4)

    # Drop unnecessary columns.
    cols_to_drop = ['Country Code', 'Indicator Code', 'Unnamed: 66']
    data_frame = data_frame.drop(cols_to_drop, axis=1)

    # Rename remaining columns.
    data_frame = data_frame.rename(columns={'Country Name': 'Country'})

    # Melt the dataframe to convert years to a single column.
    data_frame = data_frame.melt(id_vars=['Country', 'Indicator Name'],
                                 var_name='Year', value_name='Value')

    # Convert year column to integer and value column to float.
    data_frame['Year'] = pd.to_numeric(data_frame['Year'], errors='coerce')
    data_frame['Value'] = pd.to_numeric(data_frame['Value'], errors='coerce')

    # Separate dataframes with years and countries as columns.
    df_years = data_frame.pivot_table(
        index=['Country', 'Indicator Name'], columns='Year', values='Value')
    df_countries = data_frame.pivot_table(
        index=['Year', 'Indicator Name'], columns='Country', values='Value')

    # Clean the data by removing columns with all NaN values.
    df_years = df_years.dropna(how='all', axis=1)
    df_countries = df_countries.dropna(how='all', axis=1)

    return df_years, df_countries



def filter_data_subset(df_years, countries, indicators, start_year, end_year):
    """
    Subsets the data to include only the selected countries, indicators, and specified year range.
    Returns the subsetted data as a new DataFrame.
    """
    # Create a boolean mask for the specified year range
    mask_year = (df_years.columns.get_level_values('Year').astype(int) >= start_year) & (df_years.columns.get_level_values('Year').astype(int) <= end_year)

    # Apply masks to subset the data
    df_subset = df_years.loc[(countries, indicators), mask_year].transpose()

    return df_subset


def calculate_correlation_matrix(df):
    """
    Calculates the correlations between the indicators in the input DataFrame.
    Returns the correlation matrix as a new DataFrame.
    """
    corr_matrix = df.corr()
    return corr_matrix


def plot_correlation_heatmap(corr_matrix):
    """
    Plots the correlation matrix as a heatmap using Seaborn.
    """
    # Set Seaborn style
    sns.set(style="whitegrid")

    # Set the color palette
    cmap = 'coolwarm'
    
    # Set the figure size
    plt.figure(figsize=(10, 8))

    # Add more style options
    sns.heatmap(corr_matrix, cmap=cmap, annot=True, square=True, linewidths=.5, cbar_kws={"shrink": 0.8})
    
    # Set the title for the heatmap
    plt.title('Correlation Matrix of Indicators', fontsize=16)

    # Show the plot
    plt.show()




def visualize_line_chart(df_years, selected_years, selected_indicator):
    """
    Plot a line chart for a selected indicator across multiple countries for specific years.

    Parameters:
    - df_years (pd.DataFrame): DataFrame with years as columns and countries as rows.
    - selected_years (list): List of years to be plotted.
    - selected_indicator (str): The indicator to plot.

    Returns:
    - None
    """

    # List of countries to include in the chart
    country_list = ['France', 'India','South Africa', 'United Kingdom','China']

    # Set the color palette using custom colors
    custom_colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']

    # Set Seaborn style
    sns.set(style="darkgrid")

    # Set the line styles
    line_styles = ['-', '--', '-.', ':']

    # Set the figure size
    fig, ax = plt.subplots(figsize=(12, 8))

    for i, country in enumerate(country_list):
        # Extract data subset for the selected indicator and years
        df_subset = df_years.loc[(country, selected_indicator), selected_years]
        
        # Plot the line for each country with specific style and color
        ax.plot(df_subset.index, df_subset.values, label=country, linestyle=line_styles[i % len(line_styles)], color=custom_colors[i])

        # Print results for the selected years
        print(f"\nResults for {country} - {selected_indicator} for the selected years:")
        print(df_subset)

    # Set plot labels and title
    ax.set_xlabel('Year', fontsize=16)
    ax.set_ylabel(selected_indicator, fontsize=16)
    ax.set_title(f'{selected_indicator} for Selected Years', fontsize=16)

    # Move the legend to the right side
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # Display the plot
    plt.show()




def visualize_custom_bar_chart(df_years, selected_indicator):
    """
    Plot a bar chart for a specific indicator.

    Parameters:
    - df_years (pd.DataFrame): DataFrame with years as columns.
    - selected_indicator (str): The indicator to plot.

    Returns:
    - None
    """

    country_list = ['France', 'India','South Africa', 'United Kingdom','China']
    years = [1990, 1992, 1994, 1996, 1998, 2000]
    x = np.arange(len(country_list))
    width = 0.35

    # Set a custom color palette
    custom_colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']

    fig, ax = plt.subplots()
    for i, year in enumerate(years):
        indicator_values = []
        for country in country_list:
            value = df_years.loc[(country, selected_indicator), year]
            indicator_values.append(value)

            # Print the data used for each bar
            print(f"{country} - {selected_indicator} ({year}): {value}")

        rects1 = ax.bar(x - width/2 + i*width/len(years), indicator_values,
                        width/len(years), label=str(year)+" ", color=custom_colors[i])

    ax.set_xlabel('Country')
    ax.set_ylabel('Value')
    ax.set_title(f'{selected_indicator} over the years')
    ax.set_xticks(x)
    ax.set_xticklabels(country_list)
    ax.legend()

    fig.tight_layout()
    plt.show()



def visualize_custom_pie_chart(df_years, selected_year, custom_indicator, country_list=None, custom_colors=None, color_palette=None):
    """
    Plot a pie chart illustrating the distribution of a custom indicator for a selected year.

    Parameters:
    - df_years (pd.DataFrame): DataFrame with years as columns.
    - selected_year (int): The specific year to plot.
    - custom_indicator (str): The custom indicator to visualize.
    - country_list (list): List of countries to include in the plot.
    - custom_colors (list): List of custom colors for the pie chart.
    - color_palette (str): Seaborn color palette name.

    Returns:
    - None
    """

    # If country_list is not provided, use a default list
    if country_list is None:
        country_list = ['France', 'India','South Africa', 'United Kingdom','China']

    # If custom_colors is not provided, use a default list of colors
    if custom_colors is None:
        custom_colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#c2c2f0', '#ffb3e6']

    # Create a subplot for the pie chart
    fig, ax = plt.subplots(figsize=(8, 8))

    # Extract values for the selected year and custom indicator for each country
    values = [df_years.loc[(country, custom_indicator), selected_year] for country in country_list]
    
    # Filter out NaN values and corresponding countries
    valid_data = [(country, value) for country, value in zip(country_list, values) if not pd.isna(value)]
    valid_countries, valid_values = zip(*valid_data)

    # Check if there are valid data points
    if not valid_data:
        raise ValueError("No valid data points for the specified custom indicator and year.")

    # Plot the pie chart with custom colors or Seaborn color palette
    if custom_colors:
        ax.pie(valid_values, labels=valid_countries, autopct='%1.1f%%', colors=custom_colors, startangle=90)
    elif color_palette:
        colors = sns.color_palette(color_palette, n_colors=len(country_list))
        ax.pie(valid_values, labels=valid_countries, autopct='%1.1f%%', colors=colors, startangle=90)
    else:
        raise ValueError("Either custom_colors or color_palette must be provided.")

    # Ensure the pie chart has an equal aspect ratio (circular)
    ax.axis('equal')

    # Set the title for the pie chart
    plt.title(f'Distribution of {custom_indicator} in {selected_year}', fontsize=16)
    
    # Display the pie chart
    plt.show()

    # Print the results to the console
    print(f"\nResults for {custom_indicator} for the selected year {selected_year}:")
    for country, value in valid_data:
        print(f"{country}: {value:.2f}%")




def analyze_indicators(df_years, selected_countries, selected_indicators):
    """
    Analyze statistical properties of indicators for individual countries and cross-compare.

    Parameters:
    - df_years (pandas.DataFrame): DataFrame containing the yearly data for indicators.
    - selected_countries (list of str): List of countries for analysis.
    - selected_indicators (list of str): List of indicators to analyze.
    """
    # Create a dictionary to store summary statistics
    summary_statistics = {}
   
    # Analyze indicators for individual countries
    for country in selected_countries:
        for indicator in selected_indicators:
            # Get data for the specific country and indicator
            country_data = df_years.loc[(country, indicator), :]

            # Calculate summary statistics using .describe() and two additional statistical methods
            country_stats = {
                'descriptive_stats': country_data.describe(),
                'median': country_data.median(),
                'standard_deviation': country_data.std(),
            }

            # Store the statistics in the dictionary
            summary_statistics[f'{country} - {indicator}'] = country_stats

    # Analyze indicators for aggregated regions or categories
    for indicator in selected_indicators:
        # Get data for the world (you can modify this for other regions/categories)
        world_data = df_years.loc[('World', indicator), :]

        # Calculate summary statistics using .describe() and two additional statistical methods
        world_stats = {
            'descriptive_stats': world_data.describe(),
            'median': world_data.median(),
            'standard_deviation': world_data.std(),
        }

        # Store the statistics in the dictionary
        summary_statistics[f'World - {indicator}'] = world_stats

    # Print the summary statistics
    for key, stats in summary_statistics.items():
        print(f"Summary Statistics for {key}:")
        print(stats['descriptive_stats'])
        print(f"Median: {stats['median']}")
        print(f"Standard Deviation: {stats['standard_deviation']}")
        print("\n" + "=" * 50 + "\n")


def analyze_correlations(df_years, countries, indicators, start_year, end_year):
    """
    Explore and understand correlations between indicators within countries and across time.

    Parameters:
    - df_years (pandas.DataFrame): DataFrame containing the yearly data for indicators.
    - countries (list of str): List of countries for analysis.
    - indicators (list of str): List of indicators to explore.
    - start_year (int): Start year for the analysis.
    - end_year (int): End year for the analysis.
    """
    # Subset the data for the specified year range
    df_filtered = filter_data_subset(df_years, countries, indicators, start_year, end_year)

    # Calculate correlations
    corr_matrix = calculate_correlation_matrix(df_filtered)

    # Visualize correlations as a heatmap
    plot_correlation_heatmap(corr_matrix)

def main():
    # Read data from the specified CSV file
    df_years, df_countries = load_and_preprocess_data(r"C:\Users\")

    # Task 1: Calculate and print summary statistics
    indicators_heatmap1 = ['Population growth (annual %)', 'Urban population growth (annual %)']
    countries_list = ['France', 'India','South Africa', 'United Kingdom','China']
    start_year = 1990
    end_year = 2000

    # Task 2: Explore and visualize correlations
    indicators_heatmap2 = ['Urban population', 'Population, total']
  
    # Explore and visualize correlations for Task 2
    analyze_correlations(df_years, countries_list, indicators_heatmap1, start_year, end_year)
    analyze_correlations(df_years, countries_list, indicators_heatmap2, start_year, end_year)

    # Task 3: Explore indicators with six plots
    analyze_indicators(df_years, countries_list, indicators_heatmap1)
    analyze_indicators(df_years, countries_list, indicators_heatmap2)

    # Specify the selected years for plotting line charts
    selected_years = [1990, 1992, 1994, 1996, 1998, 2000]

    # Example: Call the plot_line_CO2 emissions from solid fuel consumption (% of total)
    line_selected_indicator1 = 'Energy use (kg of oil equivalent per capita)'
    visualize_line_chart(df_years, selected_years, line_selected_indicator1)
    
   
   
   # Example: Call the plot_line_CO2 emissions from liquid fuel consumption (% of total)
    line_selected_indicator2 = 'Electricity production from renewable sources, excluding hydroelectric (% of total)'
    visualize_line_chart(df_years, selected_years, line_selected_indicator2)
   
    
    # Call the modified function with a specific indicator for urban population growth
    selected_indicator_urban_population_growth = 'Agricultural land (% of land area)'
    visualize_custom_bar_chart(df_years, selected_indicator_urban_population_growth)
  
    
    selected_indicator_urban_population_growth = 'Forest area (% of land area)'
    visualize_custom_bar_chart(df_years, selected_indicator_urban_population_growth)

     
    selected_year_pie = 1990  # Replace with the desired year
    selected_indicator_pie = 'Population growth (annual %)'
    color_palette = 'Set2'  # Replace with the desired color palette
    visualize_custom_pie_chart(df_years, selected_year_pie, selected_indicator_pie, color_palette=color_palette)
    
    selected_year_pie = 2000  # Replace with the desired year
    selected_indicator_pie = 'Population growth (annual %)'
    color_palette = 'deep'  # Replace with the desired color palette
    visualize_custom_pie_chart(df_years, selected_year_pie, selected_indicator_pie, color_palette=color_palette)
    

if __name__ == '__main__':
    main()
