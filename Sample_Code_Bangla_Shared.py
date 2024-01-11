import os

# List of required packages
required_packages = ['pandas', 'numpy', 'seaborn', 'matplotlib']

# Install missing packages
for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        print(f"Installing {package}...")
        os.system(f"pip install {package}")

# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


# Library from cod-package
from cod_prep.claude.formatting import finalize_formatting, update_nid_metadata_status
from cod_prep.downloaders.ages import get_cod_ages
from cod_prep.claude.claude_io import get_claude_data

from cod_prep.downloaders import (
    get_current_cause_hierarchy,
    add_cause_metadata,
    get_cod_ages,
    add_age_metadata,
    get_all_related_causes,
    add_nid_metadata, 
    get_cause_map
)



def preprocess_data(df_bangla, iso3,cause_level):
    # Read the data from data/Bangladesh_sources.csv
    df_bangla = pd.read_csv(df_bangla)
    
    # Fetch data using get_claude_data with specific filters
    df = get_claude_data("formatted", iso3=iso3, is_active=True, nid=df_bangla.nid.unique().tolist(),
                         data_type_id=df_bangla.data_type_id.unique().tolist(),
                         year_id=df_bangla.year_id.unique().tolist())
    
    # Add NID metadata to the dataframe
    df = add_nid_metadata(df, "code_system_id")
    
    # Fetch unique code system IDs and build a code map
    csids = df["code_system_id"].unique()
    code_map_list = []  # Initialize a list to store code maps

    for id in csids:
        cm = get_cause_map(id)
        code_map_list.append(cm)  # Append each code map to the list

    code_map = pd.concat(code_map_list, ignore_index=True)  # Concatenate code maps

    
    # Merge code map into the dataframe
    df = df.merge(code_map[['code_id', "cause_id", "code_system_id", "code_system"]],
                  how="left", on=["code_id", "code_system_id"])
    
    # Fetch the current cause hierarchy
    ch = get_current_cause_hierarchy()
    
    # Merge cause hierarchy data into the dataframe
    df = df.merge(ch[["cause_id", "acause", "level", "path_to_top_parent"]],
                  how="left", on="cause_id")
    
    # Split the path_to_top_parent into separate columns
    df[["all_cause", "level_1", "level_2", "level_3", "level_4", "level_5"]] = df['path_to_top_parent'].str.split(pat=',', n=5, expand=True)

    # Conditioned the level of cause list detail by 2
    if cause_level == "Level 2":
        
        # Fill missing values and convert columns to appropriate types
        df["cause_id_level_2"] = df["level_2"].fillna(df["level_1"]).astype(int)
        df["cause_id_level_3"] = df["level_3"].fillna(df["level_2"]).fillna(df["level_1"]).astype(int)

        # Rename cause_id to cause_id_level_2 for merge
        ch2 = ch.rename(columns={"cause_id": "cause_id_level_2", "acause": "acause_level_2"})
    
        # Merge cause hierarchy data for level 2
        df = df.merge(ch2[["cause_id_level_2", "acause_level_2", "cause_name"]],
                      on="cause_id_level_2", how="left")

    # Conditioned the level of cause list detail by 2    
    elif cause_level == "Level 3":
        
        df["cause_id_level_3"] = df["level_3"].fillna(df["level_2"])
        df["cause_id_level_3"] = df["cause_id_level_3"].fillna(df["level_1"])
        df["cause_id_level_3"] = df["cause_id_level_3"].astype(int)
        # Rename cause_id to cause_id_level_3 for merge
        ch3 = ch.rename(columns={"cause_id": "cause_id_level_3", "acause": "acause_level_3"})

        # Merge cause hierarchy data for level 3
        df = df.merge(ch3[["cause_id_level_3", "acause_level_3", "cause_name"]],
                      on="cause_id_level_3", how="left")
    else:
        pass
    
    
    # Define age groups
    neonatal = [-1, 2, 3, 4, 28, 238, 388, 389, 42, 179]
    under_five = [1, 5]
    five_to_fifteen = [5, 6, 7, 23]
    fifteen_to_fifty = [8, 9, 10, 11, 12, 13, 14, 24, 30, 31, 32, 34, 156]
    fifty_plus = [16, 17, 18, 19, 20, 21, 160, 235, 154, 225]
    All_age = [22]
    
    # Function to map age group IDs to names
    def assign_name(age):
        if age in neonatal:
            return 'neonatal'
        elif age in under_five:
            return 'under_five'
        elif age in five_to_fifteen:
            return 'five_to_fifteen'
        elif age in fifteen_to_fifty:
            return 'fifteen_to_fifty'
        elif age in All_age:
            return 'All age'
        else:
            return 'fifty_plus'
    
    # Apply the function to create a new column for age group names
    df['Age_Group_name'] = df['age_group_id'].apply(assign_name)
    
    return df




# Define the function to generate cause proportions
def generate_cause_prop(df, results):
    if results == "for_heat_map":
        # Calculate the total deaths for each source (code_system) and year_id
        total_deaths_by_code_system_and_year = df.groupby(['code_system', 'year_id'])['deaths'].sum()
        df['total_deaths_by_code_system_and_year'] = df.apply(lambda row: total_deaths_by_code_system_and_year[row['code_system'], row['year_id']], axis=1)

        # Calculate the cause fraction (Cf) for each row
        df['Cf_source_year'] = (df['deaths'] / df['total_deaths_by_code_system_and_year']) * 100

        # Calculate the total deaths for each source (code_system)
        total_deaths_by_code_system = df.groupby('code_system')['deaths'].sum()
        df['total_deaths_by_code_system'] = df['code_system'].map(total_deaths_by_code_system)

        # Calculate the cause fraction (Cf) for each row
        df['Cf_source'] = (df['deaths'] / df['total_deaths_by_code_system']) * 100

    elif results == "CSV":
        df = df.groupby(['code_system', "year_id", "cause_name", "Age_Group_name"])["deaths"].sum().reset_index()
        
        total_deaths_by_code_system_and_year = df.groupby(['code_system', 'year_id'])['deaths'].sum()
        df['total_deaths_by_code_system_and_year'] = df.apply(lambda row: total_deaths_by_code_system_and_year[row['code_system'], row['year_id']], axis=1)
        df['Cf_source_year'] = (df['deaths'] / df['total_deaths_by_code_system_and_year']) * 100
        total_deaths_by_code_system = df.groupby(['code_system'])['deaths'].sum()
        df['total_deaths_by_code_system'] = df['code_system'].map(total_deaths_by_code_system)
        df['Cf_source'] = (df['deaths'] / df['total_deaths_by_code_system']) * 100

        # Select relevant columns for result dataframe
        df = df[["code_system", "cause_name", "Age_Group_name", "year_id", "deaths",
                 "total_deaths_by_code_system_and_year", "Cf_source_year",
                 "total_deaths_by_code_system", "Cf_source"]]

        # Group by specified columns and calculate aggregates
        df = df.groupby(['code_system', 'Age_Group_name', "year_id", "cause_name"])[
            "deaths", "total_deaths_by_code_system_and_year", "Cf_source_year",
            "total_deaths_by_code_system", "Cf_source"].sum().reset_index()
    else:
        pass

    return df

def generate_heatmap(df, output):
    # Define the desired order of age groups
    age_group_order =['neonatal','under_five','five_to_fifteen','fifteen_to_fifty','fifty_plus',
                      'All age']

    # Group by cause_name, age group, and code_system to get total deaths (Cf) and source deaths (Cf_source)
    grouped_data = df.groupby(['cause_name', 'Age_Group_name', 'code_system'])[str(output)].sum().reset_index()

    # Create a PDF to save the heatmaps
    pdf_filename = f"Heatmap_output:{output}.pdf"
    with PdfPages(pdf_filename) as pdf:
        # Iterate through each unique cause_name
        for idx, cause_name in enumerate(grouped_data['cause_name'].unique(), 1):
            # Filter the data for the current cause_name
            cause_data = grouped_data.loc[grouped_data['cause_name'] == cause_name]

            # Set the desired order for age groups
            # Set the desired order for age groups
            cause_data.loc[:, 'Age_Group_name'] = pd.Categorical(cause_data['Age_Group_name'], categories=age_group_order, ordered=True)

            # Pivot the table for better visualization
            pivot_table = cause_data.pivot_table(index='code_system', columns='Age_Group_name',
                                                 values= str(output), aggfunc=np.sum, fill_value=np.nan)

            # Rearrange the pivot table to show source names on the y-axis
            pivot_table = pivot_table.reindex(df['code_system'].unique(), axis=0)
            # Rearrange the pivot table to show age group names on the x-axis
            pivot_table = pivot_table.reindex(age_group_order, axis=1)

            # Create the heatmap
            plt.figure(figsize=(17, 12))
            sns.heatmap(pivot_table, annot=True, fmt='.1f', cmap='YlGnBu')
            plt.title(f'BANGLADESH: Cf_source % by Age Group and Data Source for Cause: {cause_name}')
            plt.xlabel('Age Group')
            plt.ylabel('Data Source (Code System)')

            # Set the figure description
            plt.figtext(0.5, 0.95, f'Cause of death {idx} by level 2 : {cause_name}',
                        fontsize=16, ha='center')

            # Save the heatmap to the PDF
            pdf.savefig()
            plt.close()
    return

# Apply the function to get first pre process data
preprocessed_data = preprocess_data("data/Bangladesh_sources.csv", "BGD", "Level 2")



# Save the file in CSV format
df = generate_cause_prop(preprocessed_data, "CSV") #Use to export CSV

# To save the result in CSV
# Create the output directory if it doesn't exist
if not os.path.exists("output"):
    os.makedirs("output")

# Save the DataFrame to CSV in the output directory
output_path = os.path.join("output", "result.csv")
# Save the grouped result to a CSV file
df.to_csv(output_path, index = False)

# Define the data used to generate heat map
df1 = generate_cause_prop(preprocessed_data, "for_heat_map")


# generate Heat Map based on the data df1, by death number (i.e, use "Cf_source", for cause fraction)
generate_heatmap(df1, "deaths")