#this is an extra credit assignment
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Define the path to your dataset file
dataset_file_path = "/workspaces/Extra-Credit/MOMv3.txt"

# Define the headers you want to assign
headers = [
    "Continent", "Status", "Order", "Family", "Genus", "Species",
    "Log Mass (grams)", "Combined Mass (grams)", "Reference"
]

# Read the dataset with assigned headers using Pandas
df = pd.read_csv(dataset_file_path, sep='\t', header=None, names=headers)

# Print the first few rows of the DataFrame to verify the headers
print(df.head())


# Print the first 5 rows of the dataset
print("\nFirst 5 rows of the dataset:")
print(df.head(5))

# finding the number of extinct and extant species
# Separate data into extinct and extant components
extinct_species = df[df['Status'] == 'extinct']
extant_species = df[df['Status'] == 'extant']

# Count the number of species in each category
num_extinct_species = len(extinct_species)
num_extant_species = len(extant_species)

# Print the results
print(f"Number of extinct species: {num_extinct_species}")
print(f"Number of extant species: {num_extant_species}")

# Get the total number of species in the dataset
total_species = len(df)

# Print the result
print(f"Total number of species in the dataset: {total_species}")

# Get the number of unique families in the dataset
num_unique_families = df['Family'].nunique()

# Print the result
print(f"Number of unique families in the dataset: {num_unique_families}")

# Replace -999 with NaN in the 'Combined Mass (grams)' column
df['Combined Mass (grams)'].replace(-999, np.nan, inplace=True)

# Find the index of the largest and smallest species based on mass
index_largest_species = df['Combined Mass (grams)'].idxmax()
index_smallest_species = df['Combined Mass (grams)'].idxmin()

# Get information about the largest and smallest species
largest_species_info = df.loc[index_largest_species, ['Genus', 'Species', 'Combined Mass (grams)']]
smallest_species_info = df.loc[index_smallest_species, ['Genus', 'Species', 'Combined Mass (grams)']]

# Print the results
print("Information about the largest species:")
print(largest_species_info)

print("\nInformation about the smallest species:")
print(smallest_species_info)

# Filter data for Chiroptera Order and Pteropodidae Family
filtered_df = df[(df['Order'] == 'Chiroptera') & (df['Family'] == 'Pteropodidae')]

# Calculate mean, median, and mode mass for each genus
grouped = filtered_df.groupby('Genus')['Combined Mass (grams)'].agg(['mean', 'median'])

# Calculate mode manually for each genus
def calculate_mode(series):
    return series.mode().iloc[0] if not series.mode().empty else np.nan

mode_series = filtered_df.groupby('Genus')['Combined Mass (grams)'].apply(calculate_mode)

# Combine mode results with other statistics
grouped['mode'] = mode_series

# Drop rows with NaN mass values
grouped_cleaned = grouped.dropna()

# Plot bar graphs
plt.figure(figsize=(10, 6))
x_labels = grouped_cleaned.index
x_pos = np.arange(len(x_labels))
# Plot the mean, median, and mode mass for each genus
plt.bar(x_pos - 0.2, grouped_cleaned['mean'], yerr=grouped_cleaned['mean'].std(), width=0.2, label='Mean')
plt.bar(x_pos, grouped_cleaned['median'], yerr=grouped_cleaned['median'].std(), width=0.2, label='Median')
plt.bar(x_pos + 0.2, grouped_cleaned['mode'], width=0.2, label='Mode')
# Add axis labels and title
plt.xlabel('Genus')
plt.ylabel('Mass (grams)')
plt.title('Mean, Median, and Mode Mass of Each Genus in Pteropodidae Family')
plt.xticks(x_pos, x_labels, rotation=45)
plt.legend()
plt.tight_layout()
#save the plot as a png file
plt.savefig('plot.png')
