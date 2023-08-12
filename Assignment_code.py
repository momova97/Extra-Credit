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

# Create subplots
fig, axs = plt.subplots(3, 1, figsize=(8, 18))

# Mean chart
axs[0].bar(grouped_cleaned.index, grouped_cleaned['mean'], yerr=grouped_cleaned['mean'].std())
axs[0].set_title('Mean Mass')
axs[0].set_xlabel('Genus')
axs[0].set_ylabel('Mass (grams)')
axs[0].tick_params(axis='x', rotation=45)

# Median chart
axs[1].bar(grouped_cleaned.index, grouped_cleaned['median'], yerr=grouped_cleaned['median'].std())
axs[1].set_title('Median Mass')
axs[1].set_xlabel('Genus')
axs[1].set_ylabel('Mass (grams)')
axs[1].tick_params(axis='x', rotation=45)

# Mode chart
axs[2].bar(grouped_cleaned.index, grouped_cleaned['mode'])
axs[2].set_title('Mode Mass')
axs[2].set_xlabel('Genus')
axs[2].set_ylabel('Mass (grams)')
axs[2].tick_params(axis='x', rotation=45)

# Set y-label for all subplots
for ax in axs:
    ax.set_ylabel('Mass (grams)')

# Adjust layout
plt.tight_layout()

# Save the combined plot as an image
plt.savefig('combined_charts.png')


