#this is an extra credit assignment
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Define the path to the dataset file 
dataset_file_path = "/workspaces/Extra-Credit/MOMv3.txt"

# Define the headers to assign to the dataset
headers = [
    "Continent", "Status", "Order", "Family", "Genus", "Species",
    "Log Mass (grams)", "Combined Mass (grams)", "Reference"
]

# Read the dataset with assigned headers using Pandas
df = pd.read_csv(dataset_file_path, sep='\t', header=None, names=headers)


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
    # Calculate mode using scipy's mode() function
    # If mode is empty, return NaN, else return the first mode value
    return series.mode().iloc[0] if not series.mode().empty else np.nan

# Apply the calculate_mode function to each genus and create a mode series
mode_series = filtered_df.groupby('Genus')['Combined Mass (grams)'].apply(calculate_mode)

# Combine mode results with mean and median statistics
grouped['mode'] = mode_series

# Drop rows with NaN mass values, keeping only cleaned data
grouped_cleaned = grouped.dropna()

# Create subplots for mean, median, and mode
fig, axs = plt.subplots(3, 1, figsize=(8, 18))

# Plot the mean mass with standard deviation error bars
axs[0].bar(grouped_cleaned.index, grouped_cleaned['mean'], yerr=grouped_cleaned['mean'].std())
axs[0].set_title('Mean Mass')
axs[0].set_xlabel('Genus')
axs[0].set_ylabel('Mass (grams)')
axs[0].tick_params(axis='x', rotation=45)

# Plot the median mass with standard deviation error bars
axs[1].bar(grouped_cleaned.index, grouped_cleaned['median'], yerr=grouped_cleaned['median'].std())
axs[1].set_title('Median Mass')
axs[1].set_xlabel('Genus')
axs[1].set_ylabel('Mass (grams)')
axs[1].tick_params(axis='x', rotation=45)

# Plot the mode mass
axs[2].bar(grouped_cleaned.index, grouped_cleaned['mode'])
axs[2].set_title('Mode Mass')
axs[2].set_xlabel('Genus')
axs[2].set_ylabel('Mass (grams)')
axs[2].tick_params(axis='x', rotation=45)

# Set y-label for all subplots
for ax in axs:
    ax.set_ylabel('Mass (grams)')

# Adjust layout for better visualization
plt.tight_layout()

# Save the combined plot as an image
plt.savefig('combined_charts.png')

# Filter data for Carnivora Order, Felidae Family, and Felis Genus
filtered_df = df[(df['Order'] == 'Carnivora') & (df['Family'] == 'Felidae') & (df['Genus'] == 'Felis')]

# Create a box plot
plt.figure(figsize=(10, 6))

# Create the box plot with mean indicators and descriptions
bplot = plt.boxplot(
    [group['Combined Mass (grams)'] for continent, group in filtered_df.groupby('Continent')],
    vert=False,
    showmeans=True,
    meanprops=dict(marker='o', markeredgecolor='black', markersize=8),
    patch_artist=True
)

# Set y-axis labels to continent names
plt.yticks(range(1, len(filtered_df['Continent'].unique()) + 1), filtered_df['Continent'].unique())

# Add title and labels
plt.xlabel('Mass (grams)')
plt.ylabel('Continent')
plt.title('Mass of Felis (Genus) Across Continents')

# Add legend for mean indicators
plt.legend([bplot["means"][0]], ['Mean'], loc="upper right")

# Add grid
plt.grid()

# Adjust layout
plt.tight_layout()

# Save the box plot as an image
plt.savefig('felis_mass_boxplot.png')

# Filter data for rows with non-negative 'Combined Mass (grams)'
filtered_df = df[df['Combined Mass (grams)'] >= 0]

# Group b 'Genus' and count the number of unique 'Continent' values for each genus
genus_continents_count = filtered_df.groupby('Genus')['Continent'].nunique()

# Count the number of genera with more than one unique 'Continent' value
genera_in_multiple_continents = (genus_continents_count > 1).sum()

print(f"Number of genera found in multiple continents: {genera_in_multiple_continents}")

# Filter data for Canis Genus
canis_df = df[df['Genus'] == 'Canis']


# Filter data for rows with non-negative 'Combined Mass (grams)'
filtered_df = df[df['Combined Mass (grams)'] >= 0]

# Group b 'Genus' and count the number of unique 'Continent' values for each genus
genus_continents_count = filtered_df.groupby('Genus')['Continent'].nunique()

# Count the number of genera with more than one unique 'Continent' value
genera_in_multiple_continents = (genus_continents_count > 1).sum()

print(f"Number of genera found in multiple continents: {genera_in_multiple_continents}")

# Filter data for Canis Genus
canis_df = df[df['Genus'] == 'Canis']

# Convert 'Continent' column values to strings
canis_df['Continent'] = canis_df['Continent'].astype(str)

# Create a scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(canis_df['Continent'], canis_df['Combined Mass (grams)'], alpha=0.7)
plt.xlabel('Continent')
plt.ylabel('Combined Mass (grams)')
plt.title('Mass of Genus "Canis" Across Continents')
plt.xticks(rotation=45)
plt.tight_layout()

# Save the scatter plot as an image
plt.savefig('canis_mass_scatter_plot.png')