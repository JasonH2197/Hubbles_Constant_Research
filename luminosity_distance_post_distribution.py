import os
import json
from parameter_post_distribution_graph import graph_posterior_distribution

# Search for the first JSON file in the current working directory,
# and load its content into the 'data' variable,
current_directory = os.getcwd()
files_in_directory = os.listdir(current_directory)
json_file_path = None

for filename in files_in_directory:
    if filename.endswith(".json"):
        json_file_path = os.path.join(current_directory, filename)
        break

if json_file_path is None:
    print("No JSON files found in the current directory.")
else:
    data = json_file_path

# Call the function to graph the luminosity distance
graph_posterior_distribution(data, 'luminosity_distance')

