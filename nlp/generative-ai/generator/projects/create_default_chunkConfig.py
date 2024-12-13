import os
import json
import sys

# Default settings for each file
default_config = {
    "max_words": 500,
    "overlap": 0,
    "separator": r"(?<=[.!?])\s+"
}

def generate_file_config(project_name):

    directory_path = f"{project_name}/knowledge/"

    # Dictionary to hold the configurations for each file
    file_configs = {}

    # Walk through the directory and subdirectories
    for root, _, files in os.walk(directory_path):
        for filename in files:
            if filename.endswith(('.md', '.pdf', '.docx')):
                # Create a relative file path
                file_path = os.path.join(os.path.basename(root), filename)
                file_configs[file_path] = default_config.copy()

    # Write the configurations to a JSON file in the current directory
    json_path = os.path.join(f"{project_name}/config/chunk-strategy.json")
    if file_configs:  # Only write if there are configurations
        with open(json_path, 'w') as json_file:
            json.dump(file_configs, json_file, indent=4)
        print(f"Configurations saved to {json_path}")

# Check if a directory path was provided as an argument
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python generate_file_config.py <project_name>")
        sys.exit(1)

    # Get the directory path from command-line arguments
    directory_path = sys.argv[1]

    # Run the function with the provided directory path
    generate_file_config(directory_path)
