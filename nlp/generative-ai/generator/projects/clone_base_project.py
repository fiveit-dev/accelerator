import shutil
import os
import sys

def clone_folder(source_folder, destination_folder):
    """
    Clone a folder and its subdirectories to a new location.

    Parameters:
    source_folder (str): The path to the folder to clone.
    destination_folder (str): The path to the destination folder.
    """
    # Ensure the source folder exists
    if not os.path.exists(source_folder):
        print(f"Source folder '{source_folder}' does not exist.")
        sys.exit(1)

    # Ensure the destination folder is empty or doesn't exist
    if os.path.exists(destination_folder):
        print(f"Destination folder '{destination_folder}' already exists. Choose another destination.")
        sys.exit(1)

    try:
        # Clone the folder
        shutil.copytree(source_folder, destination_folder)
        print(f"Folder '{source_folder}' cloned successfully to '{destination_folder}'.")
    except Exception as e:
        print(f"An error occurred while cloning the folder: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Ensure the script has the correct number of arguments
    if len(sys.argv) != 3:
        print("Usage: python clone_folder.py <source_folder> <destination_folder>")
        sys.exit(1)

    # Parse command-line arguments
    source_folder = sys.argv[1]
    destination_folder = sys.argv[2]

    # Run the cloning function
    clone_folder(source_folder, destination_folder)
