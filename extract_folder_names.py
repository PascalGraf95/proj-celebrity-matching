import os


def extract_folder_names(directory, output_file):
    with open(output_file, 'w', encoding="utf-8") as f:
        for item in os.listdir(directory):
            item_path = os.path.join(directory, item)
            if os.path.isdir(item_path):
                f.write(item + '\n')

if __name__ == "__main__":
    directory_path = "downloaded_images"  # Replace with the actual directory path
    output_file_path = "folder_names.txt"  # Name of the output text file

    extract_folder_names(directory_path, output_file_path)
    print("Folder names extracted and saved to", output_file_path)