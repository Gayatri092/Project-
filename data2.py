import os
import csv

# Folder containing images
image_folder = "Project/potholes"
csv_file = "dataa.csv"

# Ensure the folder exists
if not os.path.exists(image_folder):
    print(f"Error: Folder '{image_folder}' not found!")
else:
    # Get all image file names (supports common image formats)
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
    
    # Write to CSV file
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Lab", "Path"])  # CSV headers

        for image in image_files:
            image_path = os.path.join(image_folder, image)  # Full path
            writer.writerow(["normal", image_path])  # "normal" as label

    print(f"CSV file '{csv_file}' created successfully with {len(image_files)} entries!")
