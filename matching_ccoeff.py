import cv2
import csv
import glob
import yaml
import numpy as np
from tqdm import tqdm


# Read config file
with open('config.yml', "r") as f:
    config = yaml.safe_load(f)


# Define parameters
target_img_path = config['storage']['target_img_path']
objects_dir = config['storage']['objects_dir']
output_dir = config['storage']['output_dir']
threshold = config['ccoeff']['threshold']


# Define a dictionary that maps object classes to colors
opacity = 150
colors = {
    'base_cabinet': (0, 255, 255, opacity),
    'wall_cabinet': (0, 255, 255, opacity),
    'countertop': (0, 255, 255, opacity),
    'shelf': (0, 255, 255, opacity),
    'sink': (0, 255, 255, opacity),
    'tall_cabinet': (0, 255, 255, opacity)
}


# Initialize a dictionary to keep track of the count for each object class
counts = {}
for object_type in colors.keys():
    counts[object_type] = 0

# Read all the object images and store them in a dictionary
templates = {}
for object_type in ['base_cabinet', 'wall_cabinet', 'countertop', 'shelf', 'sink', 'tall_cabinet']:
    object_dir = objects_dir + object_type + '/'
    object_files = glob.glob(object_dir + '*.png')
    for object_file in object_files:
        object_name = object_file.split('/')[-1].split('.')[0]
        object_image = cv2.imread(object_file, 0)
        templates[f'{object_type}-{object_name}'] = object_image

# Load the target image
target_img = cv2.imread(target_img_path, 0)

# Detect the objects in the target image using template matching
with open(output_dir + 'object_coordinates_CCOEFF.csv', mode='w') as file:
    writer = csv.writer(file)
    writer.writerow(['object_name', 'x1', 'y1', 'x2', 'y2'])
    for object_name, template in tqdm(templates.items()):
        object_class = object_name.split('-')[0]

        result = cv2.matchTemplate(target_img, template, cv2.TM_CCOEFF_NORMED)
        locations = np.where(result >= threshold)
        color = colors[object_class]

        for pt in zip(*locations[::-1]):
            # Draw a bounding box around the detected object with transparency
            overlay = target_img.copy()
            cv2.rectangle(overlay, pt, (pt[0] + template.shape[1], pt[1] + template.shape[0]), color, cv2.FILLED)
            cv2.addWeighted(overlay, 0.5, target_img, 0.5, 0, target_img)

            # Write the object coordinates to a CSV file
            writer.writerow([object_name, pt[0], pt[1], pt[0] + template.shape[1], pt[1] + template.shape[0]])

            x1, y1 = pt[0], pt[1]
            x2, y2 = pt[0] + template.shape[1], pt[1] + template.shape[0]

            # Add object number and class name to the image
            label = f"{object_class.capitalize()} {counts[object_class]}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.9
            font_thickness = 1
            (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
            cv2.rectangle(target_img, (x1, y1 - text_height), (x1 + text_width, y1), color[:3], -1)
            cv2.putText(target_img, label, (x1, y1), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

            # Update object counts
            counts[object_class] += 1

# Save the marked-up image
cv2.imwrite(output_dir + 'room_marked_up_CCOEFF.jpg', target_img)


# Print summary statistics
print("Object detection summary:")
print("-------------------------")
total_objects = sum(counts.values())
print(f"Total number of objects detected: {total_objects}")
for object_class, count in counts.items():
    print(f"{object_class.capitalize()}: {count}")

print("\nArtifcats:")
print("----------")
print(f"CSV with detections: {output_dir + 'object_coordinates_CCOEFF.csv'}")
print(f"Image with marked objects: {output_dir + 'drawing_marked_up_CCOEFF.jpg'}")
