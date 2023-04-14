import cv2
import csv
import glob
import yaml
import math
import numpy as np
from tqdm import tqdm


# Read config file
with open('config.yml', "r") as f:
    config = yaml.safe_load(f)


# Define parameters
target_img_path = config['storage']['target_img_path']
objects_dir = config['storage']['objects_dir']
output_dir = config['storage']['output_dir']
ratio_threshold = config['sift']['ratio_threshold']
angle_tolerance = config['sift']['angle_tolerance']
min_good_matches = config['sift']['min_good_matches']


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

# Initialize SIFT detector
sift = cv2.SIFT_create()

# Get the target image dimensions
height, width = target_img.shape

# Detect keypoints and compute descriptors for the target image
kp2, des2 = sift.detectAndCompute(target_img, None)

# Define the FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

# Detect the objects in the target image using SIFT
with open(output_dir + 'object_coordinates_SIFT.csv', mode='w') as file:
    writer = csv.writer(file)
    writer.writerow(['object_name', 'x1', 'y1', 'x2', 'y2'])
    for object_name, template in tqdm(templates.items()):
        object_class = object_name.split('-')[0]
        
        kp1, des1 = sift.detectAndCompute(template, None)
        matches = flann.knnMatch(des1, des2, k=2)
        
        good_matches = []
        for m, n in matches:
            if m.distance < ratio_threshold * n.distance:
                good_matches.append(m)

        if len(good_matches) > min_good_matches:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            h, w = template.shape
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)

            # Validate that the object shape is close to a rectangle using angles
            def angle_cos(p0, p1, p2):
                d1, d2 = (p0 - p1).astype('float'), (p2 - p1).astype('float')
                return abs(np.dot(d1, d2) / np.sqrt(np.dot(d1, d1) * np.dot(d2, d2)))

            dst = np.squeeze(np.int32(dst))
            angles = [angle_cos(dst[i], dst[(i + 1) % 4], dst[(i + 2) % 4]) for i in range(4)]
            min_angle = math.degrees(math.acos(max(angles)))
            max_angle = math.degrees(math.acos(min(angles)))

            if 90 - angle_tolerance <= min_angle <= 90 + angle_tolerance and \
                90 - angle_tolerance <= max_angle <= 90 + angle_tolerance:

                # Check if coordinates are within the image boundaries
                min_x, min_y = np.min(dst, axis=0)
                max_x, max_y = np.max(dst, axis=0)
                
                if min_x >= 0 and min_y >= 0 and max_x <= width and max_y <= height:
                    # Draw the detected object with the specified color and opacity
                    color = colors[object_class]
                    overlay = target_img.copy()
                    cv2.fillPoly(overlay, [np.int32(dst)], color[:3])
                    cv2.addWeighted(overlay, color[3] / 255.0, target_img, 1 - (color[3] / 255.0), 0, target_img)

                    x1, y1 = np.int32(dst[0])
                    x2, y2 = np.int32(dst[2])
                    writer.writerow([object_name, x1, y1, x2, y2])

                    # Add object number and class name to the image
                    label = f"{object_class.capitalize()} {counts[object_class]}"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.9
                    font_thickness = 1
                    (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
                    cv2.rectangle(target_img, (x1, y1 - text_height), (x1 + text_width, y1), color[:3], -1)
                    cv2.putText(target_img, label, (x1, y1), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

                    counts[object_class] += 1


# Save the marked-up image
cv2.imwrite(output_dir + 'room_marked_up_SIFT.jpg', target_img)

# Print summary statistics
print("Object detection summary:")
print("-------------------------")
total_objects = sum(counts.values())
print(f"Total number of objects detected: {total_objects}")
for object_class, count in counts.items():
    print(f"{object_class.capitalize()}: {count}")

print("\nArtifacts:")
print("----------")
print(f"CSV with detections: {output_dir + 'object_coordinates_SIFT.csv'}")
print(f"Image with marked objects: {output_dir + 'drawing_marked_up_SIFT.jpg'}")

