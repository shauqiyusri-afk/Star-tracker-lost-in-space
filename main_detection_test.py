import os
import cv2
from detection import detection
from pixel_to_vector import pixel_to_unit_vectors
from build_tab_image import build_tab_image

image_path = r"C:\Users\shauqi\Desktop\Startracker\images\test.png"
output_path = r"C:\Users\shauqi\Desktop\Startracker\images\detected_output.png"

img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    print("ERROR: Could not load the image.")
    raise SystemExit

reg_img, region_count, centroids = detection(img, 180)

print("Detected regions:", region_count)
print("Number of centroids:", len(centroids))

output = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

for idx, (x, y) in enumerate(centroids, start=1):
    cx = int(round(x))
    cy = int(round(y))

    cv2.circle(output, (cx, cy), 6, (0, 0, 255), 1)
    cv2.putText(
        output,
        str(idx),
        (cx + 5, cy - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (0, 255, 0),
        1,
        cv2.LINE_AA
    )

os.makedirs(os.path.dirname(output_path), exist_ok=True)
cv2.imwrite(output_path, output)

fov_deg = 12.0
min_sep_deg = 0.30

star_vectors = pixel_to_unit_vectors(centroids, img.shape, fov_deg)
tab_image = build_tab_image(star_vectors, fov_deg, min_sep_deg)

print("\nFirst 10 unit vectors:")
print(star_vectors[:10])

print("\nFirst 10 rows of tab_image [i, j, angle_deg]:")
print(tab_image[:10])

print("\nTotal tab_image entries:", len(tab_image))