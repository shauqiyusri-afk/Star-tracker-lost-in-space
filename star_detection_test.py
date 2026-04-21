import cv2
import numpy as np


def grow_region(img, reg_img, row, col, thresh, region_count):
    """
    8-neighbour recursive region growing.
    """
    r, c = img.shape

    neighbors = [
        (0, 1), (1, 0), (0, -1), (-1, 0),
        (1, 1), (1, -1), (-1, -1), (-1, 1)
    ]

    reg_img[row, col] = region_count

    for dx, dy in neighbors:
        rown = row + dx
        coln = col + dy

        if 0 <= rown < r and 0 <= coln < c:
            if reg_img[rown, coln] == 0 and img[rown, coln] > thresh:
                reg_img[rown, coln] = region_count
                grow_region(img, reg_img, rown, coln, thresh, region_count)

    return reg_img


def centroid(img, reg_img, region_count):
    """
    Brightness-weighted centroid of one labeled region.
    """
    rows, cols = img.shape

    num_row = 0.0
    num_col = 0.0
    den = 0.0

    for i in range(rows):
        for j in range(cols):
            if reg_img[i, j] == region_count:
                intensity = float(img[i, j])
                num_row += i * intensity
                num_col += j * intensity
                den += intensity

    if den == 0:
        return 0.0, 0.0

    x = num_col / den
    y = num_row / den
    return x, y


def detection(img, thresh):
    """
    Detect bright regions and compute centroids.
    """
    rows, cols = img.shape

    region_count = 0
    reg_img = np.zeros_like(img, dtype=np.int32)

    for i in range(rows):
        for j in range(cols):
            if reg_img[i, j] == 0 and img[i, j] > thresh:
                region_count += 1
                reg_img = grow_region(img, reg_img, i, j, thresh, region_count)

    cent = np.zeros((region_count, 2), dtype=float)
    for region_id in range(1, region_count + 1):
        cent[region_id - 1, 0], cent[region_id - 1, 1] = centroid(img, reg_img, region_id)

    return reg_img, region_count, cent


def main():
    image_path = r"C:\Users\shauqi\Desktop\Startracker\images\test.png"   # change this to your image filename
    threshold = 180                      # adjust if needed

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(f"Error: could not load image '{image_path}'")
        return

    reg_img, region_count, centroids = detection(img, threshold)

    print("Detected regions:", region_count)
    print("Centroids:")
    for idx, (x, y) in enumerate(centroids, start=1):
        print(f"Region {idx}: x={x:.2f}, y={y:.2f}")

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

    cv2.imwrite(r"C:\Users\shauqi\Desktop\Startracker\images\detected_output.png", output)
    print("Saved output image as detected_output.png")


if __name__ == "__main__":
    main()