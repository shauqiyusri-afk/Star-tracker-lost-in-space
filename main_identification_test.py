import cv2
import numpy as np
import os

from detection import detection
from pixel_to_vector import pixel_to_unit_vectors
from build_tab_image import build_tab_image
from read_catalog import read_catalog
from gen_table import gen_table
from solver_utils import run_gvalgo_variant, solve_with_all_methods, quaternion_angle_difference_deg

image_path = r"C:\Users\shauqi\Desktop\Startracker\images\test.png"
catalog_path = r"C:\Users\shauqi\Desktop\Startracker\asu_hipparcos_catalog_ra_dec_deg_wo_header.txt"

# Parameters
GV_ALGO = "ransac_hough"   # base, ransac, hough, ransac_hough
threshold = 180
fov_deg = 12.0
min_sep_deg = 0.25
limiting_mag = 5.2
loc_err_deg = 0.10
min_votes = 2
max_solver_stars = 20   # use only strongest unique matches for QUEST / Davenport

print("=== Identification Test Start ===")
print(f"GV_ALGO={GV_ALGO}, threshold={threshold}, fov_deg={fov_deg}, "
      f"min_sep_deg={min_sep_deg}, limiting_mag={limiting_mag}, "
      f"loc_err_deg={loc_err_deg}, min_votes={min_votes}, "
      f"max_solver_stars={max_solver_stars}")

img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if img is None:
    print("ERROR: Could not load image.")
    raise SystemExit

reg_img, region_count, centroids = detection(img, threshold)
print("Detected regions:", region_count)

star_vectors = pixel_to_unit_vectors(centroids, img.shape, fov_deg)

tab_image = build_tab_image(star_vectors, fov_deg, min_sep_deg)
print("tab_image entries:", len(tab_image))
if len(tab_image) == 0:
    print("ERROR: No valid image star pairs found.")
    raise SystemExit

cat = read_catalog(catalog_path, limiting_mag)
tab_cat = gen_table(cat, fov_deg, min_sep_deg)
tab_cat = tab_cat[np.argsort(tab_cat[:, 2])]
print("tab_cat entries:", len(tab_cat))
if len(tab_cat) == 0:
    print("ERROR: No valid catalog star pairs found.")
    raise SystemExit

ids, v2 = run_gvalgo_variant(GV_ALGO, cat, tab_cat, tab_image, loc_err_deg, star_vectors=star_vectors)
print(f"\nUsing gvalgo variant: {GV_ALGO}")

print("\nFirst 20 estimated catalog row IDs:")
print(ids[:20])

print("\nFirst 20 second-round votes:")
print(v2[:20])

hip_ids = []
for row_id in ids:
    if row_id > 0:
        hip_ids.append(int(cat[row_id - 1, 1]))
    else:
        hip_ids.append(0)

print("\nFirst 20 estimated HIP IDs:")
print(hip_ids[:20])

print("\nRaw match statistics")
print("Non-zero identified stars:", np.sum(ids != 0), "/", len(ids))
print("Stars with second-round support > 0:", np.sum(v2 > 0), "/", len(v2))
print(f"Stars with second-round support >= {min_votes}:", np.sum(v2 >= min_votes), "/", len(v2))

# ------------------------------------------------------------------
# FIX 1: remove duplicate catalog matches (one image star per catalog row)
# keep the match with the highest vote; if tied, keep the first one
# ------------------------------------------------------------------
best_match_for_row = {}
for i, row_id in enumerate(ids):
    if row_id <= 0:
        continue

    vote = v2[i]
    if row_id not in best_match_for_row:
        best_match_for_row[row_id] = (i, vote)
    else:
        prev_i, prev_vote = best_match_for_row[row_id]
        if vote > prev_vote:
            best_match_for_row[row_id] = (i, vote)

keep_unique_mask = np.zeros(len(ids), dtype=bool)
for row_id, (i, vote) in best_match_for_row.items():
    keep_unique_mask[i] = True

ids_unique = np.where(keep_unique_mask, ids, 0)
v2_unique = np.where(keep_unique_mask, v2, 0)

# ------------------------------------------------------------------
# FIX 2: use only strongest unique matches for the solvers
# sort by vote descending and keep top N
# ------------------------------------------------------------------
candidate_indices = [i for i in range(len(ids_unique)) if ids_unique[i] > 0 and v2_unique[i] >= min_votes]
candidate_indices = sorted(candidate_indices, key=lambda i: v2_unique[i], reverse=True)

selected_indices = candidate_indices[:max_solver_stars]

solver_mask = np.zeros(len(ids_unique), dtype=bool)
solver_mask[selected_indices] = True

ids_solver = np.where(solver_mask, ids_unique, 0)
v2_solver = np.where(solver_mask, v2_unique, 0)

# Diagnostics after filtering
hip_ids_unique = [int(cat[row_id - 1, 1]) for row_id in ids_unique if row_id > 0]
hip_ids_solver = [int(cat[ids_solver[i] - 1, 1]) for i in range(len(ids_solver)) if ids_solver[i] > 0]

print("\nAfter unique-match filtering")
print("Unique identified stars:", np.sum(ids_unique != 0), "/", len(ids_unique))
print(f"Unique stars with v2 >= {min_votes}:", np.sum(v2_unique >= min_votes), "/", len(v2_unique))

print("\nAfter solver top-N filtering")
print("Solver-selected stars:", np.sum(ids_solver != 0), "/", len(ids_solver))
print("Top solver HIP IDs:")
print(hip_ids_solver[:max_solver_stars])

solver_results = solve_with_all_methods(cat, star_vectors, ids_solver, v2_solver, min_votes=min_votes)
for solver_name, result in solver_results.items():
    print(f"\n{solver_name} used stars with v2 >= {min_votes}")
    print("Reliable stars used:", result["used"])
    if result["q"] is not None:
        print("Estimated quaternion q = [q0 q1 q2 q3]:")
        print(result["q"])
    else:
        print(f"{solver_name} failed or rejected:", result["error"])

if solver_results["QUEST"]["q"] is not None and solver_results["Davenport Q"]["q"] is not None:
    angle_diff_deg = quaternion_angle_difference_deg(
        solver_results["QUEST"]["q"],
        solver_results["Davenport Q"]["q"]
    )
    print("\n=== QUEST vs Davenport Q Comparison ===")
    print("QUEST used stars:    ", solver_results["QUEST"]["used"])
    print("Davenport used stars:", solver_results["Davenport Q"]["used"])
    print("Angular difference (deg):", angle_diff_deg)

annotated_path = r"C:\Users\shauqi\Desktop\Startracker\images\identified_output.png"
output = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

# Plot unique matches, not raw matches
for idx, (x, y) in enumerate(centroids, start=1):
    cx = int(round(x))
    cy = int(round(y))

    row_id = ids_unique[idx - 1]
    vote = v2_unique[idx - 1]

    circle_color = (128, 128, 128)
    text = ""

    if row_id > 0:
        hip_id = int(cat[row_id - 1, 1])

        if vote >= min_votes:
            circle_color = (0, 255, 0)
            text = f"HIP {hip_id} [v{vote}]"
        elif vote > 0:
            circle_color = (0, 255, 255)
            text = f"HIP {hip_id} [v{vote}]"
        else:
            circle_color = (0, 0, 255)
            text = f"HIP {hip_id} [v{vote}]"

    cv2.circle(output, (cx, cy), 6, circle_color, 1)

    if text != "":
        cv2.putText(output, text, (cx + 5, cy - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, circle_color, 1, cv2.LINE_AA)

os.makedirs(os.path.dirname(annotated_path), exist_ok=True)
cv2.imwrite(annotated_path, output)

print(f"\nSaved identified image to: {annotated_path}")
print(f"Green  = strong match (v2 >= {min_votes})")
print("Yellow = weak match (v2 > 0)")
print("Red    = candidate only (v2 = 0)")
print("Gray   = detected star with no catalog assignment")
print("=== Identification Test End ===")