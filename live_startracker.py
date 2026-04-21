import cv2
import numpy as np
import time

from detection import detection
from pixel_to_vector import pixel_to_unit_vectors
from build_tab_image import build_tab_image
from read_catalog import read_catalog
from gen_table import gen_table
from solver_utils import run_gvalgo_variant, solve_with_all_methods


# =========================
# USER SETTINGS
# =========================
GV_ALGO = "ransac_hough"   # base, ransac, hough, ransac_hough
camera_index = 0
catalog_path = r"C:\Users\shauqi\Desktop\Startracker\asu_hipparcos_catalog_ra_dec_deg_wo_header.txt"

threshold = 180
fov_deg = 12.0
min_sep_deg = 0.30
limiting_mag = 6.0
loc_err_deg = 0.05
min_votes_for_quest = 2
ENABLE_DAVENPORT = True

resize_width = 960   # set None to keep original webcam width
show_all_labels = False   # True = show weak labels too, False = only strong labels

# =========================
# LOAD CATALOG ONCE
# =========================
print("Loading catalog...")
cat = read_catalog(catalog_path, limiting_mag)

print("Generating catalog angle table...")
tab_cat = gen_table(cat, fov_deg, min_sep_deg)
tab_cat = tab_cat[np.argsort(tab_cat[:, 2])]
print("Catalog ready.")
print("tab_cat entries:", len(tab_cat))

if len(tab_cat) == 0:
    raise RuntimeError("Catalog pair table is empty.")


def process_frame(frame_bgr):
    """
    Process one frame and return annotated frame + status info.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    reg_img, region_count, centroids = detection(gray, threshold)

    if len(centroids) < 2:
        annotated = frame_bgr.copy()
        cv2.putText(
            annotated,
            "Too few stars detected",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA
        )
        return annotated, region_count, 0, 0, None, None

    star_vectors = pixel_to_unit_vectors(centroids, gray.shape, fov_deg)
    tab_image = build_tab_image(star_vectors, fov_deg, min_sep_deg)

    if len(tab_image) == 0:
        annotated = frame_bgr.copy()
        cv2.putText(
            annotated,
            "No valid star pairs",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA
        )
        return annotated, region_count, 0, 0, None, None

    ids, v2 = run_gvalgo_variant(GV_ALGO, cat, tab_cat, tab_image, loc_err_deg, star_vectors=star_vectors)

    strong_matches = int(np.sum(v2 >= min_votes_for_quest))
    weak_matches = int(np.sum(v2 > 0))

    solver_results = solve_with_all_methods(cat, star_vectors, ids, v2, min_votes_for_quest)
    q = solver_results["QUEST"]["q"]
    q_davenport = solver_results["Davenport Q"]["q"] if ENABLE_DAVENPORT else None

    annotated = frame_bgr.copy()

    # Draw detected stars
    for idx, (x, y) in enumerate(centroids, start=1):
        cx = int(round(x))
        cy = int(round(y))

        row_id = ids[idx - 1] if idx - 1 < len(ids) else 0
        vote = v2[idx - 1] if idx - 1 < len(v2) else 0

        if row_id > 0:
            hip_id = int(cat[row_id - 1, 1])

            if vote >= min_votes_for_quest:
                color = (0, 255, 0)      # green
                text = f"HIP {hip_id} [v{vote}]"
            elif vote > 0:
                color = (0, 255, 255)    # yellow
                text = f"HIP {hip_id} [v{vote}]"
            else:
                color = (0, 0, 255)      # red
                text = f"HIP {hip_id} [v0]" if show_all_labels else ""
        else:
            color = (128, 128, 128)
            text = ""

        cv2.circle(annotated, (cx, cy), 6, color, 1)

        if text != "":
            cv2.putText(
                annotated,
                text,
                (cx + 5, cy - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                color,
                1,
                cv2.LINE_AA
            )

    # Overlay summary
    cv2.putText(
        annotated,
        f"Detected: {region_count}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        annotated,
        f"Weak matches (>0): {weak_matches}",
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        annotated,
        f"Strong matches (>={min_votes_for_quest}): {strong_matches}",
        (20, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )

    if q is not None:
        q_text = f"q=[{q[0]:.3f}, {q[1]:.3f}, {q[2]:.3f}, {q[3]:.3f}]"
        cv2.putText(
            annotated,
            q_text,
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )
    else:
        cv2.putText(
            annotated,
            "QUEST: not enough reliable stars",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
            cv2.LINE_AA
        )

    if q_davenport is not None:
        qd_text = f"qd=[{q_davenport[0]:.3f}, {q_davenport[1]:.3f}, {q_davenport[2]:.3f}, {q_davenport[3]:.3f}]"
        cv2.putText(annotated, qd_text, (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 255), 2, cv2.LINE_AA)
    elif ENABLE_DAVENPORT:
        cv2.putText(annotated, "Davenport: unavailable", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2, cv2.LINE_AA)

    cv2.putText(annotated, f"GV: {GV_ALGO}", (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    return annotated, region_count, weak_matches, strong_matches, q, q_davenport


def main():
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print("ERROR: Could not open camera.")
        return

    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Failed to read camera frame.")
            break

        if resize_width is not None:
            h, w = frame.shape[:2]
            scale = resize_width / w
            frame = cv2.resize(frame, (resize_width, int(h * scale)))

        annotated, region_count, weak_matches, strong_matches, q, q_davenport = process_frame(frame)

        # FPS
        current_time = time.time()
        fps = 1.0 / max(current_time - prev_time, 1e-6)
        prev_time = current_time

        cv2.putText(
            annotated,
            f"FPS: {fps:.2f}",
            (20, annotated.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.imshow("Live Star Tracker", annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f"live_startracker_snapshot_{int(time.time())}.png"
            cv2.imwrite(filename, annotated)
            print("Saved:", filename)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()