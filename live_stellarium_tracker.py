import cv2
import numpy as np
import time
import os
import mss

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
catalog_path = r"C:\Users\shauqi\Desktop\Startracker\asu_hipparcos_catalog_ra_dec_deg_wo_header.txt"

threshold = 150
fov_deg = 10.0
min_sep_deg = 0.5
limiting_mag = 4.5
loc_err_deg = 0.05
min_votes_for_quest = 2

resize_width = 480
show_all_labels = True

# Performance tuning
MAX_STARS = 60
PROCESS_EVERY_N_FRAMES = 5
ENABLE_QUEST = True
ENABLE_DAVENPORT = True

# Which monitor to capture:
# 1 = laptop
# 2 = extended monitor
MONITOR_INDEX = 2

# Crop margins inside selected monitor
TOP_CROP = 160
BOTTOM_CROP = 180
LEFT_CROP = 20
RIGHT_CROP = 20

# Save precomputed catalog table
USE_SAVED_TABCAT = True
TABCAT_FILE = "tab_cat_fov4_mag45_sep05.npy"

# =========================
# AUTO SAVE SETTINGS
# =========================
SAVE_FIRST_N_SOLVED_FRAMES = 5
SAVE_ALL_SOLVED_FRAMES = False
OUTPUT_FOLDER = "saved_frames"

# =========================
# NOISE TEST SETTINGS
# =========================
ENABLE_NOISE = True       # turn noise on/off
NOISE_SIGMA = 5          # noise strength (try 5–25)

# =========================
# CAMERA BLUR TEST SETTINGS
# =========================
ENABLE_BLUR = True
BLUR_KERNEL = 3   # try 3,5,7


def keep_brightest_stars(gray, centroids, max_stars=25, patch_radius=4):
    """
    Keep only the brightest detected stars based on local patch brightness.
    """
    if len(centroids) <= max_stars:
        return np.array(centroids, dtype=float)

    h, w = gray.shape
    scored = []

    for (x, y) in centroids:
        cx = int(round(x))
        cy = int(round(y))

        x1 = max(0, cx - patch_radius)
        x2 = min(w, cx + patch_radius + 1)
        y1 = max(0, cy - patch_radius)
        y2 = min(h, cy + patch_radius + 1)

        patch = gray[y1:y2, x1:x2]
        score = float(np.sum(patch))
        scored.append((score, (x, y)))

    scored.sort(key=lambda t: t[0], reverse=True)
    filtered = [item[1] for item in scored[:max_stars]]
    return np.array(filtered, dtype=float)


print("Loading catalog...")
cat = read_catalog(catalog_path, limiting_mag)

if USE_SAVED_TABCAT and os.path.exists(TABCAT_FILE):
    print("Loading saved tab_cat...")
    tab_cat = np.load(TABCAT_FILE)
else:
    print("Generating catalog angle table...")
    tab_cat = gen_table(cat, fov_deg, min_sep_deg)
    tab_cat = tab_cat[np.argsort(tab_cat[:, 2])]
    print("tab_cat entries:", len(tab_cat))

    if USE_SAVED_TABCAT:
        np.save(TABCAT_FILE, tab_cat)
        print("Saved tab_cat to", TABCAT_FILE)

print("Catalog ready.")

if len(tab_cat) == 0:
    raise RuntimeError("Catalog pair table is empty.")


def get_capture_region(sct, monitor_index):
    monitors = sct.monitors

    if monitor_index >= len(monitors):
        raise ValueError(
            f"Monitor {monitor_index} not found. Available monitors: 1 to {len(monitors)-1}"
        )

    mon = monitors[monitor_index]

    left = mon["left"] + LEFT_CROP
    top = mon["top"] + TOP_CROP
    width = mon["width"] - LEFT_CROP - RIGHT_CROP
    height = mon["height"] - TOP_CROP - BOTTOM_CROP

    if width <= 0 or height <= 0:
        raise ValueError("Crop margins are too large for this monitor.")

    return {
        "left": left,
        "top": top,
        "width": width,
        "height": height
    }

def add_gaussian_noise(image, sigma):
    noise = np.random.normal(0, sigma, image.shape).astype(np.float32)
    noisy = image.astype(np.float32) + noise
    noisy = np.clip(noisy, 0, 255)
    return noisy.astype(np.uint8)

def add_gaussian_blur(image, kernel):
    return cv2.GaussianBlur(image, (kernel, kernel), 0)


def process_frame(frame_bgr):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    # Detect star-like blobs
    reg_img, region_count_raw, centroids = detection(gray, threshold)

    # Keep only brightest stars for faster live solving
    centroids = keep_brightest_stars(gray, centroids, max_stars=MAX_STARS)
    region_count = len(centroids)

    annotated = frame_bgr.copy()

    # Need at least a few stars to form enough pairs
    if len(centroids) < 4:
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

    # Convert image centroids to unit vectors
    star_vectors = pixel_to_unit_vectors(centroids, gray.shape, fov_deg)

    # Build image star-pair table
    tab_image = build_tab_image(star_vectors, fov_deg, min_sep_deg)

    if len(tab_image) == 0:
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

    # Match with catalog
    ids, v2 = run_gvalgo_variant(GV_ALGO, cat, tab_cat, tab_image, loc_err_deg, star_vectors=star_vectors)

    strong_matches = int(np.sum(v2 >= min_votes_for_quest))
    weak_matches = int(np.sum(v2 > 0))

    q = None
    q_davenport = None
    if ENABLE_QUEST or ENABLE_DAVENPORT:
        solver_results = solve_with_all_methods(cat, star_vectors, ids, v2, min_votes_for_quest)
        if ENABLE_QUEST:
            q = solver_results["QUEST"]["q"]
        if ENABLE_DAVENPORT:
            q_davenport = solver_results["Davenport Q"]["q"]

    # Draw results
    for idx, (x, y) in enumerate(centroids, start=1):
        cx = int(round(x))
        cy = int(round(y))

        row_id = ids[idx - 1] if idx - 1 < len(ids) else 0
        vote = v2[idx - 1] if idx - 1 < len(v2) else 0

        if row_id > 0 and vote >= min_votes_for_quest:
            color = (0, 255, 0)      # strong match
        elif row_id > 0 and vote > 0:
            color = (0, 255, 255)    # weak match
        else:
            color = (0, 0, 255)      # unmatched

        cv2.circle(annotated, (cx, cy), 4, color, 1)

        if show_all_labels and row_id > 0 and vote >= min_votes_for_quest:
            hip_id = int(cat[row_id - 1, 1])
            text = f"HIP {hip_id} [v{vote}]"
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

    # Stats
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

    if ENABLE_QUEST:
        if q is not None:
            q_text = f"q=[{q[0]:.3f}, {q[1]:.3f}, {q[2]:.3f}, {q[3]:.3f}]"
            color = (255,255,255)
        else:
            q_text = "q = unavailable"
            color = (0,0,255)

        cv2.putText(
            annotated,
            q_text,
            (20,120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA
        )

    if ENABLE_NOISE:
        cv2.putText(
            annotated,
            f"Noise sigma: {NOISE_SIGMA}",
            (20, 180),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (200, 200, 200),
            2,
            cv2.LINE_AA
        )

    if ENABLE_DAVENPORT:
        if q_davenport is not None:
            qd_text = f"qd=[{q_davenport[0]:.3f}, {q_davenport[1]:.3f}, {q_davenport[2]:.3f}, {q_davenport[3]:.3f}]"
            cv2.putText(annotated, qd_text, (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(annotated, "qd = unavailable", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2, cv2.LINE_AA)

    cv2.putText(annotated, f"GV: {GV_ALGO}", (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 2, cv2.LINE_AA)

    return annotated, region_count, weak_matches, strong_matches, q, q_davenport


def main():
    sct = mss.mss()

    print("\nAvailable monitors:")
    for i, mon in enumerate(sct.monitors):
        print(f"Monitor {i}: {mon}")

    capture_region = get_capture_region(sct, MONITOR_INDEX)
    print("\nCapturing region:", capture_region)

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    frame_counter = 0
    solved_frame_counter = 0
    prev_display_time = time.time()

    # last successful solve results
    last_stats = {
        "detected": 0,
        "weak": 0,
        "strong": 0,
        "q": None,
        "qd": None
    }

    while True:
        screenshot = sct.grab(capture_region)
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        # Add simulated camera noise
        if ENABLE_NOISE:
         frame = add_gaussian_noise(frame, NOISE_SIGMA)

        if ENABLE_BLUR:
         frame = add_gaussian_blur(frame, BLUR_KERNEL)

        if resize_width is not None:
            h, w = frame.shape[:2]
            scale = resize_width / w
            frame = cv2.resize(frame, (resize_width, int(h * scale)))

        frame_counter += 1
        annotated = frame.copy()

        if ENABLE_BLUR:
            cv2.putText(
                annotated,
                f"Blur kernel: {BLUR_KERNEL}",
                (20, 210),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (200, 200, 200),
                2,
                cv2.LINE_AA
            )

        # Only run heavy processing every Nth frame
        if frame_counter % PROCESS_EVERY_N_FRAMES == 0:
            annotated, region_count, weak_matches, strong_matches, q, q_davenport = process_frame(frame)

            last_stats["detected"] = region_count
            last_stats["weak"] = weak_matches
            last_stats["strong"] = strong_matches
            last_stats["q"] = q
            last_stats["qd"] = q_davenport

            solved_frame_counter += 1

            # Save first N solved frames automatically
            if solved_frame_counter <= SAVE_FIRST_N_SOLVED_FRAMES:
                filename = os.path.join(
                    OUTPUT_FOLDER,
                    f"solved_frame_{solved_frame_counter:03d}.png"
                )
                cv2.imwrite(filename, annotated)
                print("Saved:", filename)

            # Optional: save every solved frame after that too
            elif SAVE_ALL_SOLVED_FRAMES:
                filename = os.path.join(
                    OUTPUT_FOLDER,
                    f"solved_frame_{solved_frame_counter:03d}.png"
                )
                cv2.imwrite(filename, annotated)
                print("Saved:", filename)

        else:
            # show latest known stats on a fresh frame
            cv2.putText(
                annotated,
                f"Detected: {last_stats['detected']}",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )
            cv2.putText(
                annotated,
                f"Weak matches (>0): {last_stats['weak']}",
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                cv2.LINE_AA
            )
            cv2.putText(
                annotated,
                f"Strong matches (>={min_votes_for_quest}): {last_stats['strong']}",
                (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )
            cv2.putText(
                annotated,
                f"Skipping solve frame ({frame_counter % PROCESS_EVERY_N_FRAMES}/{PROCESS_EVERY_N_FRAMES - 1})",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (200, 200, 200),
                2,
                cv2.LINE_AA
            )

            if ENABLE_QUEST:
                if last_stats["q"] is not None:
                    q = last_stats["q"]
                    q_text = f"q=[{q[0]:.3f}, {q[1]:.3f}, {q[2]:.3f}, {q[3]:.3f}]"
                    cv2.putText(annotated, q_text, (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
                else:
                    cv2.putText(annotated, "q = unavailable", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)

            if ENABLE_DAVENPORT:
                if last_stats["qd"] is not None:
                    qd = last_stats["qd"]
                    qd_text = f"qd=[{qd[0]:.3f}, {qd[1]:.3f}, {qd[2]:.3f}, {qd[3]:.3f}]"
                    cv2.putText(annotated, qd_text, (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 255), 2, cv2.LINE_AA)
                else:
                    cv2.putText(annotated, "qd = unavailable", (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2, cv2.LINE_AA)

            if ENABLE_NOISE:
                cv2.putText(
                    annotated,
                    f"Noise sigma: {NOISE_SIGMA}",
                    (20, 180),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (200, 200, 200),
                    2,
                    cv2.LINE_AA
                )

        current_time = time.time()
        display_fps = 1.0 / max(current_time - prev_display_time, 1e-6)
        prev_display_time = current_time

        cv2.putText(
            annotated,
            f"Display FPS: {display_fps:.2f}",
            (20, annotated.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.imshow("Live Stellarium Star Tracker", annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = os.path.join(
                OUTPUT_FOLDER,
                f"manual_snapshot_{int(time.time())}.png"
            )
            cv2.imwrite(filename, annotated)
            print("Saved:", filename)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()