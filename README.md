# Star Tracker Attitude Determination in Lost-In-Space Mode

This project implements a **star sensor pattern recognition pipeline** capable of determining spacecraft orientation in **Lost-In-Space (LIS) mode**, where no prior attitude information is available.

The system detects stars from an image frame, identifies star patterns by matching them with a **Hipparcos star catalog**, and estimates spacecraft orientation by solving **Wahba’s problem** using quaternion-based attitude estimation methods.

---

## Project Overview

Star trackers are among the most accurate attitude determination sensors used in modern space missions. They determine spacecraft orientation by observing star fields and matching detected stars with a known star catalog.

In **Lost-In-Space (LIS) mode**, the star tracker must identify stars from an observed image **without any prior attitude information**. This requires reliable star pattern recognition to match detected stars against a star catalog. The efficiency of star identification directly influences the accuracy and robustness of spacecraft navigation, especially in deep-space or lunar missions where traditional navigation aids (e.g., GNSS/GPS) are unavailable.

This project focuses on developing and evaluating a **star pattern recognition pipeline for LIS mode attitude determination** using simulated star images and catalog-based matching techniques.

Pipeline stages:

1. Star detection from captured frame
2. Centroid extraction of detected stars
3. Star pair generation and angular distance computation
4. Catalog matching with Hipparcos star database
5. Outlier rejection using RANSAC
6. Attitude estimation using quaternion solvers

Image Frame
     ↓
Star Detection
     ↓
Centroid Extraction
     ↓
Star Pair Generation
     ↓
Pattern Identification
(Geometric / Hough)
     ↓
Outlier Filtering (Optional)
(RANSAC)
     ↓
Attitude Determination
(QUEST / Davenport)

---

## Algorithms Implemented

The system evaluates multiple star identification approaches:

- **Geometric Voting Algorithm**
- **Hough Transform Voting**
- **RANSAC-based Outlier Filtering**

For attitude estimation, the project solves **Wahba’s Problem** using:

- **QUEST Algorithm**
- **Davenport q-method**

Both solvers produce consistent quaternion solutions when sufficient star matches are available.

---

## Key Features

- Star detection and centroiding pipeline
- Star pair angular separation matching
- Catalog-based star identification
- Robust outlier rejection with RANSAC
- Quaternion-based spacecraft attitude estimation
- Algorithm comparison and evaluation

---

## Technologies Used

- Python
- NumPy
- OpenCV
- Hipparcos Star Catalog
- Stellarium simulation for testing

---

## Results

Preliminary testing shows that geometric voting and Hough-based identification both provide sufficient star matches for reliable attitude estimation. When enough stars are identified, QUEST and Davenport methods produce identical quaternion solutions, confirming solver consistency.

Further evaluation focuses on robustness, processing speed, and performance under noisy star fields.

---

## Future Work

- Deployment on **Raspberry Pi (COTS hardware)** for real-time testing
- Integration with **spacecraft ADCS simulation frameworks**
- Performance optimization for onboard processing
- Hardware testing with optical sensors

---

## Applications

- Spacecraft attitude determination
- Star sensor development
- CubeSat navigation systems
- Autonomous spacecraft orientation

---

## Author

**Hafiz Shauqi**

Electrical & Electronic Engineering  
Final Year Project — Spacecraft Attitude Determination

---

## License

MIT License
