# Star Tracker Attitude Determination in Lost-In-Space Mode

This project implements a **star sensor pattern recognition pipeline** capable of determining spacecraft orientation in **Lost-In-Space (LIS) mode**, where no prior attitude information is available.

The system detects stars from an image frame, identifies star patterns by matching them with a **Hipparcos star catalog**, and estimates spacecraft orientation by solving **Wahba’s problem** using quaternion-based attitude estimation methods.

---

## Project Overview

Modern spacecraft rely on **star sensors (star trackers)** to determine their orientation in space with extremely high accuracy. In Lost-In-Space mode, the spacecraft has no prior knowledge of its attitude, requiring the system to identify star patterns directly from the observed star field.

This project implements a complete pipeline for star pattern recognition and attitude determination.

Pipeline stages:

1. Star detection from captured frame
2. Centroid extraction of detected stars
3. Star pair generation and angular distance computation
4. Catalog matching with Hipparcos star database
5. Outlier rejection using RANSAC
6. Attitude estimation using quaternion solvers

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
