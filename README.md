# Homography-Based Image Stitching

A robust computer vision pipeline for **automatic panorama generation** by stitching multiple overlapping input images using **Homography estimation**.

This project implements the classical feature-based image alignment technique, including keypoint detection, feature matching, and RANSAC-based homography calculation, to achieve seamless photo mosaics.

---

## Results

| Input Images (Example Pair) | Feature Matching & RANSAC Inliers | Final Stitched Panorama (Mosaic) |
| :-------------------------: | :-------------------------------: | :------------------------------: |
| [Insert Image of two input images here] | [Insert Image of matched features here] | [Insert Image of stitched panorama here] |

*Note: Please replace the bracketed tags with actual images or GIFs showing the project's results. This visual demonstration is highly recommended for a computer vision repository.*

---

## Key Features

This implementation focuses on the core stages of a feature-based image stitching pipeline:

* **Feature Detection:** Utilizes a scale-invariant detector (e.g., **SIFT** or **ORB**) for extracting robust keypoints and descriptors from each input image.
* **Feature Matching:** Employs a distance-based matching strategy (e.g., **k-NN** with **Lowe's Ratio Test**) to find correspondence pairs between adjacent images.
* **Homography Estimation:** The 3x3 projective transformation matrix (Homography) is calculated using the **Random Sample Consensus (RANSAC)** algorithm, ensuring robust estimation despite noisy feature matches (outliers).
* **Image Warping:** Applies the computed Homography matrix to warp the source image into the coordinate system of the destination image.
* **Seam Blending:** Combines the warped images, often including a method (like linear blending) to smooth the transition in the overlapping region and reduce visible seams.

---

## Installation and Setup

### Prerequisites

* Python 3.x
* OpenCV (`cv2`)
* NumPy

### Setup

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Noam-Diamant/image-stitching-homography.git](https://github.com/Noam-Diamant/image-stitching-homography.git)
    cd image-stitching-homography
    ```

2.  **Install required libraries:**
    ```bash
    pip install opencv-python numpy
    ```

---

## Usage

1.  **Prepare Input Images:** Create an `input_images/` directory and place the images you wish to stitch (e.g., `image1.jpg`, `image2.jpg`) inside it.
2.  **Run the main script:**
    ```bash
    python stitcher.py --input_dir ./input_images
    ```
    *(Note: Please ensure the script name, e.g., `stitcher.py`, matches your primary execution file.)*

3.  **Output:** The script will save the final panoramic image to a specified output location (e.g., `output_panorama.jpg`).

---

## Contribution

Contributions are welcome! Feel free to fork the repository and submit pull requests for:

* Implementing advanced blending techniques (e.g., Multi-band blending).
* Adding support for automatic chaining and ordering of multiple images (more than two).
* Improving robustness to parallax.
