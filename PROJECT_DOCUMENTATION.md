# DepthWizard: High-Resolution Urban DEM Reconstruction 

> **SIH 2026 ISRO · Problem Statement SIH26175**

This document outlines the end-to-end architecture, dataset, model training, and frontend implementation for the DepthWizard hackathon prototype. The project implements a monocular foundation model approach (inspired by *Prompt2DEM*) to generate high-resolution Digital Elevation Models (DEM) from single aerial/satellite RGB images.

---

## 1. Dataset & Preprocessing
We utilized the **DFC2019 (Data Fusion Contest 2019) Track 1 dataset**, specifically focusing on the urban environments of Jacksonville (JAX) and Omaha (OMA).

*   **Inputs**: High-resolution RGB aerial images (`*RGB.tif`).
*   **Ground Truth**: LiDAR-derived Above Ground Level (AGL) height maps (`*AGL.tif`).
*   **Scale**: The dataset provides metric heights (in meters), which is critical for generating accurate, real-world Urban DEMs.
*   **Processing**: 
    * No complex patching was required. We directly paired raw RGB images with their corresponding AGL reference maps.
    * Invalid values (NaN or < 0) in the LiDAR ground truth were clipped, and max heights were capped at 100m to stabilize training in dense urban settings.

## 2. Model Architecture: QuickHeightNet
To ensure a fast, robust, and working prototype for the hackathon, we implemented a custom architecture named **QuickHeightNet**.

### Backbone: Depth-Anything-V2
We leveraged `Depth-Anything-V2-Small` (via HuggingFace) as a frozen foundation model. 
*   **Why**: Foundation models trained on millions of diverse images have incredible zero-shot geometric understanding. Depth-Anything-V2 excels at relative depth estimation.
*   **Implementation**: The backbone weights were completely frozen during our training to prevent catastrophic forgetting and to speed up the training process significantly.

### Metric Decoder (Refinement CNN)
Since the foundation model outputs *relative* inverse depth, we added a lightweight, trainable decoding head to convert relative depth into absolute metric height (AGL).
*   **Scale & Shift**: We introduced learned `scale` and `shift` parameters to linearly map the normalized relative depth into an approximate absolute metric space (initializing the shift to ~5m, representing a typical urban building height).
*   **CNN Head**: A 4-layer Convolutional Neural Network (with ReLU activations) processes the coarse metric depth to refine edges and output the final single-channel height map in meters. 

## 3. Training Strategy
The model was trained using a highly optimized setup tailored for a hackathon timeframe:

*   **Data Split**: 120 training pairs and 20 validation pairs (sampled from the DFC2019 raw dataset).
*   **Loss Function**: 
    *   **Valid Pixel L1 Loss**: Standard Mean Absolute Error (MAE) on valid LiDAR pixels.
    *   **Gradient Loss**: A scale-invariant edge loss (L1 loss on the X and Y gradients of the image) to ensure sharp building footprints and crisp urban geometry.
*   **Hyperparameters**: 
    *   Optimizer: AdamW (Learning Rate: 2e-4) with Cosine Annealing.
    *   Epochs: 15 (Completed in ~20 minutes on an RTX 4060).
*   **Results**: The model converged beautifully, achieving a final **Validation MAE of ~3.54 meters**, which is highly competitive for a zero-shot, fast-trained prototype.

## 4. Backend & API (FastAPI)
The backend is powered by a high-performance **FastAPI** server running in Python.
*   **Inference Pipeline**: Receives an image via a POST request (`/api/infer-form`), resizes it to 518x518 (the native resolution for Depth-Anything), and passes it through the QuickHeightNet.
*   **Multi-modal Output Generation**: To match the rich visual outputs of state-of-the-art papers (like Prompt2DEM), the backend dynamically generates four visualizations on the fly:
    1.  **RGB Input**: The original image.
    2.  **Predicted Elevation**: Normalized and mapped using the `turbo` colormap.
    3.  **Hillshade**: A simulated 3D grey-relief shading computed directly from the DEM gradients (Azimuth 315°, Altitude 45°).
    4.  **Surface Normals**: A vibrant mapping of the X, Y, and Z gradients.
*   All visual maps are base64-encoded and returned to the frontend alongside the raw metric height array.

## 5. Frontend UI
We built a clean, zero-dependency vanilla HTML/JS/CSS frontend to showcase the model's capabilities.
*   **4-Panel Grid**: The primary view is a responsive 2x2 grid displaying the RGB, Elevation, Hillshade, and Surface Normal maps simultaneously.
*   **Interactive 3D Viewer**: Using `Three.js`, the frontend dynamically generates a 3D terrain mesh in the browser. It uses the predicted height array to displace the vertices of a plane and maps the hillshade texture onto it, allowing the user to orbit and explore the generated urban environment in full 3D.
*   **Aesthetics**: The UI uses modern design tokens, smooth micro-animations, and GitHub-style badges to create a premium, hackathon-winning presentation. 
