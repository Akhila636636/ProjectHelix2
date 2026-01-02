🧠 Project Helix
Project Helix is an interactive brain MRI visualization and analysis tool that enables users to explore MRI scans in both high‑quality 2D and interactive 3D views, with advanced features like tumor highlighting and 3D peeling for deeper inspection.

The project focuses on making medical imaging more intuitive, visual, and accessible, while maintaining a clean and scalable architecture suitable for future expansion.

🚀 Features
🔹 MRI Upload
Upload brain MRI scans in .nii or .nii.gz (NIfTI) format
Secure cloud storage of uploaded scans

🔹 High‑Quality 2D Slice Viewer
Smooth, radiology‑style 2D MRI slices
Slice navigation using a slider
Tumor‑like regions highlighted with overlays
Improved contrast and readability using windowing techniques

🔹 Interactive 3D Visualization
3D reconstruction of the brain from MRI volume
Interactive rotation, zoom, and pan
Clear visualization of internal structures

🔹 3D Peeling (Advanced Visualization)
Gradually remove outer layers of the brain using a peel‑depth slider
Reveal deeper internal structures and tumor regions
Helps understand tumor depth and spatial location

🔹 Tumor Segmentation (ML‑Ready)
Tumor regions detected using a modular, ML‑ready abstraction
Designed to be replaced with a real ML model in the future
Stable and demo‑safe implementation

🧩 System Overview
Workflow:
User uploads an MRI scan
Scan is stored securely in the cloud
MRI volume is processed for visualization
User explores:
2D slices with overlays
3D reconstruction

3D peeling for deeper inspection
