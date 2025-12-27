import streamlit as st
import os
import tempfile
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

from google.cloud import storage
import firebase_admin
from firebase_admin import credentials, firestore

from skimage import measure
import plotly.graph_objects as go

# ---------------- PAGE SETUP ----------------
st.set_page_config(page_title="Project Helix", layout="wide")
st.title("🧠 Project Helix")
st.subheader("MRI Visualization: 2D + 3D")
st.write("App version: STEP 11 – 3D Reconstruction")

# ---------------- GCP AUTH ----------------
gcp_creds = dict(st.secrets["gcp"])
cred = credentials.Certificate(gcp_creds)

if not firebase_admin._apps:
    firebase_admin.initialize_app(cred)

db = firestore.client()
storage_client = storage.Client.from_service_account_info(gcp_creds)

# ---------------- CONFIG ----------------
BUCKET_NAME = "project-helix-mri"

# ---------------- UI ----------------
uploaded_file = st.file_uploader(
    "Upload MRI scan (.nii or .nii.gz)",
    type=["nii", "nii.gz"]
)

if uploaded_file is not None:
    st.info("Uploading MRI to Google Cloud...")

    # ---- SAVE TEMP FILE WITH EXTENSION ----
    suffix = ".nii.gz" if uploaded_file.name.endswith(".nii.gz") else ".nii"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name

    # ---- UPLOAD TO GCS ----
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(f"uploads/{uploaded_file.name}")
    blob.upload_from_filename(tmp_path)

    db.collection("uploads").add({
        "filename": uploaded_file.name,
        "gcs_path": f"gs://{BUCKET_NAME}/uploads/{uploaded_file.name}",
    })

    st.success("✅ MRI uploaded successfully")

    # ---- LOAD MRI ----
    nii = nib.load(tmp_path)
    volume = nii.get_fdata()

    # ---------------- 2D VIEW ----------------
    st.subheader("🖼️ 2D Slice View with Tumor Overlay")

    max_slice = volume.shape[2] - 1
    slice_idx = st.slider("Select slice", 0, max_slice, max_slice // 2)

    slice_img = volume[:, :, slice_idx]
    threshold = np.percentile(slice_img, 99)
    tumor_mask_2d = slice_img > threshold

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(slice_img.T, cmap="gray", origin="lower")
    ax.imshow(tumor_mask_2d.T, cmap="Reds", alpha=0.4, origin="lower")
    ax.axis("off")
    st.pyplot(fig)

    # ---------------- 3D VIEW ----------------
    st.subheader("🧊 3D Brain & Tumor Reconstruction")

    # Normalize volume
    vol_norm = (volume - volume.min()) / (volume.max() - volume.min())

    # Brain surface
    verts_b, faces_b, _, _ = measure.marching_cubes(vol_norm, level=0.2)

    # Tumor mask (3D placeholder)
    tumor_mask_3d = volume > np.percentile(volume, 99.5)
    verts_t, faces_t, _, _ = measure.marching_cubes(tumor_mask_3d, level=0.5)

    brain_mesh = go.Mesh3d(
        x=verts_b[:, 0],
        y=verts_b[:, 1],
        z=verts_b[:, 2],
        i=faces_b[:, 0],
        j=faces_b[:, 1],
        k=faces_b[:, 2],
        color="lightgray",
        opacity=0.15,
        name="Brain"
    )

    tumor_mesh = go.Mesh3d(
        x=verts_t[:, 0],
        y=verts_t[:, 1],
        z=verts_t[:, 2],
        i=faces_t[:, 0],
        j=faces_t[:, 1],
        k=faces_t[:, 2],
        color="red",
        opacity=0.8,
        name="Tumor"
    )

    fig3d = go.Figure(data=[brain_mesh, tumor_mesh])
    fig3d.update_layout(
        scene=dict(aspectmode="data"),
        height=600
    )

    st.plotly_chart(fig3d, use_container_width=True)

    os.remove(tmp_path)
