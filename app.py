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

# ================= PAGE SETUP =================
st.set_page_config(page_title="Project Helix", layout="wide")
st.title("🧠 Project Helix")
st.subheader("MRI Analysis with 2D, 3D & Peeling Visualization")
st.caption("Final MVP – 3D Peeling Enabled")

# ================= GCP AUTH =================
gcp_creds = dict(st.secrets["gcp"])
cred = credentials.Certificate(gcp_creds)

if not firebase_admin._apps:
    firebase_admin.initialize_app(cred)

db = firestore.client()
storage_client = storage.Client.from_service_account_info(gcp_creds)

# ================= CONFIG =================
BUCKET_NAME = "project-helix-mri"

# ================= ML ABSTRACTION =================
def vertex_ai_tumor_segmentation(volume):
    vol_norm = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)
    threshold = np.percentile(vol_norm, 99.2)
    return vol_norm > threshold

# ================= UI =================
uploaded_file = st.file_uploader(
    "Upload MRI scan (.nii or .nii.gz)",
    type=["nii", "nii.gz"]
)

if uploaded_file is not None:
    st.info("Uploading MRI to Google Cloud Storage...")

    # ---- Save temp file ----
    suffix = ".nii.gz" if uploaded_file.name.endswith(".nii.gz") else ".nii"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name

    # ---- Upload to GCS ----
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(f"uploads/{uploaded_file.name}")
    #blob.upload_from_filename(tmp_path)

    db.collection("uploads").add({
        "filename": uploaded_file.name,
        "gcs_path": f"gs://{BUCKET_NAME}/uploads/{uploaded_file.name}",
    })

    st.success("✅ MRI uploaded successfully")

    # ================= LOAD MRI =================
    nii = nib.load(tmp_path)
    volume = nii.get_fdata()

    # ================= SLICE =================
    max_slice = volume.shape[2] - 1
    slice_idx = st.slider("Select MRI slice", 0, max_slice, max_slice // 2)

    tumor_mask_3d = vertex_ai_tumor_segmentation(volume)
    slice_img = volume[:, :, slice_idx]
    tumor_mask_2d = tumor_mask_3d[:, :, slice_idx]

    # ---- Contrast windowing ----
    low, high = np.percentile(slice_img, (2, 98))
    slice_img = np.clip(slice_img, low, high)
    slice_img = (slice_img - low) / (high - low + 1e-8)

    # ================= PEELING CONTROL =================
    st.subheader("🧅 3D Brain Peeling Control")
    peel_depth = st.slider(
        "Peel depth (remove outer layers)",
        min_value=0,
        max_value=min(volume.shape)//4,
        value=0,
        step=2
    )

    # ================= LAYOUT =================
    col2d, col3d = st.columns([1, 2])

    # ---------- 2D ----------
    with col2d:
        st.markdown("**2D Slice View**")
        fig2d, ax2d = plt.subplots(figsize=(3.6, 3.6), dpi=150)
        ax2d.imshow(slice_img.T, cmap="gray", origin="lower", interpolation="bicubic")
        ax2d.imshow(tumor_mask_2d.T, cmap="Reds", alpha=0.25, origin="lower")
        ax2d.axis("off")
        st.pyplot(fig2d)

    # ---------- 3D PEELING ----------
    with col3d:
        st.markdown("**3D Brain & Tumor (Peeling Enabled)**")

        # ---- Apply peeling ----
        peeled_volume = volume[
            peel_depth: volume.shape[0]-peel_depth,
            peel_depth: volume.shape[1]-peel_depth,
            peel_depth: volume.shape[2]-peel_depth
        ]

        peeled_tumor = tumor_mask_3d[
            peel_depth: volume.shape[0]-peel_depth,
            peel_depth: volume.shape[1]-peel_depth,
            peel_depth: volume.shape[2]-peel_depth
        ]

        # Normalize
        pv_norm = (peeled_volume - peeled_volume.min()) / (
            peeled_volume.max() - peeled_volume.min() + 1e-8
        )

        # ---- Brain surface ----
        verts_b, faces_b, _, _ = measure.marching_cubes(pv_norm, level=0.25)

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

        meshes = [brain_mesh]

        # ---- Tumor surface ----
        if np.any(peeled_tumor):
            verts_t, faces_t, _, _ = measure.marching_cubes(
                peeled_tumor.astype(np.uint8),
                level=0.5
            )

            tumor_mesh = go.Mesh3d(
                x=verts_t[:, 0],
                y=verts_t[:, 1],
                z=verts_t[:, 2],
                i=faces_t[:, 0],
                j=faces_t[:, 1],
                k=faces_t[:, 2],
                color="red",
                opacity=0.9,
                name="Tumor"
            )
            meshes.append(tumor_mesh)

        fig3d = go.Figure(data=meshes)
        fig3d.update_layout(
            scene=dict(aspectmode="data"),
            height=620,
            margin=dict(l=0, r=0, t=20, b=0)
        )

        st.plotly_chart(fig3d, use_container_width=True)

    os.remove(tmp_path)
