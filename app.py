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
st.subheader("MRI Analysis with 2D & 3D Tumor Visualization")
st.caption("Final MVP – Cloud + ML Ready")

# ================= GCP AUTH =================
gcp_creds = dict(st.secrets["gcp"])
cred = credentials.Certificate(gcp_creds)

if not firebase_admin._apps:
    firebase_admin.initialize_app(cred)

db = firestore.client()
storage_client = storage.Client.from_service_account_info(gcp_creds)

# ================= CONFIG =================
BUCKET_NAME = "project-helix-mri"

# ================= ML ABSTRACTION (VERTEX‑READY) =================
def vertex_ai_tumor_segmentation(volume):
    """
    Vertex‑AI‑ready tumor segmentation abstraction.
    Replace internals with real Vertex endpoint later.
    """
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

    # ---- Save temp file with correct extension ----
    suffix = ".nii.gz" if uploaded_file.name.endswith(".nii.gz") else ".nii"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name

    # ---- Upload to GCS ----
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(f"uploads/{uploaded_file.name}")
    blob.upload_from_filename(tmp_path)

    # ---- Log metadata ----
    db.collection("uploads").add({
        "filename": uploaded_file.name,
        "gcs_path": f"gs://{BUCKET_NAME}/uploads/{uploaded_file.name}",
    })

    st.success("✅ MRI uploaded successfully")

    # ================= LOAD MRI =================
    nii = nib.load(tmp_path)
    volume = nii.get_fdata()

    # ================= SLICE SELECTION =================
    max_slice = volume.shape[2] - 1
    slice_idx = st.slider("Select MRI slice", 0, max_slice, max_slice // 2)

    tumor_mask_3d = vertex_ai_tumor_segmentation(volume)

    slice_img = volume[:, :, slice_idx]
    tumor_mask_2d = tumor_mask_3d[:, :, slice_idx]

    # Normalize for clean visuals
    slice_img = (slice_img - slice_img.min()) / (slice_img.max() - slice_img.min() + 1e-8)

    # ================= 2D + 3D SIDE‑BY‑SIDE =================
    st.subheader("🧠 2D & 3D MRI Visualization")

    col2d, col3d = st.columns([1, 2])

    # ---------- 2D VIEW ----------
    with col2d:
        st.markdown("**2D Slice View**")

        fig2d, ax2d = plt.subplots(figsize=(3.2, 3.2))
        ax2d.imshow(
            slice_img.T,
            cmap="gray",
            origin="lower",
            interpolation="bilinear"
        )
        ax2d.imshow(
            tumor_mask_2d.T,
            cmap="Reds",
            alpha=0.35,
            origin="lower",
            interpolation="bilinear"
        )
        ax2d.set_title(f"Slice {slice_idx}", fontsize=9)
        ax2d.axis("off")

        st.pyplot(fig2d)

    # ---------- 3D VIEW ----------
    with col3d:
        st.markdown("**3D Brain & Tumor Reconstruction**")

        # Brain surface
        vol_norm = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)
        verts_b, faces_b, _, _ = measure.marching_cubes(vol_norm, level=0.2)

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

        # Tumor surface (safe)
        if np.any(tumor_mask_3d):
            verts_t, faces_t, _, _ = measure.marching_cubes(
                tumor_mask_3d.astype(np.uint8),
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
                opacity=0.8,
                name="Tumor"
            )
            meshes.append(tumor_mesh)
        else:
            st.info("No prominent tumor region detected for 3D view.")

        fig3d = go.Figure(data=meshes)
        fig3d.update_layout(
            scene=dict(aspectmode="data"),
            height=600,
            margin=dict(l=0, r=0, t=20, b=0)
        )

        st.plotly_chart(fig3d, use_container_width=True)

    os.remove(tmp_path)
