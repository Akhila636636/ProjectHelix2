import streamlit as st
import os
import tempfile
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

from google.cloud import storage
import firebase_admin
from firebase_admin import credentials, firestore

# ---------------- PAGE SETUP ----------------
st.set_page_config(page_title="Project Helix", layout="centered")
st.title("🧠 Project Helix")
st.subheader("MRI Upload & Slice Viewer")
st.write("App version: STEP 9 – Slice Viewer (Stable)")

# ---------------- GCP AUTH ----------------
gcp_creds = dict(st.secrets["gcp"])
cred = credentials.Certificate(gcp_creds)

if not firebase_admin._apps:
    firebase_admin.initialize_app(cred)

db = firestore.client()
storage_client = storage.Client.from_service_account_info(gcp_creds)

# ---------------- CONFIG ----------------
BUCKET_NAME = "project-helix-mri"

st.write("Using bucket:", BUCKET_NAME)

# ---------------- UI ----------------
uploaded_file = st.file_uploader(
    "Upload MRI scan (.nii or .nii.gz)",
    type=["nii", "nii.gz"]
)

if uploaded_file is not None:
    st.info("Uploading MRI to Google Cloud...")

    # -------- SAVE TEMP FILE WITH CORRECT EXTENSION --------
    suffix = ".nii.gz" if uploaded_file.name.endswith(".nii.gz") else ".nii"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name

    # -------- UPLOAD TO CLOUD STORAGE --------
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(f"uploads/{uploaded_file.name}")
    blob.upload_from_filename(tmp_path)

    # -------- LOG TO FIRESTORE --------
    db.collection("uploads").add({
        "filename": uploaded_file.name,
        "gcs_path": f"gs://{BUCKET_NAME}/uploads/{uploaded_file.name}",
    })

    st.success("✅ MRI uploaded successfully to Google Cloud")

    # -------- LOAD MRI WITH NIBABEL --------
    nii = nib.load(tmp_path)
    volume = nii.get_fdata()

    st.subheader("🖼️ Slice‑by‑Slice MRI View")

    max_slice = volume.shape[2] - 1
    slice_idx = st.slider(
        "Select slice",
        0,
        max_slice,
        max_slice // 2
    )

    slice_img = volume[:, :, slice_idx]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(slice_img.T, cmap="gray", origin="lower")
    ax.set_title(f"Slice {slice_idx}")
    ax.axis("off")

    st.pyplot(fig)

    # -------- CLEANUP --------
    os.remove(tmp_path)
