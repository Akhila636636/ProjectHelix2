import streamlit as st
import os
import tempfile
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from google.cloud import storage
import firebase_admin
from firebase_admin import credentials, firestore

# ---------- PAGE SETUP ----------
st.set_page_config(page_title="Project Helix", layout="centered")
st.title("🧠 Project Helix")
st.subheader("MRI Upload & Slice Viewer")

# ---------- GCP AUTH ----------
gcp_creds = st.secrets["gcp"]
cred = credentials.Certificate(dict(gcp_creds))

if not firebase_admin._apps:
    firebase_admin.initialize_app(cred)

db = firestore.client()
storage_client = storage.Client.from_service_account_info(dict(gcp_creds))

# ---------- CONFIG ----------
BUCKET_NAME = "project-helix-mri" 

# ---------- UI ----------
uploaded_files = st.file_uploader(
    "Upload MRI scan (.nii or .nii.gz)",
    type=["nii", "nii.gz"]
)

if uploaded_files:
    st.info("Uploading MRI to Google Cloud...")

    # Save temp file
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_files.getbuffer())
        tmp_path = tmp.name

    # Upload to GCS
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(f"uploads/{uploaded_files.name}")
    blob.upload_from_filename(tmp_path)

    # Log metadata
    db.collection("uploads").add({
        "filename": uploaded_files.name,
        "gcs_path": f"gs://{BUCKET_NAME}/uploads/{uploaded_files.name}",
    })

    st.success("✅ MRI uploaded successfully")

    # ---------- LOAD MRI ----------
    nii = nib.load(tmp_path)
    volume = nii.get_fdata()

    st.subheader("🖼️ Slice-by-Slice MRI View")

    max_slice = volume.shape[2] - 1
    slice_idx = st.slider("Select slice", 0, max_slice, max_slice // 2)

    slice_img = volume[:, :, slice_idx]

    fig, ax = plt.subplots()
    ax.imshow(slice_img.T, cmap="gray", origin="lower")
    ax.set_title(f"Slice {slice_idx}")
    ax.axis("off")

    st.pyplot(fig)

    os.remove(tmp_path)
