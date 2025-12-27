import streamlit as st
import os
import tempfile
from google.cloud import storage
import firebase_admin
from firebase_admin import credentials, firestore

# ---------- PAGE SETUP ----------
st.set_page_config(page_title="Project Helix", layout="centered")
st.title("🧠 Project Helix")
st.subheader("MRI Upload & Secure Storage")

# ---------- GCP AUTH (from Streamlit Secrets) ----------
gcp_creds = st.secrets["gcp"]

cred = credentials.Certificate(dict(gcp_creds))
if not firebase_admin._apps:
    firebase_admin.initialize_app(cred)

db = firestore.client()
storage_client = storage.Client.from_service_account_info(dict(gcp_creds))

# ---------- CONFIG ----------
BUCKET_NAME = "project-helix-mri-akhila"  # 🔴 change if needed

# ---------- UI ----------
uploaded_files = st.file_uploader(
    "Upload MRI scans (.nii or .nii.gz)",
    type=["nii", "nii.gz"],
    accept_multiple_files=True
)

if uploaded_files:
    st.info("Uploading files to Google Cloud…")

    for file in uploaded_files:
        # Save temporarily
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(file.getbuffer())
            tmp_path = tmp.name

        # Upload to GCS
        bucket = storage_client.bucket(BUCKET_NAME)
        blob = bucket.blob(f"uploads/{file.name}")
        blob.upload_from_filename(tmp_path)

        # Save metadata to Firestore
        db.collection("uploads").add({
            "filename": file.name,
            "gcs_path": f"gs://{BUCKET_NAME}/uploads/{file.name}",
        })

        os.remove(tmp_path)
        st.success(f"✅ Uploaded: {file.name}")

    st.success("🎉 All files uploaded and logged successfully!")
