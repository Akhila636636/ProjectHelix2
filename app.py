import streamlit as st

st.set_page_config(page_title="Project Helix", layout="centered")

st.title("🧠 Project Helix")
st.subheader("MRI Upload & Analysis Platform")

uploaded_files = st.file_uploader(
    "Upload MRI scans (.nii or .nii.gz)",
    type=["nii", "nii.gz"],
    accept_multiple_files=True
)

if uploaded_files:
    st.success(f"{len(uploaded_files)} file(s) uploaded successfully!")
    for f in uploaded_files:
        st.write("📄", f.name)
