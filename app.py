import streamlit as st
import cv2
import pytesseract
import re
import pandas as pd
from pdf2image import convert_from_bytes
import numpy as np
from io import BytesIO

# Streamlit UI Setup
st.title("Claims Document Extractor (Batch Mode)")
st.write("Upload scanned claims documents (PDF with multiple pages or multiple images) to extract details.")

# File uploader
uploaded_file = st.file_uploader("Upload file", type=["png", "jpg", "jpeg", "pdf"])

# Helper Functions
def preprocess_image(image):
    """Convert image to grayscale and apply thresholding"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    return thresh

def extract_text_from_image(image):
    """Extract text from an image using Tesseract OCR"""
    processed_image = preprocess_image(image)
    text = pytesseract.image_to_string(processed_image)
    return text

def extract_claim_details(text):
    """Extract required details using regex"""
    fields = {
        "Insured": re.search(r"Insured[:\-]?\s*(.+)", text, re.IGNORECASE),
        "Reg No": re.search(r"Reg No[:\-]?\s*([A-Za-z0-9]+\s*[A-Za-z0-9]*)", text, re.IGNORECASE),
        "Claim No": re.search(r"Claim No[:\-]?\s*(\S+)", text, re.IGNORECASE),
        "Policy No": re.search(r"Policy No[:\-]?\s*(\S+)", text, re.IGNORECASE),
        "Type of Cover": re.search(r"Type of Cover[:\-]?\s*(.+)", text, re.IGNORECASE),
        "Date of Loss": re.search(r"DATE OF LOSS[:\-]?\s*(\w+,\s+\w+\s+\d{1,2},\s+\d{4})", text),
        "Date of Notification": re.search(r"DATE OF NOTIFICATION[:\-]?\s*(\w+,\s+\w+\s+\d{1,2},\s+\d{4})", text),
        "Agency Name": re.search(r"Agency Name[:\-]?\s*(.+)", text, re.IGNORECASE),
        "Prepared by": re.search(r"Prepared by[:\-]?\s*(.+)", text, re.IGNORECASE),
    }
    return {key: (match.group(1) if match else "Not found") for key, match in fields.items()}

# Main Processing
if uploaded_file is not None:
    all_claims = []  # List to hold details for each document

    if uploaded_file.type == "application/pdf":
        images = convert_from_bytes(uploaded_file.read())
        st.write(f"Processing {len(images)} pages from uploaded PDF...")
        for idx, img in enumerate(images):
            st.write(f"Processing page {idx + 1}...")
            img_array = np.array(img)
            text = extract_text_from_image(img_array)
            claim_details = extract_claim_details(text)
            all_claims.append(claim_details)
    else:
        # For a single image file
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        text = extract_text_from_image(image)
        claim_details = extract_claim_details(text)
        all_claims.append(claim_details)
    
    # Combine all extracted claims into a DataFrame
    df = pd.DataFrame(all_claims)

    # Display results
    st.subheader("Extracted Claims Data")
    st.dataframe(df)

    # Download options
    json_data = df.to_json(orient="records", indent=2)
    csv_data = df.to_csv(index=False)

    st.download_button("Download JSON", json_data, "claims_data.json", "application/json")
    st.download_button("Download CSV", csv_data, "claims_data.csv", "text/csv")
