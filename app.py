import streamlit as st
import cv2
import pytesseract
import re
import pandas as pd
from pdf2image import convert_from_bytes
import numpy as np

# Inject simple blue-themed CSS
def local_css():
    st.markdown(f"""
        <style>
            body {{
                background-color: #ffffff;
                color: #000000;
                font-family: 'Arial', sans-serif;
            }}
            h1, h2, h3 {{
                color: #1E90FF;
                text-align: center;
            }}
            .stButton > button, .stDownloadButton > button {{
                background-color: #1E90FF;
                color: white;
                border: none;
                padding: 10px 20px;
                font-size: 16px;
                border-radius: 8px;
                margin: 6px;
            }}
            .stButton > button:hover, .stDownloadButton > button:hover {{
                background-color: #1C86EE;
            }}
        </style>
    """, unsafe_allow_html=True)

# Streamlit UI Setup
st.set_page_config(page_title="Claims Extractor", layout="wide")
local_css()

st.title("📄 Claims Document Extractor")
st.markdown("Upload scanned claims documents (JPG, PNG, PDF) to extract details and download them easily.")

# File uploader
uploaded_file = st.file_uploader("Upload file", type=["png", "jpg", "jpeg", "pdf"])

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

if uploaded_file is not None:
    extracted_texts = []
    claim_records = []

    if uploaded_file.type == "application/pdf":
        images = convert_from_bytes(uploaded_file.read())
        progress = st.progress(0)
        total_images = len(images)

        for idx, img in enumerate(images):
            img_array = np.array(img)
            text = extract_text_from_image(img_array)
            extracted_texts.append(text)
            claim_details = extract_claim_details(text)
            claim_records.append(claim_details)
            progress.progress((idx + 1) / total_images)
    else:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        text = extract_text_from_image(image)
        extracted_texts.append(text)
        claim_details = extract_claim_details(text)
        claim_records.append(claim_details)

    st.success("✅ Extraction Completed!")

    st.subheader("📋 Extracted Claims Data")
    df = pd.DataFrame(claim_records)
    st.dataframe(df, use_container_width=True)

    # Download buttons
    json_data = df.to_json(orient="records", indent=2)
    csv_data = df.to_csv(index=False)
    st.download_button("Download JSON", json_data, "claims_data.json", "application/json")
    st.download_button("Download CSV", csv_data, "claims_data.csv", "text/csv")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center;'>Developed with ❤️ by "
    "<a href='https://github.com/kelvinkioi/' target='_blank'>Kelvin Kioi</a></div>",
    unsafe_allow_html=True
)
