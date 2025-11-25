import streamlit as st
import easyocr
import fitz  # PyMuPDF
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.cluster import KMeans
import torch
import re
import io
import base64
from PIL import Image  # Fixed import
from sklearn.metrics.pairwise import cosine_similarity
import os

# Create data directory if not exists
os.makedirs("data", exist_ok=True)

# Load Nepali model and tokenizer (do this once)
@st.cache_resource
def load_nepali_model():
    tokenizer = AutoTokenizer.from_pretrained("Sakonii/distilbert-base-nepali")
    model = AutoModel.from_pretrained("Sakonii/distilbert-base-nepali")
    return tokenizer, model

tokenizer, nepali_model = load_nepali_model()

# Initialize EasyOCR reader with caching
@st.cache_resource
def get_ocr_reader(_lang):
    return easyocr.Reader([_lang], gpu=torch.cuda.is_available())

def extract_text_from_scanned_pdf(pdf_path, language):
    reader = get_ocr_reader(language)
    try:
        text = ''
        pdf_document = fitz.open(pdf_path)
        for page_num in range(pdf_document.page_count):
            page = pdf_document[page_num]
            pix = page.get_pixmap(dpi=200)  # Higher DPI = better OCR
            img_bytes = pix.tobytes("png")
            result = reader.readtext(img_bytes, detail=0)  # detail=0 returns only text
            text += ' '.join(result) + '\n'
        return text.strip()
    except Exception as e:
        return f"Error: {str(e)}"

def read_text_from_pillow_image(image, language):
    reader = get_ocr_reader(language)
    try:
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_bytes = img_byte_arr.getvalue()
        result = reader.readtext(img_bytes, detail=0)
        return ' '.join(result)
    except Exception as e:
        return f"Error: {str(e)}"

def summarize_text(model, tokenizer, text, num_clusters=3):
    if not text or not isinstance(text, str) or len(text.strip()) < 50:
        return "Insufficient text to summarize."

    # Split Nepali/English sentences properly
    sentences = re.split(r'[।?\.!\n]\s*', text)
    sentences = [s.strip() for s in sentences if s.strip() and len(s) > 10]

    if len(sentences) <= num_clusters:
        return " । ".join(sentences)

    # Generate embeddings
    embeddings = []
    for sentence in sentences:
        inputs = tokenizer(sentence, return_tensors='pt', truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()  # [CLS] token
        embeddings.append(embedding)

    # Clustering
    kmeans = KMeans(n_clusters=min(num_clusters, len(sentences)), n_init=10, random_state=42)
    kmeans.fit(embeddings)
    
    # Select one representative sentence per cluster (closest to centroid)
    summary_sentences = []
    for i in range(kmeans.n_clusters):
        cluster_sentences_idx = np.where(kmeans.labels_ == i)[0]
        cluster_embeddings = np.array([embeddings[idx] for idx in cluster_sentences_idx])
        centroid = kmeans.cluster_centers_[i]
        similarities = cosine_similarity([centroid], cluster_embeddings)[0]
        closest_idx = cluster_sentences_idx[np.argmax(similarities)]
        summary_sentences.append(sentences[closest_idx])

    return " । ".join(summary_sentences)

# Display PDF function (fixed F-string syntax)
def displayPDF(file_path):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

# Streamlit UI
st.set_page_config(page_title="Nepali Document Summarizer", layout="wide")
st.title("📄 Nepali Document Summarization App")
st.markdown("Extract text from PDFs/Images and generate summaries in Nepali using AI.")

option = st.selectbox("Choose Input Type", ('PDF', 'Image', 'Text'))  # Fixed typo

if option == 'PDF':
    uploaded_file = st.file_uploader("Upload Scanned or Text PDF", type=['pdf'])
    language = st.text_input("OCR Language Code (e.g., 'ne' for Nepali, 'en' for English)", value="ne")

    if uploaded_file and st.button("Extract Text & Summarize"):
        filepath = f"data/{uploaded_file.name}"
        with open(filepath, "wb") as f:
            f.write(uploaded_file.getbuffer())

        col1, col2 = st.columns(2)

        with col1:
            st.info("Uploaded PDF")
            displayPDF(filepath)

        with col2:
            with st.spinner("Extracting text using OCR..."):
                extracted_text = extract_text_from_scanned_pdf(filepath, language)
            st.success("Text Extracted!")
            st.text_area("Extracted Text", extracted_text, height=300)

            if extracted_text and not extracted_text.startswith("Error"):
                with st.spinner("Generating summary..."):
                    summary = summarize_text(nepali_model, tokenizer, extracted_text)
                st.success("Summary Generated!")
                st.markdown(f"**Summary:**\n\n{summary}")

elif option == 'Image':
    uploaded_file = st.file_uploader("Upload Image (JPG/PNG)", type=['jpg', 'jpeg', 'png'])
    language = st.text_input("OCR Language Code", value="ne")

    if uploaded_file and st.button("Extract & Summarize"):
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Uploaded Image", use_column_width=True)

        with col2:
            with st.spinner("Reading text from image..."):
                text = read_text_from_pillow_image(image, language)
            st.success("Text Extracted!")
            st.text_area("Extracted Text", text, height=200)

            if text and not text.startswith("Error"):
                with st.spinner("Summarizing..."):
                    summary = summarize_text(nepali_model, tokenizer, text)
                st.success("Summary:")
                st.write(summary)

elif option == 'Text':
    text = st.text_area("Enter Nepali text to summarize:", height=200)
    if st.button("Summarize Text") and text.strip():
        with st.spinner("Generating summary..."):
            summary = summarize_text(nepali_model, tokenizer, text)
        st.success("Summary:")
        st.markdown(f"**{summary}**")

st.caption("Powered by EasyOCR + Sakonii/distilbert-base-nepali + KMeans Clustering")


























