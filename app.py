import streamlit as st
import pytesseract
from PIL import Image
from pdf2image import convert_from_path
from grammer import perfect_grammer  # Ensure you have your grammar correction function
from summerise import summarize  # Ensure you have your summarization function
from translator import convert  # Ensure you have your translation function

# Path to the Tesseract executable (update this path as necessary)
pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe"

# Function to extract text from uploaded file (image or pdf)
def extract_text(file):
    if file is None:
        return "No file uploaded. Please upload an image or PDF."

    try:
        if file.name.endswith(".pdf"):
            # Save the PDF temporarily and extract images from it
            with open("temp.pdf", "wb") as temp_file:
                temp_file.write(file.read())
            images = convert_from_path("temp.pdf")
            extracted_text = ""
            for page_image in images:
                extracted_text += pytesseract.image_to_string(page_image)
            return extracted_text.strip()
        else:
            # Extract text from an image file
            image = Image.open(file)
            return pytesseract.image_to_string(image).strip()
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Function to handle the translation using a button
def translate_text(text, target_language):
    if target_language:
        translated_text = convert(text, target_language)  # Use your translation function here
        return translated_text
    return "Please select a language."

# Streamlit layout
st.title("Text Extractor, Summarizer, and Translator")
st.write("Upload an image or PDF file to extract text, summarize, and translate.")

# File uploader for image or PDF
uploaded_file = st.file_uploader("Upload Image or PDF", type=["jpg", "jpeg", "png", "pdf"])

if uploaded_file is not None:
    st.write("Extracting text...")

    # Extract text from the file
    extracted_text = extract_text(uploaded_file)
    st.text_area("Extracted Text", extracted_text, height=300)

    # Summarize the extracted text
    summarized_text = summarize(extracted_text)  # Make sure the summarizer function is properly defined
    st.text_area("Summarized Text", summarized_text, height=200)

    # Option to select a language for translation
    languages = ['fr', 'de', 'es', 'it', 'pt', 'zh']  # Add more languages as needed
    target_language = st.selectbox("Select Target Language", languages)
    
    if st.button("Translate"):
        translated_text = translate_text(summarized_text, target_language)
        st.text_area("Translated Text", translated_text, height=200)
