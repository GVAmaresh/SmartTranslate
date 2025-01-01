import streamlit as st
import pytesseract
from PIL import Image
from pdf2image import convert_from_path
from grammer import perfect_grammer  
from summerise import summarize  
from translator import convert  

pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe"

def extract_text(file):
    if file is None:
        return "No file uploaded. Please upload an image or PDF."

    try:
        if file.name.endswith(".pdf"):
            with open("temp.pdf", "wb") as temp_file:
                temp_file.write(file.read())
            images = convert_from_path("temp.pdf")
            extracted_text = ""
            for page_image in images:
                extracted_text += pytesseract.image_to_string(page_image)
            return extracted_text.strip()
        else:
            image = Image.open(file)
            return pytesseract.image_to_string(image).strip()
    except Exception as e:
        return f"An error occurred: {str(e)}"

def translate_text(text, target_language):
    if target_language:
        translated_text = convert(text, target_language)  
        return translated_text
    return "Please select a language."

st.title("Text Extractor, Summarizer, and Translator")
st.write("Upload an image or PDF file to extract text, summarize, and translate.")

uploaded_file = st.file_uploader("Upload Image or PDF", type=["jpg", "jpeg", "png", "pdf"])

if uploaded_file is not None:
    st.write("Extracting text...")

    extracted_text = extract_text(uploaded_file)
    st.text_area("Extracted Text", extracted_text, height=300)

    summarized_text = summarize(extracted_text) 
    st.text_area("Summarized Text", summarized_text, height=200)

    languages = [
    "English",
    "Hindi",
    "Bengali",
    "Telugu",
    "Marathi",
    "Tamil",
    "Gujarati",
    "Urdu",
    "Kannada",
    "Odia",
    "Punjabi",
    "Malayalam",
    "Assamese",
    "Maithili",
    "Sanskrit",
    "Nepali",
    "Konkani",
    "Sindhi",
    "Dogri",
    "Kashmiri",
    "Manipuri",
    "Rajasthani",
    "Santali",
    "Bodo",
    "Mizo",
    "Haryanvi"
]

    target_language = st.selectbox("Select Target Language", languages)
    
    if st.button("Translate"):
        translated_text = translate_text(summarized_text, target_language)
        st.text_area("Translated Text", translated_text, height=200)
