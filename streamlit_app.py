from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_id = 0 if torch.cuda.is_available() else -1  

def summarize(article):
    models = [
        "facebook/bart-large-cnn",
        "sshleifer/distilbart-cnn-12-6",
        "allenai/led-base-16384",
        "google/pegasus-xsum",
        "t5-small"
    ]

    similarity_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    similarity_model.to(device)  

    summaries = []

    for model_name in models:
        try:
            summarizer = pipeline("summarization", model=model_name, device=device_id)
            
            summary = summarizer(article, max_length=130, min_length=30, do_sample=False)
            summaries.append(summary[0]['summary_text'])
        except Exception as e:
            summaries.append(f"Error generating summary with {model_name}: {str(e)}")
    
    article_embedding = similarity_model.encode([article], convert_to_tensor=True)
    summary_embeddings = similarity_model.encode(summaries, convert_to_tensor=True)
    similarities = cosine_similarity(article_embedding.cpu().numpy(), summary_embeddings.cpu().numpy())[0]

    best_index = similarities.argmax()
    return summaries[best_index]



from googletrans import Translator
lan_dict = {
    "English": "en",
    'Hindi': 'hi',
    'Bengali': 'bn',
    'Telugu': 'te',
    'Marathi': 'mr',
    'Tamil': 'ta',
    'Gujarati': 'gu',
    'Urdu': 'ur',
    'Kannada': 'kn',
    'Odia': 'or',
    'Punjabi': 'pa',
    'Malayalam': 'ml',
    'Assamese': 'as',
    'Maithili': 'mai',
    'Sanskrit': 'sa',
    'Nepali': 'ne',
    'Konkani': 'kok',
    'Sindhi': 'sd',
    'Dogri': 'doi',
    'Kashmiri': 'ks',
    'Manipuri': 'mni',
    'Rajasthani': 'raj',
    'Santali': 'sat',
    'Bodo': 'bodo',
    'Mizo': 'lus',
    'Haryanvi': 'hyn'
}

translator = Translator()
# text = 'Shinchan sat on the couch, mischievously eyeing his mom as she prepared dinner. With a sly grin, he grabbed a spoon and started to make funny faces in the mirror, distracting his baby sister. His mom, turning around, caught him mid-antics, shaking her head but secretly laughing. "Shinchan, one day your pranks will get you into trouble!" she warned, but Shinchan just winked and ran off to the next adventure.'
# lan = 'Kannada'

def convert(txt, lan):
    lan = lan_dict[lan]
    result = translator.translate(txt, dest=lan)
    return result.text
# convert(text, lan)



from transformers import pipeline, T5ForConditionalGeneration, T5Tokenizer
from happytransformer import TTSettings
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_id = 0 if torch.cuda.is_available() else -1 

grammar_correction_model = pipeline(task="text2text-generation", model="hassaanik/grammar-correction-model", device=device_id)
args = TTSettings(num_beams=5, min_length=1)

model_name = "samadpls/t5-base-grammar-checker"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name).to(device) 

corrector = pipeline('text2text-generation', model="pszemraj/grammar-synthesis-small", device=device_id)

def correct_grammer(txt):
    results = []

    results.append((
        "Model 1",
        grammar_correction_model(txt, max_length=200, num_beams=5, no_repeat_ngram_size=2)[0]['generated_text']
    ))

    inputs = tokenizer.encode(txt, return_tensors="pt").to(device) 
    outputs = model.generate(inputs)
    results.append((
        "Model 3",
        tokenizer.decode(outputs[0], skip_special_tokens=True)
    ))

    results.append((
        "Model 4",
        corrector(txt, max_length=200, num_beams=5, no_repeat_ngram_size=2)[0]['generated_text']
    ))

    scored_results = [(model_name, result, "") for model_name, result in results]
    best_result = min(scored_results, key=lambda x: x[2])  
    return best_result[1]

# test_sentences = [
#     "I has a apple on the table.",
#     "She don't likes to play soccer.",
#     "They was going to the park yesterday.",
#     "He have two car and one bike.",
#     "The child throwed the ball to his friend.",
#     "Where is you going right now?",
#     "This is the book what I bought yesterday.",
#     "I doesn't know the answer to your question.",
#     "We was happy to see the movie together.",
#     "You should studies hard for the exam."
# ]

# for sentence in test_sentences:
#     print(f"Input: {sentence}")
#     print(f"Corrected: {correct_grammer(sentence)}\n")

def perfect_grammer(txt):
    return correct_grammer(txt)




import streamlit as st
import pytesseract
from PIL import Image
from pdf2image import convert_from_path
from grammer import perfect_grammer  
from summerise import summarize  
from translator import convert  
import torch
import os

pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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
