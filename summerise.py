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

# ARTICLE = """John Wick walked into the dimly lit bar, the low hum of conversation stopping as he entered. His eyes, cold and focused, scanned the room for the one person he had come to see. The bartender, a grizzled man with years of experience in the business, nodded without a word.

# "You shouldn't have come here," the voice echoed from the shadows. It was a familiar voice, but one he never hoped to hear again.

# John's hand moved to the holster under his jacket, fingers brushing the cold steel of his gun.

# "You left me no choice," John said, his voice low but steady.

# A figure stepped forward, smirking, but the smirk faded as he saw the fire in John's eyes.

# "Get ready to pay the price," John whispered, his hand already moving faster than the eye could follow. The room erupted into chaos."""

# print(summarize(ARTICLE))
