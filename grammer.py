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
