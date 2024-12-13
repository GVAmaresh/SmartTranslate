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
text = 'Shinchan sat on the couch, mischievously eyeing his mom as she prepared dinner. With a sly grin, he grabbed a spoon and started to make funny faces in the mirror, distracting his baby sister. His mom, turning around, caught him mid-antics, shaking her head but secretly laughing. "Shinchan, one day your pranks will get you into trouble!" she warned, but Shinchan just winked and ran off to the next adventure.'
lan = 'Kannada'

def convert(txt, lan):
    lan = lan_dict[lan]
    result = translator.translate(txt, dest=lan)
    reverse_dict = {v: k for k, v in lan_dict.items()}
    dct_lan = reverse_dict.get(result.src, "Unknown Language")
    # print(f"Sentence converted from {dct_lan} to {reverse_dict.get(lan)}")
    # print("Translated Text: ", result.text)
    return result.text
convert(text, lan)
