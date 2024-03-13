import spacy
from langdetect import detect
spacy_en = spacy.load("en_core_web_sm")
sentence = "cold sore time lysine vitamin help prevent help heal fast buy abreva drugstore help healing process virus cause cold sore remain body dormant cold sore go away"
print(type(detect(sentence)))
'''
parsed_sentence = spacy_en(sentence)

for token in parsed_sentence:
    if detect(token.text) != "en":
        print("NOT ENGLISH")
    else:
        print(token.text)
        '''