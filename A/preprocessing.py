import unicodedata
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning
import spacy

from tqdm import tqdm
import csv
import pandas as pd
import re
import warnings
warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

class Preprocess():
    def __init__(self, data_dir, read=False, sample_size=20000, include_content=False, max_length = 15, min_length = 5):
        self._data_list = []
        if read:
            with open('Datasets/data.csv', mode='r', encoding='utf-8') as file:
                reader = csv.reader(file)
                next(reader)  # Skip header row
                for row in reader:
                    label, title = row
                    self._data_list.append((int(label), title))
            print(f"Loaded {len(self._data_list)} samples from CSV.")
        else:
            self._spacy_en = spacy.load("en_core_web_sm")
            self._load_csv(data_dir, sample_size, include_content, max_length, min_length)
        
        
    def _load_csv(self, data_dir, sample_size, include_content, max_length, min_length):
        data_df = pd.read_csv(data_dir)
        print(f"Loaded {len(data_df)} data entries.")

        sampled_df = data_df.sample(n=sample_size, random_state=42, ignore_index=True)
        print(f"Sampled {sample_size} data entries.")

        print(f"Filtering {sample_size} samples...")
        for i in tqdm(range(len(sampled_df))):
            label = sampled_df["class_index"].loc[i]
            title = self._sentence_normaliser(sampled_df["question_title"].loc[i])

            if include_content:
                content = self._sentence_normaliser(sampled_df["question_content"].loc[i])
                data = title + " " + content
            else: 
                data = title

            length = len(data.split())
            if length >= min_length and length <= max_length:
                self._data_list.append((label, data))

        print(f"Appended {len(self._data_list)} samples.")

        with open('Datasets/data.csv', mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(["label", "data"])
            writer.writerows(self._data_list)
        print(f"Saved {len(self._data_list)} samples to CSV.")

    def _sentence_normaliser(self, sentence):
        if isinstance(sentence, str):
            # Filter HTML tags
            if BeautifulSoup(sentence, "html.parser").find():
                sentence = BeautifulSoup(sentence, 'html.parser').get_text()
            
            # Filter hyperlinks
            sentence = re.sub(r"https?://\S+", "", sentence)

            # ASCII conversion
            sentence = self._ascii_converter(sentence)

            # Lowercase conversion
            sentence = sentence.lower()

            # Stop word removal
            parsed_sentence = self._spacy_en(sentence)
            sentence = " ".join([token.text for token in parsed_sentence if not token.is_stop])

            # Filter punctuations and special characters ([] is set, ^ is negate, \w is word, \s is space)
            sentence = re.sub(r"[^\w\s]", "", sentence)
            sentence = re.sub(r"\d+", "", sentence)
            sentence = re.sub(r"\b\w{1,2}\b", "", sentence)
            
            # Lemmatisation
            parsed_sentence = self._spacy_en(sentence)
            sentence = " ".join([token.lemma_ for token in parsed_sentence])

            # Remove leading, trailing and multiple space
            sentence = re.sub(r"\s+", " ", sentence)
            sentence = sentence.strip()
        else:
            sentence = ""

        return sentence
    
    def _ascii_converter(self, sentence):
        ascii_sentence = ""
        for character in unicodedata.normalize("NFD", sentence):
            if unicodedata.category(character) != "Mn":
                ascii_sentence += character

        return ascii_sentence
    
    def get_preprocessed_data(self):
        return self._data_list
    
    def get_class_distribution(self):
        class_count = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        for (label, _) in self._data_list:
            class_count[label-1] += 1

        class_distribution = {"Society & Culture":     class_count[0], 
                              "Science & Mathematics": class_count[1], 
                              "Health":                class_count[2], 
                              "Education & Reference": class_count[3], 
                              "Computers & Internet":  class_count[4], 
                              "Sports":                class_count[5], 
                              "Business & Finance":    class_count[6], 
                              "Entertainment & Music": class_count[7], 
                              "Family & Relationships":class_count[8], 
                              "Politics & Government": class_count[9]}
        
        return class_distribution

    def get_length_distribution(self):
        length_distribution = {}
        for (_, data) in self._data_list:
            length = len(data.split())
            if length in length_distribution:
                length_distribution[length] += 1
            else:
                length_distribution[length] = 1

        return sorted(length_distribution.items())