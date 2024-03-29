import unicodedata
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning
import spacy

from tqdm import tqdm
import csv
import pandas as pd
import re
import warnings

warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

class Preprocessor():
    '''
    A class for preprocessing the input dataset.

    Attributes:
        data_dir (str): 
            The directory of the csv file.
        read (bool): 
            Use existing preprocessed data.
        sample_size (int): 
            Number of samples taken from the dataset.
        include_content (bool): 
            Include question contents from the dataset.
        max_length (int): 
            Maximum length of the question title.
        min_length (int): 
            Minimum length of the question title.
    
    Methods:
        get_preprocessed_data:
            Return the preprocessed dataset.
        get_class_distribution:
            Return the individual class count of the preprocessed dataset.
        get_length_distribution:
            Return the question length distribution of the preprocessed dataset.
        get_sampled_data
            Return the sampled dataset before preprocessing. 
    '''
    def __init__(self, data_dir, read=False, sample_size=20000, 
                 include_content=False, max_length = 15, min_length = 5):
        '''
        Initialise the Preprocessor class. Carry out preprocessing or load existing data based on input.

        Parameters:
            data_dir (str): 
                The directory of the csv file.
            read (bool): 
                Use existing preprocessed data.
            sample_size (int): 
                Number of samples taken from the dataset.
            include_content (bool): 
                Include question contents from the dataset.
            max_length (int): 
                Maximum length of the question title.
            min_length (int): 
                Minimum length of the question title.
        '''
        self._data_list = []
        if read:
            with open("Datasets/preprocessed/data.csv", mode="r", encoding="utf-8") as file:
                reader = csv.reader(file)
                next(reader)  # Skip header row
                for row in reader:
                    index, label, question, answer = row
                    self._data_list.append((index, int(label), question, answer))
            print(f"Loaded {len(self._data_list)} samples from CSV.")
        else:
            self._spacy_en = spacy.load("en_core_web_sm")
            self._load_csv(data_dir, sample_size, include_content, max_length, min_length)
        
    def _load_csv(self, data_dir, sample_size, include_content, max_length, min_length):
        '''
        Load the CSV file from the specified directory and perform sampling.

        Parameter:
            data_dir (str): 
                The directory of the csv file.
            sample_size (int): 
                Number of samples taken from the dataset.
            include_content (bool): 
                Include question contents from the dataset.
            max_length (int): 
                Maximum length of the question title.
            min_length (int): 
                Minimum length of the question title.
        '''
        data_df = pd.read_csv(data_dir)
        print(f"Loaded {len(data_df)} data entries.")

        self._sampled_df = data_df.sample(n=sample_size, random_state=42, ignore_index=True)
        print(f"Sampled {sample_size} data entries.")

        print(f"Filtering {sample_size} samples...")
        for i in tqdm(range(len(self._sampled_df))):
            label = self._sampled_df["class_index"].loc[i]
            question = self._sentence_normaliser(self._sampled_df["question_title"].loc[i])

            length = len(question.split())
            if length < min_length or length > max_length:
                continue

            answer = self._sentence_normaliser(self._sampled_df["best_answer"].loc[i])
            
            if include_content:
                content = self._sentence_normaliser(self._sampled_df["question_content"].loc[i])
                question = " ".join([question, content])

            self._data_list.append((i, label, question, answer))
        print(f"Appended {len(self._data_list)} samples.")

        with open("Datasets/preprocessed/data.csv", mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["index", "label", "question", "answer"])
            writer.writerows(self._data_list)
        print(f"Saved {len(self._data_list)} samples to CSV.")

    def _sentence_normaliser(self, sentence):
        '''
        The core of the preprocessor. Remove unnecessary components from a string.

        Parameter:
            sentence (str):
                The sentence to be normalised.
        
        Return:
            sentence (str):
                The normalised sentence.
        '''
        if isinstance(sentence, str) and sentence != "":
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
        '''
        Ensure all characters in the sentence are ascii characters.

        Parameter:
            sentence (str):
                The sentence to be checked.
        Return:
            ascii_sentence (str):
                The ascii converted sentence.
        '''
        ascii_sentence = ""
        for character in unicodedata.normalize("NFD", sentence):
            if unicodedata.category(character) != "Mn":
                ascii_sentence += character

        return ascii_sentence
    
    def get_preprocessed_data(self):
        '''
        Return the preprocessed data.

        Return:
            self._data_list (list):
                The sampled dataset with normalised sentences.
        '''
        return self._data_list
    
    def get_class_distribution(self):
        '''
        Return the class distribution of the preprocessed data

        Return:
            class_distribution (dict):
                The count and distribution of each class.
        '''
        class_count = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        for (_, label, _, _) in self._data_list:
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
        '''
        Return the length distribution of the question titles of the sampled data.

        Return:
            sorted(length_distribution.items()) (tuple):
                The counter of each title length in ascending order.
        '''
        length_distribution = {}
        for (_, _, data, _) in self._data_list:
            length = len(data.split())
            if length in length_distribution:
                length_distribution[length] += 1
            else:
                length_distribution[length] = 1

        return sorted(length_distribution.items())
    
    def get_sampled_data(self):
        '''
        Return the sampled data before preprocessing.

        Return:
            self._sampled_df (dataframe):
                The sampled data. 
        '''
        return self._sampled_df