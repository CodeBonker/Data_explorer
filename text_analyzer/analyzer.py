import string
import re
from nltk.corpus import stopwords
from collections import Counter


class TextAnalyzer:
    """
    A class that reads a text file and computes simple statistics
    """

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.text = ""

    def read_file(self) -> str:
        try:
            with open(self.file_path, "r", encoding="utf-8") as file:
                self.text = file.read()
            print("File loaded successfully")
            return self.text
        except FileNotFoundError:
            print("File not found. Please check the path.")
            return ""
        except Exception as e:
            print(f"Error while reading file: {e}")
            return ""

    def count_lines(self) -> int:
        if not self.text:
            self.read_file()
        return len(self.text.splitlines())

    def count_words(self) -> int:
        if not self.text:
            self.read_file()
        return len(self.text.split())

    def count_characters(self) -> int:
        if not self.text:
            self.read_file()
        return len(self.text)
    
    def clean_text(self) -> str:
        """
        Cleans the text by:
        1. Converting to lowercase
        2. Removing URLs, numbers and special characters using regex
        3. Removing punctuation using translate()
        4. Removing extra whitespace
        """

        if not self.text:
            self.read_file() 

        cleaned = self.text.lower()

        cleaned = re.sub(r"http\S+|www\S+", "", cleaned)

        cleaned = re.sub(r"\d+", "", cleaned)

        cleaned = cleaned.translate(str.maketrans("", "", string.punctuation))

        cleaned = re.sub(r"[^a-z\s]", "", cleaned)


        stop_words = set(stopwords.words("english"))
        cleaned_words = [word for word in cleaned.split() if word not in stop_words]
        
        return " ".join(cleaned_words)
        
    def tokenize(self) -> list:
        """
        Splits the cleaned text into tokens
        """
        cleaned = self.clean_text()
        return cleaned.split()
       

    def get_word_frequency(self, n=10) -> list:
        """
        Returns the n most common words from the cleaned text
        """
        tokens = self.tokenize()
        word_counts = Counter(tokens)
        return word_counts.most_common(n)