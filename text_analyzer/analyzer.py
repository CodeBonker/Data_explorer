class TextAnalyzer:
    """
    A class that reads a text file and computes simple statistics.
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
