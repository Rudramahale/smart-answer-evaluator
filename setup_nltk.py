import nltk

if __name__ == "__main__":
    print("Downloading NLTK packages...")
    nltk.download('punkt')
    nltk.download('punkt_tab') # Adding punkt_tab to prevent recent NLTK errors
    nltk.download('stopwords')
    nltk.download('wordnet')
    print("NLTK setup complete.")