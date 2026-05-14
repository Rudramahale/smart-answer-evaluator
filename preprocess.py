import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Common AI abbreviations expanded before tokenization
AI_SYNONYMS = {
    "ai":    "artificial intelligence",
    "ml":    "machine learning",
    "dl":    "deep learning",
    "nlp":   "natural language processing",
    "nn":    "neural network",
    "cnn":   "convolutional neural network",
    "llm":   "large language model",
    "genai": "generative artificial intelligence",
    "tf":    "tensorflow",
    "pt":    "pytorch"
}

# Additional AI-related synonyms for better keyword matching
AI_KEYWORD_SYNONYMS = {
    "data": ["information", "dataset", "statistics"],
    "algorithm": ["method", "technique", "process"],
    "automation": ["automated", "automatic", "robotics"],
    "technology": ["tech", "system", "platform"],
    "intelligence": ["smart", "intelligent", "cognitive"],
    "learning": ["train", "trained", "training"],
    "model": ["framework", "architecture", "system"],
    "prediction": ["forecast", "forecasting", "estimate"],
    "network": ["web", "system", "architecture"],
    "processing": ["analysis", "analyzing", "computation"],
    "computer": ["machine", "device", "system"],
    "vision": ["image", "visual", "recognition"],
    "language": ["text", "communication", "speech"],
    "deep": ["hierarchical", "layered", "complex"],
    "neural": ["brain-inspired", "connectionist"],
    "generative": ["creative", "producing", "creating"],
    "chatbot": ["assistant", "bot", "conversational"],
    "recognition": ["identification", "detection", "classification"]
}

# Pre-load stopwords once (avoid reloading on every call)
_STOP_WORDS = set(stopwords.words('english'))
_LEMMATIZER = WordNetLemmatizer()


def preprocess(text: str) -> str:
    """
    Preprocess text for keyword matching:
      1. Lowercase
      2. Expand AI abbreviations
      3. Remove non-alphanumeric characters
      4. Tokenize, remove stopwords, lemmatize
    """
    text = str(text).lower()

    for abbr, full_form in AI_SYNONYMS.items():
        text = re.sub(r'\b' + re.escape(abbr) + r'\b', full_form, text)

    # Remove everything except letters, numbers and spaces
    text = re.sub(r'[^a-z0-9 ]', '', text)

    words = word_tokenize(text)
    words = [w for w in words if w not in _STOP_WORDS]
    words = [_LEMMATIZER.lemmatize(w) for w in words]

    return " ".join(words)
