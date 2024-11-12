import nltk
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy

# Descargar el tokenizer (si es necesario)
nltk.download("punkt")

# Ejemplo de datos etiquetados
data = [
    ("Me encanta esta película", "positivo"),
    ("No soporto esta película", "negativo"),
    ("La trama fue interesante y cautivadora", "positivo"),
    ("El final fue muy decepcionante", "negativo"),
    ("Las actuaciones fueron brillantes", "positivo"),
    ("No recomiendo ver esta película", "negativo"),
    ("Esta película fue increíble", "positivo"),
    ("La historia es aburrida", "negativo"),
    ("Los efectos especiales fueron impresionantes", "positivo"),
    ("El diálogo fue flojo", "negativo"),
    ("El protagonista actuo de forma sossa", "positivo"),
]
"""data = [
    ("I love this movie", "positive"),
    ("I can't stand this movie", "negative"),
    ("The plot was interesting and captivating", "positive"),
    ("The ending was very disappointing", "negative"),
    ("The acting was brilliant", "positive"),
    ("I wouldn't recommend watching this movie", "negative"),
    ("This movie was amazing", "positive"),
    ("The story is boring", "negative"),
    ("The special effects were impressive", "positive"),
    ("The dialogue was weak", "negative"),
]"""
# Función para transformar oraciones en palabras
def to_feature_dict(text):
    words = nltk.word_tokenize(text)
    return {word: True for word in words}

# Transform data to feature dictionary format
labeled_data = [(to_feature_dict(text), label) for text, label in data]
# Split data into training and testing sets (80% training, 20% testing)
train_data = labeled_data[:8]
test_data = labeled_data[8:]

# Train the classifier
classifier = NaiveBayesClassifier.train(train_data)

# Evaluate accuracy on the test set
print(f"Accuracy: {accuracy(classifier, test_data) * 100:.2f}%")
# Classify a new text
new_text = "I really enjoyed this movie"
features = to_feature_dict(new_text)
print("Sentiment:", classifier.classify(features))
