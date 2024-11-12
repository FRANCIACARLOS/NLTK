# División de una oración en palabras individuales
import nltk
from nltk.tokenize import word_tokenize
nltk.download("punkt")
sentence="NLKT es una biblioteca de procesamiento de lenguaje natural."
tokens=word_tokenize(sentence)
print(tokens)
