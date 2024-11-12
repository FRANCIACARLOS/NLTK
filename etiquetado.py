# asignar etiquetas o categorías a los textos en función de ciertos criterios. Pag.10
import nltk
nltk.download('averaged_perceptron_tagger')
from nltk import pos_tag
from nltk.tokenize import word_tokenize

sentence = "NLTK es una biblioteca de Python para el procesamiento de lenguaje natural."
tokens = word_tokenize (sentence)
tagged_words = pos_tag (tokens)
print (tagged_words)