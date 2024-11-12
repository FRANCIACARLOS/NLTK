# asignar etiquetas a las palabras en un texto para indicar su función gramatical o su categoría semántica
from nltk.stem import PorterStemmer

words=["running", "plays", "jumped"]
stemmer=PorterStemmer()
stems =[stemmer.stem(word) for word in words]
print (stems)

