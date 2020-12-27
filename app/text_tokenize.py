import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re


def text_tokenizer(text):
    """
       Process the text and Lemmatize it

       Parameters
       ----------
       text(str)

       Returns
       -------
       lem(list): Lemmatized text

    """
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower()) #Text to lower and remove punct
    text = word_tokenize(text)
    text = [w for w in text if w not in stopwords.words("english")] #remove stopwords

    lem = [WordNetLemmatizer().lemmatize(w) for w in text] # Lemmatize with v pos tag

    return lem