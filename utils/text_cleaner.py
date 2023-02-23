import re
import html
import nltk
import string
import unicodedata
from nltk.corpus import stopwords


class Cleaner():
    ''' clean text before feed it into Bert model for embedding'''
    def __init__(self):
        nltk.download('stopwords')
        self.stop = stopwords.words('english')

    def remove_digits(self, doc):
        '''there are numbers concatenated to words so we need remove them'''
        pattern = '[0-9]'
        doc = re.sub(pattern, '', doc)
        return doc

    def remove_special_chars(self, doc):
        re1 = re.compile(r'  +')
        x1 = doc.replace('#39;', "'").replace('amp;', '&').replace('#146;', "'").replace(
            'nbsp;', ' ').replace('#36;', '$').replace('\\n', "\n").replace('quot;', "'").replace(
            '<br />', "\n").replace('\\"', '"').replace('<unk>', 'u_n').replace(' @.@ ', '.').replace(
            ' @-@ ', '-').replace('\\', ' \\ ')
        return re1.sub(' ', html.unescape(x1))

    def remove_non_ascii(self, doc):
        """Remove non-ASCII characters from list of tokenized words"""
        return unicodedata.normalize('NFKD', doc).encode('ascii', 'ignore').decode('utf-8', 'ignore')

    def remove_punctuation(self, doc):
        """Remove punctuation from list of tokenized words"""
        translator = str.maketrans('', '', string.punctuation)
        return doc.translate(translator)
    
    def remove_double_spaces(self, doc):
        return re.sub(' +', ' ', doc)

    def remove_stop_words(self, doc):
        doc = ' '.join([word for word in doc.split() if word not in (self.stop)])
        return doc
    
    def remove_double_spaces(self, doc):
        return re.sub(' +', ' ', doc)

    def clean(self, doc):
        ''' put all the above functions togther to complete the cleaning process '''
        # the received doc as dict
        # we will be using content and titles only
        
        doc = self.remove_punctuation(str(doc))
        doc = self.remove_double_spaces(str(doc))
        doc = self.remove_special_chars(str(doc))
        doc = self.remove_non_ascii(str(doc))
        doc = str(doc).lower()
        doc = self.remove_stop_words(str(doc))
        doc = str(doc).strip()
        return doc