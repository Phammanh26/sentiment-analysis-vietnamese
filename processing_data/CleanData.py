import unicodedata
import re
from pyvi import ViTokenizer

class CleanData:
    def __init__(self) -> None:
        pass 
    
    def processing_list_text(self, list_texts):
        print('start processing list text, size: {}'.format(len(list_texts)))
        new_list_texts = []
        for text in list_texts:
            clean_text = self.preprocess_sentence(text)
            new_list_texts.append(clean_text[:-1])
        print('finish processing list text, size: {}'.format(len(list_texts)))
        return new_list_texts

    def split_special_character(self, word: str) -> str:
        '''
        add white space between normal word and special character.
        '''
        _tail = '""''v@_!#$%^&*()<>?/\|}{~:;[],.+-\n'
        _head = '@_!#$%^&*()<>?/\|}{~:;[],.+-'
        word_size = len(word)

        #split tail character in word
        for i in reversed(range(word_size)):
            if word[i] not in _tail:
                break
        if i != word_size - 1:
            word = word[0: i + 1] + " " + word[i + 1:]
       
        #split head character in word
        for i in range(word_size):
            if word[i] not in _head:
                break
        if i != 0:
            word = word[0: i] + " " + word[i:]            
            return word
        return word


    def replace_special_character(self, word: str):

        characters = {'™': '*', '‘': "'", '®': 'x', '×': 'x', '😀': 'x', '‑': '-',
                    '́': 'x', '—': ' - ', '̣': 'x', '–': '-', '`': "'", '“': '"', '̉': 'x',
                    '’': "'", '̃': 'x', '\u200b': 'x', '̀': 'x', '”': '"', '…': '...',
                    '\ufeff': 'x', '″': '"'}
  
        for c in characters:
            if c in word:
                word = word.replace(c, characters[c])

        
        return word


    def preprocess_sentence(self, sentence) -> str:

        # sentence = preprocess_stopword(sentence, list_stopwords)
        words = sentence.split(' ')
        norm_words = []

        for i in range(len(words)):
            word = words[i]
            word = unicodedata.normalize('NFC', word)

            word = self.split_special_character(word)
            word = self.replace_special_character(word)

            norm_words.append(word)
        
        norm_text = " ".join(norm_words)

        return norm_text


    def split_sentence(self, text: str) -> str:

        text = text.replace('\r', '')
        text = text.replace('\n', '. ')
        text += ' '
        sentences = text.split('. ')
        sentences = [sen.strip(' .') for sen in sentences]
        sentences = [sen for sen in sentences if sen != '']
        sentences = [sen + '.' for sen in sentences]

        return sentences


    def preprocess_stopword(sentence: str, list_stopwords: list) -> str:

        parts = []
        special_char = '@_!#$%^&*()<>?/\|}{~:;[],.+-'

        for i in special_char:
            sentence =  sentence.replace(i, ' ')

            sentence = re.sub(' +',' ', sentence)
            sent_segment = ViTokenizer.tokenize(sentence)
            words = sent_segment.split()
            
            for word in words:
                if "_" in word:
                    word_checking = " ".join(word.split("_"))
                else:
                    word_checking = word

                if word_checking.lower() not in list_stopwords:
                    parts.append(word)
        
        return " ".join(parts)

