
import ipdb
import nltk
import random
import string

from nltk.corpus import wordnet
from nltk.corpus import stopwords
from translate import Translator

from nltk.tokenize import word_tokenize, sent_tokenize
nltk.download('averaged_perceptron_tagger')

def remove_stop_words(input_text):
    """
    """
    stop_words = set(stopwords.words("english"))
    tokens = word_tokenize(input_text)
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text

def remove_punctuations(input_text):
    filtered_text = input_text.translate(str.maketrans('', '', string.punctuation))
    return filtered_text

def synonym_replacement(sentence, n=5):
    # Remove the punctuations 
    filtered_sentence = remove_punctuations(sentence) 

    # Remove the stop words 
    filtered_sentence = remove_stop_words(filtered_sentence)

    words = nltk.word_tokenize(filtered_sentence)
    for _ in range(n):
        word_to_replace = random.choice(words)
        synonyms = wordnet.synsets(word_to_replace)
        if synonyms:
            synonym = random.choice(synonyms).lemma_names()[0]
            sentence = sentence.replace(word_to_replace, synonym)
            
    return sentence

def random_insertion(sentence, n=2): 
    # Remove the punctuations 
    filtered_sentence = remove_punctuations(sentence) 

    # Remove the stop words 
    filtered_sentence = remove_stop_words(filtered_sentence)
    words = nltk.word_tokenize(filtered_sentence)
    for _ in range(n):
        word_to_insert = random.choice(words)
        word_to_insert_with = random.choice(words)
        sentence = sentence.replace(word_to_insert_with, word_to_insert_with + ' ' + word_to_insert)
    return sentence

def random_deletion(sentence, n=2): 
    # Remove the punctuations 
    filtered_sentence = remove_punctuations(sentence) 

    # Remove the stop words 
    filtered_sentence = remove_stop_words(filtered_sentence)
    words = nltk.word_tokenize(filtered_sentence)
    for _ in range(n):
        word_to_delete = random.choice(words)
        sentence = sentence.replace(word_to_delete,'')

    return sentence

def random_masking(sentence, n=2):
    # Remove the punctuations 
    filtered_sentence = remove_punctuations(sentence) 

    # Remove the stop words 
    filtered_sentence = remove_stop_words(filtered_sentence)
    words = nltk.word_tokenize(filtered_sentence)    
    for _ in range(n):
        word_to_mask = random.choice(words)
        sentence = sentence.replace(word_to_mask, "<MASK>")

    return sentence

def textual_entailment(sentence, n = 5):
    # Example: Change active voice to passive voice

    tokens = nltk.word_tokenize(sentence)
    tagged = nltk.pos_tag(tokens)
    random.shuffle(tagged)
    tagged = tagged[0:n]

    for word, tag in tagged:
        if tag.startswith('VB'):  # Verbs
            morhped_word = wordnet.morphy(word, wordnet.VERB)
            if morhped_word != None:
                new_word = morhped_word
            else:
                new_word = word
            sentence = sentence.replace(word, new_word)
        elif tag.startswith('NN'):  # Nouns
            morhped_word = wordnet.morphy(word, wordnet.NOUN)
            if morhped_word != None:
                new_word = morhped_word
            else:
                new_word = word
            sentence = sentence.replace(word, new_word)
        else:
            continue
        
    return sentence

def back_translate(text, target_language='fr', source_language='en'):
    translator = Translator(to_lang=target_language, from_lang=source_language)
    
    # Translate the text to the target language
    translated_text = translator.translate(text)
    
    # Translate the translated text back to the source language
    back_translated_text = translator.translate(translated_text, to_lang=source_language, from_lang=target_language)
    
    return back_translated_text