import nltk
import numpy as np
import random
import string 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ChatBot():

    def __init__(self):
        f=open('chatbot.txt','r')
        raw=f.read()
        raw=raw.lower()# converts to lowercase
        #nltk.download('punkt') # first-time use only
        #nltk.download('wordnet') # first-time use only
        self.sent_tokens = nltk.sent_tokenize(raw)# converts to list of sentences 
        self.word_tokens = nltk.word_tokenize(raw)# converts to list of words
        #self.GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up","hey",)
        #self.GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me"]
        print("Inside init")

    #WordNet is a semantically-oriented dictionary of English included in NLTK.
    @staticmethod
    def LemTokens(tokens):
        lemmer = nltk.stem.WordNetLemmatizer()
        return [lemmer.lemmatize(token) for token in tokens]

    @staticmethod
    def LemNormalize(text):
        remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
        return ChatBot.LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

    @staticmethod
    def greeting(sentence):
        GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up","hey",)
        GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me"]
        for word in sentence.split():
            if word.lower() in GREETING_INPUTS:
                return random.choice(GREETING_RESPONSES)

    @staticmethod            
    def response(self,user_response):
        robo_response=''
        TfidfVec = TfidfVectorizer(tokenizer= ChatBot.LemNormalize, stop_words='english')
        tfidf = TfidfVec.fit_transform(self.sent_tokens)
        vals = cosine_similarity(tfidf[-1], tfidf)
        idx=vals.argsort()[0][-2]
        flat = vals.flatten()
        flat.sort()
        req_tfidf = flat[-2]
        if(req_tfidf==0):
            robo_response=robo_response+"I am sorry! I don't understand you"
            return robo_response
        else:
            robo_response = robo_response+self.sent_tokens[idx]
            return robo_response

    
    def Chat_with_Bot(self,user_response):
        bot_resp = []
        if(user_response!='bye'):
            if(user_response=='thanks' or user_response=='thank you' ):
                bot_resp = "E-Com: You are welcome.."
            else:
                if(ChatBot.greeting(user_response)!=None):
                    bot_resp = "E-Com: "+ ChatBot.greeting(user_response)
                else:
                    self.sent_tokens.append(user_response)
                    self.word_tokens=  self.word_tokens+nltk.word_tokenize(user_response)
                    final_words=list(set(self.word_tokens))
                    #print("ROBO: ",end="")
                    bot_resp = "E-Com: "+ ChatBot.response(self,user_response) 
                    self.sent_tokens.remove(user_response)
        return bot_resp