# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 09:59:35 2019

@author: yoliveira\
"""

import pandas as pd #dataframe manipulations
import nltk #tokenizer
import re #re.sub() - Data Cleaner
import unidecode #remover acentos


#Lê os dados do arquivo CSV
df = pd.read_excel("C:/Users/Yuri Oliveira/Desktop/TCC/tabela_codigo_do_objeto.xls", sep = ';')
df2 = pd.read_csv("C:/Users/Yuri Oliveira/Desktop/TCC/Licitacoes_2019.csv", sep = ';')

#Converte todas as palavras para letra minuscula
df.Especificação = df.Especificação.str.lower()

#Remove acentos
df['Especificação'] = df.Especificação.apply(lambda x: unidecode.unidecode(x))

#Remove caracteres especiais e numeros
df['Especificação'] = df.Especificação.apply(lambda x: re.sub('[^a-zA-Z]+', ' ', x))

#Removing StopWords from the strings
stop = nltk.corpus.stopwords.words('portuguese')
df['Especificação'] = df.Especificação.apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

#Tokenizing
df['tokenized_sents'] = df.apply(lambda row: nltk.word_tokenize(row['Especificação']), axis=1)

#transforma numma lista de lista para alimentar o LDA
teste = list(df.tokenized_sents.values)

from gensim import corpora
dct = corpora.Dictionary(teste)
corpus = [dct.doc2bow(line) for line in teste]

#https://www.machinelearningplus.com/nlp/gensim-tutorial/#11howtocreatetopicmodelswithlda
from gensim import models
from gensim import similarities
lsi = models.LsiModel(corpus, id2word=dct, num_topics=100)


index = similarities.MatrixSimilarity(lsi[corpus])


doc = "gas ENGARRAFADO DIVERSOS USOS"
vec_bow = dct.doc2bow(doc.lower().split())
vec_lsi = lsi[vec_bow]  # convert the query to LSI space
#print(vec_lsi)

sims = index[vec_lsi]
#print(list(enumerate(sims)))

sims = sorted(enumerate(sims), key=lambda item: -item[1])
for i, s in enumerate(sims):
    print(s, df.Descrição[i])


#pesquisar uma string no dataframe
#df[df['de_Obs'].str.contains('oi celular')]

#Conta a frequencia de todas as palavras do dataframe
df["freq"] = df.tokenized_sents.apply(lambda x: ' '.join(x))
freq = df.freq.str.split(expand=True).stack().value_counts()

#Frequencia contratacao/servico
df_contratacao = df[df['de_Obs'].str.contains('contratacao|servico')]
df["freq"] = df.tokenized_sents.apply(lambda x: ' '.join(x))
freq_contratacao = df_contratacao.freq.str.split(expand=True).stack().value_counts()

df_locServ = df_contratacao[df_contratacao['de_Obs'].str.contains('locacao')]


'''
df['de_Obs'] = df['de_Obs'].apply(lambda x: nlp(x))


tokens = []
lemma = []
pos = []

for doc in nlp.pipe(df['de_Obs'].astype('unicode').values, n_threads=3):
    if doc.is_parsed:
        tokens.append([n.text for n in doc])
        lemma.append([n.lemma_ for n in doc])
        pos.append([n.pos_ for n in doc])
    else:
        # We want to make sure that the lists of parsed results have the
        # same number of entries of the original Dataframe, so add some blanks in case the parse fails
        tokens.append(None)
        lemma.append(None)
        pos.append(None)

df['species_tokens'] = tokens
df['species_lemma'] = lemma
df['species_pos'] = pos
'''






'''
#Lemmatization
from nltk.stem import WordNetLemmatizer 
lemmatizer = WordNetLemmatizer() 


#Tokenizing the text
df['tokenized_sents'] = df.apply(lambda row: nltk.word_tokenize(row['df_without_stopwords']), axis=1)





#stemming the text (demora pra carai)
stemmer = nltk.stem.RSLPStemmer()
df['stemmed'] = df["tokenized_sents"].apply(lambda x: [stemmer.stem(y) for y in x])





from nltk.corpus import wordnet

syns = wordnet.synsets("program")

print(syns[0].name())
print(syns[0].lemmas()[0].name())
print(syns[0].definition())
print(syns[0].examples())

#http://wordnet.pt/
#https://babelnet.org/guide
#http://compling.hss.ntu.edu.sg/omw/summx.html
#http://ontopt.dei.uc.pt/index.php?sec=consultar
#http://www.clul.ulisboa.pt/en/
#http://multiwordnet.fbk.eu/online/multiwordnet.php
#https://github.com/own-pt/openWordnet-PT/wiki

#http://babelscape.com/doc/pythondoc/pybabelnet.html
#https://sites.google.com/site/renatocorrea/temas-de-interesse/processamento-de-linguagem-natural
#https://imasters.com.br/back-end/aprendendo-sobre-web-scraping-em-python-utilizando-beautifulsoup
'''
