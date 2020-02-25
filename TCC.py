# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 09:59:35 2019

@author: yoliveira\
"""

import numpy as np #mathematics manipulations
from matplotlib import pyplot as plt #plots 
import pandas as pd #dataframe manipulations
import nltk #tokenizer
import re #re.sub() - Data Cleaner
#from nltk.corpus import wordnet #synonymous 
#nltk.set_proxy('https://10.10.11.15:8080', ('yoliveira', 'password'))
#nltk.download()
#import spacy
#from spacy.lang.pt.examples import sentences 
import unidecode #remover acentos

#nlp = spacy.load("pt_core_news_sm")

#Lê os dados do arquivo CSV
df = pd.read_csv("C:/Users/Yuri Oliveira/Desktop/TCC/dados_tcc.csv", sep = ';')
#file_loc = "C:/Users/Yuri Oliveira/Desktop/TCC/DadosIBGE.xls"
#municipios = pd.read_excel(file_loc, na_values=['NA'], sheet_name = "MunicipioIBGE", parse_cols = [2])
#municipios pode ser usado para limpar mais ainda os dados, retirando o nome de municipios do modelo


#testes com uma amostra
#df = df.head(1000)

#Procura por NAs
df.loc[df['de_Obs'].isnull()]

#Elimina os NAs. inplace = True -> alteração irá modificiar o objeto sem a necessidade de atribuição
df.dropna(inplace = True)

#Converte todas as palavras para letra minuscula
df.de_Obs = df.de_Obs.str.lower()

#Remove acentos
df['de_Obs'] = df.de_Obs.apply(lambda x: unidecode.unidecode(x))

#Remove caracteres especiais e numeros
df['de_Obs'] = df.de_Obs.apply(lambda x: re.sub('[^a-zA-Z]+', ' ', x))

#Removing StopWords from the strings
stop = nltk.corpus.stopwords.words('portuguese')
df['de_Obs'] = df.de_Obs.apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

#Tokenizing
df['tokenized_sents'] = df.apply(lambda row: nltk.word_tokenize(row['de_Obs']), axis=1)

#Removendo "palavras" menores que 3
df['tokenized_sents'] = df.tokenized_sents.apply(lambda x:[x.remove(palavra) if len(palavra) < 3 else palavra for palavra in x])

#Transforma celulas vazias em NaN para serem excluidas
#df['Tenant'].replace('', np.nan, inplace=True)

#removing Nones
df['tokenized_sents'] = df.tokenized_sents.apply(lambda x: list(filter(None, x)))

#transforma numma lista de lista para alimentar o LDA
teste = list(df.tokenized_sents.values)

from gensim import corpora
dct = corpora.Dictionary(teste)
corpus = [dct.doc2bow(line) for line in teste]

#https://www.machinelearningplus.com/nlp/gensim-tutorial/#11howtocreatetopicmodelswithlda
from gensim.models import LdaModel, LdaMulticore
lda_model = LdaMulticore(corpus=corpus,
                         id2word=dct,
                         random_state=100,
                         num_topics=50,
                         passes=10,
                         chunksize=1000,
                         batch=False,
                         alpha='asymmetric',
                         decay=0.5,
                         offset=64,
                         eta=None,
                         eval_every=0,
                         iterations=100,
                         gamma_threshold=0.001,
                         per_word_topics=True)

lda_model.print_topics(-1)
topics = lda_model.show_topics()

i=0
for c in lda_model[corpus[:2]]:
    print("Document: {}".format(i))
    print("Document Topics      : ", c[0])      # [(Topics, Perc Contrib)]
    print("Word id, Topics      : ", c[1][:3])  # [(Word id, [Topics])]
    print("Phi Values (word id) : ", c[2][:2])  # [(Word id, [(Topic, Phi Value)])]
    print("Word, Topics         : ", [(dct[wd], topic) for wd, topic in c[1][:2]])   # [(Word, [Topics])]
    print("Phi Values (word)    : ", [(dct[wd], topic) for wd, topic in c[2][:2]])  # [(Word, [(Topic, Phi Value)])]
    print("------------------------------------------------------\n")
    i = i+1

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
