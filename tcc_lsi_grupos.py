# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 09:59:35 2019

@author: yoliveira\
"""

import pandas as pd #dataframe manipulations
import nltk #tokenizer
import re #re.sub() - Data Cleaner
import unidecode #remover acentos

'''
        Tratamento dos dados:
            *Converte todas as palavras para letra minuscula
            *Remove acentos
            *Remove caracteres especiais e numeros
            *Remove StopWords
            *Tokenizing
            *Lista 
            
'''
#Lê os dados do arquivo CSV
df = pd.read_excel("C:/Users/Yuri Oliveira/Desktop/TCC_TCE/tabela_codigo_do_objeto.xls", sep = ';')
df_licitacoes2019 = pd.read_csv("C:/Users/Yuri Oliveira/Desktop/TCC_TCE/Licitacoes_2019.csv", encoding = "ISO-8859-1", sep = ';', usecols = ["objeto"])

#Converte todas as palavras para letra minuscula
df.Especificação = df.Especificação.str.lower()
df_licitacoes2019.objeto = df_licitacoes2019.objeto.str.lower()

#Remove acentos
df['Especificação'] = df.Especificação.apply(lambda x: unidecode.unidecode(x))
df_licitacoes2019['objeto'] = df_licitacoes2019.objeto.apply(lambda x: unidecode.unidecode(x))

#Remove caracteres especiais e numeros
df['Especificação'] = df.Especificação.apply(lambda x: re.sub('[^a-zA-Z]+', ' ', x))
df_licitacoes2019['objeto'] = df_licitacoes2019.objeto.apply(lambda x: re.sub('[^a-zA-Z]+', ' ', x))

#Remove StopWords
stop = nltk.corpus.stopwords.words('portuguese')
df['Especificação'] = df.Especificação.apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
df_licitacoes2019['objeto'] = df_licitacoes2019.objeto.apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

#Tokenizing
df['tokenized_sents'] = df.apply(lambda row: nltk.word_tokenize(row['Especificação']), axis=1)
df_licitacoes2019['tokenized_sents'] = df_licitacoes2019.apply(lambda row: nltk.word_tokenize(row['objeto']), axis=1)

#stemming the text (se quiser usar o stemming, só descomentar as 3 linhas abaixo)
#stemmer = nltk.stem.RSLPStemmer()
#df['tokenized_sents'] = df["tokenized_sents"].apply(lambda x: [stemmer.stem(y) for y in x])
#df_licitacoes2019['tokenized_sents'] = df_licitacoes2019["tokenized_sents"].apply(lambda x: [stemmer.stem(y) for y in x])

#transforma numma lista de lista para alimentar o LDA
lista = list(df.tokenized_sents.values)
lista_licitacoes = list(df_licitacoes2019.tokenized_sents.values)

'''
    Fim do Tratamento dos dados
'''

'''
    LSI 
'''
from gensim import corpora
from gensim import models
from gensim import similarities

#https://www.machinelearningplus.com/nlp/gensim-tutorial/#11howtocreatetopicmodelswithlda

dct = corpora.Dictionary(lista)
corpus = [dct.doc2bow(line) for line in lista]

#Modelo LSI (20 topicos)
lsi = models.LsiModel(corpus, id2word=dct, num_topics=250)

#cria a matriz de similaridade dos grupos
index = similarities.MatrixSimilarity(lsi[corpus])

#descricao da licitacao que sera comparada com os grupos
#transforma a descricao no espaco vetorial do LSI
licitacao_entrada = "aquisicao generos alimenticios diretamente agricultura familiar empreendedor familiar rural organizacoes destinado programa nacional alimentacao escolar pnae"
vec_bow = dct.doc2bow(licitacao_entrada.lower().split())
vec_lsi = lsi[vec_bow]  # convert the query to LSI space

#Armazena a similaridade da entrada com cada um dos grupos
sims = index[vec_lsi]

#Mostra os 5 grupos mais similares com a licitacao de entrada
sims = sorted(enumerate(sims), key=lambda item: -item[1])
for i, s in enumerate(sims[0:5]):
    print(s, df.Descrição[s[0]])
'''
    Fim do LSI
'''

'''
     Rotula as Licitacoes
'''
#cria a coluna "classificacao" no dataframe
df_licitacoes2019['classificacao'] = ""

def maiorSimilaridade(licitacao_entrada):
    vec_bow = dct.doc2bow(licitacao_entrada)
    vec_lsi = lsi[vec_bow]  # convert the query to LSI space
    #Armazena a similaridade da entrada com cada um dos grupos
    sims = index[vec_lsi]
    #ordena as similaridades em ordem decrescente
    sims = sorted(enumerate(sims), key=lambda item: -item[1])
    #retorna o grupo que possui a maior similaridade
    return df.Descrição[sims[0][0]]

#Classificando todas as licitacoes
df_licitacoes2019['classificacao'] = df_licitacoes2019.apply(lambda row: maiorSimilaridade(row['tokenized_sents']), axis=1)


df_testando = df_licitacoes2019[df_licitacoes2019['classificacao'].str.contains('GÊNEROS ALIMENTÍCIOS')]

'''
    Fim do Rotula as Licitacoes
'''
#pesquisar uma string no dataframe
#df[df['de_Obs'].str.contains('oi celular')]
'''
#Conta a frequencia de todas as palavras do dataframe
df["freq"] = df.tokenized_sents.apply(lambda x: ' '.join(x))
freq = df.freq.str.split(expand=True).stack().value_counts()

#Frequencia contratacao/servico
df_contratacao = df[df['de_Obs'].str.contains('contratacao|servico')]
df["freq"] = df.tokenized_sents.apply(lambda x: ' '.join(x))
freq_contratacao = df_contratacao.freq.str.split(expand=True).stack().value_counts()

df_locServ = df_contratacao[df_contratacao['de_Obs'].str.contains('locacao')]
'''

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
df_licitacoes2019['tokenized_sents'] = df.apply(lambda row: nltk.word_tokenize(row['df_without_stopwords']), axis=1)





#stemming the text (demora pra carai)
stemmer = nltk.stem.RSLPStemmer()
df_licitacoes2019['stemmed'] = df_licitacoes2019["tokenized_sents"].apply(lambda x: [stemmer.stem(y) for y in x])





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
