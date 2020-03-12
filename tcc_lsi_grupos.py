# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 09:59:35 2019

@author: yoliveira\
"""

import pandas as pd #dataframe manipulations
import nltk #tokenizer
import re #re.sub() - Data Cleaner
import unidecode #remover acentos
import gc #garbage collector (para remover variaveis da memória que não estão sendo mais utilizadas)

'''
        Tratamento dos dados:
            *Converte todas as palavras para letra minuscula
            *Remove acentos
            *Remove caracteres especiais e numeros
            *Remove StopWords
            *Tokenizing
            *Stemming
            *Remove palavras de tamanho < 3
            *Lista que alimenta LSI
            
'''
#Lê os dados do arquivo CSV
df = pd.read_excel("C:/Users/Yuri Oliveira/Desktop/TCC_TCE/tabela_codigo_do_objeto.xls", sep = ';')
df_licitacoes2019 = pd.read_csv("C:/Users/Yuri Oliveira/Desktop/TCC_TCE/Licitacoes_2019.csv", encoding = "ISO-8859-1", sep = ';', usecols = ["objeto"])
#df_licitacoes2019 = pd.read_csv("C:/Users/Yuri Oliveira/Desktop/TCC_TCE/licitacoes.csv", sep = ';', usecols = ["de_Obs"])
#df_licitacoes2019.columns = ['objeto']
#Coloca a descrição do grupo na especificação também
df['Especificação'] = df.Especificação + " " + df.Descrição

#Converte todas as palavras para letra minuscula
df.Especificação = df.Especificação.str.lower()
df_licitacoes2019.objeto = df_licitacoes2019.objeto.str.lower()

#Remove acentos
df['Especificação'] = df.Especificação.apply(lambda x: unidecode.unidecode(x))
df_licitacoes2019['objeto'] = df_licitacoes2019.objeto.apply(lambda x: unidecode.unidecode(str(x)))

#Remove caracteres especiais e numeros
df['Especificação'] = df.Especificação.apply(lambda x: re.sub('[^a-zA-Z]+', ' ', x))
df_licitacoes2019['objeto'] = df_licitacoes2019.objeto.apply(lambda x: re.sub('[^a-zA-Z]+', ' ', x))

#Remove StopWords
stop = nltk.corpus.stopwords.words('portuguese')
newStopWords = ['adesao','aquisicao','servico','servicos','afins',
                'destinada','geral','via','etc','utilizados',
                'outros','uso','nao','caso','tais','qualquer',
                'neste','compreende','publicos','ate','todos',
                'ser','destinacao','prestados','diversos','usos',
                'abastecimento','zona','rural','pregao','presencial',
                'contratacao','municipio','municipal','empresa',
                'atender','necessidades','destinados','registro',
                'especializada','conforme','fornecimento','prestacao',
                'secretarias','sao','municipio','destinado','joao',
                'execucao','forma','grande','tipo','demanda','jose','ata',
                'rede','redes','leva','fim','menores','parcela','parcelas',
                'populacao','produtos','bem','derivado','derivados',
                'pb','aquisicoes']
stop.extend(newStopWords)
df['Especificação'] = df.Especificação.apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
df_licitacoes2019['objeto'] = df_licitacoes2019.objeto.apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

#Tokenizing
df['tokenized_sents'] = df.apply(lambda row: nltk.word_tokenize(row['Especificação']), axis=1)
df_licitacoes2019['tokenized_sents'] = df_licitacoes2019.apply(lambda row: nltk.word_tokenize(row['objeto']), axis=1)

#stemming the text (se quiser usar o stemming, só descomentar as 3 linhas abaixo)
stemmer = nltk.stem.RSLPStemmer()
df['tokenized_sents'] = df["tokenized_sents"].apply(lambda x: [stemmer.stem(y) for y in x])
df_licitacoes2019['tokenized_sents'] = df_licitacoes2019["tokenized_sents"].apply(lambda x: [stemmer.stem(y) for y in x])

#Removendo "palavras" menores que 3
#df_licitacoes2019['tokenized_sents'] = df_licitacoes2019.tokenized_sents.apply(lambda x:[x.remove(palavra) if len(palavra) < 3 else palavra for palavra in x])
#df['tokenized_sents'] = df.tokenized_sents.apply(lambda x:[x.remove(palavra) if len(palavra) < 3 else palavra for palavra in x])

#removing Nones
df_licitacoes2019['tokenized_sents'] = df_licitacoes2019.tokenized_sents.apply(lambda x: list(filter(None, x)))
df['tokenized_sents'] = df.tokenized_sents.apply(lambda x: list(filter(None, x)))

#retira tokens duplicados
df_licitacoes2019['tokenized_sents'] = df_licitacoes2019.tokenized_sents.apply(lambda x: list(set(x)))
df['tokenized_sents'] = df.tokenized_sents.apply(lambda x: list(set(x)))


#transforma numma lista de lista para alimentar o LSI
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

#Modelo LSI (100 topicos e 100 power_iterations)
lsi = models.LsiModel(corpus, id2word=dct, num_topics=100, power_iters = 100)

#cria a matriz de similaridade dos grupos
index = similarities.MatrixSimilarity(lsi[corpus])

'''
    Fim do LSI
'''
#Funcao pra testar com uma unica licitacao(index do dataframe) e mostra os 5 grupos mais similares
def maisSimilares(index_licitacao):
    #transforma a descricao da licitacao no espaco vetorial do LSI
    vec_bow = dct.doc2bow(df_licitacoes2019.tokenized_sents[index_licitacao])
    vec_lsi = lsi[vec_bow]  # convert the query to LSI space
    
    #Armazena a similaridade da entrada com cada um dos grupos
    sims = index[vec_lsi]
    
    #Mostra os 5 grupos mais similares com a licitacao de entrada
    sims = sorted(enumerate(sims), key=lambda item: -item[1])
    for i, s in enumerate(sims[0:5]):
        print(s, df.Descrição[s[0]])

'''
     Rotula as Licitacoes
'''
#cria a coluna "classificacao" no dataframe
def maiorSimilaridade(licitacao_entrada):
    #transforma a descricao no espaco vetorial do LSI
    vec_bow = dct.doc2bow(licitacao_entrada)
    vec_lsi = lsi[vec_bow]  # convert the query to LSI space
    #Armazena a similaridade da entrada com cada um dos grupos
    sims = index[vec_lsi]
    #ordena as similaridades em ordem decrescente
    sims = sorted(enumerate(sims), key=lambda item: -item[1])
    #retorna o grupo que possui a maior similaridade
    #if sims[0][1] > 0.65:
    if sims[0][1] != 0:
        return df.Descrição[sims[0][0]]
    else:
        return "outro"

#retorna a similaridade do grupo mais similar a licitacao
def maiorSimilaridade1(licitacao_entrada):
    #transforma a descricao no espaco vetorial do LSI
    vec_bow = dct.doc2bow(licitacao_entrada)
    vec_lsi = lsi[vec_bow]  # convert the query to LSI space
    #Armazena a similaridade da entrada com cada um dos grupos
    sims = index[vec_lsi]
    #ordena as similaridades em ordem decrescente
    sims = sorted(enumerate(sims), key=lambda item: -item[1])
    #retorna o grupo que possui a maior similaridade
    return sims[0][1]

#Classificando todas as licitacoes
df_licitacoes2019['classificacao'] = df_licitacoes2019.apply(lambda row: maiorSimilaridade(row['tokenized_sents']), axis=1)
df_licitacoes2019['similaridade'] = df_licitacoes2019.apply(lambda row: maiorSimilaridade1(row['tokenized_sents']), axis=1)

freq_grupos = df_licitacoes2019.classificacao.value_counts()

#Top 10 licitações escolhidas para compor a pesquisa que será mostrada nos resultados
#Esta pesquisa tem como objetivo verificar a porcentagem de acerto que o algoritmo teve
#Os dados serão utilizados em uma matriz de confusão
df_ga = df_licitacoes2019[(df_licitacoes2019.classificacao == 'GÊNEROS ALIMENTÍCIOS') & (df_licitacoes2019.similaridade > 0.65)]
df_lv = df_licitacoes2019[(df_licitacoes2019.classificacao == 'LOCAÇÃO DE VEÍCULOS') & (df_licitacoes2019.similaridade > 0.65)]
df_li = df_licitacoes2019[(df_licitacoes2019.classificacao == 'LOCAÇÃO DE IMÓVEIS') & (df_licitacoes2019.similaridade > 0.65)]
df_c = df_licitacoes2019[(df_licitacoes2019.classificacao == 'CONSULTORIA') & (df_licitacoes2019.similaridade > 0.65)]
df_o = df_licitacoes2019[(df_licitacoes2019.classificacao == 'OBRAS') & (df_licitacoes2019.similaridade > 0.65)]
df_cp = df_licitacoes2019[(df_licitacoes2019.classificacao == 'FORNECIMENTO DE ÁGUA POTÁVEL EM CAMINHÃO-PIPA') & (df_licitacoes2019.similaridade > 0.65)]
df_sa = df_licitacoes2019[(df_licitacoes2019.classificacao == 'SERVIÇOS PRESTADOS POR PROFISSIONAL DO SETOR ARTÍSTICO') & (df_licitacoes2019.similaridade > 0.65)]
df_st = df_licitacoes2019[(df_licitacoes2019.classificacao == 'SERVIÇO DE MANUTENÇÃO E SUPORTE TÉCNICO DE EQUIPAMENTOS DE INFORMÁTICA') & (df_licitacoes2019.similaridade > 0.65)]
df_tp = df_licitacoes2019[(df_licitacoes2019.classificacao == 'SERVIÇOS DE TRANSPORTE DE PASSAGEIROS') & (df_licitacoes2019.similaridade > 0.65)]
df_cl = df_licitacoes2019[(df_licitacoes2019.classificacao == 'COMBUSTÍVEIS E LUBRIFICANTES') & (df_licitacoes2019.similaridade > 0.65)]


#pega uma amostra de cada grupo que será utilizado na pesquisa para obtencao dos resultados
df_pesquisa = pd.concat([df_ga.sample(50), df_lv.sample(50), df_li.sample(50), df_c.sample(50), df_o.sample(50),
                         df_cp.sample(50), df_sa.sample(50), df_st.sample(50), df_tp.sample(50), df_cl.sample(50)])
    
#deleta os dataframes não mais utilizados
del [[df_ga,df_lv,df_li,df_c,df_o,df_cp,df_sa,df_st,df_tp,df_cl]]
gc.collect()

#dataframes de referencia
df_gref = pd.read_excel("C:/Users/Yuri Oliveira/Desktop/TCC_TCE/tabela_codigo_do_objeto.xls", sep = ';')
df_lref = pd.read_csv("C:/Users/Yuri Oliveira/Desktop/TCC_TCE/Licitacoes_2019.csv", encoding = "ISO-8859-1", sep = ';', usecols = ["objeto"])

df_gref.columns = ['codigo', 'nome_grupo', 'especificacao']

#joining dataframes
df_pesquisa = pd.merge(df_pesquisa, df_lref, left_index=True, right_index=True)
df_pesquisa = pd.merge(df_pesquisa, df_gref, left_on = 'classificacao', right_on = 'nome_grupo')

#Extraindo apenas as colunas que serão utilizadas na pesquisa
cols = [4,6,7]
df_pesquisa = df_pesquisa[df_pesquisa.columns[cols]]

#escreve o dataframe num arquivo csv
df_pesquisa.to_csv(r'C:/Users/Yuri Oliveira/Desktop/TCC_TCE/dados_pesquisa.csv', index = False, sep = ';')
'''
    Fim de Rotula as Licitacoes
'''

'''
    Testando Licitacoes
'''



#freq_grupos = df_pesquisa.classificacao.value_counts()

#mostra todas as classificacoes de um determinado tipo
#df_testando = df_licitacoes2019[df_licitacoes2019['classificacao'].str.contains('MATERIAL PEDAGÓGICO E DE RECREAÇÃO')]

#pesquisa quais sao os 5 grupos mais relevantes de uma determinada licitacao(pegar indice do dataframe)
#maisSimilares(3)

#Conta a frequencia de todas as palavras do dataframe
#df_pesquisa["freq"] = df_licitacoes2019.tokenized_sents.apply(lambda x: ' '.join(x))
#freq = df_pesquisa.freq.str.split(expand=True).stack().value_counts()

'''
    Fim de Testando Licitacoes
'''

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
