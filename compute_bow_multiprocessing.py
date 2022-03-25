import spacy
import timeit
import math
import json
from tqdm import tqdm
import pandas as pd
import shutil
import matplotlib.pyplot as plt
from os import path
from collections import Counter
from lxml import etree
from glob import glob
from unicodedata import normalize
import multiprocessing
import concurrent.futures


def create_batches(path_name, n_processors):
    
    path = glob(path_name)
    
    batch_size = round(len(path)/80 - 1)
    
    batches = []
    
    marker = 0
    for i in range(n_processors):
        batch = path[marker:marker+batch_size]
        if len(path[marker+batch_size:]) < batch_size:
            batch += path[marker+batch_size:]
        batches.append(batch)
        marker += len(batch)
        
    return batches


def clean_text(txt):
    txt_res = normalize("NFKD", txt.replace('\xa0', ' '))
    txt_res = txt_res.replace('\\xa0', '')
    return txt_res


def pipeline_spacy(path):
    pos_ko = ["NUM", "X", "SYM", "PUNCT", "SPACE"]
    list_lemma = []
    list_pos = []
    nombre_tokens = 0
    with open(path, encoding="utf8") as file:
        text = file.readlines()
        text_clean = clean_text(str(text).lower())
        docs = nlp(text_clean)
        nombre_tokens += len(docs)
        for token in docs:
            #si le token est bien un mot on récupère son lemme
            if token.pos_ not in pos_ko:
                list_lemma.append(token.lemma_)
    return list_lemma, nombre_tokens


def bigrammize(list_token):
    """fonction qui prend en parametre une liste de tokens et retourne une liste de bi-grammes"""
    list_bigram = []
    for indice_token in range(len(list_token)-1):
        bigram = list_token[indice_token]+'_'+list_token[indice_token+1]
        list_bigram.append(bigram)
    return list_bigram


def dict_freq_token(list_select, list_lemma):
    
    dict_result = dict.fromkeys(list_select)
    
    dict_temp = Counter(list_lemma)
        
    for key in dict_temp.keys():
        if key in dict_result.keys():
            dict_result[key] = dict_temp[key]/len(list_lemma)
    
    return dict_result


def moulinette(batch, list_lemma_select, list_bigram_lemma_select):
    """fonction main qui utilise les fonctions précédentes et tourne sur le corpus"""
    
    nombre_total_tokens = 0
 
    dict_results = {}

    df_main = pd.DataFrame()
    
    #print("\n\nBEGIN PROCESSING CORPUS-----------")
    
    for doc in tqdm(batch):
        
        #print("\n\nBEGIN PROCESSING NOVEL-----------")

        doc_name = path.splitext(path.basename(doc))[0]
        #print(doc_name)
        
        #On recupere le texte des romans sous forme de listes de lemmes grâce à spacy
        
        list_lemma_temp, nombre_tokens = pipeline_spacy(doc)
        
        #print("PIPELINE SPACY ----------- OK")
        
        #print("NOMBRE TOKENS = ", nombre_tokens)
        
        nombre_total_tokens += nombre_tokens
        
        list_bigram_lemma_temp = bigrammize(list_lemma_temp)
        
        dict_results.update(dict_freq_token(list_lemma_select, list_lemma_temp))
        dict_results.update(dict_freq_token(list_bigram_lemma_select, list_bigram_lemma_temp))
        
        dict_results["index"] = doc_name
        
        df_temp = pd.DataFrame(dict_results, index=[0])
        
        df_main = df_main.append(df_temp, ignore_index = True)
        
        #print("PROCESS RESULTS ----------- OK")

        #print("END PROCESSING NOVEL --------------\n\n")
        #print("PROGRESSION ", round(nombre_total_tokens/31857823,2),'% COMPLETED\n')#*100 round 4
        
    df_main.set_index("index", inplace = True)
    
    #print("\n NOMBRE TOTAL TOKENS = ", nombre_total_tokens)   
    #print("\n\n END PROCESSING CORPUS --------------\n\n")
    
    
    return df_main


print('Importing language model')
nlp = spacy.load('fr_core_news_sm')


if __name__ == '__main__':


    print('Importing lists')
    with open('../auxiliary_lists/list_lemma.json', 'r', encoding='utf8') as f:
        lemma_selected = json.load(f)
    
    with open('../auxiliary_lists/list_bigram.json', 'r', encoding='utf8') as f:    
        bigram_lemma_selected = json.load(f)

    print('Creating batches')
    path_name = r"../chunked_corpus_lemma_txt/*.txt"
    n_processors = multiprocessing.cpu_count()
    batches = create_batches(path_name, n_processors)

    batches = [batch[:10] for batch in batches[:n_processors]]

    batch_dfs = []

    print('Starting multiprocessing')
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_processors) as executor:
        
        results = [executor.submit(moulinette, batch, lemma_selected, bigram_lemma_selected) for batch in batches]
        
        for f in concurrent.futures.as_completed(results):
            #print(type(f.result()))
            batch_dfs.append(f.result())
            df_final = pd.concat(batch_dfs, axis=0)

    df_final.to_csv(r'chunks_BoW_features.csv', index = True)


