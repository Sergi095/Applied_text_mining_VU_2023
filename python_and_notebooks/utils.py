import pandas as pd 
import numpy as np
import spacy
import functools
import warnings
import re

warnings.filterwarnings("ignore")

nlp = spacy.load("en_core_web_sm")

def build_dataframe(file_path: str) -> pd.DataFrame:
    data = []
    final_df = pd.DataFrame() 
    with open(file_path, 'r') as file:
        for line in file:
            if line != '\n' :
                data.append(line.strip().split('\t'))
    df = pd.DataFrame(data)
    df = df.rename(columns={0: 'file',
                            1: 'nSentence',
                            2: 'nToken',
                            3: 'token',
                            4: 'negCue'})
    
    final_df = pd.DataFrame()
    for _file in df['file'].unique():
        file_df = df[df['file']==_file]
        sentences = file_df.groupby('nSentence')['token'].apply(lambda x: ' '.join(x)).reset_index()
        new_df = pd.DataFrame({'nSentence': sentences['nSentence'], 'sentence': sentences['token']})
        new_df = pd.merge(new_df, file_df, on='nSentence')
        df_ = applySentenceGroupBy(new_df)
        final_df = final_df.append(df_)
    negCue_dict = {"negCue":     {"O": 0, "B-NEG": 1, "I-NEG": 2}}
    final_df = final_df.replace(negCue_dict)
    return final_df

def check_affixes(token: str) -> bool:


    prefixes = ["un", "dis", "ir", "im", "in"]

    if any(token.lower().startswith(prefix) for prefix in prefixes):
        if token != 'in': #chekcing special case    
            return True
    elif token.lower().endswith("less"):
        return True
    else:
        return False
    return False

def set_features_values(row, spacy_tokenized, idx) -> dict:
    new_row = {}
    
    # Full sentence
    # new_row['sentence'] = ' '.join([token.text for token in spacy_tokenized])
    
    # Lemma
    new_row['lemma'] = spacy_tokenized[idx].lemma_
    new_row['prev_lemma'] = spacy_tokenized[idx-1].lemma_ if idx > 0 else ""
    
    # Dependency Parser
    new_row['tag'] = spacy_tokenized[idx].tag_
    new_row['dependency'] = spacy_tokenized[idx].dep_
    new_row['head'] = spacy_tokenized[idx].head.text
    new_row['root_path'] = len(list(spacy_tokenized[idx].ancestors))
    
    # Negative Expression List
    NegExpList = ['nor',
                  'neither',
                  'without',
                  'nobody',
                  'none',
                  'nothing',
                  'never',
                  'not',
                  'no',
                  'nowhere',
                  'non']
    token = spacy_tokenized[idx].text.lower()
    new_row['neg_exp_list'] = token in NegExpList
    
    # Check negative affixes
    new_row['affix_cue'] = check_affixes(spacy_tokenized[idx].text)
    
    return new_row


def applySentenceGroupBy(sentence_df) -> pd.DataFrame:
    tokens = sentence_df['token'].tolist()
    string = ' '.join(tokens)
    spacy_tokenized = nlp(string)
    
    new_rows = []
    for idx, row in sentence_df.iterrows():
        new_row = set_features_values(row, spacy_tokenized, idx)
        new_rows.append(new_row)
        
    new_df = pd.DataFrame(new_rows)
    return pd.concat([sentence_df, new_df], axis=1)
