import pandas as pd
from PyRuSH import RuSH
import nltk
from nltk.corpus import stopwords 
import re


def split_datasets(data, conf):
    """
    Split the list of dictionaries into datasets
    """

    dict_dfs = dict()

    for item in data.get("result"):
        model = item.get("model")
        if "," in model: 
            model = model.split(",")
        else:
            model = [model]
        for m in model:
            if m not in set(conf["modelname"]):
                print(f"Error: couldn't find {m} in the config file. Make sure the model name in the input file is correct and that the config file is up to date.")
            else:
                if m not in dict_dfs.keys():
                    dict_dfs[m] = [pd.DataFrame(columns = ["ID", "Text"]), 0]
                    dict_dfs.get(m)[0].loc[dict_dfs.get(m)[1]] = [item.get("ID"), item.get("note")]
                    dict_dfs.get(m)[1] +=1
                else:
                    dict_dfs.get(m)[0].loc[dict_dfs.get(m)[1]] = [item.get("ID"), item.get("note")]
                    dict_dfs.get(m)[1] +=1
            


    return dict_dfs

def preprocess(df):
    
    """Output a preprocessed dataframe"""
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)

    stop_words = stopwords.words('english')

    #removal non-alfa characters
    df['Text'] = df['Text'].apply(lambda x: re.sub("\n+", " ", x))
    df['Text'] = df['Text'].apply(lambda x: re.sub("[^0-9a-zA-Z ]+", "", x))

    
    return(df)


def split_sentences(df):
    """
    Split the notes in the DataFrame into sentences and return a new DataFrame
    """
    rush = RuSH('src/rush_rules.tsv')
    new_df = pd.DataFrame(columns = ["ID", "FullText", "Text"])
    df_it = 0

    for index, row in df.iterrows():
        for sentence in rush.segToSentenceSpans(row.Text):
            new_df.loc[df_it] = [row.ID, row.Text, row.Text[sentence.begin:sentence.end]]
            df_it +=1

    return new_df


def convert_to_dict(df, model_name):
    """
    
    """
    lst = list()
    log = set()
    for index, row in df.iterrows():
        if row.ID not in log:
            sentences = list()
            log.add(row.ID)
            for index1, row1 in df.loc[df.ID == row.ID].iterrows():
                if row1.Preds == 1:
                    sentences.append(row1.Text)

        lst.append({"ID": row.ID, "note": row.Text, "sentences": sentences, "model": model_name})

    return lst
        
