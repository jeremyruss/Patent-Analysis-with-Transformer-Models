import pandas as pd
import numpy as np
import random
import json
import seaborn as sns

from scripts.utils import *
from tqdm import tqdm
from datetime import datetime

sns.set_palette("viridis", 10)

def calculate_embeddings():
    df = pd.read_csv('./datasets/compiled.csv')

    sentences = df['patent_title'].values.tolist()
    embeddings = bert_embeddings(sentences)
    s = pd.Series(embeddings.tolist())
    df['title_embeddings'] = s

    sentences = df['patent_abstract'].values.tolist()
    embeddings = bert_embeddings(sentences)
    s = pd.Series(embeddings.tolist())
    df['abstract_embeddings'] = s

    summaries = []
    for i in range(0, len(sentences), 128):
        summaries += bart_summarisation(sentences[i:i+128])
        print(len(summaries))
    s = pd.Series(summaries)
    df['summary'] = s

    embeddings = bert_embeddings(summaries)
    s = pd.Series(embeddings.tolist())
    df['summary_embeddings'] = s

    df.to_csv('./datasets/embeddings.csv', index=False)

def record_similarity():
    df = pd.read_csv('./datasets/embeddings.csv', index_col=0)
    df['patent_date'] = pd.to_datetime(df['patent_date'])

    sample_dataframes = []
    for year in range(1969, 2029, 10):
        df_copy = df.copy()
        lb = datetime(year, 12, 31)
        ub = datetime(year+10, 12, 31)
        df_year = df_copy.loc[(df_copy['patent_date'] > lb) & (df_copy['patent_date'] <= ub)]
        row = df_year.sample()
        df_copy = df_copy.drop(row.index[0], axis=0)
        sample_dataframes.append(row)

    df_sample = pd.concat(sample_dataframes)
    
    cos_sim = [[] for x in range(6)]
    mnh_sim = [[] for x in range(6)]
    euc_sim = [[] for x in range(6)]
    for i, _ in tqdm(df.iterrows()):    # Takes about 1 minute
        n = 0
        for j, _ in df_sample.iterrows():
            u, v = preprocess_embeddings(df, df_sample, i, j, 'abstract_embeddings', 'abstract_embeddings')
            cos, mnh, euc = calculate_similarity(u, v)
            cos_sim[n].append(cos)
            mnh_sim[n].append(mnh)
            euc_sim[n].append(euc)
            n += 1
    
    cos_titles = ['1970_cos', '1980_cos', '1990_cos', '2000_cos', '2010_cos', '2020_cos']
    mnh_titles = ['1970_mnh', '1980_mnh', '1990_mnh', '2000_mnh', '2010_mnh', '2020_mnh']
    euc_titles = ['1970_euc', '1980_euc', '1990_euc', '2000_euc', '2010_euc', '2020_euc']
    for i, series in enumerate(cos_sim):
        s = pd.Series(series)
        df_copy[cos_titles[i]] = s
    for i, series in enumerate(mnh_sim):
        s = pd.Series(series)
        df_copy[mnh_titles[i]] = s
    for i, series in enumerate(euc_sim):
        s = pd.Series(series)
        df_copy[euc_titles[i]] = s

    cos_values = []
    mnh_values = []
    euc_values = []
    for i, _ in df_sample.iterrows():
        u, v = preprocess_embeddings(df_sample, df_sample, i, i, 'abstract_embeddings', 'summary_embeddings')
        cos, mnh, euc = calculate_similarity(u, v)
        cos_values.append(cos)
        mnh_values.append(mnh)
        euc_values.append(euc)

    s_cos = pd.Series(cos_values)
    s_mnh = pd.Series(mnh_values)
    s_euc = pd.Series(euc_values)
    s_cos.index, s_mnh.index, s_euc.index = df_sample.index, df_sample.index, df_sample.index

    df_sample['summary_cos'] = s_cos
    df_sample['summary_mnh'] = s_mnh
    df_sample['summary_euc'] = s_euc

    df_full = pd.concat([df_copy, df_sample]).drop_duplicates(keep=False)

    print(len(df_full.index), len(df_copy.index), len(df_sample.index))

    df_full.to_csv('./datasets/full.csv')
    df_copy.to_csv('./datasets/query.csv')
    df_sample.to_csv('./datasets/samples.csv')

def create_json(df_query, df_sample, df):
    df_copy = df_query.copy()
    #df_sample.index = range(len(df_sample.index))
    #print(len(df_copy.index), len(df_sample.index))

    years = ['1970', '1980', '1990', '2000', '2010', '2020']
    sims = ['_cos', '_mnh', '_euc']

    data = {}

    n = 0
    for i, row in df_sample.iterrows():
        patent = create_obj(row, df, i)
        data[years[n]] = patent
        n += 1
    
    for year in years:
        for sim in sims:
            sort_by = year + sim
            heading = 'matches' + sim
            if sim == '_cos':
                ascending = False
            else:
                ascending = True
            df_sorted = df_copy.sort_values(by=sort_by, ascending=ascending)
            df_top = df_sorted.head(6)[1:]
            #df_top.index = range(len(df_top.index))
            data[year][heading] = []
            for i, row in df_top.iterrows():
                score = row[sort_by]
                patent = create_obj(row, df, i, score=score)
                data[year][heading].append(patent)

    with open('./datasets/curated.json', 'w') as f:
        json.dump(data, f)

if __name__ == "__main__":

    #record_similarity()

    #df = pd.read_csv('./datasets/full.csv', index_col=0)
    #df_query = pd.read_csv('./datasets/query.csv', index_col=0)
    #df_sample = pd.read_csv('./datasets/samples.csv', index_col=0)
    #print(len(df.index), len(df_query.index), len(df_sample.index))

    #df = df[:50] # For testing

    #create_json(df_query, df_sample, df)

    #plot_2d_cluster(df)

    #create_3d_scatter(df)

    #create_stacked_bar(df)

    #heatmap('matches_cos')
    #heatmap('matches_mnh')
    #heatmap('matches_euc')

    #donut_chart(df)

    #create_wordcloud(df)

    #draw_dist_diagram()

    #extract_frames()
    #resize_frames()
    create_gif()
