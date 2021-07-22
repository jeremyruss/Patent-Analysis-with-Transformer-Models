import os
import json
import gif
import math
import random
import cv2
import imageio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
import glob

from transformers import BartTokenizer, BartForConditionalGeneration
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE   # Test?
from sklearn.cluster import KMeans
from wordcloud import WordCloud, STOPWORDS
from PIL import Image
from matplotlib.patches import Arc
from PIL import Image

sns.set_palette("viridis", 10)

def compile_dataframe(save=False):
    df = pd.DataFrame({})
    for fn in os.listdir('./datasets/'):
        #print(fn[:-4])
        datadir = os.path.join('./datasets/raw/', fn)
        new_df = pd.read_csv(datadir)
        df = df.append(new_df)

    df = df[df.patent_number.apply(lambda x: str(x).isnumeric())]
    df = df[df.patent_abstract.apply(lambda x: len(x.split())<512)]
    df["patent_number"] = pd.to_numeric(df["patent_number"], errors="coerce")
    df = df.sort_values(by='patent_number', ignore_index=True)
    df = df.drop_duplicates("patent_number")
    df.reset_index(inplace=True)
    df = df.drop(["index"], axis=1)
    df = df[['patent_date', 'patent_number', 'patent_title', 'patent_abstract']]

    print(f'Number of Rows: {len(df.index)}')
    print(df.tail())
    if save:
        df.to_csv('./datasets/final.csv')
    return df

def bert_embeddings(texts):
    model = SentenceTransformer('paraphrase-TinyBERT-L6-v2')
    embeddings = model.encode(texts, convert_to_tensor=True).cpu().numpy()
    return embeddings

def calculate_similarity(u, v):
    u = np.asarray(u, dtype='float64')
    v = np.asarray(v, dtype='float64')
    cos = np.dot(u, v)/(np.linalg.norm(u)*np.linalg.norm(v))
    #print("Cosine similarity: ", cos)
    mnh = np.abs(u-v).sum()
    #print("Manhattan distance: ", mnh)
    euc = np.linalg.norm(u-v)
    #print("Euclidean distance: ", euc)
    return cos, mnh, euc

def bart_summarisation(texts):
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
    inputs = tokenizer(texts, max_length=1024, return_tensors='pt', padding=True, truncation=True)
    summary_ids = model.generate(inputs['input_ids'], num_beams=2, max_length=64, min_length=8, early_stopping=True)
    outputs = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids]
    return outputs

def preprocess_embeddings(df1, df2, i, j, h1, h2):
    u = df1.loc[i, h1]
    v = df2.loc[j, h2]
    u = u[1:-1].split(',')
    v = v[1:-1].split(',')
    u = [float(i) for i in u]
    v = [float(i) for i in v]
    return u, v

def pca(X):
    pca = PCA(n_components=5)
    X = pca.fit_transform(X)
    #print("Explained variance ratios: ", pca.explained_variance_ratio_)
    return X

def tsne(X):
    tsne = TSNE(n_components=2)
    X = tsne.fit_transform(X)
    return X

def np_embeddings(df, title):
    embeddings = []
    for i, row in df.iterrows():
        em = row[title]
        em = em[1:-1].split(',')
        em = [float(i) for i in em]
        embeddings.append(em)
    return np.array(embeddings)

def compute_clusters(X):
    kmeans = KMeans(n_clusters=10, random_state=42).fit(X)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    return labels, centroids

def assign_clusters(df):
    X = np_embeddings(df, 'title_embeddings')
    labels, centroids = compute_clusters(X)
    labels, centroids = labels.tolist(), centroids.tolist()

    s = pd.Series(labels)
    df['labels'] = s

    series = []
    for label in labels:
        c = centroids[label]
        series.append(c)

    s = pd.Series(series)
    df['centroids'] = s
    return df

def dim_reduction(df):
    X = np_embeddings(df, 'abstract_embeddings')
    s = pd.Series(pca(X).tolist())
    df['abstract_pca'] = s
    s = pd.Series(tsne(X).tolist())
    df['abstract_tsne'] = s

    X = np_embeddings(df, 'title_embeddings')
    s = pd.Series(pca(X).tolist())
    df['title_pca'] = s
    s = pd.Series(tsne(X).tolist())
    df['title_tsne'] = s

    centroids = []
    for c in df['centroids']:
        centroids.append(c)
    X = np.asarray(centroids, dtype='float64')
    s = pd.Series(pca(X).tolist())
    df['centroid_pca'] = s
    s = pd.Series(tsne(X).tolist())
    df['centroid_tsne'] = s
    return df

def create_obj(row, df, i, score=None, em=True, c=True):
    # pca_em = df.iloc[i][1:-1].split(',')
    # pca_em = [float(i) for i in pca_em]
    # tsne_em = df.iloc[i]['tsne_embeddings'][1:-1].split(',')
    # tsne_em = [float(i) for i in tsne_em]
    obj = {
        "patent_date": row['patent_date'],
        "patent_number": row['patent_number'],
        "patent_title": row['patent_title'],
        "patent_abstract": row['patent_abstract'],
        "summary": row['summary'],
    }
    if score is not None:
        obj["score"] = score
    if em:
        obj["pca_embedding"] = df.iloc[i]['abstract_pca']
        obj["tsne_embedding"] = df.iloc[i]['abstract_tsne']
    if c:
        obj["cluster"] = str(df.iloc[i]['labels'])
    return obj

def create_wordcloud(df):
    values = df['patent_title'].values
    text = ""
    for value in values:
        text += value + " "
    mask = np.array(Image.open('./images/uspto.png'))
    wordcloud = WordCloud(width=1800,
                          height=800, 
                          random_state=42, 
                          font_path='./datasets/fonts/Merriweather-Black.ttf',
                          background_color=None, 
                          stopwords=STOPWORDS, 
                          contour_width=0,
                          colormap='viridis',
                          mode="RGBA",
                          mask=mask)
    generated = wordcloud.generate(text)
    generated.to_file('./images/output/wordcloud.png')

def create_stacked_bar(df):
    df = df.copy()
    df = df[:-6]
    values = {year: [0 for i in range(10)] for year in range(1976, 2021)}
    for i, row in df.iterrows():
        label = row['labels']
        year = int(row['patent_date'][:4])
        values[year][label] += 1

    dates = list(values.keys())
    numbers = list(values.values())
    numbers = np.array(numbers).T.tolist()

    fig, ax = plt.subplots()

    bottom = [0 for i in range(len(numbers[0]))]
    for i, num in enumerate(numbers):
        ax.bar(dates, num, bottom=bottom, label=i)
        bottom = np.array(num) + np.array(bottom)
        bottom = bottom.tolist()

    ax.set_ylabel('Number of Patents')
    ax.set_xlabel('Patent Date')
    ax.set_title('Number & Distribution Each Year')
    ax.legend(title='Cluster')

    plt.tight_layout()
    #plt.yscale('log')

    plt.savefig('./images/output/bar.png',  bbox_inches='tight')
    plt.show()

@gif.frame
def create_3d_scatter(df):
    x = df['title_pca'].apply(lambda x: x[0])
    y = df['title_pca'].apply(lambda x: x[1])
    z = df['title_pca'].apply(lambda x: x[2])

    u = df['title_pca'].apply(lambda x: x[3])
    v = df['title_pca'].apply(lambda x: x[4])
    vmax, vmin = v.max(), v.min()
    v = v.apply(lambda x: (1/(vmax-vmin)*(x-vmax)+1))

    cluster = df['labels']
    
    fig = px.scatter_3d(
        x = x,
        y = y,
        z = z,
        color = u,
        size = v,
        symbol = cluster,
        opacity = 1,
        size_max = 25,
        template = 'seaborn',
        color_continuous_scale = px.colors.sequential.Viridis,
        #title = '5D PCA BERT Title Embeddings'
    )

    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0), showlegend=False)
    fig.update(layout_coloraxis_showscale=False)

    x_eye = -1.25
    y_eye = 2
    z_eye = 0.5

    fig.update_layout(scene_camera_eye=dict(x=x_eye, y=y_eye, z=z_eye),
                  updatemenus=[dict(type='buttons',
                                    showactive=False,
                                    y=1,
                                    x=0.8,
                                    xanchor='left',
                                    yanchor='bottom',
                                    pad=dict(t=45, r=10),
                                    buttons=[dict(label='Play',
                                                  method='animate',
                                                  args=[None, dict(frame=dict(duration=100, redraw=True),
                                                                   transition=dict(duration=0),
                                                                   fromcurrent=True,
                                                                   mode='immediate'
                                                                   )]
                                                  )
                                             ]
                                    )
                               ]
                  )

    def rotate_z(x, y, z, theta):
        w = x + 1j * y
        return np.real(np.exp(1j * theta) * w), np.imag(np.exp(1j * theta) * w), z

    frames = []
    for t in np.arange(0, 7, 0.01):
        xe, ye, ze = rotate_z(x_eye, y_eye, z_eye, -t)
        fig.update_layout(scene_camera_eye=dict(x=x_eye, y=y_eye, z=z_eye))
        frames.append(go.Frame(layout=dict(scene_camera_eye=dict(x=xe, y=ye, z=ze))))

    fig.frames = frames

    #gif.save(frames, './images/output/3dscatter.gif', duration=50)

    fig.write_html('./images/output/scatter.html', include_plotlyjs="cdn")
    #fig.show()

def plot_2d_cluster(df):
    tsne_values = df['title_tsne'].values
    x = [i[0] for i in tsne_values]
    y = [i[1] for i in tsne_values]
    c = df['labels'].values

    sns.scatterplot(
        x = x,
        y = y,
        hue = c,
        palette = sns.color_palette("viridis", 10),
        alpha = 0.7
    )

    plt.ylabel('Y Component')
    plt.xlabel('X Component')

    plt.legend(title="Cluster", loc="center left", bbox_to_anchor=(1, 0.5))
    plt.title('t-SNE 2D Title Embeddings')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()

    plt.savefig('./images/output/tsne.png',  bbox_inches='tight')
    plt.show()

def heatmap(matches):
    with open('./datasets/curated.json', encoding='utf-8') as f:
        data = json.load(f)
        f.close()

    fmt = '.2f'
    if matches == 'matches_cos':
        title = 'Cosine Similarity Matches'
        suffix = '_cos'
    elif matches == 'matches_mnh':
        title = 'Manhattan Distance Matches'
        suffix = '_mnh'
        fmt = '.3g'
    elif matches == 'matches_euc':
        title = 'Euclidean Distance Matches'
        suffix = '_euc'

    ylabels = ['First', 'Second', 'Third', 'Fourth', 'Fifth']

    matrix = []
    p_numbers = []
    
    xlabels = []
    for year in data:
        xlabel = year + "\n" + str(data[year]['patent_number'])
        xlabels.append(xlabel)
        
        # data[year]['matches_cos'] = list
        scores = []
        for patent in data[year][matches]:
            score = patent['score']
            num = patent['patent_number']
            scores.append(score)
            p_numbers.append(num)

        matrix.append(scores)

    matrix = np.array(matrix).T.tolist()

    ax = sns.heatmap(matrix, annot=True, annot_kws={"size": 8}, fmt=fmt, cmap="viridis")

    p_numbers = np.array(p_numbers).reshape(6,5).tolist()
    p_labels = []

    for i in range(5):
        for numbers in p_numbers:
            p_labels.append(numbers[i])

    for t, num in zip(ax.texts, p_labels):
        t.set_text(t.get_text() + "\n" + str(num))

    ax.set_ylabel('Top 5 Matches')
    ax.set_xlabel('Random Patent Sample From Each Decade')
    ax.set_title(title)
    ax.set_xticklabels(xlabels)
    ax.set_yticklabels(ylabels)
    #plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(f'./images/output/heatmap{suffix}.png',  bbox_inches='tight')
    plt.show()

def donut_chart(df):
    count = df['labels'].value_counts()
    total = count.sum()
    labels = [f'{l}, {(s/total)*100:0.1f}%' for l, s in zip(count.index, count.values)]

    circle = plt.Circle((0,0), 0.7, color='white')

    plt.pie(count.values, labels=count.index, wedgeprops={'linewidth': 1, 'edgecolor': 'white'})
    p = plt.gcf()
    p.gca().add_artist(circle)

    plt.title('Proportion Of Patents Assigned To Each Cluster')
    plt.legend(labels=labels, loc="center left", title="Cluster",  bbox_to_anchor=(1.04, 0.5))
    plt.tight_layout()

    plt.savefig('./images/output/donut.png',  bbox_inches='tight')
    plt.show()

def draw_dist_diagram():
    axes = plt.gca()
    axes.set_xlim([0,10])
    axes.set_ylim([0,10])
    axes.set_aspect('equal', adjustable='box')

    major_ticks = np.arange(0, 10, 1)
    minor_ticks = np.arange(0, 10, 0.1)
    axes.set_xticks(major_ticks)
    axes.set_xticks(minor_ticks, minor=True)
    axes.set_yticks(major_ticks)
    axes.set_yticks(minor_ticks, minor=True)

    x1, y1 = [0,4], [0,7]
    x2, y2 = [0,7], [0,4]
    x3, y3 = [7,4], [4,7]
    x4, y4 = [4,4], [4,7]
    x5, y5 = [4,7], [4,4]

    theta1 = math.degrees(math.atan2(7, 4))
    theta2 = math.degrees(math.atan2(4, 7))
    
    arc = Arc((0,0), 3, 3, theta1=theta2, theta2=theta1)
    axes.add_patch(arc)

    #angle = f'{(theta1-theta2):.1f}\u00b0'
    axes.text(x=1.2, y=1.2, s='\u03B8')
    axes.text(x=3.5, y=5.4, s='a')
    axes.text(x=5.4, y=3.5, s='b')
    axes.text(x=5.6, y=5.6, s='c')

    cos_sim = 'Cosine Similarity = cos(\u03B8)'
    mnh_dis = 'Manhattan Distance = a + b'
    euc_dis = 'Euclidean Distance = c'

    axes.text(x=4.4, y=9.3, s=cos_sim)
    axes.text(x=4.4, y=8.3, s=mnh_dis)
    axes.text(x=4.4, y=7.3, s=euc_dis)

    plt.plot(x1, y1, x2, y2, marker = 'o')
    plt.plot(x3, y3, linestyle=':', marker = 'o')
    plt.plot(x4, y4, x5, y5, linestyle='--', marker = 'o')
    plt.title('Measures Of Similarity')
    plt.grid()

    plt.savefig('./images/output/similarity.png',  bbox_inches='tight')
    plt.show()

def extract_frames():
    vidcap = cv2.VideoCapture('./images/video/video.mp4')
    success, image = vidcap.read()
    count = 0
    while success:
        cv2.imwrite("./images/video/frames/frame%d.jpg" % count, image)     
        success,image = vidcap.read()
        count += 1
        #print('Read a new frame: ', success)
        if count % 500 == 0:
            print(count)

def resize_frames():
    x, y, h, w = 50, 300, 725, 800
    for fn in os.listdir('./images/video/frames/'):
        img = cv2.imread(os.path.join('./images/video/frames/', fn))
        crop_img = img[y:y+h, x:x+w]
        cv2.imwrite(os.path.join('./images/video/frames/', fn), crop_img)
        print(fn)
    print("Completed resizing!")

def create_gif():
    filenames = []

    def sorter(item):
        num = int(item[5:-4])
        return num

    for fn in sorted(os.listdir('./misc/video/frames/'), key=sorter):
        filenames.append(os.path.join('./misc/video/frames/', fn))

    with imageio.get_writer('./images/output/3dscatter.gif', mode='I', duration=0.1) as writer:
        for i, filename in enumerate(filenames):
            if i % 26 == 0:
                image = imageio.imread(filename)
                writer.append_data(image)
            if i % 512 == 0:
                print(i)