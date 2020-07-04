import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings(action="ignore")


def prepocesing_data(data):
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    #data = data.dropna() #odbacili smo redove u kojima nedostaje vrednost
    data = data.fillna(data.median()) #popunili smo prazna polja prosecnim vrednostima
    data.drop('CUST_ID', axis=1, inplace=True)
    #df = data.describe()
    #print(df)
    return data


def normalize_data(data):
    X = np.asarray(data)
    normalize = StandardScaler()
    X = normalize.fit_transform(X)
    return X


def vizualization(X):

    """Metodom lakta cemo odrediti broj klastera"""
    n_clusters=30
    wcss=[]
    for i in range(1,n_clusters):
        kmean= KMeans(i, init='random', random_state=101)
        kmean.fit(X)
        wcss.append(kmean.inertia_)

    plt.plot(wcss, 'yo-')
    plt.title('Medota lakta')
    plt.xlabel('Broj klastera')
    plt.ylabel('SSE')
    #plt.show()

    """Posmatranjem krive vidimo da se pad moze uociti kod broja 7,
    tako da cemo odabrati 7 klastera"""

    kmean= KMeans(n_clusters=7, init='k-means++',random_state=101)
    kmean.fit(X)
    labels=kmean.labels_

    clusters=pd.concat([data, pd.DataFrame({'cluster':labels})], axis=1)
    clusters.head()

    for c in clusters:
        grid= sns.FacetGrid(clusters, col='cluster')
        grid.map(plt.hist, c, color="r")
    #plt.show()


    dist = 1 - cosine_similarity(X)

    pca = PCA(2)
    pca.fit(dist)
    X_PCA = pca.transform(dist)
    X_PCA.shape

    x, y = X_PCA[:, 0], X_PCA[:, 1]

    colors = {
        0: 'red',
        1: 'blue',
        2: 'green',
        3: 'purple',
        4: 'brown',
        5: 'pink',
        6: 'cyan'
        }

    names = {
         0: 'ucestalo kupuju uglavnom na rate, ali ne trose puno novca, imaju prosecan limit',
         1: 'vrlo cesto kupuju na rate, trose manje novca, imaju prosecan limit',
         2: 'ucestalo kupuju i trose vise novca, imaju visoki limit, cesto uplacuju novac unapred',
         3: 'retko kupuju, vise jednokratno, imaju prosecan limit',
         4: 'ucestalo kupuju, vise na rate, imaju nizak limit',
         5: 'vrlo cesto kupuju raznovrsno, imaju visok limit',
         6: 'vrlo cesto kupuju raznovrsno, imaju veliki limit'
         }

    df = pd.DataFrame({'x': x, 'y': y, 'label': labels})
    groups = df.groupby('label')

    fig, ax = plt.subplots(figsize=(20, 13))

    for name, group in groups:
        ax.plot(group.x, group.y, marker='o', linestyle='', ms=5,
                color=colors[name], label=names[name], mec='none')
        ax.set_aspect('auto')
        ax.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')
        ax.tick_params(axis='y', which='both', left='off', top='off', labelleft='off')

    ax.legend()
    ax.set_title("Grupacija korisnika na osnovu nacina njihovog koriscenja kreditne kartice.")
    plt.show()

    return X


if __name__ == '__main__':
    data = pd.read_csv("credit_card_data.csv")
    data = prepocesing_data(data)
    X = normalize_data(data)
    X = vizualization(X)


