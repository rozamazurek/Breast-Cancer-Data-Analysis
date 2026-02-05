import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.manifold import TSNE
from matplotlib.backends.backend_pdf import PdfPages



def run_tsne(data_set, hue_column, file_name):
    data_set = data_set.dropna()
    categorical_cols = ['differentiate', 'Grade', 'Marital Status', 'Race', 'T Stage ', 'N Stage',
                        '6th Stage', 'A Stage', 'Estrogen Status', 'Progesterone Status', 'Status']

    encoder = OneHotEncoder()
    encoded_categories = encoder.fit_transform(data_set[categorical_cols]).toarray()

    numeric_cols = ['Age', 'Tumor Size', 'Regional Node Examined', 'Reginol Node Positive', 'Survival Months']
    encoded_data = np.hstack((encoded_categories, data_set[numeric_cols].values))

    scaled_data = StandardScaler().fit_transform(encoded_data)

    tsne = TSNE(n_components=2, perplexity=3, random_state=42)
    tsne_results = tsne.fit_transform(scaled_data)

    tsne_df = pd.DataFrame(tsne_results, columns=['TSNE1', 'TSNE2'])
    tsne_df[hue_column] = data_set[hue_column]
    with PdfPages(file_name) as pdf:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x='TSNE1', y='TSNE2', hue=hue_column, data=tsne_df, palette='coolwarm', s=100, edgecolor='k')
        plt.title(f"t-SNE Visualization colored by {hue_column}")
        #pdf.savefig()
        plt.savefig(file_name, format="pdf", bbox_inches="tight", dpi=300)
        plt.show()
        plt.close()


def pca(data_set, file_name):
    run_tsne(data_set, 'differentiate', file_name)
    run_tsne(data_set, 'A Stage', file_name)
    run_tsne(data_set, 'Grade', file_name)
    #run_tsne(data_set, 'Tumor Size', file_name)
    #run_tsne(data_set, 'Reginol Node Positive', file_name)
    #run_tsne(data_set, 'Survival Months', file_name)

