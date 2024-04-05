# --------------------------------------------------------------------------------------------
# LIBRARIES
# --------------------------------------------------------------------------------------------


from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd


# --------------------------------------------------------------------------------------------
# CLASS DEFINITIONS
# --------------------------------------------------------------------------------------------


def show_tsne_2d(data, labels, tsne_perplexity = 40, tsne_niter = 1000, tsne_random_state = 42):

    tsne = TSNE(n_components = 2, verbose = 1, perplexity = tsne_perplexity,
                n_iter = tsne_niter, random_state = tsne_random_state)
    
    tsne_results_2D = tsne.fit_transform(data)

    df_show = pd.DataFrame()
    df_show['T-SNE 2D First Component'] = tsne_results_2D[:, 0]
    df_show['T-SNE 2D Second Component'] = tsne_results_2D[:, 1]
    df_show['labels'] = labels

    plt.figure(figsize = (16, 10))

    ax = sns.scatterplot(
        x = "T-SNE 2D First Component", y = "T-SNE 2D Second Component",
        hue = "labels",
        style = df_show['labels'],
        palette = sns.color_palette("hls", len(df_show['labels'].factorize()[1])),
        data = df_show,
        legend = "full",
        alpha = 1
    )

    plt.title("2D T-SNE Representation",fontweight = 'bold')
    plt.xlabel('T-SNE 2D First Component',fontweight = 'bold')
    plt.ylabel('T-SNE 2D Second Component',fontweight = 'bold')
    plt.grid(True)

    return tsne_results_2D