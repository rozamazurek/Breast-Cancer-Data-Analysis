import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages



def heatmaps(data_set,file_name):
    sns.set(style='whitegrid')

    plots = [
        ("Estrogen Status", "Progesterone Status", "Reginol Node Positive", "mean",
         "Heatmap of Reginol Node Positive by Estrogen Status and Progesterone Status", sns.cubehelix_palette(as_cmap=True), True, None, None),

        ("Estrogen Status", "Progesterone Status", "Tumor Size", "mean",
         "Heatmap of Tumor Size by Estrogen Status and Progesterone Status", sns.cubehelix_palette(as_cmap=True), True, None, None),

        ("Race", "Age", "Tumor Size", "mean",
         "Heatmap of Tumor Size by Race and Age", "coolwarm", True, None, None),

        ("Race", "Age", "Reginol Node Positive", "median",
         "Heatmap of Reginol Node Positive by Race and Age", "coolwarm", False, 2, 6),
    ]
    with PdfPages(file_name) as pdf:
        for index, columns, values, aggfunc, title, cmap, annot, vmin, vmax in plots:
            plt.figure(figsize=(20, 5))
            data = data_set.pivot_table(index=index, columns=columns, values=values, aggfunc=aggfunc)
            sns.heatmap(data, annot=annot, linewidth=.5, cmap=cmap, vmin=vmin, vmax=vmax)
            plt.title(title)
            #pdf.savefig()
            plt.savefig(file_name, format="pdf", bbox_inches="tight", dpi=300)
            plt.show()
            plt.close()
