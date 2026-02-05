import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages



def histograms(data_set,file_name):
    sns.set(style='whitegrid')

    plots = [
        ("T Stage ", 30, None, "Histogram of Breast Cancer by T Stage"),
        ("A Stage", 30, None, "Histogram of Breast Cancer by A Stage"),
        ("6th Stage", 30, None, "Histogram of Breast Cancer by 6th Stage"),
        ("N Stage", 30, None, "Histogram of Breast Cancer by N Stage"),
        ("Grade", 15, "Progesterone Status", "Histogram of Breast Cancer by Grade"),
        ("Tumor Size", 20, "Progesterone Status", "Histogram of Breast Cancer by Tumor Size and Progesterone Status"),
    ]
    with PdfPages(file_name) as pdf:
        for x_var, bins, hue, title in plots:
            plt.figure(figsize=(20, 10))
            sns.displot(data_set, x=x_var, bins=bins, hue=hue, multiple="stack" if hue else "layer")
            plt.title(title)
            #pdf.savefig()
            plt.savefig(file_name, format="pdf", bbox_inches="tight", dpi=300)
            plt.show()
            plt.close()
