import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages



def boxplots(data_set,file_name):
    sns.set(style='whitegrid')
    plt.figure(figsize=(12, 6))

    plots = [
        ("Age", "Tumor Size", "Boxplot of Tumor Size by Age"),
        ("T Stage ", "Survival Months", "Boxplot of Survival Months by T Stage"),
        ("N Stage", "Survival Months", "Boxplot of Survival Months by N Stage"),
        ("6th Stage", "Survival Months", "Boxplot of Survival Months by 6th Stage"),
        ("A Stage", "Survival Months", "Boxplot of Survival Months by A Stage"),
        ("Reginol Node Positive", "N Stage", "Boxplot of N Stage by Reginol Node Positive"),
        ("Reginol Node Positive", "T Stage ", "Boxplot of T Stage by Reginol Node Positive"),
        ("Reginol Node Positive", "6th Stage", "Boxplot of 6th Stage by Reginol Node Positive"),
        ("Reginol Node Positive", "A Stage", "Boxplot of A Stage by Reginol Node Positive"),
    ]
    with PdfPages(file_name) as pdf:
        for x_var, y_var, title in plots:
            plt.figure(figsize=(12, 6))
            sns.boxplot(x=x_var, y=y_var, data=data_set)

            if "Survival Months" in y_var:  # Dodaj średnią i odchylenie standardowe tylko tam, gdzie to ma sens
                means = data_set.groupby(x_var)[y_var].mean()
                stds = data_set.groupby(x_var)[y_var].std()
                plt.errorbar(x=means.index, y=means, yerr=stds, fmt='o', color='red', capsize=5)

            plt.title(title)
            #pdf.savefig()
            plt.savefig(file_name, format="pdf", bbox_inches="tight", dpi=300)
            plt.show()
            plt.close()
