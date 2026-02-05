import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages



def violinplots(data_set,file_name):
    sns.set(style='whitegrid')

    plots = [
        ("T Stage ", "Survival Months", "Violinplot of Survival Months by T Stage divided into Progesterone Status"),
        ("N Stage", "Survival Months", "Violinplot of Survival Months by N Stage divided into Progesterone Status"),
        ("6th Stage", "Survival Months", "Violinplot of Survival Months by 6th Stage divided into Progesterone Status"),
        ("A Stage", "Survival Months", "Violinplot of Survival Months by A Stage divided into Progesterone Status"),
    ]
    with PdfPages(file_name) as pdf:
        for x_var, y_var, title in plots:
            plt.figure(figsize=(12, 6))
            sns.catplot(
                data=data_set, x=x_var, y=y_var, hue="Progesterone Status", kind="violin"
            )

            means = data_set.groupby(x_var)[y_var].mean()
            stds = data_set.groupby(x_var)[y_var].std()
            plt.errorbar(x=means.index, y=means, yerr=stds, fmt='o', color='red', capsize=5)

            plt.title(title)
            #pdf.savefig()
            plt.savefig(file_name, format="pdf", bbox_inches="tight", dpi=300)
            plt.show()
            plt.close()
