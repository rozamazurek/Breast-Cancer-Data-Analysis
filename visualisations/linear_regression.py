import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages



def linear_regression(data_set,file_name):
    sns.set(style='whitegrid')


    plots = [
        ("Reginol Node Positive", "Survival Months", None, "Linear regression of Survival Months by Reginol Node Positive"),
        ("Tumor Size", "Survival Months", None, "Linear regression of Survival Months by Tumor Size"),
    ]
    with PdfPages(file_name) as pdf:
        for x_var, y_var, hue, title in plots:
            plt.figure(figsize=(15, 10))
            data_set_grouped = data_set.groupby(x_var, as_index=False)[y_var].mean()
            sns.regplot(x=x_var, y=y_var, data=data_set_grouped)
            plt.title(title)
            #pdf.savefig()
            plt.savefig(file_name, format="pdf", bbox_inches="tight", dpi=300)
            plt.show()
            plt.close()

    # Regresja liniowa z hue
    lm_plots = [
        ("Tumor Size", "Reginol Node Positive", "Estrogen Status", "Linear regression of Reginol Node Positive by Tumor Size and Estrogen Status"),
        ("Tumor Size", "Reginol Node Positive", "Progesterone Status", "Linear regression of Reginol Node Positive by Tumor Size and Progesterone Status"),
    ]
    with PdfPages(file_name) as pdf:
        for x_var, y_var, hue, title in lm_plots:
            data_set_grouped = data_set.groupby([x_var, hue], as_index=False)[y_var].mean()
            plt.figure(figsize=(15, 15))
            sns.lmplot(x=x_var, y=y_var, data=data_set_grouped, hue=hue, height=8, aspect=1.5)
            plt.title(title)
            #pdf.savefig()
            plt.savefig(file_name, format="pdf", bbox_inches="tight", dpi=300)
            plt.show()
            plt.close()
