from statistics.read_data import read_data, basic_statistics
from visualisations.boxplot import boxplots
from visualisations.violinplot import violinplots
from visualisations.histograms import histograms
from visualisations.heatmap import heatmaps
from visualisations.linear_regression import linear_regression
from visualisations.pca import pca


def save_data(csv_file_name, file_name):
    data_set = read_data()
    if data_set is None:
        return

    try:
        basic_statistics(data_set, csv_file_name)
        boxplots(data_set, file_name)
        violinplots(data_set, file_name)
        histograms(data_set, file_name)
        heatmaps(data_set, file_name)
        linear_regression(data_set, file_name)
        pca(data_set, file_name)
    except Exception as e:
        print(f"Wystąpił błąd: {e}")