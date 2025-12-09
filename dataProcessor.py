import string

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def data_cleaning(dataset):
    # Búsqueda de Nan por columnas del dataset
    for comp in dataset.features_name:
        if dataset.features[comp].isna().any():
            preguntaRespondida = False
            while not preguntaRespondida:
                print(comp + " contiene valores NA")
                limpiar = input("¿Limpiar " + comp + "? (S/N)")
                if str.capitalize(limpiar) == "S":
                    dataset.features[comp].dropna()
                    print(comp + " limpiado de NA")
                    preguntaRespondida = True
                elif string.capwords(limpiar) == "N":
                    preguntaRespondida = True
                else:
                    print("Responder S/N")

    # Búsqueda de valores 0 por columnas del dataset
    for comp in dataset.features_name:
        if dataset.features[comp].__contains__(0):
            preguntaRespondida = False
            while not preguntaRespondida:
                print(comp + " contiene resultados con valor 0")
                limpiar = input("¿Reemplazar 0 por la media de " + comp + "? (S/N)")
                if str.capitalize(limpiar) == "S":
                    dataset.features[comp] = dataset.features[comp].replace(0, dataset.features[comp].mean())
                    print(comp + " Valores 0 reemplazados")
                    preguntaRespondida = True
                elif str.capitalize(limpiar) == "N":
                    preguntaRespondida = True
                else:
                    print("Responder S/N")


def data_summary(dataset):
    for comp in dataset.features:
        print(dataset.features[comp].describe())
        print(dataset.features[comp].unique())


def diabetes_rescaling(dataset):
    scaler = StandardScaler()
    scaler.fit(dataset.features)
    # diabetes.features_scaled = scaler.transform(diabetes.features)
    dataset.features_scaled = pd.DataFrame(scaler.transform(dataset.features), columns=dataset.features_name)


def create_plot_pca_analysis(dataset):
    pca = PCA(n_components=len(dataset.features_name), random_state=2020)
    pca.fit(dataset.features_scaled)
    X_pca = pca.transform(dataset.features_scaled)
    plt.plot(np.cumsum(pca.explained_variance_ratio_ * 100))
    plt.xlabel("Número de componentes")
    plt.ylabel("Porcentaje de varianza")
    plt.show()


def create_sub_plots_scattered(dataset, component):
    fig, axes = plt.subplots(2, 4, figsize=(15, 10))

    xPos = 0
    yPos = 0
    for comp2 in dataset.features_name:
        if comp2 != component:
            sns.scatterplot(x=dataset.features_scaled[component], y=dataset.features_scaled[comp2], hue=dataset.target,
                            ax=axes[xPos % 2][yPos % 4])
            axes[xPos % 2][yPos % 4].set_xlabel(component)
            axes[xPos % 2][yPos % 4].set_ylabel(comp2)
            if yPos == 3: xPos += 1
            yPos += 1

    plt.tight_layout()
    plt.show()


def create_sub_plots_histplot(dataset):
    fig, axes = plt.subplots(2, 4, figsize=(15, 10))

    xPos = 0
    yPos = 0
    for comp in dataset.features_name:
        media = np.mean(dataset.features_scaled[comp])
        desviacion = np.std(dataset.features_scaled[comp])

        # Crear histograma con histplot ya que distplot es obsoleto
        sns.set(style="whitegrid")
        # sns.distplot(dataset.features[comp], kde=False, fit=norm, fit_kws={'color': 'r', 'linewidth': 2.5})
        sns.histplot(dataset.features_scaled[comp], bins=50, stat='density', alpha=0.5, ax=axes[xPos % 2][yPos % 4])
        # Superponer curva normal
        x = np.linspace(min(dataset.features_scaled[comp]), max(dataset.features_scaled[comp]), 1000)
        y = norm.pdf(x, media, desviacion)

        axes[xPos % 2][yPos % 4].plot(x, y, color='red', linewidth=2,
                                      label=f'Normal (μ={media:.2f}, σ={desviacion:.2f})')
        axes[xPos % 2][yPos % 4].legend()

        if yPos == 3: xPos += 1
        yPos += 1

        # # Personalización
        # grafico.set_axis_labels(comp, 'Densidad')
        # grafico.set_titles('Histograma con displot y Curva Normal')

    plt.tight_layout()
    plt.show()


def create_sub_plots_boxplot(dataset):
    fig, axes = plt.subplots(2, 4, figsize=(15, 10))
    rng = np.random.default_rng(42)
    palette = [
        "#{:02x}{:02x}{:02x}".format(*rng.integers(0, 256, size=3))
        for _ in range(len(dataset.features_name))
    ]

    xPos = 0
    yPos = 0
    numColor = 0

    for comp in dataset.features_name:
        sns.boxplot(data=dataset.features[comp], ax=axes[xPos % 2][yPos % 4], color=palette[numColor])
        numColor += 1
        axes[xPos % 2][yPos % 4].set_xlabel(comp)
        if yPos == 3: xPos += 1
        yPos += 1

    plt.tight_layout()
    plt.show()


def create_sub_plots_boxplot_Target(dataset):
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(20, 70), sharex=True)
    data = pd.concat([dataset.features, dataset.target], axis=1)

    xPos = 0
    yPos = 0
    for comp in dataset.features_name:
        sns.boxplot(x='Outcome', y=comp, data=data, ax=axes[xPos % 4][yPos % 4], width=0.6)
        axes[xPos % 4][yPos % 4].grid(axis='y', alpha=0.25)
        if yPos == 3: xPos += 1
        yPos += 1

    plt.show()


class DataProcessor:

    def __init__(self, file_path, number_files, prefix, separator, digits):
        self.dataframe = None
        self.filePath = file_path
        self.numberFiles = number_files
        self.prefix = prefix
        self.separator_character = separator
        self.numberDigits = digits

    def load_file_set(self):
        files = []
        fileName = self.filePath + '/' + self.prefix + self.separator_character
        loaded = 1
        while loaded <= self.numberFiles:
            files.append(fileName + str(loaded).zfill(2))
            loaded += 1

        self.dataframe = pd.concat([pd.read_csv(file, sep='\t', header=None) for file in files], ignore_index=True)
