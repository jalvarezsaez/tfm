from matplotlib import pyplot as plt
import pandas as pd
import dataProcessing.DiabetesDataset as dd
import dataProcessing.dataProcessor as dp
import seaborn as sns
import dataProcessing.dataTrainer as dt
from sklearn.datasets import load_breast_cancer


if __name__ == '__main__':

    # Carga de ficheros diabetes (no necesario, se utilizará el csv de diabetes en lugar de los ficheros originales)
    # datasource = dp.DataProcessor("./datasource", 70, "data", "-", 2)
    # datasource.load_file_set()

    # Carga de fichero csv PIMA
    diabetes = dd.DiabetesDataset("Fichero PIMA de Diabetes", "./datasource/diabetes.csv",
                                  ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI',
                                   'DiabetesPedigreeFunction', 'Age'], ['DiabetesDiagnosed'])

    # ============================================================
    # 1.1 LIMPIEZA DE DATOS
    # ============================================================
    # Limpieza de Nan por columnas del dataset
    dp.data_cleaning(diabetes)
    # dp.data_summary(diabetes)

    # ============================================================
    # 1.2 TRANSFORMACIÓN DE DATOS
    # ============================================================
    # Rescaling
    dp.diabetes_rescaling(diabetes)

    # ============================================================
    # 1.3 MANEJO DE VALORES ATÍPICOS (no need)
    # ============================================================

    # ============================================================
    # 1.4 CODIFICACIÓN DE VARIABLES CATEGÓRICAS (no need)
    # ============================================================

    # ============================================================
    # 2. EXTRACCIÓN DE CARACTERÍSTICAS
    # ============================================================
    # PCA
    # dp.create_plot_pca_analysis(diabetes)
    # El 100% de la varianza se explica con las 8 variables. Reducir a 7 supondría bajar a un 90%.
    # Se descarta por tanto la eliminación de componentes

    # ============================================================
    # 3. EXPLORACIÓN DE DATOS
    # ============================================================

    # ============================================================
    # 3.1 Gráficos de dispersión para explorar la relación entre variables
    # ============================================================
    # Relación de cada componente con el resto
    # for comp in diabetes.features_name:
    #     dp.create_sub_plots_scattered(diabetes, comp)
    # # No se aprecian componentes principales al estar todos los target muy dispersos y solapados entre sí

    # ============================================================
    # 3.2 Gráficos de distribución para cada variable
    # ============================================================
    # dp.create_sub_plots_histplot(diabetes)

    # ============================================================
    # 3.3 Gráficos de cajas
    # ============================================================

    # Análisis individual de cada variable
    # dp.create_sub_plots_boxplot(diabetes)

    # Análisis de cada variable con respecto al diagnóstico de la diabtes
    # dp.create_sub_plots_boxplot_Target(diabetes)

    # Análisis de correlación entre las variables usando un mapa de calor
    # plt.figure(figsize=(12,12))
    # sns.heatmap(diabetes.features_scaled.corr(), annot=True, cmap='coolwarm')
    # plt.show()

    # ============================================================
    # 3.4 Pairplots
    # ============================================================

    # Los gráficos obtenidos en el punto 3.1 son iguales a los incluidos en este pairplot
    # scaledData = pd.concat([diabetes.features_scaled, diabetes.target], axis=1)
    # sns.pairplot(scaledData, palette='Pastel2', hue='Outcome')
    # plt.show()

    # ============================================================
    # 4. AUMENTO DE DATOS (no procede)
    # ============================================================

    diabetes.splitDataSetBasic(t_size=0.1, rnd_state=42, scaled_features=True)

    dt.train_predict_Logistic_Regression(diabetes.X_train, diabetes.y_train, diabetes.X_test, diabetes.y_test)
    dt.train_predict_svm(diabetes.X_train, diabetes.y_train, diabetes.X_test, diabetes.y_test)
    dt.train_predict_lgbmc(diabetes.X_train, diabetes.y_train, diabetes.X_test, diabetes.y_test)
    dt.optimise_svm(diabetes.X_train, diabetes.y_train, diabetes.X_test, diabetes.y_test)