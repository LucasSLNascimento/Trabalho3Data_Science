import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from pickle import dump

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data'
names = ['ID', 'Diagnosis', 'radius1', 'texture1', 'perimeter1', 'area1', 'smoothness1', 'compactness1', 'concavity1', 'concave_points1', 'symmetry1', 'fractal_dimension1', 'radius2', 'texture2', 'perimeter2', 'area2', 'smoothness2', 
         'compactness2', 'concavity2', 'concave_points2', 'symmetry2', 'fractal_dimension2', 'radius3', 'texture3', 'perimeter3', 'area3', 'smoothness3', 'compactness3', 'concavity3', 'concave_points3', 'symmetry3', 'fractal_dimension3']

dadosFull = pd.read_csv(url, header=None, names=names)

dados = dadosFull.drop(columns=['Diagnosis'])
dados_cat = dadosFull['Diagnosis']

normalizador = preprocessing.MinMaxScaler()
modelo_normalizador = normalizador.fit(dados)
dados_normalizados = modelo_normalizador.fit_transform(dados)

dadosFinal = pd.DataFrame(data = dados_normalizados, columns=['ID', 'radius1', 'texture1', 'perimeter1', 'area1', 'smoothness1', 'compactness1', 'concavity1', 'concave_points1', 'symmetry1', 'fractal_dimension1', 'radius2', 'texture2', 'perimeter2', 'area2', 'smoothness2', 
         'compactness2', 'concavity2', 'concave_points2', 'symmetry2', 'fractal_dimension2', 'radius3', 'texture3', 'perimeter3', 'area3', 'smoothness3', 'compactness3', 'concavity3', 'concave_points3', 'symmetry3', 'fractal_dimension3'])

dadosFinal = dadosFinal.join(dados_cat, how='left')

print(dadosFinal)

dados_atributos = dados_normalizados
dados_classes = dados_cat

tree = DecisionTreeClassifier()
atr_train, atr_test, class_train, class_test = train_test_split(dados_atributos, dados_classes, test_size =0.3)
fertility_tree = tree.fit(atr_train, class_train)
class_predict = fertility_tree.predict(atr_test)

print(class_predict)

dump(fertility_tree,open('C:/Users/lucas/OneDrive/Documentos/Estudos/Faculdade/Semestre_6/Data_Science/Trabalho3Data_Science/modelo_fertility_tree.pkl', 'wb'))

acuracia = accuracy_score(class_test, class_predict)
taxa_erro = 1 - acuracia

print(f'Acurácia: {acuracia:.2f}')
print(f'Taxa de Erro: {taxa_erro:.2f}')

cm=confusion_matrix(class_test, class_predict)
disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = fertility_tree.classes_)
disp.plot()
plt.show()

