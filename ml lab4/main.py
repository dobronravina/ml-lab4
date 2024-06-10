import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import *
from sklearn.model_selection import ShuffleSplit
from sklearn.neighbors import KNeighborsClassifier

# Завантаження даних
data = pd.read_csv('WQ-R.csv', sep=';')

# Показники даних
total_rows = data.shape[0]
total_columns = data.shape[1]
print('Кількість записів:', total_rows)

# Показ назв колонок
for idx, column in enumerate(data.columns, start=1):
    print(f'{idx}) {column}')

# Поділ даних на навчальну і тестову вибірки
splitter = ShuffleSplit(n_splits=10, train_size=0.8, random_state=1)
split_indices = list(splitter.split(data))[7]

train_data = data.iloc[split_indices[0]]
test_data = data.iloc[split_indices[1]]

print(train_data['quality'].value_counts())
print(test_data['quality'].value_counts())

X_train, y_train = train_data.iloc[:, :-1], train_data.iloc[:, -1]
X_test, y_test = test_data.iloc[:, :-1], test_data.iloc[:, -1]

# Навчання моделі
knn_model = KNeighborsClassifier()
knn_model.fit(X_train, y_train)

# Функція для обчислення метрик
def calculate_metrics(model, features, labels):
    preds = model.predict(features)
    return {
        'Accuracy': accuracy_score(labels, preds),
        'Precision': precision_score(labels, preds, average='weighted', zero_division=0),
        'Recall': recall_score(labels, preds, average='weighted', zero_division=0),
        'F1-Score': f1_score(labels, preds, average='weighted', zero_division=0),
        'MCC': matthews_corrcoef(labels, preds),
        'Balanced Accuracy': balanced_accuracy_score(labels, preds),
    }

# Обчислення метрик для навчальної і тестової вибірки
train_metrics = calculate_metrics(knn_model, X_train, y_train)
test_metrics = calculate_metrics(knn_model, X_test, y_test)

# Відображення метрик
plt.bar(test_metrics.keys(), test_metrics.values())
plt.title('Показники класифікації на тестовій вибірці')
plt.show()

# Аналіз впливу кількості сусідів
neighbors_range = range(1, 21)
train_scores = []
test_scores = []

for k in neighbors_range:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    train_scores.append(balanced_accuracy_score(y_train, model.predict(X_train)))
    test_scores.append(balanced_accuracy_score(y_test, model.predict(X_test)))

# Побудова графіків після завершення циклу
plt.plot(neighbors_range, test_scores, label='Тестова вибірка')
plt.plot(neighbors_range, train_scores, label='Навчальна вибірка')
plt.xlabel('Кількість сусідів')
plt.ylabel('Збалансована точність')
plt.legend()
plt.title('Вплив кількості сусідів на результати класифікації')
plt.xticks(neighbors_range)
plt.show()
