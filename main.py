import matplotlib.pyplot as plt
import numpy as np  # linear algebra
import pandas as pd
import seaborn as sns

df = pd.read_csv('data/Crop_recommendation.csv')
df.head()
print(df.describe())
sns.heatmap(df.isnull(), cmap="coolwarm")
plt.show()
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
# sns.distplot(df_setosa['sepal_length'],kde=True,color='green',bins=20,hist_kws={'alpha':0.3})
sns.distplot(df['temperature'], color="purple", bins=15, hist_kws={'alpha': 0.2})
plt.subplot(1, 2, 2)
sns.distplot(df['ph'], color="green", bins=15, hist_kws={'alpha': 0.2})
sns.pairplot(df, hue='label')
plt.show()
sns.jointplot(x="rainfall", y="humidity", data=df[(df['temperature'] < 30) & (df['rainfall'] > 120)], hue="label")
plt.show()
sns.jointplot(x="K", y="N", data=df[(df['N'] > 40) & (df['K'] > 40)], hue="label")
plt.show()
sns.jointplot(x="K", y="humidity", data=df, hue='label', size=8, s=30, alpha=0.7)
plt.show()
sns.boxplot(y='label', x='ph', data=df)
plt.show()
sns.boxplot(y='label', x='P', data=df[df['rainfall'] > 150])
plt.show()
sns.lineplot(data=df[(df['humidity'] < 65)], x="K", y="rainfall", hue="label")
plt.show()
c = df.label.astype('category')
targets = dict(enumerate(c.cat.categories))
df['target'] = c.cat.codes

y = df.target
X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
sns.heatmap(X.corr())
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)

# we must apply the scaling to the test set as well that we are computing for the training set
X_test_scaled = scaler.transform(X_test)
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(X_train_scaled, y_train)
knn.score(X_test_scaled, y_test)
from sklearn.metrics import confusion_matrix

mat = confusion_matrix(y_test, knn.predict(X_test_scaled))
df_cm = pd.DataFrame(mat, list(targets.values()), list(targets.values()))
sns.set(font_scale=1.0)  # for label size
plt.figure(figsize=(12, 8))
sns.heatmap(df_cm, annot=True, annot_kws={"size": 12}, cmap="terrain")
plt.show()
k_range = range(1, 11)
scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    scores.append(knn.score(X_test_scaled, y_test))

plt.xlabel('k')
plt.ylabel('accuracy')
plt.scatter(k_range, scores)
plt.vlines(k_range, 0, scores, linestyle="dashed")
plt.ylim(0.96, 0.99)
plt.xticks([i for i in range(1, 11)]);
from sklearn.svm import SVC

plt.show()
svc_linear = SVC(kernel='linear').fit(X_train_scaled, y_train)
print("Linear Kernel Accuracy: ", svc_linear.score(X_test_scaled, y_test))

svc_poly = SVC(kernel='rbf').fit(X_train_scaled, y_train)
print("Rbf Kernel Accuracy: ", svc_poly.score(X_test_scaled, y_test))

svc_poly = SVC(kernel='poly').fit(X_train_scaled, y_train)
print("Poly Kernel Accuracy: ", svc_poly.score(X_test_scaled, y_test))
from sklearn.model_selection import GridSearchCV

parameters = {'C': np.logspace(-3, 2, 6).tolist(), 'gamma': np.logspace(-3, 2, 6).tolist()}
# 'degree': np.arange(0,5,1).tolist(), 'kernel':['linear','rbf','poly']

model = GridSearchCV(estimator=SVC(kernel="linear"), param_grid=parameters, n_jobs=-1, cv=4)
model.fit(X_train, y_train)
print(model.best_score_)
print(model.best_params_)
