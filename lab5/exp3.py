import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, classification_report

column_names = ["ID", "Diagnosis"] + [f"Feature_{i}" for i in range(1, 31)]
df = pd.read_csv("wdbc.data", header=None, names=column_names)

df["Diagnosis"] = df["Diagnosis"].map({"M": 0, "B": 1})

X = df.drop(["ID", "Diagnosis"], axis=1)
y = df["Diagnosis"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

nb_model = GaussianNB()
dt_model = DecisionTreeClassifier(random_state=42)

nb_model.fit(X_train, y_train)
dt_model.fit(X_train, y_train)

nb_train_acc = accuracy_score(y_train, nb_model.predict(X_train))
nb_test_acc = accuracy_score(y_test, nb_model.predict(X_test))

dt_train_acc = accuracy_score(y_train, dt_model.predict(X_train))
dt_test_acc = accuracy_score(y_test, dt_model.predict(X_test))

labels = ["Naïve Bayes", "Decision Tree"]
train_acc = [nb_train_acc, dt_train_acc]
test_acc = [nb_test_acc, dt_test_acc]

x = np.arange(len(labels))
width = 0.35

plt.figure()
plt.bar(x - width/2, train_acc, width, label="Training Accuracy", color="#5D6D7E")
plt.bar(x + width/2, test_acc, width, label="Testing Accuracy", color="#D98880")
plt.xticks(x, labels)
plt.ylabel("Accuracy")
plt.title("Training vs Testing Accuracy Comparison")
plt.legend()
plt.show()

nb_prob = nb_model.predict_proba(X_test)[:, 1]
dt_prob = dt_model.predict_proba(X_test)[:, 1]

nb_fpr, nb_tpr, _ = roc_curve(y_test, nb_prob)
dt_fpr, dt_tpr, _ = roc_curve(y_test, dt_prob)

nb_auc = auc(nb_fpr, nb_tpr)
dt_auc = auc(dt_fpr, dt_tpr)

plt.figure()
plt.plot(nb_fpr, nb_tpr, color="#34495E", linewidth=2, label="Naïve Bayes (AUC = %0.2f)" % nb_auc)
plt.plot(dt_fpr, dt_tpr, color="#138D75", linewidth=2, label="Decision Tree (AUC = %0.2f)" % dt_auc)
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.show()

nb_cm = confusion_matrix(y_test, nb_model.predict(X_test))
dt_cm = confusion_matrix(y_test, dt_model.predict(X_test))

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
sns.heatmap(nb_cm, annot=True, fmt="d", cmap="Pastel1", cbar=False)
plt.title("Naïve Bayes Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.subplot(1,2,2)
sns.heatmap(dt_cm, annot=True, fmt="d", cmap="Pastel2", cbar=False)
plt.title("Decision Tree Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()
plt.show()

print("Naïve Bayes Test Accuracy:", nb_test_acc)
print("Decision Tree Test Accuracy:", dt_test_acc)

print("\nNaïve Bayes Classification Report:\n")
print(classification_report(y_test, nb_model.predict(X_test)))

print("\nDecision Tree Classification Report:\n")
print(classification_report(y_test, dt_model.predict(X_test)))
