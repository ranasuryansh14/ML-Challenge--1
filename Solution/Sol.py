import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


X_train = np.load(".npy/X_train.npy")
X_test = np.load(".npy/X_test.npy")
y_train = np.load(".npy/y_train.npy")
y_test = np.load(".npy/y_test.npy")


models = {
    "Logistic Regression": LogisticRegression(),
    "SVM (Linear Kernel)": SVC(kernel='linear'),
    "K-Nearest Neighbors": KNeighborsClassifier(),
  
}


accuracy_results = {}

for name, model in models.items():
    model.fit(X_train, y_train) 
    y_pred = model.predict(X_test) 
    accuracy = accuracy_score(y_test, y_pred) 
    accuracy_results[name] = accuracy
    print(f"{name} Accuracy: {accuracy:.2%}")

# Save accuracy results to a file
with open("accuracy_results.txt", "w") as file:
    for name, acc in accuracy_results.items():
        file.write(f"{name} Accuracy: {acc:.2%}\n")

# Display accuracy in a table format
df_results = pd.DataFrame(accuracy_results.items(), columns=["Model", "Accuracy"])
df_results.set_index("Model", inplace=True)

print("\nModel Performance Summary:\n")
print(df_results)
