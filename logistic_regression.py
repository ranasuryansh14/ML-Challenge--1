import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load preprocessed data
X_train = np.load("X_train.npy")
X_test = np.load("X_test.npy")
y_train = np.load("y_train.npy")
y_test = np.load("y_test.npy")

# Train Logistic Regression Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict on Test Data
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Save accuracy results
with open("accuracy_results.txt", "a") as file:
    file.write(f"Logistic Regression Accuracy: {accuracy:.2%}\n")

# Save trained model
with open("models/logistic_regression.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

# Print accuracy
print(f"Logistic Regression Accuracy: {accuracy:.2%}")
