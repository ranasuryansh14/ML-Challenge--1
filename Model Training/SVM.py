import pickle
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# Load preprocessed data
X_train = np.load(".npy/X_train.npy")
X_test = np.load(".npy/X_test.npy")
y_train = np.load(".npy/y_train.npy")
y_test = np.load(".npy/y_test.npy")

# Train SVM Model (Linear Kernel)
model = SVC(kernel="linear")
model.fit(X_train, y_train)

# Predict on Test Data
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Save accuracy results
with open("accuracy_results.txt", "a") as file:
    file.write(f"SVM (Linear) Accuracy: {accuracy:.2%}\n")

# Save trained model
with open("models/svm.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

# Print accuracy
print(f"SVM (Linear) Accuracy: {accuracy:.2%}")
