from sklearn.svm import LinearSVC
import numpy as np

data = np.load("sift_histogram.npz", allow_pickle=True)

# Create an SVM model
svm = LinearSVC(random_state=42)

# Train the model
svm.fit(data["X_train"], data["y_train"])

# Evaluate the model
accuracy = svm.score(data["X_test"], data["y_test"])
print(f'SVM accuracy: {accuracy:.2f}')