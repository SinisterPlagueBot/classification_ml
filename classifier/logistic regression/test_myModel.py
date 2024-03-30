from mymodel import LogisticRegression,accuracy
import numpy as np
from sklearn.model_selection import train_test_split

# Define sample data
np.random.seed(42)
X = np.random.randn(100, 2)  # 100 samples, 2 features
y = np.random.randint(2, size=100)  # Binary labels (0 or 1)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instantiate LogisticRegression object
model = LogisticRegression(proba='tanh', learning_rate=0.1, max_iter=1000, random_state=42)

# Fit the model to the training data
model.fit(X_train, y_train)
print('accuracy is :',accuracy(y_train,model.predict(X_train)))
