from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

def create_model():
    mlp = MLPClassifier(
        max_iter=2000,
        random_state=42
    )

    param_grid = {
        "hidden_layer_sizes": [(50,), (100,), (50, 50)],
        "activation": ["relu", "tanh"],
        "alpha": [0.0001, 0.001, 0.01]
    }

    grid = GridSearchCV(
        mlp,
        param_grid,
        cv=5,
        scoring="accuracy"
    )

    return grid
