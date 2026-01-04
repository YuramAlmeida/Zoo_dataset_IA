from data import load_and_prepare_data
from model import create_model

def train():
    X_train, X_test, y_train, y_test = load_and_prepare_data()

    grid = create_model()
    grid.fit(X_train, y_train)

    print("Melhores hiperparâmetros encontrados:")
    print(grid.best_params_)

    return grid, X_test, y_test

if __name__ == "__main__":
    train()
