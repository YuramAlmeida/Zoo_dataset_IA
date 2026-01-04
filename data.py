from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_and_prepare_data():
    X, y = load_iris(return_X_y=True)

    # Normalização
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Separar treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test