from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def train_val_test_split(X, y, test_size=0.2, val_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size, random_state=random_state
    )

    return X_train, X_val, X_test, y_train, y_val, y_test

def standardize(X_train, X_val, X_test):
    """
    Standardizes each feature to zero mean and unit variance. 
    It learns per‑feature μ and σ from the training data, then applies x' = (x − μ) / σ to train, validation, and test.

    Why only X_train is fit: To avoid data leakage. Validation/test data must simulate unseen data; 
    using their values to compute μ/σ would leak information, making performance estimates optimistically biased. 
    You fit on X_train once, then use the same scaler to transform X_val and X_test.

    Parameters: 
        *Arrays: Any
    
    Return:
        *Arrays
        *Scaler
    """
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)
    return X_train_s, X_val_s, X_test_s, scaler