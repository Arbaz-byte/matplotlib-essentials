import numpy as np

def mean_absolute_error(y_true, y_pred):
    """
    Compute the Mean Absolute Error (MAE).

    Parameters:
    y_true : array-like
        Ground truth (correct) target values.
    y_pred : array-like
        Estimated target values.

    Returns:
    float
        MAE loss.
    """
    return np.mean(np.abs(y_true - y_pred))


def mean_squared_error(y_true, y_pred):
    """
    Compute the Mean Squared Error (MSE).

    Parameters:
    y_true : array-like
        Ground truth (correct) target values.
    y_pred : array-like
        Estimated target values.

    Returns:
    float
        MSE loss.
    """
    return np.mean((y_true - y_pred) ** 2)


def huber_loss(y_true, y_pred, delta=1.0):
    """
    Compute the Huber loss.

    Huber loss is a combination of MSE and MAE:
        - Quadratic for small errors (|error| ≤ delta)
        - Linear for large errors (|error| > delta)

    This makes it robust to outliers while maintaining smooth gradients.

    Parameters:
    y_true : array-like
        Ground truth (correct) target values.
    y_pred : array-like
        Estimated target values.
    delta : float, default=1.0
        Threshold parameter that controls where the loss transitions
        from quadratic to linear.

    Returns:
    float
        Huber loss averaged over all samples.
    """
    residual = y_true - y_pred
    abs_res = np.abs(residual)

    # Split into quadratic and linear parts
    quadratic = np.minimum(abs_res, delta)          # errors within delta
    linear = abs_res - quadratic                    # errors beyond delta

    # Loss: 0.5 * quadratic^2 + delta * linear
    return np.mean(0.5 * quadratic ** 2 + delta * linear)


def loss_function(y_true, y_pred, loss_type='mse', **kwargs):
    """
    Unified interface to compute MAE, MSE, or Huber loss.

    Parameters:
    y_true : array-like
        Ground truth (correct) target values.
    y_pred : array-like
        Estimated target values.
    loss_type : str, default='mse'
        Type of loss to compute. Options: 'mae', 'mse', 'huber'.
    **kwargs : additional arguments
        Passed to the specific loss function (e.g., `delta` for Huber).

    Returns:
    float
        The computed loss.

    Raises:
    ValueError
        If an unknown loss_type is provided.
    """
    if loss_type == 'mae':
        return mean_absolute_error(y_true, y_pred)
    elif loss_type == 'mse':
        return mean_squared_error(y_true, y_pred)
    elif loss_type == 'huber':
        return huber_loss(y_true, y_pred, **kwargs)
    else:
        raise ValueError(f"Unsupported loss_type '{loss_type}'. Choose from 'mae', 'mse', 'huber'.")


# ------------------- Example Usage -------------------
if __name__ == "__main__":
    # Sample data
    y_true = np.array([3.0, -0.5, 2.0, 7.0])
    y_pred = np.array([2.5, 0.0, 2.0, 8.0])

    print("MAE   :", mean_absolute_error(y_true, y_pred))
    print("MSE   :", mean_squared_error(y_true, y_pred))
    print("Huber :", huber_loss(y_true, y_pred, delta=1.0))

    # Using the unified function
    print("Via unified (MAE)   :", loss_function(y_true, y_pred, loss_type='mae'))
    print("Via unified (Huber) :", loss_function(y_true, y_pred, loss_type='huber', delta=0.5))
