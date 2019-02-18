from tensorflow_probability import bijectors


class Transform:
    """
    A transformation of a random variable from one space into another.

    Attributes
    ----------
    name : str
    """

    def __init__(self, name="", bijector=None):
        self.name = name
        self.bijector = bijector

    def forward(self, x):
        """
        Apply transformation forward to input variable `x`.

        When transform is used on some distribution `p`, it will transform the random variable `x` after sampling
        from `p`.

        Parameters
        ----------
        x : tensor
            Input tensor to be transformed.

        Returns
        -------
        tensor
            Transformed tensor.
        """
        return self.bijector.forward(x)

    def backward(self, x):
        """
        Apply inverse of transformation to input variable `z`.

        When transform is used on some distribution `p`, which has observed values `z`, it is used to
        transform the values of `z` correctly to the support of `p`.

        Parameters
        ----------
        z : tensor
            Input tensor to be inverse transformed.

        Returns
        -------
        tensor
            Inverse transformed tensor.
        """
        return self.bijector.inverse(x)

    def logjacdet(self, x):
        """
        Calculate logarithm of the absolute value of the Jacobian determinant for input `x`.

        Parameters
        ----------
        x : tensor
            Input to calculate Jacobian determinant of.

        Returns
        -------
        tensor
            The log abs Jacobian determinant of `x` w.r.t. this transform.
        """
        return self.bijector.forward_log_det_jacobian(x, x.shape.ndims)


logodds = Transform(name="logodds", bijector=bijectors.Invert(bijectors.Sigmoid))
