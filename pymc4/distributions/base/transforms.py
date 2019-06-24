class Transform(object):
    def forward(self, x):
        """Applies transformation forward to input variable `x`.
        When transform is used on some distribution `p`, it will transform the random variable `x` after sampling
        from `p`.

        Parameters
        ----------
        x : tensor
            Input tensor to be transformed.

        Returns
        --------
        tensor
            Transformed tensor.
        """
        raise NotImplementedError

    def backward(self, z):
        """
        Applies inverse of transformation to input variable `z`.
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
        raise NotImplementedError

    def jacobian_log_det(self, x):
        """
        Calculates logarithm of the absolute value of the Jacobian determinant for input `x`.

        Parameters
        ----------
        x : tensor
            Input to calculate Jacobian determinant of.

        Returns
        -------
        tensor
            The log abs Jacobian determinant of `x` w.r.t. this transform.
        """
        raise NotImplementedError

    def inverse_jacobian_log_det(self, z):
        """
        Calculates logarithm of the absolute value of the Jacobian determinant for output `z`.

        Parameters
        ----------
        z : tensor
            Output to calculate Jacobian determinant of.

        Returns
        -------
        tensor
            The log abs Jacobian determinant of `z` w.r.t. this transform.

        Notes
        -----
        May be desired to be implemented efficiently
        """
        raise -self.jacobian_log_det(self.backward(z))


class Invert(Transform):
    def __init__(self, transform):
        self.transform = transform

    def forward(self, x):
        return self.transform.backward(x)

    def backward(self, z):
        return self.transform.forward(z)

    def jacobian_log_det(self, x):
        return self.transform.inverse_jacobian_log_det(x)

    def inverse_jacobian_log_det(self, z):
        return self.transform.jacobian_log_det(z)


class Exp(Transform):
    ...
