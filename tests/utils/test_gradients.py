from unittest import TestCase

import pytest
import numpy as np

from morpheus.utils.gradients import perturb, num_grad_batch


class GradientsTest(TestCase):
    def test_perturb_basic(self):
        X = np.array([[1, 2], [3, 4]], dtype=float)
        eps = 0.01
        X_pert_pos, X_pert_neg = perturb(X, eps)

        self.assertEqual(X_pert_pos.shape,(4, 2))
        self.assertEqual(X_pert_neg.shape , (4, 2))
        np.testing.assert_allclose(X_pert_pos[0], [1.01, 2])
        np.testing.assert_allclose(X_pert_neg[0], [0.99, 2])

    def test_perturb_probability_mode(self):
        X = np.array([[0.25, 0.75]])
        eps = 0.02
        X_pert_pos, X_pert_neg = perturb(X, eps, proba=True)
        expected_pos = [[0.27, 0.73], [0.23, 0.77]]
        expected_neg = [[0.23, 0.77], [0.27, 0.73]]
        np.testing.assert_allclose(X_pert_pos, expected_pos)
        np.testing.assert_allclose(X_pert_neg, expected_neg)

    def test_num_grad_batch(self):
        def sample_function(X, multiplier=1.0):
            return X * multiplier

        X = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)
        args = (2.0,)
        eps = 0.01
        gradients = num_grad_batch(sample_function, X, args, eps)
        expected_gradients = np.array([[[2.0, 0.0], [0.0, 2.0]], [[2.0, 0.0], [0.0, 2.0]]])
        np.testing.assert_allclose(gradients, expected_gradients)

    def test_num_grad_batch_with_zero_epsilon(self):
        def sample_function(X, multiplier=0.0):
            raise ZeroDivisionError

        X = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)
        args = (2.0,)
        eps = 0
        with pytest.raises(ZeroDivisionError):
            _ = num_grad_batch(sample_function, X, args, eps)
