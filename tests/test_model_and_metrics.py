"""
Unit tests for FedActuary Neural Network Model & Actuarial Metrics.
"""

import unittest
import numpy as np
import torch
import torch.nn as nn

from src.model import MultipleRegression, get_parameters, set_parameters
from src.metrics import percentage_deviance_explained, gini_coefficient
from src.data import preprocess_single_record, load_preprocessing_artifacts
from src.config import CKPT_DIR


class TestModelAndMetrics(unittest.TestCase):
    def test_model_forward_pass_and_shape(self):
        """Verify model output is strictly non-negative count with expected shape."""
        batch_size = 16
        num_features = 39
        model = MultipleRegression(num_features=num_features)
        x = torch.randn(batch_size, num_features)
        y_pred = model(x)

        self.assertEqual(y_pred.shape, (batch_size, 1))
        # Ensure outputs are strictly positive due to exp activation
        self.assertTrue((y_pred > 0).all())

    def test_model_parameters_getter_and_setter(self):
        """Verify weight extraction and restoration for federated weight aggregation."""
        model1 = MultipleRegression(num_features=10)
        model2 = MultipleRegression(num_features=10)

        params1 = get_parameters(model1)
        set_parameters(model2, params1)

        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            self.assertTrue(torch.allclose(p1, p2))

    def test_percentage_deviance_explained(self):
        """Verify %PDE calculation on synthetic oracle and flat null cases."""
        rng = np.random.default_rng(42)
        n = 1000
        exposure = rng.uniform(0.2, 1.0, n)
        true_freq = rng.gamma(2.0, 0.04, n)
        y = rng.poisson(true_freq * exposure).astype(float)

        # Oracle count
        pde_oracle = percentage_deviance_explained(y, true_freq * exposure, exposure)
        self.assertGreater(pde_oracle, 0.0)

        # Flat null should score near 0%
        flat_null = np.full(n, np.average(y, weights=exposure))
        pde_null = percentage_deviance_explained(y, flat_null, exposure)
        self.assertAlmostEqual(pde_null, 0.0, delta=1.5)

    def test_gini_coefficient(self):
        """Verify Gini calculation bounds and ranking ability."""
        rng = np.random.default_rng(42)
        n = 1000
        exposure = rng.uniform(0.2, 1.0, n)
        true_freq = rng.gamma(2.0, 0.04, n)
        y = rng.poisson(true_freq * exposure).astype(float)

        gini = gini_coefficient(y, true_freq * exposure, exposure)
        self.assertGreaterEqual(gini, 0.0)
        self.assertLessEqual(gini, 1.0)

    def test_single_record_preprocessing(self):
        """Verify that single raw record creates exact (1, 39) numpy array."""
        scaler, feature_names = load_preprocessing_artifacts(CKPT_DIR)
        self.assertEqual(len(feature_names), 39)

        sample = {
            "Exposure": 1.0,
            "Area": "C",
            "VehPower": 6,
            "VehAge": 4,
            "DrivAge": 35,
            "BonusMalus": 50,
            "VehBrand": "B12",
            "VehGas": "Regular",
            "Density": 1000.0,
            "Region": "R82"
        }
        x_array = preprocess_single_record(sample, scaler, feature_names)
        self.assertEqual(x_array.shape, (1, 39))
        self.assertEqual(x_array.dtype, np.float32)


if __name__ == "__main__":
    unittest.main()
