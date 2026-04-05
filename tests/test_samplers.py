"""Integration tests for samplers with dummy models."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import pytest
from schedules import make_schedule, make_edm2_schedule, make_flow_schedule
from samplers.ddim_reheating import ddim_sample, adaptive_sample
from samplers.edm2_reheating import edm2_sample
from samplers.flow_reheating import flow_sample


# --- Dummy models ---

class DummyEpsModel(torch.nn.Module):
    """Dummy noise predictor for DDIM testing."""
    def forward(self, x, t):
        return torch.randn_like(x)


class DummyDenoiser(torch.nn.Module):
    """Dummy denoiser D(x, sigma) -> denoised x for EDM2 testing."""
    def forward(self, x, sigma):
        return x * 0.5  # Just shrink toward 0


class DummyVelocityModel(torch.nn.Module):
    """Dummy velocity model v(x, t) for flow matching testing."""
    def forward(self, x, t):
        return -x * 0.1  # Simple contraction


# --- DDIM tests ---

class TestDDIMSampler:
    def test_output_shape(self):
        model = DummyEpsModel()
        sched = make_schedule('monotonic', N=10)
        out = ddim_sample(model, sched, shape=(4, 3, 8, 8), device='cpu', use_fp16=False)
        assert out.shape == (4, 3, 8, 8)

    def test_no_nan(self):
        model = DummyEpsModel()
        sched = make_schedule('monotonic', N=10)
        out = ddim_sample(model, sched, shape=(2, 3, 8, 8), device='cpu', use_fp16=False)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_trajectory(self):
        model = DummyEpsModel()
        sched = make_schedule('monotonic', N=10)
        out, traj = ddim_sample(model, sched, shape=(2, 3, 8, 8), device='cpu',
                                return_trajectory=True, use_fp16=False)
        assert len(traj) == 10  # N steps = N-1 transitions + initial
        assert traj[0].shape == (2, 3, 8, 8)

    @pytest.mark.parametrize("family", ['monotonic', 'single_reheat', 'damped_oscillation'])
    def test_all_families(self, family):
        model = DummyEpsModel()
        sched = make_schedule(family, N=15)
        out = ddim_sample(model, sched, shape=(2, 3, 8, 8), device='cpu', use_fp16=False)
        assert out.shape == (2, 3, 8, 8)
        assert not torch.isnan(out).any()

    def test_stochastic_eta(self):
        model = DummyEpsModel()
        sched = make_schedule('monotonic', N=10)
        out = ddim_sample(model, sched, shape=(2, 3, 8, 8), device='cpu', eta=1.0, use_fp16=False)
        assert out.shape == (2, 3, 8, 8)
        assert not torch.isnan(out).any()

    def test_2d_shape(self):
        """Test with 2D data (for toy experiments)."""
        model = DummyEpsModel()
        sched = make_schedule('monotonic', N=10)
        out = ddim_sample(model, sched, shape=(16, 2), device='cpu', use_fp16=False)
        assert out.shape == (16, 2)


# --- Adaptive sampler tests ---

class TestAdaptiveSampler:
    def test_output_shape(self):
        model = DummyEpsModel()
        sched = make_schedule('monotonic', N=10)
        out = adaptive_sample(model, sched, shape=(2, 3, 8, 8), tau=0.5,
                              device='cpu', use_fp16=False)
        assert out.shape == (2, 3, 8, 8)

    def test_stats(self):
        model = DummyEpsModel()
        sched = make_schedule('monotonic', N=10)
        out, stats = adaptive_sample(model, sched, shape=(2, 3, 8, 8), tau=0.5,
                                     device='cpu', return_stats=True, use_fp16=False)
        assert stats.total_nfe > 0
        assert stats.total_nfe <= len(sched) + 15  # max_reheats default
        assert isinstance(stats.reheat_positions, list)
        assert isinstance(stats.prediction_deltas, list)

    def test_nfe_budget(self):
        model = DummyEpsModel()
        sched = make_schedule('monotonic', N=10)
        # Very low tau => many reheats triggered, but budget should cap it
        out, stats = adaptive_sample(model, sched, shape=(2, 3, 8, 8), tau=0.001,
                                     max_reheats=5, device='cpu',
                                     return_stats=True, use_fp16=False)
        assert stats.total_nfe <= len(sched) + 5


# --- EDM2 tests ---

class TestEDM2Sampler:
    def test_output_shape(self):
        denoiser = DummyDenoiser()
        sigmas = make_edm2_schedule('monotonic', N=10)
        out = edm2_sample(denoiser, sigmas, shape=(2, 3, 8, 8), device='cpu', use_fp16=False)
        assert out.shape == (2, 3, 8, 8)

    def test_no_nan(self):
        denoiser = DummyDenoiser()
        sigmas = make_edm2_schedule('monotonic', N=10)
        out = edm2_sample(denoiser, sigmas, shape=(2, 3, 8, 8), device='cpu', use_fp16=False)
        assert not torch.isnan(out).any()

    def test_heun(self):
        denoiser = DummyDenoiser()
        sigmas = make_edm2_schedule('monotonic', N=10)
        out = edm2_sample(denoiser, sigmas, shape=(2, 3, 8, 8), device='cpu',
                          order=2, use_fp16=False)
        assert out.shape == (2, 3, 8, 8)

    @pytest.mark.parametrize("family", ['monotonic', 'single_reheat', 'damped_oscillation'])
    def test_reheating_families(self, family):
        denoiser = DummyDenoiser()
        sigmas = make_edm2_schedule(family, N=15)
        out = edm2_sample(denoiser, sigmas, shape=(2, 3, 8, 8), device='cpu', use_fp16=False)
        assert out.shape == (2, 3, 8, 8)

    def test_trajectory(self):
        denoiser = DummyDenoiser()
        sigmas = make_edm2_schedule('monotonic', N=10)
        out, traj = edm2_sample(denoiser, sigmas, shape=(2, 3, 8, 8), device='cpu',
                                return_trajectory=True, use_fp16=False)
        assert len(traj) > 0


# --- Flow matching tests ---

class TestFlowSampler:
    def test_output_shape(self):
        vmodel = DummyVelocityModel()
        times = make_flow_schedule('monotonic', N=10)
        out = flow_sample(vmodel, times, shape=(2, 3, 8, 8), device='cpu', use_fp16=False)
        assert out.shape == (2, 3, 8, 8)

    def test_no_nan(self):
        vmodel = DummyVelocityModel()
        times = make_flow_schedule('monotonic', N=10)
        out = flow_sample(vmodel, times, shape=(2, 3, 8, 8), device='cpu', use_fp16=False)
        assert not torch.isnan(out).any()

    @pytest.mark.parametrize("family", ['monotonic', 'single_reheat', 'damped_oscillation'])
    def test_reheating_families(self, family):
        vmodel = DummyVelocityModel()
        times = make_flow_schedule(family, N=15)
        out = flow_sample(vmodel, times, shape=(2, 3, 8, 8), device='cpu', use_fp16=False)
        assert out.shape == (2, 3, 8, 8)

    def test_trajectory(self):
        vmodel = DummyVelocityModel()
        times = make_flow_schedule('monotonic', N=10)
        out, traj = flow_sample(vmodel, times, shape=(2, 3, 8, 8), device='cpu',
                                return_trajectory=True, use_fp16=False)
        assert len(traj) == 10


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
