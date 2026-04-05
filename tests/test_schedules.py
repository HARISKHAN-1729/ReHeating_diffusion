"""Unit tests for schedule construction."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from schedules import (
    make_schedule, make_edm2_schedule, make_flow_schedule,
    schedule_report, compare_all_schedules,
)


class TestScheduleShapes:
    """Verify all schedules return correct shapes."""

    @pytest.mark.parametrize("family", [
        'monotonic', 'single_reheat', 'sawtooth', 'damped_oscillation',
    ])
    @pytest.mark.parametrize("N", [10, 25, 50, 100])
    def test_shape(self, family, N):
        sched = make_schedule(family, N=N)
        assert sched.shape == (N,), f"{family} N={N}: shape {sched.shape} != ({N},)"

    @pytest.mark.parametrize("family", [
        'monotonic', 'single_reheat', 'sawtooth', 'damped_oscillation',
    ])
    def test_edm2_shape(self, family):
        sigmas = make_edm2_schedule(family, N=50)
        assert sigmas.shape == (51,)  # N + terminal 0
        assert sigmas[-1] == 0.0

    @pytest.mark.parametrize("family", [
        'monotonic', 'single_reheat', 'sawtooth', 'damped_oscillation',
    ])
    def test_flow_shape(self, family):
        times = make_flow_schedule(family, N=50)
        assert times.shape == (50,)


class TestScheduleBoundaries:
    """Verify boundary conditions."""

    @pytest.mark.parametrize("family", [
        'monotonic', 'single_reheat', 'sawtooth', 'damped_oscillation',
    ])
    def test_boundaries(self, family):
        sched = make_schedule(family, N=50)
        # Sampling order: index 0 = high noise (low alpha), index N-1 = low noise (high alpha)
        assert sched[0] < 0.01, f"{family}: first element should be near 0 (high noise)"
        assert sched[-1] > 0.99, f"{family}: last element should be near 1 (low noise)"

    @pytest.mark.parametrize("family", [
        'monotonic', 'single_reheat', 'sawtooth', 'damped_oscillation',
    ])
    def test_range(self, family):
        sched = make_schedule(family, N=100)
        assert np.all(sched > 0), f"{family}: all values must be > 0"
        assert np.all(sched < 1), f"{family}: all values must be < 1"


class TestMonotonicity:
    """Verify monotonicity properties."""

    def test_monotonic_is_monotonic(self):
        sched = make_schedule('monotonic', N=100)
        diffs = np.diff(sched)
        assert np.all(diffs >= 0), "Monotonic schedule must be non-decreasing in sampling order"

    def test_single_reheat_has_non_monotonic(self):
        sched = make_schedule('single_reheat', N=100, delta=0.15, t_reheat=0.4)
        diffs = np.diff(sched)
        n_reheat = np.sum(diffs < -1e-6)
        assert n_reheat > 0, "Single reheat should have at least one reheat step"

    def test_sawtooth_has_non_monotonic(self):
        sched = make_schedule('sawtooth', N=100, period=25, sawtooth_delta=0.08)
        diffs = np.diff(sched)
        n_reheat = np.sum(diffs < -1e-6)
        assert n_reheat > 0, "Sawtooth should have reheat steps"

    def test_damped_oscillation_has_non_monotonic(self):
        sched = make_schedule('damped_oscillation', N=100, A=0.2, gamma=2.5, freq=4)
        diffs = np.diff(sched)
        n_reheat = np.sum(diffs < -1e-6)
        assert n_reheat > 0, "Damped oscillation should have reheat steps"


class TestScheduleReport:
    """Verify schedule reporting."""

    def test_report_keys(self):
        sched = make_schedule('monotonic', N=50)
        stats = schedule_report(sched, name='test')
        assert 'N' in stats
        assert 'reheat_steps' in stats
        assert 'denoise_steps' in stats
        assert 'overhead' in stats
        assert 'critical_zone_steps' in stats

    def test_monotonic_zero_overhead(self):
        sched = make_schedule('monotonic', N=50)
        stats = schedule_report(sched)
        assert stats['reheat_steps'] == 0
        assert stats['overhead'] < 0.01  # Should be ~0

    def test_reheat_positive_overhead(self):
        sched = make_schedule('single_reheat', N=50, delta=0.15)
        stats = schedule_report(sched)
        assert stats['overhead'] > 0, "Reheating schedule should have positive overhead"


class TestEdgeCases:
    """Edge cases and error handling."""

    def test_unknown_family(self):
        with pytest.raises(ValueError):
            make_schedule('nonexistent', N=50)

    def test_small_N(self):
        sched = make_schedule('monotonic', N=5)
        assert len(sched) == 5

    def test_single_reheat_extreme_params(self):
        # Very large delta
        sched = make_schedule('single_reheat', N=50, delta=0.5, t_reheat=0.5)
        assert np.all(sched > 0) and np.all(sched < 1)

    def test_edm2_sigma_ordering(self):
        sigmas = make_edm2_schedule('monotonic', N=50)
        # Should start high, end at 0
        assert sigmas[0] > sigmas[-2] > sigmas[-1] == 0.0


class TestCompareAll:
    """Verify compare_all_schedules runs."""

    def test_compare_runs(self, capsys):
        compare_all_schedules(N=50)
        captured = capsys.readouterr()
        assert 'Monotonic' in captured.out
        assert 'Single Reheat' in captured.out
        assert 'Sawtooth' in captured.out
        assert 'Damped Oscillation' in captured.out


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
