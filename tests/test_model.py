"""Tests of the model (simulator.py and simulator_leaky.py).

Run from the package folder:  python -m pytest tests
They take a few minutes: most of them run the model once (about 10 s each).
"""
import contextlib
import datetime
import importlib.util
import io
import os
import sys
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault('MPLBACKEND', 'Agg')  # no figure windows
PACKAGE = Path(__file__).resolve().parents[1]


def load(name):
    """A simulator from this package (whatever else is on sys.path)."""
    spec = importlib.util.spec_from_file_location(name, PACKAGE / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # simulator_leaky imports simulator to draw its figures
    spec.loader.exec_module(module)
    return module


simulator = load('simulator')
simulator_leaky = load('simulator_leaky')

YEAR = 365.25
# The settings of Fig. 3: to July 2027, lockdown on 1 April 2020 (simulator_leaky's defaults end on 1 October 2026)
FIG3 = dict(tspan=27.75 * YEAR, Tpan=20.25 / 27.75)
LEAKY = dict(foi='balanced')
NAMES = ('R0V', 'R0Y', 'TV_years', 'TY_years', 'sigmaVY', 'sigmaYV', 'm', 'season_phase', 'c_pandemic',
         'Lockdown_duration')

# The best fit: rank 1 by NH likelihood, sample 128424 of the search (filtered.csv)
BEST_FIT = dict(zip(NAMES, (2.3189437683189342, 1.707655585879957, 7.223733982639554, 6.624348460662543,
                            0.3108651528504346, 0.0601745125596484, 0.0037653275557292, 3.336869159635911,
                            0.336863303976744, 390.42647145573847)))

# Samples of the search that the model stops early (shortcuts=True), one for each reason, with the outcome it
# returns; every other output is NaN
EARLY_STOPS = {
    'no coexistence before lockdown (sample 0)': (
        (1.8207929041758055, 4.730798244486241, 6.40018127261872, 2.1689587236877386, 0.6384279058226522,
         0.7443780243776483, 0.0086514601497426, 43.36632139356947, 0.2165347231318204, 162.52877480703106),
        dict(simple_coexist=False)),
    'Victoria extinct before lockdown (sample 33583)': (
        (3.764163195008196, 3.402162128347652, 8.280735073445372, 5.04294868484163, 0.4896896790269155,
         0.7304071806204776, 0.007586097234444, 18.29641046073897, 0.3458132405017099, 541.1608145410135),
        dict(simple_coexist=True, recovery_V=-1, extinct_Y=True)),
    'Victoria extinct after lockdown (sample 8)': (
        (1.810669342745839, 3.0602543744525543, 1.880165673816538, 6.809552497243721, 0.1671379772879268,
         0.9555839099521392, 0.0036766024987658, 24.973453812276976, 0.6050149032369327, 247.40574947761243),
        dict(simple_coexist=True, recovery_V=-10, extinct_Y=True)),
    'Yamagata extinct before lockdown (sample 21004)': (
        (3.428765791614966, 2.771661626855076, 5.328821968647284, 9.269859065760452, 0.4405798891220397,
         0.9523815654292638, 0.0076513565693683, 73.12724229562838, 0.775056555337043, 488.35673812821057),
        dict(simple_coexist=True, recovery_Y=-1, extinct_Y=True)),
    'Yamagata never extinct (sample 5)': (
        (4.197515200367191, 3.724567970550921, 3.3810239176806025, 8.76559901634229, 0.2992142948346213,
         0.922583219270969, 0.006511046869758, 85.70334312588041, 0.2038759161218011, 375.5429427349633),
        dict(simple_coexist=True, extinct_Y=False)),
}


def model_kwargs(theta, **changes):
    """model()'s arguments for a parameter set (as in the notebook), with some of them changed."""
    theta = {**theta, **changes}
    return dict(R0V=theta['R0V'], R0Y=theta['R0Y'], omegaV=1 / (YEAR * theta['TV_years']),
                omegaY=1 / (YEAR * theta['TY_years']), sigmaVY=theta['sigmaVY'], sigmaYV=theta['sigmaYV'],
                m=theta['m'], season_phase=theta['season_phase'], c_pandemic=theta['c_pandemic'],
                Lockdown_duration=theta['Lockdown_duration'])


def quiet(fn, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(**kwargs)


@pytest.fixture(scope='module')
def best_fit():
    """The best fit, deterministic, with Yamagata's R_e on the four reintroduction dates."""
    return quiet(simulator.model, plot=False, stochastic2=True, nsims2=0, **model_kwargs(BEST_FIT))


@pytest.mark.parametrize('module', [simulator, simulator_leaky], ids=['all-or-none', 'leaky'])
@pytest.mark.parametrize('m', [0.0, 0.001, 0.0037653275557292, 0.01, 0.3])
def test_balanced_mixing(module, m):
    """Everyone makes the same number of contacts, and contacts between any two regions balance."""
    N = (4e9, 3.4e9, 8e8)
    a, q = module.balanced_mixing(m, *N)
    assert sum(q) == pytest.approx(1)
    for i in range(3):
        assert a[i] + sum(m * q[j] for j in range(3) if j != i) == pytest.approx(1)  # at home + elsewhere
        for j in range(3):
            assert N[i] * m * q[j] == pytest.approx(N[j] * m * q[i])  # contacts i -> j = contacts j -> i


@pytest.mark.parametrize('module, extra', [
    (simulator, {}),
    (simulator_leaky, dict(LEAKY, leaky_mode='susceptibility')),
    (simulator_leaky, dict(LEAKY, leaky_mode='infectiousness')),
], ids=['all-or-none', 'leaky susceptibility', 'leaky infectiousness'])
def test_coexistence_equilibrium_is_stationary(module, extra):
    """Without seasonality or lockdown the model starts at an equilibrium and stays there."""
    res = module.model(plot=False, shortcuts=False, season_amp=0.0, **FIG3,
                       **model_kwargs(BEST_FIT, c_pandemic=0.0), **extra)
    assert res['simple_coexist'] and not res['extinct_V'] and not res['extinct_Y']
    y = res['y_all']
    state = y[:, 0] != 0  # S and I compartments (the cumulative infections start at 0)
    drift = np.abs(y[state] - y[state, :1]) / y[state, :1]
    assert drift.max() < 1e-9


def test_best_fit_reproduces_the_paper(best_fit):
    """The best fit's outputs in Table S1 and Fig. 3."""
    assert set(best_fit) == set(simulator.RESULT_KEYS)
    assert best_fit['simple_coexist'] is True and best_fit['extinct_Y'] is True and best_fit['extinct_V'] is False
    assert best_fit['min_prevalence_V'] == pytest.approx(2.84137422983934, rel=1e-6)
    assert best_fit['recovery_V'] == pytest.approx(1.8663586584531144, rel=1e-6)
    extinction = datetime.date(2015, 10, 1) + datetime.timedelta(days=best_fit['t_yamagata_extinction'] - 15.75 * YEAR)
    assert extinction == datetime.date(2020, 10, 10)
    assert isinstance(best_fit['R_eff_Y_peak_2026_27'], float)
    assert best_fit['R_eff_Y_peak_2026_27'] == pytest.approx(1.3167624204855264, rel=1e-6)
    assert best_fit['R_values'] == pytest.approx(
        [1.1753416145285938, 1.316726533370986, 1.1510201114135106, 1.0360369347793443], rel=1e-6)


@pytest.mark.parametrize('theta, outcome', EARLY_STOPS.values(), ids=EARLY_STOPS.keys())
def test_early_stop(theta, outcome):
    """A run stopped early has every output: its outcome, the region populations and NaN for the rest."""
    res = simulator.model(plot=False, **model_kwargs(dict(zip(NAMES, theta))))
    assert set(res) == set(simulator.RESULT_KEYS)
    assert res['region_pops'] == [4_000_000_000, 3_400_000_000, 800_000_000]
    for key in simulator.RESULT_KEYS:
        if key in outcome:
            assert res[key] == outcome[key], key
        elif key != 'region_pops':
            assert np.isnan(res[key]), key


@pytest.mark.filterwarnings('ignore:The iteration is not making good progress')  # leaky: no equilibrium
@pytest.mark.parametrize('module', [simulator, simulator_leaky], ids=['all-or-none', 'leaky'])
def test_search_row(module):
    """run_single_simulation returns the parameters, the search's outputs and success=True."""
    theta, outcome = EARLY_STOPS['no coexistence before lockdown (sample 0)']
    row = module.run_single_simulation(theta)
    assert list(row) == list(NAMES) + list(module.SEARCH_OUTPUTS) + ['success']
    assert row['success'] is True and row['simple_coexist'] is False
    assert [row[k] for k in NAMES] == list(theta)


@pytest.mark.parametrize('module', [simulator, simulator_leaky], ids=['all-or-none', 'leaky'])
def test_search_records_a_failed_run(module, monkeypatch):
    """A run whose model raises is recorded (success=False, with the error); the search goes on."""
    def fails(**kwargs):
        raise RuntimeError('solver failed')
    monkeypatch.setattr(module, 'model', fails)
    theta = tuple(BEST_FIT.values())
    row = module.run_single_simulation(theta)
    assert row['success'] is False and 'solver failed' in row['error']
    assert [row[k] for k in NAMES] == list(theta)


@pytest.fixture(scope='module')
def without_cross_immunity():
    """The all-or-none model with no cross-immunity and no lockdown (so that neither lineage dies out)."""
    return simulator.model(plot=False, shortcuts=False, **FIG3,
                           **model_kwargs(BEST_FIT, sigmaVY=0.0, sigmaYV=0.0, c_pandemic=0.0))


@pytest.mark.parametrize('leaky_mode', ['susceptibility', 'infectiousness'])
def test_leaky_model_without_cross_immunity_is_the_all_or_none_model(without_cross_immunity, leaky_mode):
    base = without_cross_immunity
    res = simulator_leaky.model(plot=False, shortcuts=False, leaky_mode=leaky_mode, **LEAKY, **FIG3,
                                **model_kwargs(BEST_FIT, sigmaVY=0.0, sigmaYV=0.0, c_pandemic=0.0))
    for lineage in 'VY':
        a, b = base[f'weekly_infections_{lineage}'], res[f'weekly_infections_{lineage}']
        assert a.shape == b.shape
        assert np.max(np.abs(a - b)) / np.max(a) < 1e-4  # relative to the peak (they agree to about 5e-6)


def test_stochastic_ensembles(best_fit):
    """Small ensembles for the best fit: what model() returns, and that it agrees with what it prints. The
    exact counts depend on numpy's random streams (numpy as in requirements.txt)."""
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        res = simulator.model(plot=False, stochastic=True, nsims=300, stochastic2=True, nsims2=300,
                              **model_kwargs(BEST_FIT))
    text = out.getvalue()
    assert res['stoch_extinction_pct_V'] == 95.0
    assert f"Victoria went extinct in {res['stoch_extinction_pct_V']:.2f}%" in text
    reintro = res['reintroductions']
    assert [x['date'] for x in reintro] == ['Oct 1st 2026', 'Jan 1st 2027', 'Apr 1st 2027', 'Jul 1st 2027']
    assert [x['fadeouts'] for x in reintro] == [252, 228, 269, 279]
    assert all(x['nsims'] == 300 for x in reintro)
    assert [x['R_e'] for x in reintro] == pytest.approx(best_fit['R_values'], rel=1e-12)
    for x in reintro:
        assert f"Stochastic Fade-outs (< 100 cases): {x['fadeouts']} / 300" in text
