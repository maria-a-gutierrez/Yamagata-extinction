# simulator.py: the two-lineage (Victoria, Yamagata), three-region influenza B model.
#
# Regions: Northern-Hemisphere temperate (N1 = 4 billion people), Tropics (N2 = 3.4 billion) and
# Southern-Hemisphere temperate (N3 = 800 million). For each lineage, SIRS dynamics with waning immunity
# (omegaV = 1/T_V, omegaY = 1/T_Y) and all-or-none cross-immunity (sigmaVY, sigmaYV); seasonal forcing in
# the temperate regions (peak on day season_phase in the NH, opposite phase in the SH); mixing between
# regions (m); and a COVID-19 reduction in contacts from 1 April 2020 (initial size c_pandemic, relaxing
# over Lockdown_duration days).
#
# Force of infection (balanced mixing): in region i,
#     FOI_i = beta * s_i * c * [(1 - m (N - N_i) / N) I_i / N_i + m * sum_{j != i} I_j / N],
# i.e. a region-i person makes a fraction m N_j / N of their contacts with region j and everyone makes the
# same total number of contacts, so contacts between any two regions balance. Seasonality s_i and the
# contact multiplier c act on the recipient's region. With equal prevalence everywhere the bracket reduces
# to I / N, so the single-region coexistence equilibrium is an exact equilibrium of the unforced model.
#
# model() integrates the deterministic ODEs (scipy's solve_ivp) from 1 January 2000 (t = 0, at the
# coexistence equilibrium) and by default runs to July 2027 with the lockdown on 1 April 2020
# (tspan = 27.75 years, Tpan = 20.25/27.75: the settings of Fig. 3). A lineage is extinct when it has
# fewer than `threshold` (1) infected in every region. Optionally it runs tau-leaping ensembles (0.1-day
# leaps, starting from rounded counts): Victoria from Yamagata's extinction (stochastic=True, nsims runs)
# and single Yamagata reintroductions into the NH on 1 October 2026, 1 January, 1 April and 1 July 2027
# (stochastic2=True, nsims2 runs; nsims2=0 gives only Yamagata's R_e on those dates). With plot=True it
# draws the Fig. 3 layout and saves the data it shows next to the figure (<figure>_plotdata.pkl), so that
# redraw_fig3() can restyle it without rerunning anything. It returns a dict (RESULT_KEYS), including the
# ensembles' results: stoch_extinction_pct_V (% of the Victoria runs extinct before its recovery) and
# reintroductions (per reintroduction date: R_e and fadeouts, the runs with fewer than 100 infections).
#
# run_single_simulation() is the wrapper that the Latin hypercube search calls; a run that fails is
# recorded with success=False.

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pickle
import datetime
import pandas as pd

def waning_to_susceptibles(S, I, omega):
    return omega * (1.0 - (S + I))

def coexistence_equilibrium(R0V, R0Y, gamma, omegaV, omegaY, sigmaVY, sigmaYV, mu):
    S_hat_Y = 1.0 / R0Y
    S_hat_V = 1.0 / R0V
    
    beta_Y = (gamma + mu) / S_hat_Y
    beta_V = (gamma + mu) / S_hat_V
    
    A_Y = (mu + omegaY) * (1.0 - S_hat_Y)
    A_V = (mu + omegaV) * (1.0 - S_hat_V)
    
    B_Y = beta_Y * S_hat_Y + omegaY
    B_V = beta_V * S_hat_V + omegaV
    
    C_Y = sigmaVY * beta_V * S_hat_Y
    C_V = sigmaYV * beta_Y * S_hat_V
    
    num_Y = A_Y * B_V - A_V * C_Y
    num_V = A_V * B_Y - A_Y * C_V
    denom = B_Y * B_V - C_Y * C_V
    
    if num_V > 0 and num_Y > 0:
        I_Y = num_Y / denom
        I_V = num_V / denom
        return S_hat_V, S_hat_Y, I_V, I_Y, True
    else:
        return 1-1e-6, 1-1e-6, 1e-6, 1e-6, False
    
# Contact and seasonal helper functions matching the original logic
def contact_multiplier(t, t_lock, Lockdown_duration, c_pandemic):
    if t <= t_lock: 
        return 1.0
    return 1.0 - c_pandemic * np.exp(-(t - t_lock) / Lockdown_duration)
        
def seasonal_multiplier_local(amp, phase, period, t, t_lock, Lockdown_duration):
    base= amp * np.cos(2 * np.pi * (t - phase) / period)
    if t <= t_lock:
        return 1.0 + base
    return 1.0 + base * (1 - np.exp(-(t - t_lock) / Lockdown_duration))

def seasonal_multiplier_triple(multiplier_nh):
    return multiplier_nh, 1.0, 2.0 - multiplier_nh

def balanced_mixing(m, N1, N2, N3):
    """Balanced mixing: a region-i person makes a fraction m*N_j/N of their contacts
    with region j and the rest, 1 - m*(N - N_i)/N, at home. Returns the within-region
    coefficients (a1, a2, a3) and the population shares (q1, q2, q3) = N_i/N."""
    N = N1 + N2 + N3
    q = (N1/N, N2/N, N3/N)
    return tuple(1 - m*(1 - qi) for qi in q), q


# Every output of model()
RESULT_KEYS = ('simple_coexist', 'recovery_V', 'recovery_Y', 'min_prevalence_V', 'R_eff_Y_peak_2026_27',
               'R_values', 'extinct_V', 'extinct_Y', 'weekly_infections_V', 'weekly_infections_Y',
               'weekly_times', 't_eval', 'y_all', 'region_pops', 't_yamagata_extinction',
               't_min_prevalence_V', 't_R_eff_Y_peak_2026_27', 'stoch_extinction_pct_V', 'reintroductions')


def stopped_early(region_pops, **outcome):
    """model()'s result when shortcuts=True stops a run early: the outcome given, the region populations
    and NaN for every other output. When Victoria dies out, extinct_V is NaN and extinct_Y True whatever
    Yamagata did: recovery_V (-1 before the lockdown, -10 after) is what marks Victoria's extinction, and the
    selection rules rely on these values."""
    return {**dict.fromkeys(RESULT_KEYS, np.nan), 'region_pops': region_pops, **outcome}


def run_stochastic_ensemble(t_start, t_end, y_start_state, N1, N2, N3, betaV, gamma, mu, omegaV, m, 
                            season_amp, season_phase, season_period, t_lock, Lockdown_duration, c_pandemic,
                            n_sims=100, seed=1, tau=0.1, t_check=None):
    """
    Simulates the remaining system using Tau-Leaping for Victoria dynamics
    after Yamagata's absolute extinction.
    The runs start from the rounded counts at extinction and take leaps of tau days (with 1-day
    leaps the ensemble mean drifts below the ODE). All runs advance together; the state is stored
    once a week from the first whole day, and the total infecteds of every run are also taken at
    the first leap at or after t_check (at the end if t_check is None or past it).
    Events that move people between the same two compartments are drawn as one Poisson count
    (recoveries and deaths of infecteds; waning and births), which has the same distribution as
    drawing them separately. A run whose infecteds reach zero stays at zero (Victoria has no
    source outside the model), so it is no longer advanced: its infecteds and cumulative
    infections are final, and its stored susceptibles stay at their value when it died out.
    Returns (weekly times, cumulative infections cumIV1-3 (n_sims, n_weeks, 3), infecteds at t_check
    (n_sims,)); only these are used, so only these are stored.
    """
    rng = np.random.default_rng(seed)
    # y_start_state indices correspond to Victoria's layout:
    # 0:SV1, 1:IV1, 2:cumIV1, 3:SV2, 4:IV2, 5:cumIV2, 6:SV3, 7:IV3, 8:cumIV3
    t_days = np.arange(np.ceil(t_start), t_end, 1.0)
    t_eval = t_days[::7]
    leaps_per_day = int(round(1.0 / tau))

    # Storage for all simulations: cumulative infections, shape (n_sims, n_weeks, 3); single precision
    # keeps 10^5 runs under 0.4 GB (counts lose under 1 part in 10^7)
    all_cum = np.zeros((n_sims, len(t_eval), 3), dtype=np.float32)

    (a1, a2, a3), (q1, q2, q3) = balanced_mixing(m, N1, N2, N3)
    N = np.array([N1, N2, N3])
    a, q = np.array([a1, a2, a3]), np.array([q1, q2, q3])
    births = mu * N

    # Counts by region (columns) for every run (rows), rounded as stated in Methods
    S = np.tile(np.rint([y_start_state[0]*N1, y_start_state[6]*N2, y_start_state[12]*N3]), (n_sims, 1))
    I = np.tile(np.rint([y_start_state[1]*N1, y_start_state[7]*N2, y_start_state[13]*N3]), (n_sims, 1))
    cum = np.zeros((n_sims, 3))
    alive = np.arange(n_sims)  # runs that still have infecteds
    I_at_check = None

    for day_idx, t_day in enumerate(t_days):
        if day_idx % 7 == 0:
            all_cum[:, day_idx // 7, :] = cum
        alive = alive[I[alive].sum(axis=1) > 0]
        Sa, Ia, cuma = S[alive], I[alive], cum[alive]
        for leap in range(leaps_per_day):
            t = t_day + leap * tau
            if I_at_check is None and t_check is not None and t >= t_check:
                I[alive] = Ia
                I_at_check = I.sum(axis=1)

            # Compute current multiplier arrays
            cm = contact_multiplier(t, t_lock, Lockdown_duration, c_pandemic)
            seas = np.array(seasonal_multiplier_triple(seasonal_multiplier_local(season_amp, season_phase, season_period, t, t_lock, Lockdown_duration)))

            # Global fractional forces of infection: a_i I_i/N_i + m * sum_{j != i} q_j I_j/N_j
            prev = Ia / N
            shared = prev * q
            forceV = betaV * seas * cm * (a * prev + m * (shared.sum(axis=1, keepdims=True) - shared))

            # Sample discrete number of events occurring over window tau: infections, recoveries and
            # deaths of infecteds, waning and births, deaths of susceptibles
            # Clips to 0 to prevent issues with floating-point underflows
            e_inf, e_out, e_in, e_ms = rng.poisson(np.stack([
                np.maximum(forceV * Sa, 0), (gamma + mu) * Ia, np.maximum(omegaV * (N - (Sa + Ia)), 0) + births, mu * Sa
            ]) * tau)

            # Apply state changes securely, bounding populations at 0
            Sa = np.maximum(0.0, Sa - e_inf + e_in - e_ms)
            Ia = np.maximum(0.0, Ia + e_inf - e_out)
            cuma = cuma + e_inf # tracking cumulative infections
        S[alive], I[alive], cum[alive] = Sa, Ia, cuma

    if I_at_check is None:
        I_at_check = I.sum(axis=1)
    return t_eval, all_cum, I_at_check


def model(
        R0V=1.2, R0Y=1.2, gamma=1/3.4,
        omegaV=0.0, omegaY=0.0,
        sigmaVY=0.0, sigmaYV=0.0,
        tspan=27.75*365.25, Tpan=20.25/27.75, Lockdown_duration=500,
        c_pandemic=0.5,
        N1: int = 4_000_000_000,
        N2: int = 3_400_000_000,
        N3: int = 800_000_000, mortality_rate=1/(75*365.25),
        m = 0.0060,                        
        threshold: float = 1.0, 
        season_amp: float = 0.1, season_phase: float = 90, season_period: float = 365.25,
        plot: bool = True, transient_years=15.75, shortcuts: bool = True, name=None, title=None, style=None, stochastic: bool = False, nsims=100, stochastic2: bool = False, nsims2=100,
        **solver_kwargs):

    S_V, S_Y, I_V, I_Y, simple_coexist = coexistence_equilibrium(R0V, R0Y, gamma, omegaV, omegaY, sigmaVY, sigmaYV, mortality_rate)

    if shortcuts and not simple_coexist:
        return stopped_early([N1, N2, N3], simple_coexist=simple_coexist)
    else: 
        betaV = R0V * (gamma+mortality_rate)
        betaY = R0Y * (gamma+mortality_rate)
        t_lock = Tpan * tspan
        (a1, a2, a3), (q1, q2, q3) = balanced_mixing(m, N1, N2, N3)
        

        def force_strain(beta, seasonal, I_in, I_out_1, I_out_2):
            return beta * seasonal * (I_in + m * (I_out_1 + I_out_2))

        def force_region(t, seasonal, IV, IV_out_1, IV_out_2, IY, IY_out_1, IY_out_2):
            cm = contact_multiplier(t, t_lock, Lockdown_duration, c_pandemic)
            transmissibility_multiplier = seasonal * cm
            forceV = force_strain(betaV, transmissibility_multiplier, IV, IV_out_1, IV_out_2)
            forceY = force_strain(betaY, transmissibility_multiplier, IY, IY_out_1, IY_out_2)
            return forceV, forceY

        def derivatives_region(forceV, forceY, SV, IV, SY, IY):
            dSV = - forceV * SV - sigmaYV * forceY * SV + waning_to_susceptibles(SV, IV, omegaV) + mortality_rate - mortality_rate * SV
            dIV = forceV * SV - (gamma + mortality_rate) * IV
            dSY = - forceY * SY - sigmaVY * forceV * SY + waning_to_susceptibles(SY, IY, omegaY) + mortality_rate - mortality_rate * SY
            dIY = forceY * SY - (gamma + mortality_rate) * IY
            return dSV, dIV, dSY, dIY
        
        def new_infections_rate(forceV, forceY, SV, SY):
            return forceV * SV, forceY * SY

        def F(t, x):
            (SV1, IV1, SY1, IY1, cumIV1, cumIY1,
             SV2, IV2, SY2, IY2, cumIV2, cumIY2,
             SV3, IV3, SY3, IY3, cumIV3, cumIY3) = x

            seas1, seas2, seas3 = seasonal_multiplier_triple(seasonal_multiplier_local(season_amp, season_phase, season_period, t, t_lock, Lockdown_duration))

            # balanced mixing: force_strain gives beta*s*(a_i*i_i + m*sum_j q_j*i_j)
            forceV1, forceY1 = force_region(t, seas1, a1*IV1, IV2*q2, IV3*q3, a1*IY1, IY2*q2, IY3*q3)
            forceV2, forceY2 = force_region(t, seas2, a2*IV2, IV1*q1, IV3*q3, a2*IY2, IY1*q1, IY3*q3)
            forceV3, forceY3 = force_region(t, seas3, a3*IV3, IV1*q1, IV2*q2, a3*IY3, IY1*q1, IY2*q2)

            dSV1, dIV1, dSY1, dIY1 = derivatives_region(forceV1, forceY1, SV1, IV1, SY1, IY1)
            dSV2, dIV2, dSY2, dIY2 = derivatives_region(forceV2, forceY2, SV2, IV2, SY2, IY2)
            dSV3, dIV3, dSY3, dIY3 = derivatives_region(forceV3, forceY3, SV3, IV3, SY3, IY3)
            
            new_infV1, new_infY1 = new_infections_rate(forceV1, forceY1, SV1, SY1)
            new_infV2, new_infY2 = new_infections_rate(forceV2, forceY2, SV2, SY2)
            new_infV3, new_infY3 = new_infections_rate(forceV3, forceY3, SV3, SY3)
            
            return [dSV1, dIV1, dSY1, dIY1, new_infV1, new_infY1,
                    dSV2, dIV2, dSY2, dIY2, new_infV2, new_infY2,
                    dSV3, dIV3, dSY3, dIY3, new_infV3, new_infY3]
            
        y0 = [S_V, I_V, S_Y, I_Y, 0.0, 0.0,
              S_V, I_V, S_Y, I_Y, 0.0, 0.0,
              S_V, I_V, S_Y, I_Y, 0.0, 0.0]   
        
        V_state_idx = [1, 7, 13]   
        Y_state_idx = [3, 9, 15]   
        
        extinct_V = False
        extinct_Y = False
        t_extinct_V = np.nan
        t_extinct_Y = np.nan

        t_seg_start = 0.0
        y_current = np.array(y0, dtype=float)
        seg_ts = []
        seg_ys = []

        # Variables for managing parallel stochastic workflows
        yamagata_extinction_state_captured = None
        t_yamagata_extinction = np.nan

        t_extinction_tracking_start = t_lock - (2 * 365.25) # check two years before lockdown

        while t_seg_start < tspan:
            active_events = []
            event_labels = [] 

            if not extinct_V:
                def make_global_v_ev():
                    def ev(t, x):
                        if t < t_extinction_tracking_start: return 1.0
                        return max(x[1] * N1, x[7] * N2, x[13] * N3) - threshold
                    ev.terminal = True
                    ev.direction = -1
                    return ev
                active_events.append(make_global_v_ev())
                event_labels.append('V')
            else:
                if shortcuts:
                    return stopped_early([N1, N2, N3], simple_coexist=simple_coexist,
                                         recovery_V=-1 if t_seg_start < t_lock else -10, extinct_Y=True)

            if not extinct_Y:
                def make_global_y_ev():
                    def ev(t, x):
                        if t < t_extinction_tracking_start: return 1.0
                        return max(x[3] * N1, x[9] * N2, x[15] * N3) - threshold
                    ev.terminal = True
                    ev.direction = -1
                    return ev  
                active_events.append(make_global_y_ev())
                event_labels.append('Y')
            else:
                if shortcuts and t_seg_start < t_lock:
                    return stopped_early([N1, N2, N3], simple_coexist=simple_coexist, recovery_Y=-1, extinct_Y=True)
            
            t_start_i = int(np.ceil(t_seg_start)) if t_seg_start > 0 else 0
            t_eval_seg = np.arange(t_start_i, tspan, 1.0)
            
            if len(t_eval_seg) == 0: break
                
            sol_seg = solve_ivp(F, [t_seg_start, tspan], y_current, t_eval=t_eval_seg,
                                events=active_events if active_events else None, max_step=1.0, **solver_kwargs)

            if not sol_seg.success and sol_seg.status != 1: raise RuntimeError(sol_seg.message)

            if len(sol_seg.t) > 0:
                seg_ts.append(sol_seg.t)
                seg_ys.append(sol_seg.y)

            if sol_seg.status != 1: break  

            fired = [(sol_seg.t_events[i][0], i) for i in range(len(active_events)) if len(sol_seg.t_events[i]) > 0]
            if not fired: break

            fired.sort()
            t_event, event_idx = fired[0]
            strain = event_labels[event_idx]
            t_seg_start = t_event
            y_current = sol_seg.y_events[event_idx][0].copy()

            if strain == 'V':
                for idx in V_state_idx: y_current[idx] = 0.0
                extinct_V = True
                t_extinct_V = t_event
            else:
                # Capture system state right before wiping Yamagata fraction to feed into stochastic system
                if yamagata_extinction_state_captured is None:
                    yamagata_extinction_state_captured = y_current.copy()
                    t_yamagata_extinction = t_event
                    t_extinct_Y = t_event # Lock in first extinction time for plots
                
                for idx in Y_state_idx: y_current[idx] = 0.0
                extinct_Y = True

            if extinct_V and extinct_Y: 
                break

        if shortcuts and not extinct_Y:
            return stopped_early([N1, N2, N3], simple_coexist=simple_coexist, extinct_Y=False)

        # Concatenate deterministic segments cleanly
        if len(seg_ts) > 1:
            t_raw = np.concatenate(seg_ts)
            y_raw = np.concatenate(seg_ys, axis=1)
            _, uniq = np.unique(t_raw, return_index=True)
            t_eval = t_raw[uniq]
            y_all  = y_raw[:, uniq]
        else:
            t_eval = seg_ts[0]
            y_all  = seg_ys[0]

        IV1, IY1 = y_all[1], y_all[3]
        IV2, IY2 = y_all[7], y_all[9]
        IV3, IY3 = y_all[13], y_all[15]
        cumIV1, cumIY1 = y_all[4], y_all[5]
        cumIV2, cumIY2 = y_all[10], y_all[11]
        cumIV3, cumIY3 = y_all[16], y_all[17]
        
        # Calculate weekly infections via backward differences
        week_indices = np.arange(0, len(t_eval), 7)
        weekly_infections_V = np.zeros((3, len(week_indices)))
        weekly_infections_Y = np.zeros((3, len(week_indices)))
        weekly_times = np.zeros(len(week_indices))
        
        for i, idx in enumerate(week_indices):
            if i == 0:
                weekly_infections_V[0, i], weekly_infections_Y[0, i] = cumIV1[idx] * N1, cumIY1[idx] * N1
                weekly_infections_V[1, i], weekly_infections_Y[1, i] = cumIV2[idx] * N2, cumIY2[idx] * N2
                weekly_infections_V[2, i], weekly_infections_Y[2, i] = cumIV3[idx] * N3, cumIY3[idx] * N3
            else:
                prev_idx = week_indices[i - 1]
                weekly_infections_V[0, i] = (cumIV1[idx] - cumIV1[prev_idx]) * N1
                weekly_infections_V[1, i] = (cumIV2[idx] - cumIV2[prev_idx]) * N2
                weekly_infections_V[2, i] = (cumIV3[idx] - cumIV3[prev_idx]) * N3
                weekly_infections_Y[0, i] = (cumIY1[idx] - cumIY1[prev_idx]) * N1
                weekly_infections_Y[1, i] = (cumIY2[idx] - cumIY2[prev_idx]) * N2
                weekly_infections_Y[2, i] = (cumIY3[idx] - cumIY3[prev_idx]) * N3
            weekly_times[i] = t_eval[idx]

        # --- PRESERVED BASELINE METRIC CALCULATIONS ---
        recovery_V = np.nan
        recovery_Y = np.nan
        min_prevalence_V = np.nan
        t_min_prevalence_V = np.nan  # when it happens (days)

        if Tpan < 1.0:
            years_for_peak = 5
            pre_weeks_mask = (weekly_times >= (t_lock - years_for_peak * 365.25)) & (weekly_times < t_lock)
            post_weeks_mask = weekly_times >= t_lock
            
            if np.any(pre_weeks_mask):
                pre_times = weekly_times[pre_weeks_mask]
                masks = [(pre_times >= t_lock - (year + 1) * 365.25) & (pre_times < t_lock - year * 365.25) for year in range(years_for_peak)]
                recovery_threshold = 0.10 
                t_post_weeks = weekly_times[post_weeks_mask]
                
                if extinct_V:
                    recovery_V = -1 if t_extinct_V < t_lock else -10
                else:
                    weekly_V_r1 = weekly_infections_V[0]
                    pre_cases_V = weekly_V_r1[pre_weeks_mask]
                    peaks_V = [np.max(pre_cases_V[m]) for m in masks if np.any(m)]
                    baseline_peak_V = np.mean(peaks_V) * recovery_threshold if peaks_V else 0
                    cases_post_V = weekly_V_r1[post_weeks_mask]
                    trough_idx_V = np.argmin(cases_post_V)
                    rebound_mask_V = np.arange(len(cases_post_V)) > trough_idx_V
                    idx_V = np.where(cases_post_V[rebound_mask_V] >= baseline_peak_V)[0]
                    if len(idx_V) > 0:
                        recovery_V = (t_post_weeks[trough_idx_V + 1 + idx_V[0]] - t_lock) / 365.25

                if extinct_Y:
                    if t_extinct_Y < t_lock: recovery_Y = -1  
                    elif (t_extinct_Y > t_lock + (recovery_V if not np.isnan(recovery_V) else 0) * 365.25) and not extinct_V: recovery_Y = -100 
                    else: recovery_Y = -10 
                else:
                    weekly_Y_r1 = weekly_infections_Y[0]
                    pre_cases_Y = weekly_Y_r1[pre_weeks_mask]
                    peaks_Y = [np.max(pre_cases_Y[m]) for m in masks if np.any(m)]
                    baseline_peak_Y = np.mean(peaks_Y) * recovery_threshold if peaks_Y else 0
                    cases_post_Y = weekly_Y_r1[post_weeks_mask]
                    trough_idx_Y = np.argmin(cases_post_Y)
                    rebound_mask_Y = np.arange(len(cases_post_Y)) > trough_idx_Y
                    idx_Y = np.where(cases_post_Y[rebound_mask_Y] >= baseline_peak_Y)[0]
                    if len(idx_Y) > 0:
                        recovery_Y = (t_post_weeks[trough_idx_Y + 1 + idx_Y[0]] - t_lock) / 365.25

            abs_IV_global = IV1*N1 + IV2*N2 + IV3*N3
            # Ensure the search window for the minimum prevalence is strictly BEFORE its recovery or extinction
            trough_end_V = t_extinct_V if (extinct_V and t_extinct_V >= t_lock) else \
                           (t_lock + recovery_V * 365.25 if (not np.isnan(recovery_V) and recovery_V > 0) else tspan)
            trough_mask_V = (t_eval >= t_lock) & (t_eval < trough_end_V)
            
            if np.any(trough_mask_V):
                min_prevalence_V = float(np.min(abs_IV_global[trough_mask_V]))
                t_min_prevalence_V = float(t_eval[trough_mask_V][np.argmin(abs_IV_global[trough_mask_V])])

        # Trigger stochastic simulation if Yamagata extinction occurred
        stoch_times = None
        stoch_data = None
        stoch_extinction_pct_V = np.nan  # % of the Victoria runs extinct before its recovery (if run)
        if stochastic and extinct_Y and yamagata_extinction_state_captured is not None:
            # Calculate stochastic end time (Sep 30, 2026)
            ref_date_stoch = datetime.datetime(2015, 10, 1, tzinfo=datetime.timezone.utc)
            stoch_end_date = datetime.datetime(2026, 9, 30, tzinfo=datetime.timezone.utc)
            t_stoch_end = 15.75 * 365.25 + (stoch_end_date - ref_date_stoch).days
            actual_stoch_end = min(tspan, t_stoch_end)
            
            if t_yamagata_extinction < actual_stoch_end:
                # --- NEW ADDITION: CALCULATE VICTORIA STOCHASTIC EXTINCTION PERCENTAGE ---
                t_check_extinct = actual_stoch_end
                if not np.isnan(recovery_V) and recovery_V > 0:
                    t_check_extinct = min(actual_stoch_end, t_lock + recovery_V * 365.25)

                # final_active_V: active infecteds IV1 + IV2 + IV3 of every run at t_check_extinct
                stoch_times, stoch_cum, final_active_V = run_stochastic_ensemble(
                    t_start=t_yamagata_extinction, t_end=actual_stoch_end,
                    y_start_state=yamagata_extinction_state_captured,
                    N1=N1, N2=N2, N3=N3, betaV=betaV, gamma=gamma, mu=mortality_rate, omegaV=omegaV, m=m,
                    season_amp=season_amp, season_phase=season_phase, season_period=season_period,
                    t_lock=t_lock, Lockdown_duration=Lockdown_duration, c_pandemic=c_pandemic, n_sims=nsims, seed=1,
                    t_check=t_check_extinct)

                # Count the number of simulations where the final count of active infections is 0
                extinct_runs_count = np.sum(final_active_V == 0)
                stoch_extinction_pct_V = float(extinct_runs_count / stoch_cum.shape[0]) * 100
                print(f">> Victoria went extinct in {stoch_extinction_pct_V:.2f}% of the stochastic runs before its recovery.")

                # Weekly incidences from the weekly cumulative counts (the first week holds the count
                # at the first record, as before)
                # Shape mapping: [simulations, regions, weeks]
                stoch_weekly_inc = np.diff(stoch_cum, axis=1, prepend=0.0).transpose(0, 2, 1)
                del stoch_cum
                stoch_data = {'times': stoch_times, 'incidence': stoch_weekly_inc}
        
        R_eff_Y_peak_2026_27 = np.nan
        t_R_eff_Y_peak_2026_27 = np.nan  # when that peak is (days)
        t_ref_days = 26.75* 365.25 #arbitrary
        ref_date = datetime.datetime(2026, 10, 1, tzinfo=datetime.timezone.utc)
        _season_start = t_ref_days + (datetime.datetime(2026, 10, 1, tzinfo=datetime.timezone.utc) - ref_date).days
        _season_end   = t_ref_days + (datetime.datetime(2027, 3, 31, tzinfo=datetime.timezone.utc) - ref_date).days
        _season_mask  = (t_eval >= _season_start) & (t_eval <= _season_end)


        if np.any(_season_mask):
            _t_season = t_eval[_season_mask]
            _SY1_season = y_all[2][_season_mask]
            _seas_vals = np.array([seasonal_multiplier_local(season_amp, season_phase, season_period, _t, t_lock, Lockdown_duration) for _t in _t_season])
            _peak_idx = np.argmax(_seas_vals)
            _Reff_r1 = R0Y * a1 * _SY1_season[_peak_idx] * _seas_vals[_peak_idx] * contact_multiplier(_t_season[_peak_idx], t_lock, Lockdown_duration, c_pandemic)
            R_eff_Y_peak_2026_27 = float(_Reff_r1)
            t_R_eff_Y_peak_2026_27 = float(_t_season[_peak_idx])

        # --- NEW ADDITION: STOCHASTIC YAMAGATA TRACKING (MULTI-DATE ENSEMBLES) ---
        # Kept as None to cleanly remove stochastic Yamagata envelopes from the downstream plot system
        stoch_Y_data = None 
        R_values=[]
        reintroductions = []  # one entry per reintroduction ensemble run (nsims2 > 0)
        if stochastic2 and extinct_Y and not np.isnan(t_yamagata_extinction):
            
            hypo_dates_str = ["Oct 1st 2026", "Jan 1st 2027", "Apr 1st 2027", "Jul 1st 2027"]
            target_dates = [
                datetime.datetime(2026, 10, 1,  tzinfo=datetime.timezone.utc),
                datetime.datetime(2027, 1, 1, tzinfo=datetime.timezone.utc),
                datetime.datetime(2027, 4, 1, tzinfo=datetime.timezone.utc),
                datetime.datetime(2027, 7, 1, tzinfo=datetime.timezone.utc)
            ]

            # Safely map the calendar dates to simulation days using the model's datetime framework
            hypo_times = [(d - ref_date).days + t_ref_days for d in target_dates]
            tau = 0.1  # leap length in days (1-day leaps drift below the ODE, as in run_stochastic_ensemble)
            
            print(f"\n=============================================")
            print(f"--- YAMAGATA MULTI-DATE REINTRODUCTION STATS ---")
            print(f"=============================================")
            
            for d_str, t_reintro_hypo in zip(hypo_dates_str, hypo_times):
                if t_reintro_hypo >= tspan:
                    print(f"Date {d_str} (t={t_reintro_hypo:.1f}) skipped: exceeds simulation timeframe.")
                    continue
                
                # 1. CALCULATE EFFECTIVE REPRODUCTION NUMBER (Re) FOR SEEDED REGION (NH / Region 1)
                # Derived from the localized force of infection given a single index case: IY = [1, 0, 0]
                SY1_det_init = np.interp(t_reintro_hypo, t_eval, y_all[2]) * N1
                seas1_init = seasonal_multiplier_local(season_amp, season_phase, season_period, t_reintro_hypo, t_lock, Lockdown_duration)
                cm_init = contact_multiplier(t_reintro_hypo, t_lock, Lockdown_duration, c_pandemic)
                
                Re_Y = (betaY * a1 * seas1_init * cm_init * SY1_det_init) / (N1 * (gamma + mortality_rate))
                R_values.append(Re_Y)

                if nsims2 > 0:
                    # 2. INITIALIZE ARRAYS FOR THE CURRENT INDEPENDENT TIMEFRAME
                    t_eval_stoch_Y = np.arange(t_reintro_hypo, tspan, tau)
                    n_steps_Y = len(t_eval_stoch_Y)
                    
                    # Interpolate deterministic Victoria infection state over this specific residual timeframe
                    IV_det_frac = np.array([np.interp(t_eval_stoch_Y, t_eval, y_all[1]),
                                            np.interp(t_eval_stoch_Y, t_eval, y_all[7]),
                                            np.interp(t_eval_stoch_Y, t_eval, y_all[13])])
                    
                    # Dynamic initialization of susceptibles from deterministic system state at this date (rounded counts)
                    SY_init = np.rint([np.interp(t_reintro_hypo, t_eval, y_all[2]) * N1,
                                       np.interp(t_reintro_hypo, t_eval, y_all[8]) * N2,
                                       np.interp(t_reintro_hypo, t_eval, y_all[14]) * N3])
                    
                    cm_arr = np.array([contact_multiplier(t, t_lock, Lockdown_duration, c_pandemic) for t in t_eval_stoch_Y])
                    seas1_arr = np.array([seasonal_multiplier_local(season_amp, season_phase, season_period, t, t_lock, Lockdown_duration) for t in t_eval_stoch_Y])
                    seas2_arr = np.ones_like(seas1_arr)
                    seas3_arr = 2.0 - seas1_arr
                    
                    forceV_det = np.zeros((3, n_steps_Y))
                    for step_idx in range(n_steps_Y):
                        iv1, iv2, iv3 = IV_det_frac[0, step_idx], IV_det_frac[1, step_idx], IV_det_frac[2, step_idx]
                        forceV_det[0, step_idx] = betaV * seas1_arr[step_idx] * cm_arr[step_idx] * (a1*iv1 + m*(iv2*q2 + iv3*q3))
                        forceV_det[1, step_idx] = betaV * seas2_arr[step_idx] * cm_arr[step_idx] * (a2*iv2 + m*(iv1*q1 + iv3*q3))
                        forceV_det[2, step_idx] = betaV * seas3_arr[step_idx] * cm_arr[step_idx] * (a3*iv3 + m*(iv1*q1 + iv2*q2))
                    
                    rng = np.random.default_rng(1) # All 4 hypothetical dates can use the same NRG
                    N_vec = np.array([N1, N2, N3])
                    a_vec, q_vec = np.array([a1, a2, a3]), np.array([q1, q2, q3])
                    births_Y = mortality_rate * N_vec

                    # 3. RUN THE INDEPENDENT STOCHASTIC ENSEMBLE, all runs together (rows: runs, columns: regions).
                    # Events between the same two compartments are drawn as one Poisson count (cross-immunity
                    # and deaths of susceptibles; waning and births; recoveries and deaths of infecteds), which
                    # has the same distribution as drawing them separately. A run whose infecteds reach zero
                    # has its final outbreak size (Yamagata has no other source), so it is no longer advanced.
                    SY = np.tile(SY_init, (nsims2, 1))
                    IY = np.zeros((nsims2, 3))
                    IY[:, 0] = 1.0 # Reintroduce exactly 1 case in NH
                    cumY = np.zeros((nsims2, 3))
                    alive = np.arange(nsims2)  # runs still advanced (SY, IY, cumY hold only these)
                    cumY_done = np.zeros((nsims2, 3))  # final cumulative infections of every run

                    for step_idx in range(n_steps_Y):
                        if step_idx % 10 == 0:  # once a day, set aside the runs that have faded out
                            keep = IY.sum(axis=1) > 0
                            cumY_done[alive[~keep]] = cumY[~keep]
                            alive, SY, IY, cumY = alive[keep], SY[keep], IY[keep], cumY[keep]
                            if len(alive) == 0:
                                break
                        # Calculate Yamagata force of infection: a_i I_i/N_i + m * sum_{j != i} q_j I_j/N_j
                        prevY = IY / N_vec
                        sharedY = prevY * q_vec
                        seas_step = np.array([seas1_arr[step_idx], seas2_arr[step_idx], seas3_arr[step_idx]])
                        fY = betaY * seas_step * cm_arr[step_idx] * (a_vec * prevY + m * (sharedY.sum(axis=1, keepdims=True) - sharedY))

                        # Rates: infection; cross-immunity from (deterministic) Victoria and death of
                        # susceptibles; waning and birth; recovery and death of infecteds
                        e_inf, e_out_S, e_in_S, e_out_I = rng.poisson(np.stack([
                            np.maximum(fY * SY, 0), (sigmaVY * forceV_det[:, step_idx] + mortality_rate) * SY,
                            omegaY * np.maximum(0, N_vec - SY - IY) + births_Y, (gamma + mortality_rate) * IY
                        ]) * tau)

                        SY = np.maximum(0, SY - e_inf - e_out_S + e_in_S)
                        IY = np.maximum(0, IY + e_inf - e_out_I)
                        cumY = cumY + e_inf
                    cumY_done[alive] = cumY
                    cumY = cumY_done

                    # 4. EVALUATE OUTBREAK SIZE METRICS FOR THIS ENSEMBLE
                    outbreak_sizes_Y = cumY.sum(axis=1)
                    fadeout_count = np.sum(outbreak_sizes_Y < 100)
                    prob_fadeout = fadeout_count / nsims2
                    reintroductions.append({'date': d_str, 't': float(t_reintro_hypo), 'R_e': float(Re_Y),
                                            'fadeouts': int(fadeout_count), 'nsims': nsims2})
                    print(f"Hypothetical Introduction Date: {d_str}")
                    print(f"  Calculated Initial Re: {Re_Y:.2f} and poisson formula for extinction probability: {1/Re_Y:.2f}")
                    print(f"  Stochastic Fade-outs (< 100 cases): {fadeout_count} / {nsims2}")
                    print(f"  Probability of Fade-out: {prob_fadeout:.2%}")
                    print(f"---------------------------------------------")

       
        
        
        # --- EXECUTE EXTENDED PLOTTING SYSTEM ---
        # What the figure shows is gathered in fig3, saved next to the figure (fig3_data_path) so that
        # redraw_fig3 can redraw it without rerunning anything, and drawn by draw_fig3.
        if plot:
            t_post = transient_years * 365.25
            weekly_mask = weekly_times >= t_post

            weekly_mask_V = weekly_mask.copy()
            if extinct_V and t_extinct_V >= t_post:
                weekly_mask_V &= (weekly_times < t_extinct_V)
            elif extinct_V:
                weekly_mask_V = np.zeros(len(weekly_times), dtype=bool)

            weekly_mask_Y = weekly_mask.copy()
            if not np.isnan(t_yamagata_extinction):
                # Only draw the Yamagata deterministic line up to its very first lockdown extinction
                weekly_mask_Y &= (weekly_times <= t_yamagata_extinction)

            weekly_times_plot = weekly_times[weekly_mask]
            iso_dates = [f"{(ref_date + datetime.timedelta(days=t - t_ref_days)).isocalendar()[0]}-W{(ref_date + datetime.timedelta(days=t - t_ref_days)).isocalendar()[1]:02d}" for t in weekly_times_plot]

            # Dynamic x-coordinate mapping for vertical lines
            def t_days_to_x(t_days):
                if t_days < weekly_times_plot[0] or t_days > weekly_times_plot[-1]:
                    return None
                return np.argmin(np.abs(weekly_times_plot - t_days)).item()

            t_extinct_Y_x = None
            if not np.isnan(t_yamagata_extinction) and t_yamagata_extinction >= t_post:
                t_extinct_Y_x = t_days_to_x(t_yamagata_extinction)

            regions = []
            for r in range(3):
                v_weekly = weekly_infections_V[r, weekly_mask].copy() / 1e6
                y_weekly = weekly_infections_Y[r, weekly_mask].copy() / 1e6
                v_weekly[~weekly_mask_V[weekly_mask]] = np.nan
                y_weekly[~weekly_mask_Y[weekly_mask]] = np.nan
                region = {'v_weekly': v_weekly, 'y_weekly': y_weekly, 'stoch': None}

                # Summaries of the stochastic Victoria runs, week by week
                if stoch_data is not None:
                    stoch_times_w = stoch_data['times']
                    stoch_inc_w = stoch_data['incidence'][:, r, :]

                    # Track matching pairs explicitly to prevent data shifting bugs
                    stoch_x_indices = []
                    valid_data_indices = []
                    for data_idx, st_t in enumerate(stoch_times_w):
                        # Find the absolute closest deterministic week index (None if out of bounds)
                        closest_plot_idx = t_days_to_x(st_t)
                        if closest_plot_idx is not None:
                            # Verify it falls within the 3.5-day boundary (half a week, round to closest week)
                            if np.abs(weekly_times_plot[closest_plot_idx] - st_t) < 3.5:
                                stoch_x_indices.append(closest_plot_idx)
                                valid_data_indices.append(data_idx)
                    stoch_x_indices = np.array(stoch_x_indices)

                    if len(stoch_x_indices) > 0:
                        stoch_inc_trimmed = stoch_inc_w[:, valid_data_indices] / 1e6
                        region['stoch'] = {'x': stoch_x_indices,
                                           'mean': np.mean(stoch_inc_trimmed, axis=0),
                                           'median': np.median(stoch_inc_trimmed, axis=0),
                                           'low': np.percentile(stoch_inc_trimmed, 2.5, axis=0),
                                           'high': np.percentile(stoch_inc_trimmed, 97.5, axis=0)}

                # Background seasonal parameters
                multipliers = []
                for t in weekly_times_plot:
                    seas = seasonal_multiplier_local(season_amp, season_phase, season_period, t, t_lock, Lockdown_duration) if r==0 else \
                            (1.0 if r==1 else 2.0 - seasonal_multiplier_local(season_amp, season_phase, season_period, t, t_lock, Lockdown_duration))
                    multipliers.append(seas * contact_multiplier(t, t_lock, Lockdown_duration, c_pandemic))
                region['multipliers'] = multipliers
                regions.append(region)

            fig3 = {'iso_dates': iso_dates, 'regions': regions,
                    't_lock_x': t_days_to_x(t_lock),
                    't_testing_x': t_days_to_x(20.5 * 365.25),  # July 1st 2020, increase in surveillance
                    # September 30th 2025, end of fitted data (t = 15.75 years is 1 October 2015)
                    't_data_x': t_days_to_x(15.75 * 365.25 + (datetime.date(2025, 9, 30) - datetime.date(2015, 10, 1)).days),
                    't_extinct_Y_x': t_extinct_Y_x,
                    # Victoria's minimum prevalence after lockdown (deterministic), and the weeks shown
                    't_min_V_x': None if np.isnan(t_min_prevalence_V) else t_days_to_x(t_min_prevalence_V),
                    'weekly_times': weekly_times_plot,
                    # the four hypothetical Yamagata reintroductions
                    't_reintro_xs': [t_days_to_x(years * 365.25) for years in (26.75, 27, 27.25, 27.5)]}
            if name is not None:
                with open(fig3_data_path(name), 'wb') as f:
                    pickle.dump(fig3, f)
            draw_fig3(fig3, name=name, title=title, style=style)

            

        return {
            'simple_coexist': simple_coexist, 'recovery_V': recovery_V, 'recovery_Y': recovery_Y,
            'min_prevalence_V': min_prevalence_V, 'R_eff_Y_peak_2026_27': R_eff_Y_peak_2026_27, 'R_values': R_values,
            'extinct_V': extinct_V, 'extinct_Y': extinct_Y, 'weekly_infections_V': weekly_infections_V,
            'weekly_infections_Y': weekly_infections_Y, 'weekly_times': weekly_times, 't_eval': t_eval,
            'y_all': y_all, 'region_pops': [N1, N2, N3], 't_yamagata_extinction': t_yamagata_extinction,
            't_min_prevalence_V': t_min_prevalence_V, 't_R_eff_Y_peak_2026_27': t_R_eff_Y_peak_2026_27,
            'stoch_extinction_pct_V': stoch_extinction_pct_V, 'reintroductions': reintroductions
        }
    
    

def fig3_data_path(name):
    """Where model(plot=True, name=...) saves what its figure shows, for redraw_fig3."""
    return str(name).rsplit('.', 1)[0] + '_plotdata.pkl'


def redraw_fig3(name, title=None, out=None, style=None):
    """Redraw a figure in Fig. 3's layout from the data model() saved with it (no simulations), to out
    (default: name itself), optionally with a title and style options (see draw_fig3)."""
    with open(fig3_data_path(name), 'rb') as f:
        fig3 = pickle.load(f)
    draw_fig3(fig3, name=out or name, title=title, style=style)


def draw_fig3(fig3, name=None, title=None, style=None):
    """Draw Fig. 3's layout (three regions; deterministic lines, the stochastic Victoria summaries, the events)
    from the data model() gathers. title: (text, marker, colour) above the panels, e.g. a best fit's rank.
    style: {'labels_left': (label, ...)} puts those event labels to the left of their lines instead of
    the right, where events are only weeks apart."""
    style = style or {}
    iso_dates, regions = fig3['iso_dates'], fig3['regions']
    t_lock_x, t_testing_x, t_data_x = fig3['t_lock_x'], fig3['t_testing_x'], fig3['t_data_x']
    t_extinct_Y_x, t_reintro_xs = fig3['t_extinct_Y_x'], fig3['t_reintro_xs']
    t_min_V_x = fig3.get('t_min_V_x')  # Victoria's deterministic minimum (figures saved before 28 Oct 2026 lack it)
    t_reintro_x = t_reintro_xs[0]

    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    region_names = ["Northern Hemisphere", "Tropics", "Southern Hemisphere"]

    # Format X-Ticks to isolate and display ONLY the year
    tick_positions = []
    tick_labels = []
    prev_year = None
    for i, iso_date in enumerate(iso_dates):
        year = int(iso_date.split('-')[0])
        # Crucial: Only drop a tick line when shifting into a NEW year (skips index 0 padding)
        if prev_year is not None and year != prev_year:
            tick_positions.append(i)
            tick_labels.append(str(year))
        prev_year = year

    lim=[]
    for r in range(3):
        ax = axes[r]
        ax.text(0, 1, ['A', 'B', 'C'][r], transform=ax.transAxes,
                fontsize=20, fontname='Arial', fontweight='bold', va='bottom', ha='right')
        weeks = np.arange(len(iso_dates))
        v_weekly, y_weekly = regions[r]['v_weekly'], regions[r]['y_weekly']

        # Plot standard deterministic lines
        if len(y_weekly) > 0 and not np.all(np.isnan(y_weekly)):
            ax.plot(weeks, np.sqrt(y_weekly),
                    linewidth=2.5, label='Yamagata deterministic', color='tab:orange')
        if len(v_weekly) > 0 and not np.all(np.isnan(v_weekly)):
            ax.plot(weeks, np.sqrt(v_weekly),
                    linewidth=2.5, label='Victoria deterministic', color='tab:blue')

        safe_max_v = np.nanmax(v_weekly) if len(v_weekly) > 0 and not np.all(np.isnan(v_weekly)) else 0
        safe_max_y = np.nanmax(y_weekly) if len(y_weekly) > 0 and not np.all(np.isnan(y_weekly)) else 0
        lim.append(np.sqrt(1.55* max(safe_max_v, safe_max_y, 1)))
        # Render Yamagata Extinction Line if it happened
        if t_extinct_Y_x is not None:
            ax.axvline(x=t_extinct_Y_x, color='tab:orange', linestyle='--', linewidth=2, alpha=0.7)
        # and Victoria's minimum prevalence (deterministic)
        if t_min_V_x is not None:
            ax.axvline(x=t_min_V_x, color='tab:blue', linestyle='--', linewidth=2, alpha=0.7)
        # Overlay the stochastic summary envelopes
        stoch = regions[r]['stoch']
        if stoch is not None:
            ax.plot(stoch['x'], np.sqrt(stoch['median']), linestyle=':', linewidth=2,
                    color='tab:blue', label='Victoria median')
            ax.plot(stoch['x'], np.sqrt(stoch['mean']), linestyle='-.', linewidth=1.5,
                    color='tab:blue', label='Victoria mean')
            ax.fill_between(stoch['x'], np.sqrt(np.maximum(stoch['low'], 0)), np.sqrt(stoch['high']),
                            color='tab:blue', alpha=0.15, label='Victoria 95% interval')

        # Render April 1st 2020 Lockdown Line correctly mapped to x-axis indices
        ax.axvline(x=t_lock_x, color='red', linestyle='--', linewidth=2, alpha=0.7, label='April 1st 2020 (Lockdown)' if r==0 else "")
        # Render July 1st 2020 Testing Increase Line
        ax.axvline(x=t_testing_x, color='green', linestyle='--', linewidth=2, alpha=0.7, label='July 1st 2020 (Testing Increase)' if r==0 else "")
        # Render September 30th 2025 Data End Line
        ax.axvline(x=t_data_x, color='green', linestyle='-.', linewidth=2, alpha=0.7, label='Sept 30th 2025 (Data End)' if r==0 else "")
        #Yamagata reintroductions
        for x in t_reintro_xs:
            ax.axvline(x=x, color='tab:orange', linestyle=':', linewidth=2, alpha=1)

        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)
        ax.set_xlim(0, len(iso_dates))
        ax.set_ylabel('weekly cases (millions)', fontsize=14)

        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: f'{int(val**2):,}'))
        plt.setp(ax.xaxis.get_majorticklabels(), fontsize=12)
        plt.setp(ax.yaxis.get_majorticklabels(), fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_title(f'{region_names[r]}', fontsize=14, fontweight='bold')

        # Position middle subplot legend nicely on the right side
        if r == 1:
            ax.legend(loc=[0.02, 0.425], fontsize=14, facecolor='white', framealpha=1)

        # Background seasonal parameters
        ax2 = ax.twinx()
        ax2.plot(weeks, regions[r]['multipliers'], linestyle='--', linewidth=1, color='gray', alpha=0.7)
        ax2.set_ylabel('seasonality × contact multiplier', color='gray', fontsize=14)
        ax2.set_ylim(0.65,1.15)
        plt.setp(ax2.yaxis.get_majorticklabels(), fontsize=12, color='gray')
    labels_left = set(style.get('labels_left', ()))

    def event_label(ax, x, y, text, colour, alpha):
        """An event's label along its line: to the right of it, or to the left if in style['labels_left']."""
        if text in labels_left:  # right up to the line: the rotated text's descender space overlaps it
            ax.text(x+0.75*offset, y, text, color=colour, fontsize=14, alpha=alpha, va='top', ha='right', rotation=90)
        else:
            ax.text(x+offset, y, text, color=colour, fontsize=14, alpha=alpha, va='top', ha='left', rotation=90)

    for r in range(3):
        ax = axes[r]
        mlim = max(lim)
        ax.set_ylim(0, mlim)
        mlim=0.95*mlim #for v coordinate of text annotations
        offset=2
        if t_lock_x is not None:
            event_label(ax, t_lock_x, mlim, 'lockdown', 'red', 0.7)
        if t_testing_x is not None:
            event_label(ax, t_testing_x, mlim, 'increased reporting', 'green', 0.7)
        if t_data_x is not None:
            event_label(ax, t_data_x, mlim, 'end of data', 'green', 0.7)
        if t_extinct_Y_x is not None:
            event_label(ax, t_extinct_Y_x, mlim, 'Yam. extinction', 'tab:orange', 1)
            ax.text(t_reintro_x-offset, mlim, 'Yam. reintroductions', color='tab:orange', fontsize=14, alpha=1, va='top', ha='right', rotation=90)
        if t_min_V_x is not None:
            event_label(ax, t_min_V_x, mlim, 'Vic. minimum', 'tab:blue', 0.7)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.set_xlim(0, len(iso_dates))
    title_above(fig, title)
    if name is not None:
        plt.savefig(name)
    plt.show()


def title_above(fig, title):
    """Lay out a figure in Fig. 3's layout, with title = (text, marker, colour) above the panels (e.g. a best
    fit's rank and its marker) or none. Also used by simulator_leaky.py."""
    if title is None:
        plt.tight_layout()
        return
    text, marker, colour = title
    fig.legend(handles=[Line2D([0], [0], linestyle='none', marker=marker, markersize=16, color=colour,
                               markeredgecolor='white', label=text)],
               loc='upper center', bbox_to_anchor=(0.5, 1.0), frameon=False, handletextpad=0.3,
               prop={'size': 18, 'weight': 'bold'})
    plt.tight_layout(rect=(0, 0, 1, 0.965))


# The outputs of model() that the Latin hypercube search keeps
SEARCH_OUTPUTS = ('extinct_V', 'extinct_Y', 'recovery_V', 'recovery_Y', 'simple_coexist', 'min_prevalence_V',
                  'R_eff_Y_peak_2026_27', 'R_values')


def run_single_simulation(params, threshold=1):
    """One sample of the Latin hypercube search (what each worker process runs). Returns the parameters,
    the outputs in SEARCH_OUTPUTS and success=True or, if the model raises, the parameters, success=False
    and the error."""
    rV, rY, tV, tY, sVY, sYV,  m, season_phase, c_pandemic, Lockdown_duration = params
    row = {'R0V': rV, 'R0Y': rY, 'TV_years': tV, 'TY_years': tY, 'sigmaVY': sVY, 'sigmaYV': sYV, 'm': m,
           'season_phase': season_phase, 'c_pandemic': c_pandemic, 'Lockdown_duration': Lockdown_duration}
    try:
        res = model(
            R0V=rV, R0Y=rY, threshold=threshold,
            omegaV=1/(tV * 365.25), omegaY=1/(tY * 365.25),
            sigmaVY=sVY, sigmaYV=sYV,
            m=m, season_phase=season_phase, c_pandemic=c_pandemic, Lockdown_duration=Lockdown_duration,
            plot=False, stochastic=False  # Crucial for multiprocessing!
        )
    except Exception as e:
        return {**row, 'success': False, 'error': repr(e)}
    return {**row, **{k: res[k] for k in SEARCH_OUTPUTS}, 'success': True}
