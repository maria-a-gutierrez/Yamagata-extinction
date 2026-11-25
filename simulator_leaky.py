"""The model of simulator.py with leaky instead of all-or-none cross-immunity (SI, "All-or-none immunity
assumption"; Figs. S12, S14, S16 and S18).

Cross-immunity follows the leaky model of the companion preprint cited in the SI (its Appendix 5.1,
equations (28)-(29)). Everything else is as in simulator.py: births and deaths, waning, seasonality,
lockdown, the force of infection, extinction thresholds, metrics and figure. Only the deterministic model
is converted: the stochastic ensembles assume all-or-none cross-immunity, so they are disabled here.

Force of infection: foi='balanced' (the default) mixes the regions exactly as simulator.py does,
including the within-NH contact coefficient a1 in the Yamagata R_e; foi='published' uses
I_i + m * sum_{j != i} I_j over N_i instead, which does not balance the contacts between regions (kept
for comparison only).

Status with respect to each lineage (Yamagata shown; Victoria mirrors it with sigma_YV):
  S_Y   fully susceptible
  P_Y   partially immune after infection with Victoria (every infection, not a fraction sigma_VY)
  I_YS  infected from S_Y;  I_YP  infected from P_Y
  1 - S_Y - P_Y - I_YS - I_YP  fully immune after a Yamagata infection
Both partial and full immunity wane back to S_Y at omega_Y, as in Appendix eq. (28).

leaky_mode='susceptibility' (the default, used in the SI) is the reading in the appendix text
("susceptibility reduced by a factor 1 - sigma"): P_Y is infected at (1 - sigma_VY) times the rate, and
those infections transmit fully. leaky_mode='infectiousness' is equations (28)-(29) as written: P_Y is
infected at the full rate and those infections transmit at (1 - sigma_VY), lambda_Y = beta_Y (I_YS +
(1 - sigma_VY) I_YP). Both give R_e,Y = R0,Y (S_Y + (1 - sigma_VY) P_Y) (Appendix eq. 31), the formula
used here for R_e. Reported infections count every new infection, from S and from P.
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
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
    
def leaky_coexistence_equilibrium(R0V, R0Y, gamma, omegaV, omegaY, sigmaVY, sigmaYV, mu,
                                  mode='infectiousness'):
    """Coexistence equilibrium of the single-region leaky model, as population fractions
    (SV, PV, IVS, IVP, SY, PY, IYS, IYP), and whether it is biologically feasible.

    At equilibrium S + (1 - sigma) P = 1/R0 for each lineage, so every compartment follows from
    the two forces of infection; these are found by root-finding, starting from the all-or-none
    equilibrium (closed form, coexistence_equilibrium) or from the single-lineage equilibria."""
    g = gamma + mu
    sus_PV, sus_PY = (1.0, 1.0) if mode == 'infectiousness' else (1.0 - sigmaYV, 1.0 - sigmaVY)

    def lineage(R0, lam_own, lam_other, sigma, omega, sus_P):
        k = sus_P * lam_own + omega + mu                 # rate of leaving P
        S = (1.0 / R0) / (1.0 + (1.0 - sigma) * lam_other / k)
        P = lam_other * S / k
        I_S = lam_own * S / g
        I_P = sus_P * lam_own * P / g
        resid = mu * (1 - S) - (lam_own + lam_other) * S + omega * (1 - S - I_S - I_P)
        return (S, P, I_S, I_P), resid

    def residuals(u):
        lV, lY = np.exp(u)
        return [lineage(R0V, lV, lY, sigmaYV, omegaV, sus_PV)[1] / mu,
                lineage(R0Y, lY, lV, sigmaVY, omegaY, sus_PY)[1] / mu]

    def single(R0, omega):  # force of infection at the single-lineage equilibrium
        return R0 * g * (mu + omega) * (1 - 1 / R0) / (g + omega) if R0 > 1 else 0.0

    S_V, S_Y, I_V, I_Y, ok = coexistence_equilibrium(R0V, R0Y, gamma, omegaV, omegaY, sigmaVY, sigmaYV, mu)
    guesses = [(R0V * g * I_V, R0Y * g * I_Y)] if ok else []
    guesses += [(single(R0V, omegaV), single(R0Y, omegaY)),
                (0.5 * single(R0V, omegaV), 0.5 * single(R0Y, omegaY))]
    for lV0, lY0 in guesses:
        if lV0 <= 0 or lY0 <= 0:
            continue
        u = fsolve(residuals, np.log([lV0, lY0]), xtol=1e-13)
        if np.max(np.abs(residuals(u))) > 1e-8:
            continue
        lV, lY = np.exp(u)
        (SV, PV, IVS, IVP), _ = lineage(R0V, lV, lY, sigmaYV, omegaV, sus_PV)
        (SY, PY, IYS, IYP), _ = lineage(R0Y, lY, lV, sigmaVY, omegaY, sus_PY)
        comps = np.array([SV, PV, IVS, IVP, SY, PY, IYS, IYP])
        if np.all(comps >= 0) and SV + PV + IVS + IVP <= 1 and SY + PY + IYS + IYP <= 1:
            return tuple(float(c) for c in comps), True
    return (1 - 1e-6, 0.0, 1e-6, 0.0, 1 - 1e-6, 0.0, 1e-6, 0.0), False


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
    """Balanced mixing (as in simulator.py): a region-i person makes a fraction
    m*N_j/N of their contacts with region j and the rest, 1 - m*(N - N_i)/N, at home. Returns the
    within-region coefficients (a1, a2, a3) and the population shares (q1, q2, q3) = N_i/N."""
    N = N1 + N2 + N3
    q = (N1/N, N2/N, N3/N)
    return tuple(1 - m*(1 - qi) for qi in q), q


# Every output of model()
RESULT_KEYS = ('simple_coexist', 'recovery_V', 'recovery_Y', 'min_prevalence_V', 'R_eff_Y_peak_2026_27',
               'R_values', 'extinct_V', 'extinct_Y', 'weekly_infections_V', 'weekly_infections_Y',
               'weekly_times', 't_eval', 'y_all', 'region_pops', 't_min_prevalence_V')


def stopped_early(region_pops, **outcome):
    """model()'s result when shortcuts=True stops a run early: the outcome given, the region populations
    and NaN for every other output. When Victoria dies out, extinct_V is NaN and extinct_Y True whatever
    Yamagata did: recovery_V (-1 before the lockdown, -10 after) is what marks Victoria's extinction, and the
    selection rules rely on these values."""
    return {**dict.fromkeys(RESULT_KEYS, np.nan), 'region_pops': region_pops, **outcome}


def run_stochastic_ensemble(t_start, t_end, y_start_state, N1, N2, N3, betaV, gamma, mu, omegaV, m, 
                            season_amp, season_phase, season_period, t_lock, Lockdown_duration, c_pandemic,
                            n_sims=100, seed=1, tau=1.0):
    """
    Simulates the remaining system using Tau-Leaping for Victoria dynamics 
    after Yamagata's absolute extinction. 
    """
    np.random.seed(seed)
    # y_start_state indices correspond to Victoria's layout:
    # 0:SV1, 1:IV1, 2:cumIV1, 3:SV2, 4:IV2, 5:cumIV2, 6:SV3, 7:IV3, 8:cumIV3
    t_eval = np.arange(np.ceil(t_start), t_end, tau)
    n_steps = len(t_eval)
    
    # Storage for all simulations: shape (n_sims, n_steps, 9)
    all_trajectories = np.zeros((n_sims, n_steps, 9))

    r21, r31 = N2/N1, N3/N1
    r12, r32 = N1/N2, N3/N2
    r13, r23 = N1/N3, N2/N3

    for sim in range(n_sims):
        # Local state tracks total counts rather than fractions for Poisson accuracy
        # x vector structure: [SV1, IV1, cumIV1, SV2, IV2, cumIV2, SV3, IV3, cumIV3]
        x = np.array([
            y_start_state[0]*N1, y_start_state[1]*N1, 0.0,
            y_start_state[6]*N2, y_start_state[7]*N2, 0.0,
            y_start_state[12]*N3, y_start_state[13]*N3, 0.0
        ], dtype=float)
        
        for step_idx, t in enumerate(t_eval):
            all_trajectories[sim, step_idx, :] = x
            
            # Extract current integer counts
            SV1, IV1, _, SV2, IV2, _, SV3, IV3, _ = x
            
            # Compute current multiplier arrays
            cm = contact_multiplier(t, t_lock, Lockdown_duration, c_pandemic)
            seas1, seas2, seas3 = seasonal_multiplier_triple(seasonal_multiplier_local(season_amp, season_phase, season_period, t, t_lock, Lockdown_duration))
            
            # Global fractional forces of infection
            inf_fraction_V1 = (IV1/N1) + m * ((IV2/N2)*r21 + (IV3/N3)*r31)
            inf_fraction_V2 = (IV2/N2) + m * ((IV1/N1)*r12 + (IV3/N3)*r32)
            inf_fraction_V3 = (IV3/N3) + m * ((IV1/N1)*r13 + (IV2/N2)*r23)
            
            forceV1 = betaV * seas1 * cm * inf_fraction_V1
            forceV2 = betaV * seas2 * cm * inf_fraction_V2
            forceV3 = betaV * seas3 * cm * inf_fraction_V3
            
            # Calculate transition rates (Expected counts per unit time)
            # Region 1
            rate_inf1 = forceV1 * SV1
            rate_rec1 = gamma * IV1
            rate_wan1 = omegaV * (N1 - (SV1 + IV1))
            rate_birth1 = mu * N1
            rate_m_S1 = mu * SV1
            rate_m_I1 = mu * IV1
            
            # Region 2
            rate_inf2 = forceV2 * SV2
            rate_rec2 = gamma * IV2
            rate_wan2 = omegaV * (N2 - (SV2 + IV2))
            rate_birth2 = mu * N2
            rate_m_S2 = mu * SV2
            rate_m_I2 = mu * IV2
            
            # Region 3
            rate_inf3 = forceV3 * SV3
            rate_rec3 = gamma * IV3
            rate_wan3 = omegaV * (N3 - (SV3 + IV3))
            rate_birth3 = mu * N3
            rate_m_S3 = mu * SV3
            rate_m_I3 = mu * IV3
            
            # Sample discrete number of events occurring over window tau
            # Clips to 0 to prevent issues with floating-point underflows
            events = np.random.poisson(np.maximum([
                rate_inf1, rate_rec1, rate_wan1, rate_birth1, rate_m_S1, rate_m_I1,
                rate_inf2, rate_rec2, rate_wan2, rate_birth2, rate_m_S2, rate_m_I2,
                rate_inf3, rate_rec3, rate_wan3, rate_birth3, rate_m_S3, rate_m_I3
            ], 0) * tau)
            
            # Unpack events
            e_inf1, e_rec1, e_wan1, e_b1, e_ms1, e_mi1 = events[0:6]
            e_inf2, e_rec2, e_wan2, e_b2, e_ms2, e_mi2 = events[6:12]
            e_inf3, e_rec3, e_wan3, e_b3, e_ms3, e_mi3 = events[12:18]
            
            # Apply state changes securely, bounding populations at 0
            # Region 1 Updates
            x[0] = max(0.0, SV1 - e_inf1 + e_wan1 + e_b1 - e_ms1)
            x[1] = max(0.0, IV1 + e_inf1 - e_rec1 - e_mi1)
            x[2] += e_inf1 # tracking cumulative infections
            
            # Region 2 Updates
            x[3] = max(0.0, SV2 - e_inf2 + e_wan2 + e_b2 - e_ms2)
            x[4] = max(0.0, IV2 + e_inf2 - e_rec2 - e_mi2)
            x[5] += e_inf2
            
            # Region 3 Updates
            x[6] = max(0.0, SV3 - e_inf3 + e_wan3 + e_b3 - e_ms3)
            x[7] = max(0.0, IV3 + e_inf3 - e_rec3 - e_mi3)
            x[8] += e_inf3
            
    return t_eval, all_trajectories


def model(
        R0V=1.2, R0Y=1.2, gamma=1/3.4,
        omegaV=0.0, omegaY=0.0,
        sigmaVY=0.0, sigmaYV=0.0,
        tspan=26.75*365.25, Tpan=20.25/26.75, Lockdown_duration=500,
        c_pandemic=0.5,
        N1: int = 4_000_000_000,
        N2: int = 3_400_000_000,
        N3: int = 800_000_000, mortality_rate=1/(75*365.25),
        m = 0.0060,                        
        threshold: float = 1.0, 
        season_amp: float = 0.1, season_phase: float = 90, season_period: float = 365.25,
        plot: bool = True, transient_years=15.75, shortcuts: bool = True, name=None, title=None, style=None, stochastic: bool = False, nsims=100, stochastic2: bool = False, nsims2=100,
        leaky_mode: str = 'susceptibility', foi: str = 'balanced', **solver_kwargs):

    if stochastic or (stochastic2 and nsims2 > 0):
        raise NotImplementedError('the stochastic ensembles still assume all-or-none cross-immunity; '
                                  'run the leaky model deterministically (stochastic=False, nsims2=0)')
    if leaky_mode not in ('infectiousness', 'susceptibility'):
        raise ValueError("leaky_mode must be 'infectiousness' or 'susceptibility'")
    if foi not in ('published', 'balanced'):
        raise ValueError("foi must be 'published' or 'balanced'")
    eq, simple_coexist = leaky_coexistence_equilibrium(R0V, R0Y, gamma, omegaV, omegaY, sigmaVY, sigmaYV,
                                                       mortality_rate, leaky_mode)

    if shortcuts and not simple_coexist:
        return stopped_early([N1, N2, N3], simple_coexist=simple_coexist)
    else: 
        betaV = R0V * (gamma+mortality_rate)
        betaY = R0Y * (gamma+mortality_rate)
        t_lock = Tpan * tspan

        # Mixing: a_i weights region i's own infectious fraction, w_ij that of region j, in region i's
        # force of infection. 'published': a_i = 1, w_ij = N_j/N_i. 'balanced' (as in simulator.py):
        # a_i = 1 - m (N - N_i)/N, w_ij = N_j/N.
        if foi == 'balanced':
            (a1, a2, a3), (q1, q2, q3) = balanced_mixing(m, N1, N2, N3)
            w12, w13, w21, w23, w31, w32 = q2, q3, q1, q3, q1, q2
        else:
            a1 = a2 = a3 = 1.0
            w12, w13, w21, w23, w31, w32 = N2/N1, N3/N1, N1/N2, N3/N2, N1/N3, N2/N3


        def force_strain(beta, seasonal, I_in, I_out_1, I_out_2):
            return beta * seasonal * (I_in + m * (I_out_1 + I_out_2))

        def force_region(t, seasonal, IV, IV_out_1, IV_out_2, IY, IY_out_1, IY_out_2):
            cm = contact_multiplier(t, t_lock, Lockdown_duration, c_pandemic)
            transmissibility_multiplier = seasonal * cm
            forceV = force_strain(betaV, transmissibility_multiplier, IV, IV_out_1, IV_out_2)
            forceY = force_strain(betaY, transmissibility_multiplier, IY, IY_out_1, IY_out_2)
            return forceV, forceY

        # Leaky cross-immunity (Appendix 5.1, eqs. 28-29), births and deaths as in the all-or-none model.
        # sus_P*: relative rate at which P is infected; inf_P*: relative infectiousness of infections from P.
        if leaky_mode == 'infectiousness':
            sus_PV, sus_PY, inf_PV, inf_PY = 1.0, 1.0, 1.0 - sigmaYV, 1.0 - sigmaVY
        else:
            sus_PV, sus_PY, inf_PV, inf_PY = 1.0 - sigmaYV, 1.0 - sigmaVY, 1.0, 1.0

        def derivatives_region(forceV, forceY, SV, PV, IVS, IVP, SY, PY, IYS, IYP):
            g = gamma + mortality_rate
            dSV = mortality_rate * (1 - SV) - forceV * SV - forceY * SV + omegaV * (1 - SV - IVS - IVP)
            dPV = forceY * SV - sus_PV * forceV * PV - (omegaV + mortality_rate) * PV
            dIVS = forceV * SV - g * IVS
            dIVP = sus_PV * forceV * PV - g * IVP
            dSY = mortality_rate * (1 - SY) - forceY * SY - forceV * SY + omegaY * (1 - SY - IYS - IYP)
            dPY = forceV * SY - sus_PY * forceY * PY - (omegaY + mortality_rate) * PY
            dIYS = forceY * SY - g * IYS
            dIYP = sus_PY * forceY * PY - g * IYP
            return dSV, dPV, dIVS, dIVP, dSY, dPY, dIYS, dIYP
        
        def new_infections_rate(forceV, forceY, SV, PV, SY, PY):
            return forceV * (SV + sus_PV * PV), forceY * (SY + sus_PY * PY)

        def F(t, x):
            (SV1, PV1, IVS1, IVP1, SY1, PY1, IYS1, IYP1, cumIV1, cumIY1,
             SV2, PV2, IVS2, IVP2, SY2, PY2, IYS2, IYP2, cumIV2, cumIY2,
             SV3, PV3, IVS3, IVP3, SY3, PY3, IYS3, IYP3, cumIV3, cumIY3) = x

            seas1, seas2, seas3 = seasonal_multiplier_triple(seasonal_multiplier_local(season_amp, season_phase, season_period, t, t_lock, Lockdown_duration))

            # Infectious pressure: infections from P count at inf_P; regions mix with a_i and w_ij (above)
            JV1, JV2, JV3 = IVS1 + inf_PV * IVP1, IVS2 + inf_PV * IVP2, IVS3 + inf_PV * IVP3
            JY1, JY2, JY3 = IYS1 + inf_PY * IYP1, IYS2 + inf_PY * IYP2, IYS3 + inf_PY * IYP3

            forceV1, forceY1 = force_region(t, seas1, a1*JV1, JV2*w12, JV3*w13, a1*JY1, JY2*w12, JY3*w13)
            forceV2, forceY2 = force_region(t, seas2, a2*JV2, JV1*w21, JV3*w23, a2*JY2, JY1*w21, JY3*w23)
            forceV3, forceY3 = force_region(t, seas3, a3*JV3, JV1*w31, JV2*w32, a3*JY3, JY1*w31, JY2*w32)

            d1 = derivatives_region(forceV1, forceY1, SV1, PV1, IVS1, IVP1, SY1, PY1, IYS1, IYP1)
            d2 = derivatives_region(forceV2, forceY2, SV2, PV2, IVS2, IVP2, SY2, PY2, IYS2, IYP2)
            d3 = derivatives_region(forceV3, forceY3, SV3, PV3, IVS3, IVP3, SY3, PY3, IYS3, IYP3)

            n1 = new_infections_rate(forceV1, forceY1, SV1, PV1, SY1, PY1)
            n2 = new_infections_rate(forceV2, forceY2, SV2, PV2, SY2, PY2)
            n3 = new_infections_rate(forceV3, forceY3, SV3, PV3, SY3, PY3)

            return [*d1, *n1, *d2, *n2, *d3, *n3]
            
        # Layout per region: SV, PV, IVS, IVP, SY, PY, IYS, IYP, cumIV, cumIY; every region starts at the
        # single-region equilibrium, as in the all-or-none model
        y0 = (list(eq) + [0.0, 0.0]) * 3

        V_state_idx = [2, 3, 12, 13, 22, 23]   # IVS, IVP in each region
        Y_state_idx = [6, 7, 16, 17, 26, 27]   # IYS, IYP in each region
        
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
                        return max((x[2] + x[3]) * N1, (x[12] + x[13]) * N2, (x[22] + x[23]) * N3) - threshold
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
                        return max((x[6] + x[7]) * N1, (x[16] + x[17]) * N2, (x[26] + x[27]) * N3) - threshold
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

        IV1, IY1 = y_all[2] + y_all[3], y_all[6] + y_all[7]
        IV2, IY2 = y_all[12] + y_all[13], y_all[16] + y_all[17]
        IV3, IY3 = y_all[22] + y_all[23], y_all[26] + y_all[27]
        cumIV1, cumIY1 = y_all[8], y_all[9]
        cumIV2, cumIY2 = y_all[18], y_all[19]
        cumIV3, cumIY3 = y_all[28], y_all[29]
        
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
        t_min_prevalence_V = np.nan  # when it happens (days), as in simulator.py

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
        if stochastic and extinct_Y and yamagata_extinction_state_captured is not None:
            # Calculate stochastic end time (Sep 30, 2026)
            ref_date_stoch = datetime.datetime(2015, 10, 1, tzinfo=datetime.timezone.utc)
            stoch_end_date = datetime.datetime(2026, 9, 30, tzinfo=datetime.timezone.utc)
            t_stoch_end = 15.75 * 365.25 + (stoch_end_date - ref_date_stoch).days
            actual_stoch_end = min(tspan, t_stoch_end)
            
            if t_yamagata_extinction < actual_stoch_end:
                # Initialize extinction metric tracker
                stoch_extinction_pct_V = np.nan
                stoch_times, stoch_trajectories = run_stochastic_ensemble(
                    t_start=t_yamagata_extinction, t_end=actual_stoch_end, 
                    y_start_state=yamagata_extinction_state_captured,
                    N1=N1, N2=N2, N3=N3, betaV=betaV, gamma=gamma, mu=mortality_rate, omegaV=omegaV, m=m,
                    season_amp=season_amp, season_phase=season_phase, season_period=season_period,
                    t_lock=t_lock, Lockdown_duration=Lockdown_duration, c_pandemic=c_pandemic, n_sims=nsims, seed=1)
                
                # --- NEW ADDITION: CALCULATE VICTORIA STOCHASTIC EXTINCTION PERCENTAGE ---
                t_check_extinct = actual_stoch_end
                if not np.isnan(recovery_V) and recovery_V > 0:
                    t_check_extinct = min(actual_stoch_end, t_lock + recovery_V * 365.25)
                
                check_idx = np.searchsorted(stoch_times, t_check_extinct)
                if check_idx >= len(stoch_times):
                    check_idx = -1

                # Extract final step active infecteds across all simulations: IV1 (idx 1), IV2 (idx 4), IV3 (idx 7)
                final_active_V = (stoch_trajectories[:, check_idx, 1] + 
                                  stoch_trajectories[:, check_idx, 4] + 
                                  stoch_trajectories[:, check_idx, 7])
                
                # Count the number of simulations where the final count of active infections is 0
                extinct_runs_count = np.sum(final_active_V == 0)
                stoch_extinction_pct_V = (extinct_runs_count / stoch_trajectories.shape[0]) * 100
                print(f">> Victoria went extinct in {stoch_extinction_pct_V:.2f}% of the stochastic runs before its recovery.")

                # Map full trajectories into weekly incidences matching deterministic tracking structures
                stoch_week_indices = np.arange(0, len(stoch_times), 7)
                stoch_times_weekly = stoch_times[stoch_week_indices]
                
                # Shape mapping: [simulations, regions, weeks]
                stoch_weekly_inc = np.zeros((nsims, 3, len(stoch_week_indices)))
                for s in range(nsims):
                    cumV1_s = stoch_trajectories[s, :, 2]
                    cumV2_s = stoch_trajectories[s, :, 5]
                    cumV3_s = stoch_trajectories[s, :, 8]
                    for w_idx, idx in enumerate(stoch_week_indices):
                        if w_idx == 0:
                            stoch_weekly_inc[s, 0, w_idx] = cumV1_s[idx]
                            stoch_weekly_inc[s, 1, w_idx] = cumV2_s[idx]
                            stoch_weekly_inc[s, 2, w_idx] = cumV3_s[idx]
                        else:
                            p_idx = stoch_week_indices[w_idx - 1]
                            stoch_weekly_inc[s, 0, w_idx] = (cumV1_s[idx] - cumV1_s[p_idx])
                            stoch_weekly_inc[s, 1, w_idx] = (cumV2_s[idx] - cumV2_s[p_idx])
                            stoch_weekly_inc[s, 2, w_idx] = (cumV3_s[idx] - cumV3_s[p_idx])
                stoch_data = {'times': stoch_times_weekly, 'incidence': stoch_weekly_inc}
        
        R_eff_Y_peak_2026_27 = np.nan
        t_ref_days = 26.75* 365.25 #arbitrary 
        ref_date = datetime.datetime(2026, 10, 1, tzinfo=datetime.timezone.utc)
        _season_start = t_ref_days + (datetime.datetime(2026, 10, 1, tzinfo=datetime.timezone.utc) - ref_date).days
        _season_end   = t_ref_days + (datetime.datetime(2027, 3, 31, tzinfo=datetime.timezone.utc) - ref_date).days
        _season_mask  = (t_eval >= _season_start) & (t_eval <= _season_end)


        if np.any(_season_mask):
            _t_season = t_eval[_season_mask]
            _SY1_season = (y_all[4] + (1 - sigmaVY) * y_all[5])[_season_mask]  # S + (1 - sigma) P
            _seas_vals = np.array([seasonal_multiplier_local(season_amp, season_phase, season_period, _t, t_lock, Lockdown_duration) for _t in _t_season])
            _peak_idx = np.argmax(_seas_vals)
            _Reff_r1 = R0Y * a1 * _SY1_season[_peak_idx] * _seas_vals[_peak_idx] * contact_multiplier(_t_season[_peak_idx], t_lock, Lockdown_duration, c_pandemic)
            R_eff_Y_peak_2026_27 = float(_Reff_r1)

        # --- NEW ADDITION: STOCHASTIC YAMAGATA TRACKING (MULTI-DATE ENSEMBLES) ---
        # Kept as None to cleanly remove stochastic Yamagata envelopes from the downstream plot system
        stoch_Y_data = None 
        R_values=[]
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
            tau = 1.0
            
            r21, r31 = N2/N1, N3/N1
            r12, r32 = N1/N2, N3/N2
            r13, r23 = N1/N3, N2/N3
            print(f"\n=============================================")
            print(f"--- YAMAGATA MULTI-DATE REINTRODUCTION STATS ---")
            print(f"=============================================")
            
            for d_str, t_reintro_hypo in zip(hypo_dates_str, hypo_times):
                if t_reintro_hypo >= tspan:
                    print(f"Date {d_str} (t={t_reintro_hypo:.1f}) skipped: exceeds simulation timeframe.")
                    continue
                
                # 1. CALCULATE EFFECTIVE REPRODUCTION NUMBER (Re) FOR SEEDED REGION (NH / Region 1)
                # Derived from the localized force of infection given a single index case: IY = [1, 0, 0]
                SY1_det_init = np.interp(t_reintro_hypo, t_eval, y_all[4] + (1 - sigmaVY) * y_all[5]) * N1  # S + (1 - sigma) P
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
                    
                    # Dynamic initialization of susceptibles from deterministic system state at this date
                    SY_init = np.array([np.interp(t_reintro_hypo, t_eval, y_all[2]) * N1,
                                        np.interp(t_reintro_hypo, t_eval, y_all[8]) * N2,
                                        np.interp(t_reintro_hypo, t_eval, y_all[14]) * N3])
                    
                    cm_arr = np.array([contact_multiplier(t, t_lock, Lockdown_duration, c_pandemic) for t in t_eval_stoch_Y])
                    seas1_arr = np.array([seasonal_multiplier_local(season_amp, season_phase, season_period, t, t_lock, Lockdown_duration) for t in t_eval_stoch_Y])
                    seas2_arr = np.ones_like(seas1_arr)
                    seas3_arr = 2.0 - seas1_arr
                    
                    forceV_det = np.zeros((3, n_steps_Y))
                    for step_idx in range(n_steps_Y):
                        iv1, iv2, iv3 = IV_det_frac[0, step_idx], IV_det_frac[1, step_idx], IV_det_frac[2, step_idx]
                        forceV_det[0, step_idx] = betaV * seas1_arr[step_idx] * cm_arr[step_idx] * (iv1 + m*(iv2*r21 + iv3*r31))
                        forceV_det[1, step_idx] = betaV * seas2_arr[step_idx] * cm_arr[step_idx] * (iv2 + m*(iv1*r12 + iv3*r32))
                        forceV_det[2, step_idx] = betaV * seas3_arr[step_idx] * cm_arr[step_idx] * (iv3 + m*(iv1*r13 + iv2*r23))
                    
                    all_cum_Y = np.zeros((nsims2, 3, n_steps_Y))
                    np.random.seed(1) # All 4 hypothetical dates can use the same NRG
                    
                    # 3. RUN THE INDEPENDENT STOCHASTIC ENSEMBLE
                    for sim in range(nsims2):
                        SY = SY_init.copy()
                        IY = np.array([1.0, 0.0, 0.0]) # Reintroduce exactly 1 case in NH
                        cumY = np.array([0.0, 0.0, 0.0])
                        
                        for step_idx in range(n_steps_Y):
                            all_cum_Y[sim, :, step_idx] = cumY
                                
                            # Calculate Yamagata force of infection
                            f1Y = betaY * seas1_arr[step_idx] * cm_arr[step_idx] * (IY[0]/N1 + m*(IY[1]/N2*r21 + IY[2]/N3*r31))
                            f2Y = betaY * seas2_arr[step_idx] * cm_arr[step_idx] * (IY[1]/N2 + m*(IY[0]/N1*r12 + IY[2]/N3*r32))
                            f3Y = betaY * seas3_arr[step_idx] * cm_arr[step_idx] * (IY[2]/N3 + m*(IY[0]/N1*r13 + IY[1]/N2*r23))
                            
                            # Region 1 rates
                            r_inf1 = f1Y * SY[0]
                            r_cross1 = sigmaVY * forceV_det[0, step_idx] * SY[0]
                            r_wan1 = omegaY * max(0, N1 - SY[0] - IY[0])
                            r_b1 = mortality_rate * N1
                            r_ds1 = mortality_rate * SY[0]
                            r_di1 = mortality_rate * IY[0]
                            r_rec1 = gamma * IY[0]
                            
                            # Region 2 rates
                            r_inf2 = f2Y * SY[1]
                            r_cross2 = sigmaVY * forceV_det[1, step_idx] * SY[1]
                            r_wan2 = omegaY * max(0, N2 - SY[1] - IY[1])
                            r_b2 = mortality_rate * N2
                            r_ds2 = mortality_rate * SY[1]
                            r_di2 = mortality_rate * IY[1]
                            r_rec2 = gamma * IY[1]
                            
                            # Region 3 rates
                            r_inf3 = f3Y * SY[2]
                            r_cross3 = sigmaVY * forceV_det[2, step_idx] * SY[2]
                            r_wan3 = omegaY * max(0, N3 - SY[2] - IY[2])
                            r_b3 = mortality_rate * N3
                            r_ds3 = mortality_rate * SY[2]
                            r_di3 = mortality_rate * IY[2]
                            r_rec3 = gamma * IY[2]
                            
                            events = np.random.poisson(np.maximum([
                                r_inf1, r_cross1, r_wan1, r_b1, r_ds1, r_di1, r_rec1,
                                r_inf2, r_cross2, r_wan2, r_b2, r_ds2, r_di2, r_rec2,
                                r_inf3, r_cross3, r_wan3, r_b3, r_ds3, r_di3, r_rec3
                            ], 0) * tau)
                            
                            e_inf1, e_cr1, e_w1, e_b1, e_ds1, e_di1, e_r1 = events[0:7]
                            e_inf2, e_cr2, e_w2, e_b2, e_ds2, e_di2, e_r2 = events[7:14]
                            e_inf3, e_cr3, e_w3, e_b3, e_ds3, e_di3, e_r3 = events[14:21]
                            
                            SY[0] = max(0, SY[0] - e_inf1 - e_cr1 + e_w1 + e_b1 - e_ds1)
                            IY[0] = max(0, IY[0] + e_inf1 - e_r1 - e_di1)
                            cumY[0] += e_inf1
                            
                            SY[1] = max(0, SY[1] - e_inf2 - e_cr2 + e_w2 + e_b2 - e_ds2)
                            IY[1] = max(0, IY[1] + e_inf2 - e_r2 - e_di2)
                            cumY[1] += e_inf2
                            
                            SY[2] = max(0, SY[2] - e_inf3 - e_cr3 + e_w3 + e_b3 - e_ds3)
                            IY[2] = max(0, IY[2] + e_inf3 - e_r3 - e_di3)
                            cumY[2] += e_inf3

                    # 4. EVALUATE OUTBREAK SIZE METRICS FOR THIS ENSEMBLE
                    outbreak_sizes_Y = np.sum(all_cum_Y[:, :, -1], axis=1)
                    fadeout_count = np.sum(outbreak_sizes_Y < 100)
                    prob_fadeout = fadeout_count / nsims2
                    print(f"Hypothetical Introduction Date: {d_str}")
                    print(f"  Calculated Initial Re: {Re_Y:.2f} and poisson formula for extinction probability: {1/Re_Y:.2f}")
                    print(f"  Stochastic Fade-outs (< 100 cases): {fadeout_count} / {nsims2}")
                    print(f"  Probability of Fade-out: {prob_fadeout:.2%}")
                    print(f"---------------------------------------------")

       
        
        
        # --- EXECUTE EXTENDED PLOTTING SYSTEM ---
        # What the figure shows is gathered in fig3, saved next to the figure (fig3_data_path) so that
        # redraw_fig3 can redraw it without rerunning anything, and drawn by draw_fig3 (all three in
        # simulator.py, whose plotting this was a copy of).
        if plot:
            import pickle
            from simulator import draw_fig3, fig3_data_path
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
            'y_all': y_all, 'region_pops': [N1, N2, N3], 't_min_prevalence_V': t_min_prevalence_V
        }
    
    
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
