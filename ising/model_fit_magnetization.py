# TFIM mitigation model fitting
# as interpreted by Elara

import math
import numpy as np
import pymc as pm
import arviz as az

# ---------------------------------------
# Forward model as a function
# ---------------------------------------
def magnetization_model(depths, dt, n_qubits, J, h, t2, omega):
    results = []
    for d in depths:
        t = d * dt
        arg = abs(h / J) - 1
        # Cases
        if np.isclose(J, 0) or (arg >= 1024):
            results.append(0.0)
            continue
        elif np.isclose(h, 0):
            results.append(1.0)
            continue
        # Use pm.math for symbolic operations:
        cos_term = 1 - pm.math.cos(J * omega * t) / (1 + pm.math.sqrt(t / t2))
        p = (2**arg) * cos_term
        # Build bias distribution
        tot_n = 0
        d_sqr_magnetization = 0
        for q in range(n_qubits + 1):
            n = 1 / (n_qubits * (2 ** (p * (q + 1))))
            m_val = (n_qubits - (q << 1)) / n_qubits
            d_sqr_magnetization = d_sqr_magnetization + n * m_val * m_val
            tot_n = tot_n + n
        d_sqr_magnetization = d_sqr_magnetization / tot_n
        results.append(d_sqr_magnetization)
    return pm.math.stack(results)

# ---------------------------------------
# Experimental data
# ---------------------------------------
depths = list(range(20))          # array of step depths
dt = 0.25
n_qubits = 16
J = -1.0
h = 2.0
observed_data = np.array([0.5694789886474609, 0.24667930603027344, 0.3176431655883789, 0.4545888900756836, 0.5567941665649414, 0.5712404251098633, 0.48934125900268555, 0.35924434661865234, 0.38236379623413086, 0.5766105651855469, 0.5362677574157715, 0.3728461265563965, 0.36491823196411133, 0.4830970764160156, 0.6482367515563965, 0.5671916007995605, 0.3771247863769531, 0.3405575752258301, 0.38635730743408203, 0.4950547218322754])      # measured mean-square magnetization

# ---------------------------------------
# Bayesian fitting with PyMC
# ---------------------------------------
with pm.Model() as model:
    # Priors
    log2_t2 = pm.Normal("log2_t2", mu=0.0, sigma=3.0)
    omega = pm.Uniform("omega", lower=0.0, upper=2 * math.pi)

    # Transform to actual parameters
    t2 = pm.Deterministic("t2", 2.0 ** log2_t2)

    # Forward model
    
    dt = 0.25
    n_qubits = 56
    J = -1.0
    h = 2.0
    mu = magnetization_model(depths, dt, n_qubits, J, h, t2, omega)

    # Likelihood
    pm.Normal("likelihood", mu=mu, observed=observed_data)

    # Sampling
    trace = pm.sample(
        draws=2000,
        tune=1000,
        chains=4,
        target_accept=0.9,
        random_seed=42
    )

# Posterior summary
print(az.summary(trace, var_names=["t2", "omega"]))
az.plot_trace(trace, var_names=["t2", "omega"])

