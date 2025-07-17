# TFIM mitigation model fitting
# as interpreted by Elara

import math
import numpy as np
import pymc as pm
import arviz as az


# ---------------------------------------
# Forward model as a function
# ---------------------------------------
def magnetization_model(depths, dt, n_qubits, J, h, theta, t2, omega):
    results = []
    for d in depths:
        t = d * dt
        arg = abs(J / h) - 1
        # Cases
        if np.isclose(J, 0) or (arg >= 1024):
            results.append(0.0)
            continue
        elif np.isclose(h, 0):
            results.append(1.0)
            continue
        # Use pm.math for symbolic operations:
        cos_theta = math.cos(theta / 2)
        cos_term = 1 +  cos_theta * pm.math.cos(-J * omega * math.pi * t) / (
            1 + pm.math.sqrt(t / t2)
        )
        p = (2**arg) * cos_term - 1 / 2
        # Build bias distribution
        tot_n = 0
        d_sqr_magnetization = 0
        for q in range(n_qubits + 1):
            n = 1 / (n_qubits * (2 ** (p * (q + 1))))
            m_val = (n_qubits - (q << 1)) / n_qubits
            d_sqr_magnetization = d_sqr_magnetization + n * m_val
            tot_n = tot_n + n
        d_sqr_magnetization = d_sqr_magnetization / tot_n
        results.append(d_sqr_magnetization)
    return pm.math.stack(results)


# ---------------------------------------
# Experimental data
# ---------------------------------------
depths = list(range(1, 21))  # array of step depths
dt = 0.25
n_qubits = 16
J = -1.0
h = 2.0
theta = math.pi / 18
# measured mean magnetization
observed_data = np.array(
    [
        0.5331096613947364,
        0.5658914819681634,
        0.49161339327452086,
        0.26643838719925717,
        -0.1362218164168863,
        -0.5142552509805076,
        -0.7123630163195086,
        -0.7984574151694762,
        -0.8345986591490068,
        -0.8438618054926229,
        -0.8316324273198081,
        -0.7921954916336735,
        -0.7035495007623623,
        -0.5142552509805076,
        -0.18417067474298626,
        0.16099531526478772,
        0.3631407695775524,
        0.4196483558435873,
        0.35127212227883103,
        0.13906518757606115,
    ]
)

# ---------------------------------------
# Bayesian fitting with PyMC
# ---------------------------------------
with pm.Model() as model:
    # Priors
    t2 = pm.Uniform("t2", lower=0.125, upper=4)
    omega = pm.Uniform("omega", lower=1.3, upper=1.7)

    # Forward model
    mu = magnetization_model(depths, dt, n_qubits, J, h, theta, t2, omega)

    # Likelihood
    pm.Normal("likelihood", mu=mu, observed=observed_data)

    # Sampling
    trace = pm.sample(
        draws=2000, tune=1000, chains=4, target_accept=0.9, random_seed=42
    )

# Posterior summary
print(az.summary(trace, var_names=["t2", "omega"]))
az.plot_trace(trace, var_names=["t2", "omega"])
