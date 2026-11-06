# Cobaya theory wrapper for hi_class (module 'hiclassy', class HiClass), so it can run next to the patched
# CLASS ('classy') used for the PBH and m_e fits.
#
# By (Anthropic) Claude and Dan Strano
#
# Brans-Dicke is the simplest consistent form of a massless dilaton coupled universally to rest mass: the
# trace (spin-0) part of a rank-2 gravitational potential, coupled to T = trace of the stress-energy tensor.
# Its coupling to matter, relative to gravity, is alpha^2 = 1 / (2 omega_BD + 3). Input parameter: omega_BD
# (sample log10 of it, or alpha^2, and pass omega_BD via a lambda). The Brans-Dicke potential V0 is tuned by
# hi_class to close the energy budget (it acts as the cosmological constant), and phi_0 is tuned so that the
# effective Newton constant for non-relativistic matter today equals the local G (normalize_G_NR).
import sys
import types

import hiclassy
from cobaya.theories.classy import classy as _classy

_classy_module = sys.modules[_classy.__module__]

HICLASS_BD_ARGS = {
    "Omega_Lambda": 0.0,
    "Omega_fld": 0.0,
    "Omega_smg": -1.0,
    "gravity_model": "brans_dicke",
    "M2_tuning_smg": "yes",
    "M2_today_smg": 1.0,
    "normalize_G_NR": "yes",
    "pert_initial_conditions_smg": "zero",
    "a_min_stability_test_smg": 1e-6,
    "a_ini_over_a_today_default": 1e-10,
    "a_ini_test_qs_smg": 1e-10,
    "pert_ic_ini_z_ref_smg": 1e9,
}


class hiclass(_classy):
    def initialize(self):
        shim = types.SimpleNamespace(Class=hiclassy.HiClass,
                                     CosmoComputationError=hiclassy.CosmoComputationError,
                                     CosmoSevereError=hiclassy.CosmoSevereError)
        original = _classy_module.load_external_module
        _classy_module.load_external_module = lambda *a, **k: shim
        try:
            super().initialize()
        finally:
            _classy_module.load_external_module = original

    def set(self, params_values_dict):
        params = dict(params_values_dict)
        omega_bd = params.pop("omega_BD", None)
        if omega_bd is not None:
            # parameters_smg = V0 (tuned), omega_BD, phi_ini (tuned), phi_prime_ini (0 = radiation-era attractor)
            self.extra_args["parameters_smg"] = f"0.7, {float(omega_bd)!r}, 1., 0."
        super().set(params)

    @classmethod
    def get_class_options(cls, input_options={}):
        return super().get_class_options(input_options)
