"""Compatibility coverage for the pure-Python material layer."""

import base64
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import gmes
from gmes import material
from gmes.material import Cpml, Lorentz, LorentzPole, Pml
from gmes.pygeom import Compound, Material

# These protocol-4 payloads were produced by the Cython gmes.material module at
# bf1b7fb, immediately before it was replaced by the Python implementation.
LEGACY_CYTHON_PICKLES = {
    "cpml_uninitialized": (
        "gASVswAAAAAAAACMDWdtZXMubWF0ZXJpYWyUjARDcG1slJOUKVKUfZQojAdlcHNfaW5m"
        "lEdAAAAAAAAAAIwGbXVfaW5mlEdACAAAAAAAAIwLaW5pdGlhbGl6ZWSUiYwBbZRHQAmZ"
        "mZmZmZqMCWthcHBhX21heJRHP/gAAAAAAACMA21fYZRHQBBmZmZmZmaMBWFfbWF4lEc/"
        "5mZmZmZmZowPc2lnbWFfbWF4X3JhdGlvlEc/4zMzMzMzM3ViLg=="
    ),
    "cpml_initialized": (
        "gASVNgIAAAAAAACMDWdtZXMubWF0ZXJpYWyUjARDcG1slJOUKVKUfZQojAdlcHNfaW5m"
        "lEdAAAAAAAAAAIwGbXVfaW5mlEdACAAAAAAAAIwLaW5pdGlhbGl6ZWSUiIwGY2VudGVy"
        "lIwWbnVtcHkuX2NvcmUubXVsdGlhcnJheZSMDF9yZWNvbnN0cnVjdJSTlIwFbnVtcHmU"
        "jAduZGFycmF5lJOUSwCFlEMBYpSHlFKUKEsBSwOFlGgMjAVkdHlwZZSTlIwCZjiUiYiH"
        "lFKUKEsDjAE8lE5OTkr/////Sv////9LAHSUYolDGAAAAAAAAPA/AAAAAAAAAEAAAAAA"
        "AAAIQJR0lGKMCWhhbGZfc2l6ZZRoC2gOSwCFlGgQh5RSlChLAUsDhZRoGIlDGAAAAAAA"
        "ABBAAAAAAAAAFEAAAAAAAAAYQJR0lGKMAWSURz/oAAAAAAAAjAJkdJRHP8AAAAAAAACM"
        "AmR3lGgLaA5LAIWUaBCHlFKUKEsBSwOFlGgYiUMYAAAAAAAA0D8AAAAAAADgPwAAAAAA"
        "APA/lHSUYowJc2lnbWFfbWF4lGgLaA5LAIWUaBCHlFKUKEsBSwOFlGgYiUMYNfEo+j9W"
        "GkA18Sj6P1YKQDXxKPo/Vvo/lHSUYowBbZRHQAmZmZmZmZqMCWthcHBhX21heJRHP/gA"
        "AAAAAACMA21fYZRHQBBmZmZmZmaMBWFfbWF4lEc/5mZmZmZmZowPc2lnbWFfbWF4X3Jh"
        "dGlvlEc/4zMzMzMzM3ViLg=="
    ),
    "lorentz_uninitialized": (
        "gASVvQAAAAAAAACMDWdtZXMubWF0ZXJpYWyUjAdMb3JlbnR6lJOUKVKUfZQojAdlcHNf"
        "aW5mlEdAAAAAAAAAAIwGbXVfaW5mlEdACAAAAAAAAIwFc2lnbWGURz/QAAAAAAAAjANs"
        "cHOUaACMC0xvcmVudHpQb2xllJOUKYGUfZQojANhbXCUR0AQAAAAAAAAjAVvbWVnYZRH"
        "QBQAAAAAAACMBWdhbW1hlEdAGAAAAAAAAHVihZSMC2luaXRpYWxpemVklIl1Yi4="
    ),
    "lorentz_initialized": (
        "gASVqwEAAAAAAACMDWdtZXMubWF0ZXJpYWyUjAdMb3JlbnR6lJOUKVKUfZQojAdlcHNf"
        "aW5mlEdAAAAAAAAAAIwGbXVfaW5mlEdACAAAAAAAAIwFc2lnbWGURz/QAAAAAAAAjANs"
        "cHOUaACMC0xvcmVudHpQb2xllJOUKYGUfZQojANhbXCUR0AQAAAAAAAAjAVvbWVnYZRH"
        "QBQAAAAAAACMBWdhbW1hlEdAGAAAAAAAAHVihZSMC2luaXRpYWxpemVklIiMAmR0lEc/"
        "wAAAAAAAAIwBYZSMFm51bXB5Ll9jb3JlLm11bHRpYXJyYXmUjAxfcmVjb25zdHJ1Y3SU"
        "k5SMBW51bXB5lIwHbmRhcnJheZSTlEsAhZRDAWKUh5RSlChLAUsBSwOGlGgXjAVkdHlw"
        "ZZSTlIwCZjiUiYiHlFKUKEsDjAE8lE5OTkr/////Sv////9LAHSUYolDGBdddNFFF92/"
        "uuiiiy668j8vuuiiiy7yP5R0lGKMAWOUaBZoGUsAhZRoG4eUUpQoSwFLA4WUaCOJQxjw"
        "B/wBf8CvP/AH/AF/wN+/4A/4A/6A7z+UdJRidWIu"
    ),
}


class CustomCpml(Cpml):
    pass


class CustomLorentz(Lorentz):
    pass


class TestMaterialCompatibility:
    def load_legacy_pickle(self, name):
        return pickle.loads(base64.b64decode(LEGACY_CYTHON_PICKLES[name]))

    def test_material_is_a_python_module_with_unchanged_exports(self):
        assert Path(material.__file__).suffix == ".py"
        assert gmes.Cpml is material.Cpml
        assert gmes.Lorentz is material.Lorentz

    def test_pml_preserves_material_and_compound_relationships(self):
        assert Pml.__bases__ == (Material, Compound)
        assert isinstance(Cpml(), Material)
        assert isinstance(Cpml(), Compound)

    @pytest.mark.parametrize(
        "initialized", (False, True), ids=("before-init", "after-init")
    )
    def test_loads_legacy_cython_cpml_pickles(self, initialized):
        restored = self.load_legacy_pickle(
            f"cpml_{'initialized' if initialized else 'uninitialized'}"
        )

        assert isinstance(restored, Cpml)
        assert restored.eps_inf == 2
        assert restored.mu_inf == 3
        assert restored.initialized == initialized
        assert restored.m == 3.2
        assert restored.kappa_max == 1.5
        assert restored.m_a == 4.1
        assert restored.a_max == 0.7
        assert restored.sigma_max_ratio == 0.6
        if initialized:
            np.testing.assert_array_equal(restored.center, (1, 2, 3))
            np.testing.assert_array_equal(restored.half_size, (4, 5, 6))
            np.testing.assert_array_equal(restored.dw, (0.25, 0.5, 1.0))
            assert restored.d == 0.75
            assert restored.dt == 0.125

    @pytest.mark.parametrize(
        "initialized", (False, True), ids=("before-init", "after-init")
    )
    def test_loads_legacy_cython_lorentz_pickles(self, initialized):
        restored = self.load_legacy_pickle(
            f"lorentz_{'initialized' if initialized else 'uninitialized'}"
        )

        assert isinstance(restored, Lorentz)
        assert restored.eps_inf == 2
        assert restored.mu_inf == 3
        assert restored.sigma == 0.25
        assert restored.initialized == initialized
        assert len(restored.lps) == 1
        assert (restored.lps[0].amp, restored.lps[0].omega, restored.lps[0].gamma) == (
            4,
            5,
            6,
        )
        if initialized:
            assert restored.dt == 0.125
            assert restored.a.shape == (1, 3)
            assert restored.c.shape == (3,)

    @pytest.mark.parametrize("instance", ("cpml", "lorentz"), ids=("cpml", "lorentz"))
    def test_custom_material_subclasses_construct_initialize_and_pickle(self, instance):
        cpml = CustomCpml(eps_inf=2)
        cpml.init(
            SimpleNamespace(dt=0.125, dr=(0.25, 0.5, 1.0)),
            ((1, 2, 3), (4, 5, 6), 0.75),
        )
        lorentz = CustomLorentz(
            eps_inf=2,
            lps=(LorentzPole(amp=4, omega=5, gamma=6),),
        )
        lorentz.init(SimpleNamespace(dt=0.125))

        instance = cpml if instance == "cpml" else lorentz
        restored = pickle.loads(pickle.dumps(instance))
        assert type(restored) is type(instance)
        assert restored.eps_inf == instance.eps_inf
        assert restored.mu_inf == instance.mu_inf
        assert restored.initialized == instance.initialized
        assert restored.dt == instance.dt
        if isinstance(instance, CustomCpml):
            np.testing.assert_array_equal(restored.center, instance.center)
            np.testing.assert_array_equal(restored.half_size, instance.half_size)
            np.testing.assert_array_equal(restored.dw, instance.dw)
            np.testing.assert_array_equal(restored.sigma_max, instance.sigma_max)
        else:
            assert (
                restored.lps[0].amp,
                restored.lps[0].omega,
                restored.lps[0].gamma,
            ) == (
                instance.lps[0].amp,
                instance.lps[0].omega,
                instance.lps[0].gamma,
            )
            np.testing.assert_array_equal(restored.a, instance.a)
            np.testing.assert_array_equal(restored.c, instance.c)
