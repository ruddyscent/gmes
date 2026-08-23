"""Compatibility coverage for the pure-Python material layer."""

import base64
import pickle
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

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


class MaterialCompatibilityTest(unittest.TestCase):
    def load_legacy_pickle(self, name):
        return pickle.loads(base64.b64decode(LEGACY_CYTHON_PICKLES[name]))

    def test_material_is_a_python_module_with_unchanged_exports(self):
        self.assertEqual(Path(material.__file__).suffix, ".py")
        self.assertIs(gmes.Cpml, material.Cpml)
        self.assertIs(gmes.Lorentz, material.Lorentz)

    def test_pml_preserves_material_and_compound_relationships(self):
        self.assertEqual(Pml.__bases__, (Material, Compound))
        self.assertIsInstance(Cpml(), Material)
        self.assertIsInstance(Cpml(), Compound)

    def test_loads_legacy_cython_cpml_pickles(self):
        for initialized in (False, True):
            with self.subTest(initialized=initialized):
                restored = self.load_legacy_pickle(
                    f"cpml_{'initialized' if initialized else 'uninitialized'}"
                )

                self.assertIsInstance(restored, Cpml)
                self.assertEqual(restored.eps_inf, 2)
                self.assertEqual(restored.mu_inf, 3)
                self.assertEqual(restored.initialized, initialized)
                self.assertEqual(restored.m, 3.2)
                self.assertEqual(restored.kappa_max, 1.5)
                self.assertEqual(restored.m_a, 4.1)
                self.assertEqual(restored.a_max, 0.7)
                self.assertEqual(restored.sigma_max_ratio, 0.6)
                if initialized:
                    np.testing.assert_array_equal(restored.center, (1, 2, 3))
                    np.testing.assert_array_equal(restored.half_size, (4, 5, 6))
                    np.testing.assert_array_equal(restored.dw, (0.25, 0.5, 1.0))
                    self.assertEqual(restored.d, 0.75)
                    self.assertEqual(restored.dt, 0.125)

    def test_loads_legacy_cython_lorentz_pickles(self):
        for initialized in (False, True):
            with self.subTest(initialized=initialized):
                restored = self.load_legacy_pickle(
                    f"lorentz_{'initialized' if initialized else 'uninitialized'}"
                )

                self.assertIsInstance(restored, Lorentz)
                self.assertEqual(restored.eps_inf, 2)
                self.assertEqual(restored.mu_inf, 3)
                self.assertEqual(restored.sigma, 0.25)
                self.assertEqual(restored.initialized, initialized)
                self.assertEqual(len(restored.lps), 1)
                self.assertEqual(
                    (restored.lps[0].amp, restored.lps[0].omega, restored.lps[0].gamma),
                    (4, 5, 6),
                )
                if initialized:
                    self.assertEqual(restored.dt, 0.125)
                    self.assertEqual(restored.a.shape, (1, 3))
                    self.assertEqual(restored.c.shape, (3,))

    def test_custom_material_subclasses_construct_initialize_and_pickle(self):
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

        for instance in (cpml, lorentz):
            with self.subTest(material=type(instance).__name__):
                restored = pickle.loads(pickle.dumps(instance))
                self.assertIs(type(restored), type(instance))
                self.assertEqual(restored.eps_inf, instance.eps_inf)
                self.assertEqual(restored.mu_inf, instance.mu_inf)
                self.assertEqual(restored.initialized, instance.initialized)
                self.assertEqual(restored.dt, instance.dt)
                if isinstance(instance, CustomCpml):
                    np.testing.assert_array_equal(restored.center, instance.center)
                    np.testing.assert_array_equal(
                        restored.half_size, instance.half_size
                    )
                    np.testing.assert_array_equal(restored.dw, instance.dw)
                    np.testing.assert_array_equal(
                        restored.sigma_max, instance.sigma_max
                    )
                else:
                    self.assertEqual(
                        (
                            restored.lps[0].amp,
                            restored.lps[0].omega,
                            restored.lps[0].gamma,
                        ),
                        (
                            instance.lps[0].amp,
                            instance.lps[0].omega,
                            instance.lps[0].gamma,
                        ),
                    )
                    np.testing.assert_array_equal(restored.a, instance.a)
                    np.testing.assert_array_equal(restored.c, instance.c)


if __name__ == "__main__":
    unittest.main()
