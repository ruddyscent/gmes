import unittest
from unittest.mock import patch

import numpy as np

import gmes.fdtd as fdtd_module
import gmes.material as material_module
from gmes import (
    FDTD,
    Cartesian,
    Continuous,
    Cpml,
    DefaultMedium,
    Dielectric,
    Drude,
    DrudePole,
    Ez,
    PointSource,
    Shell,
    Sphere,
    TMzFDTD,
)
from gmes.pw_material import (
    ConstElectricParamReal,
    ConstExReal,
    DielectricElectricParamReal,
    DielectricExReal,
    DummyElectricParamReal,
    UpmlElectricParamReal,
    UpmlExReal,
)


class BulkAttachmentTest(unittest.TestCase):
    def dielectric_parameters(self, *values):
        parameters = []
        for value in values:
            parameter = DielectricElectricParamReal()
            parameter.eps_inf = value
            parameters.append(parameter)
        return parameters

    def test_bulk_attachment_preserves_index_parameter_order(self):
        indices = np.array(((2, 0, 0), (0, 0, 0), (1, 0, 0)), dtype=np.intc)
        material = DielectricExReal()

        returned = material.attach_many(
            indices, self.dielectric_parameters(4.0, 2.0, 3.0)
        )

        self.assertIs(returned, material)
        self.assertEqual(material.idx_size(), 3)
        self.assertEqual(material.get_eps_inf((2, 0, 0)), 4.0)
        self.assertEqual(material.get_eps_inf((0, 0, 0)), 2.0)
        self.assertEqual(material.get_eps_inf((1, 0, 0)), 3.0)

    def test_bulk_attachment_matches_coordinate_dependent_individual_attachment(self):
        indices = np.array(((1, 1, 1), (2, 1, 1)), dtype=np.intc)
        parameters = []
        for c1 in (0.25, 0.75):
            parameter = UpmlElectricParamReal()
            parameter.eps_inf = 1
            parameter.d = 0
            parameter.c1 = c1
            parameter.c2 = 1
            parameter.c3 = 0
            parameter.c4 = 1
            parameter.c5 = 1
            parameter.c6 = 0
            parameters.append(parameter)

        bulk = UpmlExReal()
        bulk.attach_many(indices, parameters)
        individual = UpmlExReal()
        for index, parameter in zip(indices, parameters, strict=True):
            individual.attach(index, parameter)

        bulk_fields = [np.zeros((4, 3, 3)) for _ in range(3)]
        individual_fields = [np.zeros((4, 3, 3)) for _ in range(3)]
        bulk_fields[1][2, 2, 1] = 1
        individual_fields[1][2, 2, 1] = 1
        for step in range(3):
            bulk.update_all(*bulk_fields, 1, 1, 1, step)
            individual.update_all(*individual_fields, 1, 1, 1, step)

        for actual, expected in zip(bulk_fields, individual_fields, strict=True):
            np.testing.assert_array_equal(actual, expected)

    def test_bulk_attachment_rejects_invalid_inputs_atomically(self):
        valid = np.array(((0, 0, 0),), dtype=np.intc)
        material = DielectricExReal()
        material.attach_many(valid, self.dielectric_parameters(2.0))

        invalid_cases = (
            ([(1, 0, 0)], self.dielectric_parameters(3.0), TypeError),
            (
                np.array(((1, 0, 0),), dtype=np.int64),
                self.dielectric_parameters(3.0),
                TypeError,
            ),
            (
                np.array((1, 0, 0), dtype=np.intc),
                self.dielectric_parameters(3.0),
                ValueError,
            ),
            (
                np.zeros((1, 4), dtype=np.intc),
                self.dielectric_parameters(3.0),
                ValueError,
            ),
            (
                np.zeros((2, 3), dtype=np.intc)[:, ::-1],
                self.dielectric_parameters(3.0, 4.0),
                ValueError,
            ),
            (
                np.array(((1, 0, 0),), dtype=np.intc),
                self.dielectric_parameters(3.0, 4.0),
                ValueError,
            ),
            (
                np.array(((-1, 0, 0),), dtype=np.intc),
                self.dielectric_parameters(3.0),
                IndexError,
            ),
            (
                np.array(((1, 0, 0), (1, 0, 0)), dtype=np.intc),
                self.dielectric_parameters(3.0, 4.0),
                ValueError,
            ),
            (valid, self.dielectric_parameters(3.0), ValueError),
            (
                np.array(((1, 0, 0),), dtype=np.intc),
                [DummyElectricParamReal()],
                TypeError,
            ),
        )

        for indices, parameters, exception in invalid_cases:
            with self.subTest(exception=exception.__name__, indices=indices):
                with self.assertRaises(exception):
                    material.attach_many(indices, parameters)
                self.assertEqual(material.idx_size(), 1)
                self.assertEqual(material.get_eps_inf((0, 0, 0)), 2.0)

    def test_upper_bounds_are_checked_against_the_update_field(self):
        material = ConstExReal()
        parameter = ConstElectricParamReal()
        parameter.eps_inf = 1
        parameter.value = 2
        material.attach_many(
            np.array(((3, 0, 0),), dtype=np.intc),
            [parameter],
        )

        fields = [np.zeros((3, 1, 1)) for _ in range(3)]
        with self.assertRaisesRegex(IndexError, "out of bounds"):
            material.update_all(*fields, 1, 1, 1, 0)
        self.assertFalse(fields[0].any())


class MaterialMappingFastPathTest(unittest.TestCase):
    def build_simulation(self, bloch=None):
        geometry = [
            DefaultMedium(
                material=Drude(
                    eps_inf=2.0,
                    dps=(DrudePole(omega=1.0, gamma=0.1),),
                )
            ),
            Shell(material=Cpml()),
        ]
        sources = [
            PointSource(
                src_time=Continuous(freq=0.8, width=0.5),
                center=(0, 0, 0),
                component=Ez,
            )
        ]
        kwargs = {} if bloch is None else {"bloch": bloch}
        return TMzFDTD(
            Cartesian(size=(2, 2, 0), resolution=4),
            geometry,
            sources,
            verbose=False,
            **kwargs,
        )

    def assert_material_maps_equal(self, fast, legacy):
        self.assertEqual(fast.pw_material.keys(), legacy.pw_material.keys())
        for component in fast.pw_material:
            fast_updaters = fast.pw_material[component]
            legacy_updaters = legacy.pw_material[component]
            self.assertEqual(fast_updaters.keys(), legacy_updaters.keys())
            getter_name = (
                "get_eps_inf" if component.__name__.startswith("E") else "get_mu_inf"
            )
            for updater_type, fast_updater in fast_updaters.items():
                legacy_updater = legacy_updaters[updater_type]
                self.assertEqual(fast_updater.idx_size(), legacy_updater.idx_size())
                fast_getter = getattr(fast_updater, getter_name)
                legacy_getter = getattr(legacy_updater, getter_name)
                for index in np.ndindex(fast.field[component].shape):
                    self.assertEqual(fast_getter(index), legacy_getter(index))

    def test_fast_mapping_matches_legacy_maps_and_multistep_fields(self):
        for bloch in (None, (0.1, 0.2, 0)):
            with self.subTest(bloch=bloch):
                fast = self.build_simulation(bloch)
                fast.init()
                with patch.object(fdtd_module, "_BUILTIN_MATERIAL_TYPES", ()):
                    legacy = self.build_simulation(bloch)
                    legacy.init()

                self.assert_material_maps_equal(fast, legacy)
                for _ in range(5):
                    fast.step()
                    legacy.step()
                for component in fast.field:
                    np.testing.assert_array_equal(
                        fast.field[component], legacy.field[component]
                    )

    def test_builtin_mapping_constructs_one_updater_per_material_type(self):
        constructor = material_module.CpmlEzReal
        calls = 0

        def counted_constructor():
            nonlocal calls
            calls += 1
            return constructor()

        with patch.object(material_module, "CpmlEzReal", counted_constructor):
            simulation = self.build_simulation()
            simulation.init()

        self.assertEqual(calls, 1)

    def test_custom_material_subclass_uses_legacy_signature(self):
        class CustomDielectric(Dielectric):
            calls = 0

            def get_pw_material_ez(self, idx, coords, underneath=None, cmplx=False):
                type(self).calls += 1
                return super().get_pw_material_ez(idx, coords, underneath, cmplx)

        simulation = TMzFDTD(
            Cartesian(size=(2, 2, 0), resolution=3),
            [DefaultMedium(material=CustomDielectric())],
            verbose=False,
        )
        simulation.init()

        self.assertGreater(CustomDielectric.calls, 1)
        self.assertTrue(simulation.pw_material[Ez])

    def test_inherited_builtin_material_descriptor_uses_fast_path(self):
        class InheritedDielectric(Dielectric):
            pass

        constructor = material_module.DielectricEzReal
        calls = 0

        def counted_constructor():
            nonlocal calls
            calls += 1
            return constructor()

        with patch.object(material_module, "DielectricEzReal", counted_constructor):
            simulation = TMzFDTD(
                Cartesian(size=(2, 2, 0), resolution=3),
                [DefaultMedium(material=InheritedDielectric())],
                verbose=False,
            )
            simulation.init()

        self.assertEqual(calls, 1)

    def test_batched_geometry_mapping_matches_fallback_for_all_components(self):
        def build(size, bloch):
            geometry = [
                DefaultMedium(material=Dielectric(1)),
                Sphere(material=Dielectric(2), radius=0.45),
                Sphere(material=Dielectric(3), radius=0.2),
                Shell(material=Cpml()),
            ]
            kwargs = {} if bloch is None else {"bloch": bloch}
            return FDTD(
                Cartesian(size=size, resolution=3),
                geometry,
                verbose=False,
                **kwargs,
            )

        for size in ((1, 1, 1), (2, 0, 0)):
            for bloch in (None, (0.1, 0.2, 0)):
                with self.subTest(size=size, bloch=bloch):
                    fast = build(size, bloch)
                    fast._MATERIAL_TILE_SIZE = 7
                    fast.init()
                    with patch.object(fdtd_module, "_BUILTIN_GEOMETRY_TYPES", ()):
                        fallback = build(size, bloch)
                        fallback.init()

                    self.assert_material_maps_equal(fast, fallback)
                    for _ in range(3):
                        fast.step()
                        fallback.step()
                    for component in fast.field:
                        np.testing.assert_array_equal(
                            fast.field[component], fallback.field[component]
                        )

    def test_custom_geometry_subclass_uses_pointwise_fallback(self):
        class CustomSphere(Sphere):
            pass

        simulation = FDTD(
            Cartesian(size=(1, 1, 1), resolution=2),
            [
                DefaultMedium(material=Dielectric()),
                CustomSphere(material=Dielectric(2), radius=0.25),
            ],
            verbose=False,
        )

        with patch.object(
            simulation.space,
            "component_coordinate_axes",
            side_effect=AssertionError("batch path must not be used"),
        ):
            simulation.init()

        self.assertTrue(simulation.pw_material)


if __name__ == "__main__":
    unittest.main()
