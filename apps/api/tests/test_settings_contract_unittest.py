import unittest

from app.core.config import Settings, settings
from app.core.settings import CoreSettings, PolicySettings, RuntimeSettings


class SettingsContractTestCase(unittest.TestCase):
    _MODULE_FIELD_NAMES: dict[str, tuple[str, ...]] = {
        "core": CoreSettings.FIELD_NAMES,
        "policy": PolicySettings.FIELD_NAMES,
        "runtime": RuntimeSettings.FIELD_NAMES,
    }
    _ALL_MAPPED_FIELD_NAMES: tuple[str, ...] = tuple(
        name
        for field_names in _MODULE_FIELD_NAMES.values()
        for name in field_names
    )
    _UNKNOWN_ATTR = "__settings_contract_unknown_field__"

    def setUp(self) -> None:
        self._snapshot = {
            name: getattr(settings, name)
            for name in self._ALL_MAPPED_FIELD_NAMES
        }

    def tearDown(self) -> None:
        for name, value in self._snapshot.items():
            setattr(settings, name, value)

        for module_name in self._MODULE_FIELD_NAMES.keys():
            module_proxy = getattr(settings, module_name)
            if hasattr(module_proxy, self._UNKNOWN_ATTR):
                delattr(module_proxy, self._UNKNOWN_ATTR)

        if hasattr(settings, self._UNKNOWN_ATTR):
            delattr(settings, self._UNKNOWN_ATTR)

    def test_field_name_mapping_guards_against_drift(self) -> None:
        declared_fields = {
            name
            for name, value in vars(Settings).items()
            if not name.startswith("_") and not callable(value)
        }
        mapped_fields = set(self._ALL_MAPPED_FIELD_NAMES)

        self.assertEqual(
            len(self._ALL_MAPPED_FIELD_NAMES),
            len(mapped_fields),
            "FIELD_NAMES contains duplicated entries across modules",
        )
        self.assertSetEqual(
            mapped_fields,
            declared_fields,
            "FIELD_NAMES must fully and exactly map Settings fields",
        )

    def test_bidirectional_sync_between_root_and_module_views(self) -> None:
        for module_name, field_names in self._MODULE_FIELD_NAMES.items():
            module_proxy = getattr(settings, module_name)
            for field_name in field_names:
                self.assertEqual(
                    getattr(module_proxy, field_name),
                    getattr(settings, field_name),
                    f"{module_name}.{field_name} should read from root settings",
                )

                from_root = object()
                setattr(settings, field_name, from_root)
                self.assertIs(
                    getattr(module_proxy, field_name),
                    from_root,
                    f"{module_name}.{field_name} should reflect root writes",
                )

                from_proxy = object()
                setattr(module_proxy, field_name, from_proxy)
                self.assertIs(
                    getattr(settings, field_name),
                    from_proxy,
                    f"root settings should reflect {module_name}.{field_name} writes",
                )

    def test_unknown_field_contract_for_module_views(self) -> None:
        self.assertFalse(hasattr(settings, self._UNKNOWN_ATTR))

        for module_name in self._MODULE_FIELD_NAMES.keys():
            module_proxy = getattr(settings, module_name)

            with self.assertRaises(AttributeError):
                getattr(module_proxy, self._UNKNOWN_ATTR)

            marker = object()
            setattr(module_proxy, self._UNKNOWN_ATTR, marker)
            self.assertIs(getattr(module_proxy, self._UNKNOWN_ATTR), marker)
            self.assertFalse(
                hasattr(settings, self._UNKNOWN_ATTR),
                f"unknown attr set on {module_name} proxy must not leak to root settings",
            )


if __name__ == "__main__":
    unittest.main()
