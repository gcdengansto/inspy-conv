import pytest

import inspy.__main__ as cli


@pytest.mark.parametrize('arguments', [[], ['-fit']])
def test_cli_runs_unified_fit_gui_by_default_and_with_fit_flag(
        monkeypatch, arguments):
    imported = []
    runs = []

    class FitModule:
        @staticmethod
        def main():
            runs.append(True)

    def fake_import(module_path, package=None):
        imported.append((module_path, package))
        return FitModule

    monkeypatch.setattr(cli, 'import_module', fake_import)

    cli.main(arguments)

    assert imported == [('.gui.gui_convfit_uf', 'inspy')]
    assert runs == [True]


def test_cli_rejects_resolution_gui_flag(monkeypatch):
    imported = []
    monkeypatch.setattr(
        cli, 'import_module',
        lambda *args, **kwargs: imported.append((args, kwargs)))

    with pytest.raises(SystemExit) as exc_info:
        cli.main(['-rescalc'])

    assert exc_info.value.code == 2
    assert not imported


def test_public_main_entrypoint_is_the_unified_fit_gui():
    import inspy

    assert inspy.main.__module__ == 'inspy.gui.gui_convfit_uf'
