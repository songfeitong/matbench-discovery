"""Local checkpoints use the official sharding and scoring pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import pytest
from ase.build import bulk
from ase.calculators.emt import EMT
from pymatgen.entries.compatibility import MaterialsProject2020Compatibility
from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.io.ase import AseAtomsAdaptor
from pymatviz.enums import Key

import matbench_discovery.data as data_module
import matbench_discovery.discovery as core
import models.run_discovery as runner
from matbench_discovery.calculators import CALCULATORS, CalcSpec
from matbench_discovery.enums import MbdKey

if TYPE_CHECKING:
    from ase import Atoms


def test_local_checkpoint_subset_run_merge_and_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only selected IDs are scored; changing weights cannot reuse even full shards."""
    ids = ["wbm-a", "wbm-b", "wbm-outside"]
    reference = pd.DataFrame(
        {
            Key.mat_id: ids,
            Key.formula: ["Cu", "Cu", "Cu"],
            MbdKey.each_true: [-0.1, 0.2, -0.2],
            MbdKey.e_form_dft: [-0.2, 0.1, -0.3],
            MbdKey.uniq_proto: [True, False, True],
        },
        index=pd.Index(ids, name=str(Key.mat_id)),
    )
    monkeypatch.setattr(data_module, "df_wbm", reference)
    structures = {}
    for material_id in ids:
        atoms = bulk("Cu")
        atoms.info[str(Key.mat_id)] = material_id
        structures[material_id] = atoms

    def load_selected(
        model_key: str,
        *,
        dry_run: bool = False,
        material_ids: list[str] | None = None,
    ) -> dict[str, Atoms]:
        """Supply local structures, so the test requires no WBM download."""
        assert model_key == "prophetv2_local"
        assert not dry_run
        assert material_ids == ids[:2]
        return {key: structures[key] for key in material_ids}

    def factory(device: str, checkpoint: str, dtype: str) -> EMT:
        """Exercise explicit checkpoint and precision forwarding through EMT."""
        assert device == "cpu"
        assert dtype == "float32"
        assert Path(checkpoint).is_file()
        return EMT()

    monkeypatch.setitem(
        CALCULATORS, "prophetv2_local", CalcSpec(factory, requires_checkpoint=True)
    )
    monkeypatch.setattr(runner, "load_wbm_atoms", load_selected)
    entries = pd.DataFrame(
        {
            Key.computed_structure_entry: [
                ComputedStructureEntry(
                    AseAtomsAdaptor.get_structure(structures[key]),
                    -3.0,
                    parameters={"run_type": "GGA"},
                )
                for key in ids[:2]
            ],
        },
        index=ids[:2],
    )
    write_artifacts = core.write_discovery_artifacts

    def write_local_entries(
        merged_run: core.MergedDiscoveryRun,
        *,
        pred_file_path: str,
        geo_opt_file_path: str,
    ) -> core.DiscoveryArtifacts:
        """Use real MP2020 processing with synthetic reference entries."""
        return write_artifacts(
            merged_run,
            pred_file_path=pred_file_path,
            geo_opt_file_path=geo_opt_file_path,
            df_wbm_summary=reference,
            df_wbm_cse=entries,
            compatibility=MaterialsProject2020Compatibility(check_potcar=False),
            elemental_ref_energies={"Cu": -3.0},
        )

    monkeypatch.setattr(runner, "write_discovery_artifacts", write_local_entries)
    checkpoint = tmp_path / "weights.pt"
    checkpoint.write_bytes(b"first-weights")
    selection = tmp_path / "selection.txt"
    selection.write_text("\n".join(ids[:2]) + "\n")
    run_args = [
        "--model",
        "prophetv2_local",
        "--checkpoint",
        str(checkpoint),
        "--dtype",
        "float32",
        "--device",
        "cpu",
        "--material-ids",
        str(selection),
        "--out-dir",
        str(tmp_path / "output"),
        "--n-shards",
        "2",
        "--max-steps",
        "0",
    ]
    for index in range(2):
        assert runner.main([*run_args, "--shard-index", str(index)]) == 0
    assert runner.main([*run_args, "--merge-shards"]) == 0
    report = json.loads((tmp_path / "output/discovery-metrics.json").read_text())
    assert report["scope"] == "wbm-selected"
    assert (report["n_expected"], report["n_present"], report["n_missing"]) == (2, 2, 0)
    assert report["n_converged"] == 0
    assert "selected_set" in report["metrics"]
    assert "full_test_set" not in report["metrics"]
    classes = report["metrics"]["selected_set"]
    assert sum(classes[key] for key in ("TP", "FP", "TN", "FN")) == 2
    assert report["run_metadata"]["checkpoint_sha256"]
    checkpoint.write_bytes(b"different-weights")
    with pytest.raises(ValueError, match="checkpoint_sha256"):
        runner.main([*run_args, "--shard-index", "0"])


def test_local_subset_cannot_write_leaderboard_yaml(tmp_path: Path) -> None:
    """A subset must not be registered as a complete leaderboard result."""
    selected = tmp_path / "ids.txt"
    selected.write_text("wbm-a\n")
    with pytest.raises(SystemExit):
        runner.main(
            [
                "--model",
                "prophetv2_local",
                "--material-ids",
                str(selected),
                "--merge-shards",
                "--write-yaml",
            ]
        )


def test_fixed_selection_preserves_ids_and_rejects_duplicates(tmp_path: Path) -> None:
    """Selection files fix coverage independently of checkpoint and sharding."""
    selected = tmp_path / "ids.txt"
    selected.write_text("wbm-b\nwbm-a\n")
    assert runner.read_material_ids(selected) == ["wbm-b", "wbm-a"]
    selected.write_text("wbm-a\nwbm-a\n")
    with pytest.raises(ValueError, match="unique"):
        runner.read_material_ids(selected)
