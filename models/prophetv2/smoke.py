"""Probe ProphetV2 through the shared calculator loader and relaxation code.

This uses small, untrained models and synthetic crystals. It does not produce
benchmark scores or register a leaderboard submission.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import torch
from ase.build import bulk
from ase.calculators.calculator import Calculator
from prophetv2.data import Batch, Data
from prophetv2.models import (
    MLIP,
    ModelConfig,
    MoEModelConfig,
    ProphetCalculator,
    ProphetV2Dense,
    ProphetV2MoE,
    load_mlip,
)

from matbench_discovery.calculators import CALCULATORS, CalcSpec, load_calculator
from matbench_discovery.discovery import RelaxationSettings, relax_atoms


def make_calculator(device: str, checkpoint: str, dtype: str) -> Calculator:
    """Load an exported physical model with explicit deployment precision."""
    if dtype != "float32":
        raise ValueError("This smoke test validates float32 only")
    mlip = load_mlip(checkpoint, device=device, mixed_precision="no")
    if next(mlip.parameters()).dtype != torch.float32:
        raise ValueError("The exported model must contain float32 weights")
    return ProphetCalculator(mlip)


def run_smoke(output_dir: Path, device: str) -> dict[str, object]:
    """Check Dense/MoE export, E/F/S agreement, and two variable-cell steps."""
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    torch.set_float32_matmul_precision("highest")
    CALCULATORS["prophetv2_local"] = CalcSpec(make_calculator, requires_checkpoint=True)
    options = {
        "atomic_numbers": [14],
        "num_layers": 1,
        "l_max": 2,
        "m_max": 1,
        "node_channels": 8,
        "node_hidden_channels": 8,
        "edge_hidden_channels": 8,
        "species_channels": 8,
        "radial_hidden_channels": 8,
        "decoder_hidden_channels": 8,
        "num_radial": 8,
    }
    results: dict[str, object] = {}
    for kind in ("dense", "moe"):
        model = (
            ProphetV2Dense(ModelConfig(**options))
            if kind == "dense"
            else ProphetV2MoE(MoEModelConfig(**options))
        )
        mlip = MLIP(model, element_references={14: -5.0}, energy_scale=1.7)
        checkpoint = output_dir / f"{kind}-untrained.pt"
        torch.save({"config": mlip.config, "model": mlip.state_dict()}, checkpoint)
        calculator = load_calculator(
            "prophetv2_local",
            device=device,
            dtype="float32",
            checkpoint=str(checkpoint),
        )
        atoms = bulk("Si", "diamond", a=5.43)
        atoms.positions[1] += [0.03, -0.02, 0.01]
        atoms.info["material_id"] = f"synthetic-si-{kind}"
        atoms.calc = calculator
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        stress = atoms.get_stress(voigt=False)
        batch = Batch.from_data_list([Data.from_atoms(atoms)]).to(device)
        reference = mlip.to(device).eval()(
            batch, properties=("energy", "forces", "stress"), create_graph=False
        )
        for actual, expected in (
            (energy, reference["energy"][0].detach().cpu().numpy()),
            (forces, reference["forces"].detach().cpu().numpy()),
            (stress, reference["stress"][0].detach().cpu().numpy()),
        ):
            if not np.isfinite(actual).all():
                raise ValueError(f"{kind} produced nonfinite predictions")
            np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
        # A tight smoke-only tolerance exercises both optimizer steps.
        record = relax_atoms(
            atoms, calculator, RelaxationSettings(max_force=1e-8, max_steps=2)
        )
        if record.structure is None or not np.isfinite(record.energy):
            raise ValueError(f"{kind} failed the shared relaxation path")
        if record.n_steps != 2:
            raise ValueError(f"{kind} did not exercise both relaxation steps")
        results[kind] = {
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "energy_eV": energy,
            "forces_shape": list(forces.shape),
            "stress_shape": list(stress.shape),
            "max_force_eV_per_A": float(np.abs(forces).max()),
            "relaxation_steps": record.n_steps,
            "final_energy_eV": record.energy,
            "converged": record.converged,
            "direct_mlip_agreement": True,
        }
    report = {
        "scope": "untrained Dense/MoE; synthetic Si; no benchmark scores",
        "device": device,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(device),
        "results": results,
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    print(json.dumps(run_smoke(args.output_dir, args.device), indent=2))
