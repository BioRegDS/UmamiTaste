#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem
from torch_geometric.data import Batch as PyGBatch

from kmol.core.config import Config
from kmol.data.streamers import GeneralStreamer
from kmol.model.executors import Pipeliner


def get_single_graph_and_smiles(graphs):
    if isinstance(graphs, PyGBatch):
        data_list = graphs.to_data_list()
        if len(data_list) != 1:
            raise ValueError(f"Expected one molecule, got {len(data_list)}")
        sample = data_list[0]
    else:
        sample = graphs

    smiles = getattr(sample, "smiles", None)
    if smiles is None:
        raise ValueError("Graph object does not contain .smiles")
    return sample, smiles


def scalar_output(network, batch, graph_key, target):
    out = network(batch.inputs)
    return out.reshape(-1)[target]


def manual_integrated_gradients(
    network,
    batch,
    graph_key="graph",
    target=0,
    n_steps=100,
):
    """
    Integrated Gradients for node-feature matrix x [n_atoms, n_features].

    We explicitly traverse the straight-line path from a zero node-feature
    baseline to the true node features. This avoids Captum treating n_atoms
    as the minibatch dimension.

    Other graph information (edge_index, edge_attr, molecular features, etc.)
    is held fixed, so the attribution is specifically to node features.
    """
    graph = batch.inputs[graph_key]
    x_input = graph.x.detach().clone()
    baseline = torch.zeros_like(x_input)

    # End-point outputs
    graph.x = x_input
    with torch.no_grad():
        input_logit_t = scalar_output(network, batch, graph_key, target)
        input_logit = float(input_logit_t.detach().cpu())

    graph.x = baseline
    with torch.no_grad():
        baseline_logit_t = scalar_output(network, batch, graph_key, target)
        baseline_logit = float(baseline_logit_t.detach().cpu())

    # Trapezoidal integration over alpha in [0, 1].
    alphas = torch.linspace(
        0.0, 1.0, steps=n_steps + 1, device=x_input.device, dtype=x_input.dtype
    )

    grads = []
    for alpha in alphas:
        x_alpha = (baseline + alpha * (x_input - baseline)).detach()
        x_alpha.requires_grad_(True)
        graph.x = x_alpha

        network.zero_grad(set_to_none=True)
        y = scalar_output(network, batch, graph_key, target)
        grad = torch.autograd.grad(
            y,
            x_alpha,
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )[0]
        grads.append(grad.detach())

    grads = torch.stack(grads, dim=0)  # [steps+1, atoms, features]

    # Trapezoidal rule: average adjacent gradients then mean over intervals.
    avg_grads = ((grads[:-1] + grads[1:]) / 2.0).mean(dim=0)
    attrs = (x_input - baseline) * avg_grads

    graph.x = x_input

    ig_total = float(attrs.sum().detach().cpu())
    output_difference = input_logit - baseline_logit
    completeness_error = output_difference - ig_total

    return attrs, input_logit, baseline_logit, output_difference, completeness_error


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--n-steps", type=int, default=100)
    p.add_argument("--target", type=int, default=0)
    p.add_argument("--graph-input-key", default="graph")
    args = p.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    config = Config.from_file(args.config, "visualize")
    streamer = GeneralStreamer(config=config)
    loader = streamer.get(
        split_name=config.test_split,
        batch_size=1,
        shuffle=False,
        mode=GeneralStreamer.Mode.TEST,
    )

    pipeliner = Pipeliner(config=config)
    network = pipeliner.get_network()
    network.eval()

    atom_rows = []
    mol_rows = []

    for local_idx, batch in enumerate(loader.dataset):
        pipeliner._to_device(batch)

        graphs = batch.inputs[args.graph_input_key]
        _, smiles = get_single_graph_and_smiles(graphs)

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"RDKit failed for SMILES: {smiles}")

        attrs, input_logit, baseline_logit, output_difference, completeness_error = (
            manual_integrated_gradients(
                network=network,
                batch=batch,
                graph_key=args.graph_input_key,
                target=args.target,
                n_steps=args.n_steps,
            )
        )

        a = attrs.detach().cpu().numpy()
        signed_atom = a.sum(axis=1)
        abs_atom = np.abs(a).sum(axis=1)

        max_signed_abs = np.max(np.abs(signed_atom)) if len(signed_atom) else 0.0
        max_abs = np.max(abs_atom) if len(abs_atom) else 0.0

        signed_norm = signed_atom / max_signed_abs if max_signed_abs > 0 else signed_atom
        abs_norm = abs_atom / max_abs if max_abs > 0 else abs_atom

        if mol.GetNumAtoms() != a.shape[0]:
            raise ValueError(
                f"Atom count mismatch for {smiles}: "
                f"RDKit={mol.GetNumAtoms()}, graph={a.shape[0]}"
            )

        label = float(batch.outputs.detach().cpu().reshape(-1)[args.target])
        dataset_id = (
            batch.ids[0]
            if hasattr(batch, "ids") and len(batch.ids)
            else local_idx
        )
        input_prob = float(torch.sigmoid(torch.tensor(input_logit)))

        mol_rows.append({
            "fold": args.fold,
            "dataset_id": dataset_id,
            "local_index": local_idx,
            "smiles": smiles,
            "label": label,
            "input_logit": input_logit,
            "input_probability": input_prob,
            "baseline_logit": baseline_logit,
            "output_difference": output_difference,
            "ig_total": float(a.sum()),
            "completeness_error": completeness_error,
            "abs_completeness_error": abs(completeness_error),
            "relative_completeness_error": (
                abs(completeness_error) / max(abs(output_difference), 1e-8)
            ),
            "n_atoms": mol.GetNumAtoms(),
            "n_steps": args.n_steps,
            "ig_method": "manual_trapezoidal_node_features",
        })

        for atom_idx, atom in enumerate(mol.GetAtoms()):
            atom_rows.append({
                "fold": args.fold,
                "dataset_id": dataset_id,
                "local_index": local_idx,
                "smiles": smiles,
                "label": label,
                "input_probability": input_prob,
                "atom_index": atom_idx,
                "atom_symbol": atom.GetSymbol(),
                "atomic_number": atom.GetAtomicNum(),
                "formal_charge": atom.GetFormalCharge(),
                "is_aromatic": int(atom.GetIsAromatic()),
                "signed_ig": float(signed_atom[atom_idx]),
                "abs_ig": float(abs_atom[atom_idx]),
                "signed_ig_norm": float(signed_norm[atom_idx]),
                "abs_ig_norm": float(abs_norm[atom_idx]),
                "feature_ig_json": json.dumps(
                    [float(v) for v in a[atom_idx].tolist()]
                ),
            })

    atom_path = outdir / f"fold_{args.fold}_atom_ig.csv"
    mol_path = outdir / f"fold_{args.fold}_molecule_ig_summary.csv"

    if atom_rows:
        with atom_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(atom_rows[0].keys()))
            w.writeheader()
            w.writerows(atom_rows)

    if mol_rows:
        with mol_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(mol_rows[0].keys()))
            w.writeheader()
            w.writerows(mol_rows)

    print(f"Wrote: {atom_path}")
    print(f"Wrote: {mol_path}")
    print(f"Molecules: {len(mol_rows)}")

    if mol_rows:
        abs_err = np.array([r["abs_completeness_error"] for r in mol_rows], dtype=float)
        rel_err = np.array([r["relative_completeness_error"] for r in mol_rows], dtype=float)
        print(f"Median |completeness error|: {np.median(abs_err):.6g}")
        print(f"95th pct |completeness error|: {np.percentile(abs_err, 95):.6g}")
        print(f"Median relative completeness error: {np.median(rel_err):.6g}")
        print(f"95th pct relative completeness error: {np.percentile(rel_err, 95):.6g}")


if __name__ == "__main__":
    main()
