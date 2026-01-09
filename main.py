# main.py
import argparse
import torch
from typing import Dict, Tuple, List

from src.io_xyz import read_xyz, System
from src.radial import RadialBasis

from src.training import train_energy_only_ridge, train_energy_forces_adam
from src.evaluation import eval_energy_only, eval_forces
from src.model_io import save_ace_model, load_ace_model

# NEW: prediction
from src.predict import predict_system, write_extxyz


# -----------------------------
# Helpers: species mapping
# -----------------------------
def build_species_mapping(symbols: List[str]) -> Tuple[Dict[str, int], torch.Tensor, List[str]]:
    species_list = sorted(set(symbols))
    species_map = {s: i for i, s in enumerate(species_list)}
    species_id = torch.tensor([species_map[s] for s in symbols], dtype=torch.int64)
    return species_map, species_id, species_list


def add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    p.add_argument("--dtype", type=str, default="float64", choices=["float32", "float64"])


def add_hyperparam_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--rc", type=float, default=3.0)
    p.add_argument("--lmax", type=int, default=2)
    p.add_argument("--nmax", type=int, default=4)
    p.add_argument("--include_g0", action="store_true")
    p.add_argument("--lam", type=float, default=5.0)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Linear ACE (B1+B2) training/evaluation/prediction driver."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # ---------------- train ----------------
    p_train = sub.add_parser("train", help="Train a linear ACE model and save it.")
    p_train.add_argument("--xyz", type=str, required=True, help="Training xyz/extxyz (multi-frame).")
    p_train.add_argument("--out", type=str, default="ace_linear_model.pt", help="Output model path.")
    p_train.add_argument("--train_frac", type=float, default=0.8)

    add_common_args(p_train)
    add_hyperparam_args(p_train)

    # training mode switches
    p_train.add_argument("--use_forces", action="store_true", help="Train with energies+forces using Adam.")
    p_train.add_argument("--epochs", type=int, default=200)
    p_train.add_argument("--lr", type=float, default=1e-2)
    p_train.add_argument("--w_energy", type=float, default=1.0)
    p_train.add_argument("--w_forces", type=float, default=1.0)

    # energy-only ridge
    p_train.add_argument("--ridge_lambda", type=float, default=1e-8)

    # ---------------- eval ----------------
    p_eval = sub.add_parser("eval", help="Evaluate an existing model on a xyz/extxyz file (needs labels).")
    p_eval.add_argument("--xyz", type=str, required=True, help="Eval xyz/extxyz (multi-frame).")
    p_eval.add_argument("--model", type=str, required=True, help="Path to saved model (.pt).")
    p_eval.add_argument("--eval_forces", action="store_true", help="Also evaluate forces if present in xyz.")
    add_common_args(p_eval)

    # ---------------- predict ----------------
    p_pred = sub.add_parser("predict", help="Predict energy+forces and write a new extxyz.")
    p_pred.add_argument("--xyz", type=str, required=True, help="Input xyz/extxyz (may have no labels).")
    p_pred.add_argument("--model", type=str, required=True, help="Path to saved model (.pt).")
    p_pred.add_argument("--out", type=str, required=True, help="Output extxyz with predicted energy+forces.")
    add_common_args(p_pred)

    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device(args.device)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32

    if args.cmd == "train":
        frames = read_xyz(args.xyz, dtype=dtype)
        if len(frames) < 1:
            raise RuntimeError("No frames found in xyz.")

        # Build global species map from all frames
        all_symbols = []
        for sys in frames:
            all_symbols.extend(sys.symbols)
        species_map, _, species_list = build_species_mapping(all_symbols)
        n_species = len(species_list)

        print("Species:", {i: s for i, s in enumerate(species_list)})
        print(f"Frames: total={len(frames)}  train_frac={args.train_frac}")

        # Radial basis (training hyperparams)
        rb = RadialBasis(rc=args.rc, nmax=args.nmax, lam=args.lam, include_g0=args.include_g0)

        # split train/test
        n_train = max(1, int(len(frames) * args.train_frac))
        train_frames = frames[:n_train]
        test_frames = frames[n_train:] if n_train < len(frames) else []

        print(f"Split: train={len(train_frames)}  test={len(test_frames)}")
        print(f"Hyperparams: rc={args.rc} lmax={args.lmax} nmax={args.nmax} include_g0={args.include_g0} lam={args.lam}")

        if args.use_forces:
            for k, sys in enumerate(train_frames):
                if sys.forces is None:
                    raise ValueError(
                        f"Train frame {k} has no forces. Remove --use_forces or provide forces in extxyz."
                    )
            coeff, bias = train_energy_forces_adam(
                frames=train_frames,
                rc=args.rc,
                lmax=args.lmax,
                rb=rb,
                species_map_global=species_map,
                n_species=n_species,
                device=device,
                dtype=dtype,
                epochs=args.epochs,
                lr=args.lr,
                w_energy=args.w_energy,
                w_forces=args.w_forces,
            )
        else:
            coeff, bias = train_energy_only_ridge(
                frames=train_frames,
                rc=args.rc,
                lmax=args.lmax,
                rb=rb,
                species_map_global=species_map,
                n_species=n_species,
                device=device,
                dtype=dtype,
                ridge_lambda=args.ridge_lambda,
            )

        print("Training done.")

        # eval energy (always)
        eval_energy_only(
            train_frames, coeff, bias, args.rc, args.lmax, rb,
            species_map, n_species, device, dtype, tag="train"
        )
        if test_frames:
            eval_energy_only(
                test_frames, coeff, bias, args.rc, args.lmax, rb,
                species_map, n_species, device, dtype, tag="test"
            )

        # optional: eval forces if trained with forces and labels exist
        if args.use_forces:
            eval_forces(
                train_frames, coeff, bias, args.rc, args.lmax, rb,
                species_map, n_species, device, dtype, tag="train"
            )
            if test_frames:
                eval_forces(
                    test_frames, coeff, bias, args.rc, args.lmax, rb,
                    species_map, n_species, device, dtype, tag="test"
                )

        # save model
        save_ace_model(
            path=args.out,
            coeff=coeff,
            bias=bias,
            species_map=species_map,
            species_list=species_list,
            rc=args.rc,
            lmax=args.lmax,
            nmax=args.nmax,
            include_g0=args.include_g0,
            lam=args.lam,
        )
        print(f"Saved model to {args.out}")

    elif args.cmd == "eval":
        frames = read_xyz(args.xyz, dtype=dtype)
        if len(frames) < 1:
            raise RuntimeError("No frames found in xyz.")

        ckpt = load_ace_model(args.model, device=device, dtype=dtype)

        coeff = ckpt["coeff"]
        bias = ckpt["bias"]
        species_map = ckpt["species_map"]
        species_list = ckpt["species_list"]
        n_species = len(species_list)

        # Use hyperparams stored in the model to build RB
        rc = float(ckpt["rc"])
        lmax = int(ckpt["lmax"])
        nmax = int(ckpt["nmax"])
        include_g0 = bool(ckpt["include_g0"])
        lam = float(ckpt["lam"])

        rb = RadialBasis(rc=rc, nmax=nmax, lam=lam, include_g0=include_g0)

        print(f"Loaded model: {args.model}")
        print("Species:", {i: s for i, s in enumerate(species_list)})
        print(f"Eval frames: {len(frames)}")
        print(f"Model hyperparams: rc={rc} lmax={lmax} nmax={nmax} include_g0={include_g0} lam={lam}")

        eval_energy_only(
            frames, coeff, bias, rc, lmax, rb,
            species_map, n_species, device, dtype, tag="eval"
        )

        if args.eval_forces:
            eval_forces(
                frames, coeff, bias, rc, lmax, rb,
                species_map, n_species, device, dtype, tag="eval"
            )

    elif args.cmd == "predict":
        frames = read_xyz(args.xyz, dtype=dtype)
        if len(frames) < 1:
            raise RuntimeError("No frames found in xyz.")

        ckpt = load_ace_model(args.model, device=device, dtype=dtype)

        coeff = ckpt["coeff"]
        bias = ckpt["bias"]
        species_map = ckpt["species_map"]
        species_list = ckpt["species_list"]

        rc = float(ckpt["rc"])
        lmax = int(ckpt["lmax"])
        nmax = int(ckpt["nmax"])
        include_g0 = bool(ckpt["include_g0"])
        lam = float(ckpt["lam"])

        rb = RadialBasis(rc=rc, nmax=nmax, lam=lam, include_g0=include_g0)

        print(f"Loaded model: {args.model}")
        print("Species:", {i: s for i, s in enumerate(species_list)})
        print(f"Predict frames: {len(frames)}")
        print(f"Model hyperparams: rc={rc} lmax={lmax} nmax={nmax} include_g0={include_g0} lam={lam}")

        energies: List[float] = []
        forces_list: List[torch.Tensor] = []

        for sys in frames:
            E, F = predict_system(
                sys=sys,
                coeff=coeff,
                bias=bias,
                species_map=species_map,
                rb=rb,
                rc=rc,
                lmax=lmax,
                device=device,
                dtype=dtype,
            )
            energies.append(E)
            forces_list.append(F)

        write_extxyz(args.out, frames, energies, forces_list)
        print(f"Predictions written to {args.out}")

    else:
        raise RuntimeError("Unknown command.")


if __name__ == "__main__":
    main()
