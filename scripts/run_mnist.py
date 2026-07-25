#!/usr/bin/env python3
"""MNIST common benchmark (REQUIRED for every team).

    python scripts/run_mnist.py --qubits 5 10 15 --n-train 4000 --n-test 2000

Uses real MNIST (784-d) via OpenML when reachable, otherwise the sklearn 8x8
digits set offline. Same TFIM reservoir as the weather track.
"""
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CFG
from src.mnist_benchmark import mnist_scaling

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(RESULTS, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qubits", type=int, nargs="+", default=[5, 10, 15])
    ap.add_argument("--n-train", type=int, default=4000)
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument("--jh", type=float, default=None)
    ap.add_argument("--backend", default="auto")
    args = ap.parse_args()

    jh = args.jh
    if jh is None:
        p = os.path.join(RESULTS, "edge_of_chaos.json")
        jh = json.load(open(p))["best_jh"] if os.path.exists(p) else 1.0

    print(f"MNIST QRC benchmark | qubits={args.qubits} | J/h={jh}")
    rows = mnist_scaling(CFG, qubit_counts=tuple(args.qubits),
                         n_train=args.n_train, n_test=args.n_test,
                         backend=args.backend, jh=jh)
    with open(os.path.join(RESULTS, "mnist.json"), "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nSaved results/mnist.json  (source: {rows[0]['source']})")
    if rows[0]["source"] == "sklearn_digits":
        print("NOTE: offline fallback used 8x8 digits. For the submission, run "
              "with internet so real mnist_784 is fetched.")


if __name__ == "__main__":
    main()
