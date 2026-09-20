"""
Politicas de referencia (baselines) bajo el protocolo oficial de evaluacion.

Sin estos numeros no se puede interpretar el puntaje de un agente entrenado. En
particular, en Space Invaders una politica **constante** que siempre dispara ya
obtiene un puntaje respetable, porque los invasores avanzan hacia el canon y se
colocan solos en la linea de tiro. Cualquier agente entrenado que no supere ese
piso no ha aprendido nada util, por bien que se vea su curva de entrenamiento.

Uso:
    python -m proyecto.baselines
    python -m proyecto.baselines --episodios 10
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Callable

import numpy as np

from .config import CONFIG_BASE, ConfigEntorno
from .envs import crear_entorno_evaluacion

ACCIONES = {0: "NOOP", 1: "FIRE", 2: "RIGHT", 3: "LEFT", 4: "RIGHTFIRE", 5: "LEFTFIRE"}


def correr_politica(
    politica: Callable[[np.ndarray], int],
    cfg: ConfigEntorno,
    n_episodios: int,
    seed: int,
) -> list[float]:
    """Corre `n_episodios` completos y retorna el puntaje real de cada uno."""
    env = crear_entorno_evaluacion(cfg, seed=seed)
    puntajes: list[float] = []
    try:
        obs = env.reset()
        while len(puntajes) < n_episodios:
            obs, _, dones, infos = env.step([politica(obs)])
            if dones[0]:
                puntajes.append(float(infos[0]["episode"]["r"]))
    finally:
        env.close()
    return puntajes


def main() -> None:
    p = argparse.ArgumentParser(description="Mide las politicas de referencia")
    p.add_argument("--episodios", type=int, default=5)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--salida", default="runs/baselines.csv")
    args = p.parse_args()

    rng = np.random.default_rng(0)
    politicas: dict[str, Callable[[np.ndarray], int]] = {
        "aleatoria": lambda o: int(rng.integers(0, 6)),
        **{f"constante:{n}": (lambda o, a=a: a) for a, n in ACCIONES.items()},
    }

    print(f"Protocolo: {CONFIG_BASE.para_evaluacion().resumen()}")
    print(f"{args.episodios} episodios por politica\n")
    print(f"{'politica':18s} {'media':>8} {'std':>7} {'min':>6} {'max':>6}")
    print("-" * 50)

    filas = []
    for nombre, politica in politicas.items():
        r = np.array(correr_politica(politica, CONFIG_BASE, args.episodios, args.seed))
        print(f"{nombre:18s} {r.mean():8.1f} {r.std():7.1f} {r.min():6.0f} {r.max():6.0f}")
        filas.append({
            "politica": nombre,
            "episodios": args.episodios,
            "media": f"{r.mean():.1f}",
            "std": f"{r.std():.1f}",
            "min": f"{r.min():.0f}",
            "max": f"{r.max():.0f}",
        })

    salida = Path(args.salida)
    salida.parent.mkdir(parents=True, exist_ok=True)
    with salida.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)
    print(f"\nGuardado en {salida}")


if __name__ == "__main__":
    main()
