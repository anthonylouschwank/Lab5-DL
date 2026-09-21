"""
Seleccion del agente para la competencia.

El ranking de la clase se construye con el **maximo de 5 episodios**, no con el
promedio. Son objetivos distintos: una politica con mas varianza puede tener
peor media y aun asi un maximo mas alto. Este modulo compara candidatos con la
metrica que realmente se usa.

Para cada candidato (checkpoint x tipo de politica) se corren N episodios y se
reporta:

  - media y desviacion estandar (la metrica "honesta" del informe);
  - **E[max de 5]**, estimado por bootstrap: se remuestrean 5 episodios con
    reemplazo muchas veces y se promedia el maximo. Es el valor esperado de lo
    que pasaria el dia de la competencia;
  - **p90 de max-de-5**, el escenario con suerte.

Estimar E[max de 5] con bootstrap sobre N=30 episodios es mucho mas estable que
correr 5 episodios y quedarse con ese maximo, que es basicamente una sola
muestra de una distribucion con mucha cola.

Uso:
    python -m proyecto.seleccion --run runs/ppo-base-10M --episodios 30
    python -m proyecto.seleccion --run runs/ppo-base-10M --checkpoints todos
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from sb3_contrib import QRDQN
from stable_baselines3 import A2C, DQN, PPO

from .config import ConfigEntorno
from .evaluate import evaluar

CLASES = {"ppo": PPO, "dqn": DQN, "a2c": A2C, "qrdqn": QRDQN}


def metricas_ranking(
    puntajes: list[float], n_competencia: int = 5, n_bootstrap: int = 20_000,
    semilla: int = 0,
) -> dict[str, float]:
    """Resume una muestra de episodios con la metrica del ranking.

    ``E[max de n]`` se estima remuestreando ``n_competencia`` episodios con
    reemplazo y promediando el maximo.
    """
    r = np.asarray(puntajes, dtype=float)
    rng = np.random.default_rng(semilla)
    muestras = rng.choice(r, size=(n_bootstrap, n_competencia), replace=True)
    maximos = muestras.max(axis=1)
    return {
        "episodios": len(r),
        "media": float(r.mean()),
        "std": float(r.std()),
        "error_std": float(r.std() / np.sqrt(len(r))),
        "max_observado": float(r.max()),
        "esperado_max5": float(maximos.mean()),
        "p90_max5": float(np.percentile(maximos, 90)),
    }


def listar_candidatos(dir_run: Path, cuales: str) -> list[tuple[str, Path]]:
    """Resuelve que archivos de pesos evaluar."""
    candidatos: list[tuple[str, Path]] = []

    mejor = dir_run / "mejor" / "best_model.zip"
    if mejor.exists():
        candidatos.append(("mejor", mejor))
    final = dir_run / "modelo.zip"
    if final.exists():
        candidatos.append(("final", final))

    if cuales == "todos":
        for ck in sorted(
            (dir_run / "checkpoints").glob("ckpt_*_steps.zip"),
            key=lambda p: int(p.stem.split("_")[1]),
        ):
            candidatos.append((f"ckpt-{int(ck.stem.split('_')[1]) // 1_000_000}M", ck))

    return candidatos


def main() -> None:
    p = argparse.ArgumentParser(description="Selecciona el agente para la competencia")
    p.add_argument("--run", required=True)
    p.add_argument("--episodios", type=int, default=30)
    p.add_argument("--seed", type=int, default=2000)
    p.add_argument("--checkpoints", choices=["principales", "todos"], default="principales")
    p.add_argument("--solo", default=None,
                   help="lista separada por comas de candidatos a evaluar, "
                        "ej: ckpt-8M,ckpt-9M,ckpt-10M (util para desempatar finalistas)")
    p.add_argument("--politicas", default="greedy,estocastica",
                   help="lista separada por comas: greedy, estocastica")
    p.add_argument("--salida", default=None, help="CSV de resultados")
    args = p.parse_args()

    dir_run = Path(args.run)
    cfg = ConfigEntorno.cargar(dir_run / "config_entorno.json")
    algo = json.loads(
        (dir_run / "hiperparametros.json").read_text(encoding="utf-8")
    ).get("algoritmo", "ppo")

    candidatos = listar_candidatos(dir_run, args.checkpoints)
    if args.solo:
        querido = {s.strip() for s in args.solo.split(",") if s.strip()}
        desconocidos = querido - {n for n, _ in candidatos}
        if desconocidos:
            raise SystemExit(
                f"Candidatos no encontrados: {sorted(desconocidos)}. "
                f"Disponibles: {[n for n, _ in candidatos]}"
            )
        candidatos = [(n, p) for n, p in candidatos if n in querido]
    if not candidatos:
        raise SystemExit(f"No se encontraron pesos en {dir_run}")

    politicas = [s.strip() for s in args.politicas.split(",") if s.strip()]

    print(f"Entorno : {cfg.para_evaluacion().resumen()}")
    print(f"Episodios por candidato: {args.episodios}  (E[max5] por bootstrap)\n")
    encabezado = (
        f"{'candidato':14s} {'politica':12s} {'media':>8} {'err':>6} "
        f"{'max obs':>8} {'E[max5]':>9} {'p90 max5':>9}"
    )
    print(encabezado)
    print("-" * len(encabezado))

    filas: list[dict[str, Any]] = []
    for nombre, pesos in candidatos:
        modelo = CLASES[algo].load(pesos, device="auto")
        for politica in politicas:
            resultados = evaluar(
                modelo, cfg,
                n_episodios=args.episodios,
                deterministico=(politica == "greedy"),
                seed=args.seed,
            )
            m = metricas_ranking([x["recompensa_total"] for x in resultados])
            print(
                f"{nombre:14s} {politica:12s} {m['media']:8.1f} {m['error_std']:6.1f} "
                f"{m['max_observado']:8.0f} {m['esperado_max5']:9.1f} {m['p90_max5']:9.0f}"
            )
            filas.append({"candidato": nombre, "politica": politica,
                          "pesos": str(pesos), **m})

    mejor = max(filas, key=lambda f: f["esperado_max5"])
    print()
    print(f"Mejor por E[max de 5]: {mejor['candidato']} / {mejor['politica']} "
          f"-> {mejor['esperado_max5']:.1f}")
    print(f"  pesos: {mejor['pesos']}")

    salida = Path(args.salida or (dir_run / "seleccion.csv"))
    with salida.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)
    print(f"\nGuardado en {salida}")


if __name__ == "__main__":
    main()
