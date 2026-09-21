"""
Evaluacion de un agente entrenado sobre ALE/SpaceInvaders-v5.

Este es el script de la competencia: carga los pesos, reconstruye el entorno con
EXACTAMENTE el mismo preprocesamiento del entrenamiento (leido del JSON que
quedo junto al modelo), corre 5 episodios con politica greedy, reporta el
puntaje de cada uno y graba el video. Sin pasos manuales.

Uso:
    python -m proyecto.evaluate --run runs/ppo-base
    python -m proyecto.evaluate --run runs/ppo-base --mejor --video
    python -m proyecto.evaluate --run runs/ppo-base --episodios 5 --estocastico

Adapta las funciones del Laboratorio #5 (crear_entorno / ejecutar_episodio /
generar_video_agente) al caso de una politica aprendida: la diferencia es que la
observacion ya no es el frame RGB crudo sino el tensor apilado de 4x84x84 que
espera la red, por lo que el entorno se construye desde `proyecto.envs`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sb3_contrib import QRDQN
from stable_baselines3 import A2C, DQN, PPO

from .config import ConfigEntorno
from .envs import crear_entorno_evaluacion

CLASES = {"ppo": PPO, "dqn": DQN, "a2c": A2C, "qrdqn": QRDQN}


def cargar_agente(
    dir_run: str | Path,
    usar_mejor: bool = False,
    pesos: str | Path | None = None,
):
    """Carga modelo + configuracion de entorno de una corrida.

    ``pesos`` permite apuntar a un checkpoint concreto. Hace falta porque el
    mejor agente no siempre es ``best_model.zip``: ese archivo se elige con
    evaluaciones intermedias de pocos episodios, que sobrestiman al ganador.

    Returns
    -------
    (modelo, cfg, ruta_pesos)
    """
    dir_run = Path(dir_run)

    ruta_cfg = dir_run / "config_entorno.json"
    if not ruta_cfg.exists():
        raise FileNotFoundError(
            f"No se encontro {ruta_cfg}. Sin la configuracion del entorno no se "
            "puede garantizar que la evaluacion use el mismo preprocesamiento."
        )
    cfg = ConfigEntorno.cargar(ruta_cfg)

    hp: dict[str, Any] = json.loads(
        (dir_run / "hiperparametros.json").read_text(encoding="utf-8")
    )
    algo = hp.get("algoritmo", "ppo")

    if pesos is not None:
        ruta_pesos = Path(pesos)
    elif usar_mejor:
        ruta_pesos = dir_run / "mejor" / "best_model.zip"
    else:
        ruta_pesos = dir_run / "modelo.zip"

    if not ruta_pesos.exists():
        disponibles = sorted(p.name for p in dir_run.rglob("*.zip"))
        raise FileNotFoundError(
            f"No existe {ruta_pesos}. Disponibles en {dir_run}: {disponibles}"
        )

    modelo = CLASES[algo].load(ruta_pesos, device="auto")
    return modelo, cfg, ruta_pesos


def evaluar(
    modelo,
    cfg: ConfigEntorno,
    n_episodios: int = 5,
    deterministico: bool = True,
    seed: int = 1000,
    video_folder: str | Path | None = None,
    name_prefix: str = "agente-evaluacion",
) -> list[dict[str, Any]]:
    """Corre `n_episodios` completos y retorna las metricas de cada uno.

    Un episodio son las 3 vidas completas y la recompensa es el puntaje real del
    juego, porque el entorno se construye con ``cfg.para_evaluacion()``.
    """
    env = crear_entorno_evaluacion(
        cfg,
        seed=seed,
        video_folder=video_folder,
        name_prefix=name_prefix,
        # Grabar solo los episodios pedidos: tras el ultimo, el VecEnv reinicia
        # solo y ese reset dejaria un .mp4 vacio.
        episode_trigger=lambda ep: ep < n_episodios,
    )

    resultados: list[dict[str, Any]] = []
    try:
        obs = env.reset()
        pasos = 0
        while len(resultados) < n_episodios:
            accion, _ = modelo.predict(obs, deterministic=deterministico)
            obs, _, dones, infos = env.step(accion)
            pasos += 1
            if dones[0]:
                # Monitor deja aqui el return y la longitud del episodio real.
                ep = infos[0].get("episode", {})
                resultados.append({
                    "episodio": len(resultados),
                    "pasos": int(ep.get("l", pasos)),
                    "recompensa_total": float(ep.get("r", float("nan"))),
                })
                pasos = 0
    finally:
        # Indispensable para que RecordVideo termine de escribir los .mp4.
        env.close()

    return resultados


def resumir(resultados: list[dict[str, Any]]) -> str:
    """Tabla de texto con el detalle por episodio y el resumen del ranking."""
    r = np.array([x["recompensa_total"] for x in resultados], dtype=float)
    p = np.array([x["pasos"] for x in resultados], dtype=float)

    lineas = [
        f"{'Episodio':>8} | {'Pasos':>7} | {'Puntaje':>8}",
        f"{'-'*8}-+-{'-'*7}-+-{'-'*8}",
    ]
    for x in resultados:
        lineas.append(
            f"{x['episodio']:>8} | {x['pasos']:>7} | {x['recompensa_total']:>8.0f}"
        )
    lineas += [
        f"{'-'*8}-+-{'-'*7}-+-{'-'*8}",
        f"{'media':>8} | {p.mean():>7.0f} | {r.mean():>8.1f}",
        "",
        f"  Puntaje medio : {r.mean():.1f} +- {r.std():.1f}",
        f"  Puntaje maximo: {r.max():.0f}   <- metrica del ranking de la clase",
    ]
    return "\n".join(lineas)


def main() -> None:
    p = argparse.ArgumentParser(description="Evalua un agente entrenado")
    p.add_argument("--run", required=True, help="carpeta de la corrida, ej: runs/ppo-base")
    p.add_argument("--mejor", action="store_true", help="usar mejor/best_model.zip")
    p.add_argument("--pesos", default=None,
                   help="ruta a un .zip concreto (tiene prioridad sobre --mejor)")
    p.add_argument("--episodios", type=int, default=5)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--estocastico", action="store_true",
                   help="muestrear de la politica en vez de tomar el argmax")
    p.add_argument("--video", action="store_true", help="grabar los episodios")
    p.add_argument("--video-folder", default=None,
                   help="carpeta de salida (por defecto <run>/videos)")
    args = p.parse_args()

    modelo, cfg, pesos = cargar_agente(args.run, usar_mejor=args.mejor, pesos=args.pesos)

    carpeta_video = None
    if args.video:
        carpeta_video = Path(args.video_folder or (Path(args.run) / "videos"))

    print("=" * 62)
    print(f"  Pesos    : {pesos}")
    print(f"  Entorno  : {cfg.para_evaluacion().resumen()}")
    print(f"  Politica : {'estocastica' if args.estocastico else 'greedy (argmax)'}")
    print(f"  Episodios: {args.episodios}")
    print("=" * 62)

    resultados = evaluar(
        modelo, cfg,
        n_episodios=args.episodios,
        deterministico=not args.estocastico,
        seed=args.seed,
        video_folder=carpeta_video,
        name_prefix=f"{Path(args.run).name}-eval",
    )

    print()
    print(resumir(resultados))

    if carpeta_video is not None:
        videos = sorted(carpeta_video.glob("*.mp4"))
        print(f"\n  Videos en {carpeta_video}:")
        for v in videos:
            print(f"    {v.name}  ({v.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
