"""
Analisis cualitativo del agente entrenado (seccion 2.4 del informe).

Responde con datos, no con impresiones, a tres preguntas:

  1. **Que acciones usa?** La distribucion de acciones frente a las politicas de
     referencia dice si el agente aprendio a moverse con intencion o degenero en
     algo casi constante.
  2. **Como se distribuyen los puntajes?** Un histograma multimodal revela que el
     agente tiene "modos de fallo" discretos (morir en la primera oleada frente a
     limpiarla) en vez de una degradacion continua.
  3. **Donde se posiciona la nave?** Se rastrea la posicion horizontal del canon
     en los videos ya grabados, con la misma deteccion por color del Laboratorio
     #5. Si se concentra en un extremo, el agente encontro un rincon seguro.

Uso:
    python -m proyecto.analisis --run runs/ppo-base-10M \\
        --pesos runs/ppo-base-10M/checkpoints/ckpt_9000000_steps.zip --episodios 30
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from .config import ConfigEntorno
from .envs import crear_entorno_evaluacion
from .evaluate import cargar_agente

ACCIONES = ["NOOP", "FIRE", "RIGHT", "LEFT", "RIGHTFIRE", "LEFTFIRE"]

# Colores de sprites (constantes del emulador; ver ale_utils.py del Lab 5).
_COLOR_NAVE = np.array([50, 132, 50], dtype=np.uint8)
_COLOR_ALIENS = np.array([134, 134, 29], dtype=np.uint8)
_FILAS_NAVE = slice(180, 196)
_FILAS_ALIENS = slice(25, 160)


def perfil_de_conducta(modelo, cfg: ConfigEntorno, n_episodios: int, seed: int):
    """Corre episodios registrando acciones y resultados por episodio."""
    env = crear_entorno_evaluacion(cfg, seed=seed)
    acciones = Counter()
    episodios: list[dict] = []
    try:
        obs = env.reset()
        while len(episodios) < n_episodios:
            a, _ = modelo.predict(obs, deterministic=True)
            acciones[int(a[0])] += 1
            obs, _, dones, infos = env.step(a)
            if dones[0]:
                ep = infos[0]["episode"]
                episodios.append({"puntaje": float(ep["r"]), "pasos": int(ep["l"])})
    finally:
        env.close()
    return acciones, episodios


def posicion_de_la_nave(carpeta_videos: Path, tolerancia: int = 25) -> np.ndarray:
    """Posicion horizontal del canon a lo largo de los videos ya grabados.

    Reutiliza la deteccion por color del Laboratorio #5 sobre los frames RGB, que
    es lo unico que permite ver la conducta espacial: la observacion que recibe
    la red (84x84 en gris) ya perdio esa informacion de forma legible.

    A diferencia del Lab 5, aqui la comparacion es **por tolerancia y no exacta**.
    Los videos estan codificados en H.264, que es compresion con perdida: el
    verde (50,132,50) de la nave llega al archivo como (50,115,61) o similar. Una
    igualdad exacta no encuentra un solo pixel.
    """
    import imageio.v3 as iio

    objetivo = _COLOR_NAVE.astype(int)
    posiciones: list[float] = []
    for video in sorted(carpeta_videos.glob("*.mp4")):
        for i, frame in enumerate(iio.imiter(str(video), plugin="FFMPEG")):
            if i % 4:            # 1 de cada 4 frames basta y es 4x mas rapido
                continue
            banda = frame[_FILAS_NAVE].astype(int)
            m = np.abs(banda - objetivo).max(axis=-1) <= tolerancia
            cols = np.flatnonzero(m.any(axis=0))
            if cols.size:
                posiciones.append(float(cols.mean()))
    return np.asarray(posiciones)


def figura_conducta(
    acciones: Counter, episodios: list[dict], posiciones: np.ndarray, salida: Path
) -> Path:
    """Tres paneles con el perfil de conducta, para la seccion 2.4."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))
    total = sum(acciones.values())

    ax = axes[0]
    pct = [100 * acciones[i] / total for i in range(len(ACCIONES))]
    colores = ["#95a5a6" if a == "NOOP" else "#2980b9" if "FIRE" in a else "#e67e22"
               for a in ACCIONES]
    ax.bar(ACCIONES, pct, color=colores)
    ax.set_ylabel("% de pasos")
    ax.set_title("Acciones que elige el agente")
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.grid(axis="y", alpha=.3)

    ax = axes[1]
    r = np.array([e["puntaje"] for e in episodios])
    ax.hist(r, bins=np.arange(0, r.max() + 250, 250), color="#2980b9",
            edgecolor="white")
    ax.axvline(285, color="#c0392b", ls="--", lw=1.3, label="FIRE constante (285)")
    ax.set_xlabel("puntaje del episodio")
    ax.set_ylabel("episodios")
    ax.set_title("Distribucion de puntajes (bimodal)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=.3)

    ax = axes[2]
    if posiciones.size:
        ax.hist(posiciones, bins=np.arange(0, 170, 10), color="#27ae60",
                edgecolor="white")
        ax.set_xlim(0, 160)
    ax.set_xlabel("posicion horizontal del canon (px)")
    ax.set_ylabel("frames")
    ax.set_title("Donde se posiciona la nave")
    ax.grid(axis="y", alpha=.3)

    fig.tight_layout()
    salida.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(salida, dpi=150, bbox_inches="tight")
    return salida


def main() -> None:
    p = argparse.ArgumentParser(description="Analisis cualitativo del agente")
    p.add_argument("--run", required=True)
    p.add_argument("--pesos", default=None)
    p.add_argument("--mejor", action="store_true")
    p.add_argument("--episodios", type=int, default=30)
    p.add_argument("--seed", type=int, default=5000)
    p.add_argument("--salida", default=None)
    args = p.parse_args()

    dir_run = Path(args.run)
    modelo, cfg, pesos = cargar_agente(dir_run, usar_mejor=args.mejor, pesos=args.pesos)
    print(f"Pesos: {pesos}\n")

    acciones, episodios = perfil_de_conducta(modelo, cfg, args.episodios, args.seed)

    total = sum(acciones.values())
    print("Distribucion de acciones")
    print(f"  {'accion':11s} {'%':>7}")
    print("  " + "-" * 19)
    reparto = {}
    for i, nombre in enumerate(ACCIONES):
        pct = 100 * acciones[i] / total
        reparto[nombre] = pct
        print(f"  {nombre:11s} {pct:6.1f}%")
    disparo = sum(reparto[a] for a in ("FIRE", "RIGHTFIRE", "LEFTFIRE"))
    movimiento = sum(reparto[a] for a in ("RIGHT", "LEFT", "RIGHTFIRE", "LEFTFIRE"))
    print(f"\n  acciones que disparan : {disparo:.1f}%")
    print(f"  acciones que mueven   : {movimiento:.1f}%")

    r = np.array([e["puntaje"] for e in episodios])
    l = np.array([e["pasos"] for e in episodios])
    print(f"\nPuntajes ({len(r)} episodios)")
    print(f"  media {r.mean():.1f}  std {r.std():.1f}  min {r.min():.0f}  max {r.max():.0f}")
    print(f"  cuartiles: {np.percentile(r, [25, 50, 75]).round(0)}")
    print(f"  correlacion puntaje-duracion: {np.corrcoef(r, l)[0, 1]:.3f}")

    bordes = np.arange(0, r.max() + 250, 250)
    hist, _ = np.histogram(r, bins=bordes)
    print("\n  histograma:")
    for lo, n in zip(bordes[:-1], hist):
        if n:
            print(f"    {lo:5.0f}-{lo+250:5.0f}  {'#' * n} ({n})")

    carpeta_videos = dir_run / "videos"
    resumen_pos = {}
    if carpeta_videos.exists() and any(carpeta_videos.glob("*.mp4")):
        pos = posicion_de_la_nave(carpeta_videos)
        if pos.size:
            izq = 100 * (pos < 60).mean()
            centro = 100 * ((pos >= 60) & (pos <= 100)).mean()
            der = 100 * (pos > 100).mean()
            resumen_pos = {"izquierda": izq, "centro": centro, "derecha": der,
                           "media": float(pos.mean()), "std": float(pos.std())}
            print(f"\nPosicion horizontal de la nave ({pos.size} frames)")
            print(f"  media {pos.mean():.1f}  std {pos.std():.1f}   (pantalla: 0-160)")
            print(f"  izquierda (<60) {izq:.0f}%   centro {centro:.0f}%   derecha (>100) {der:.0f}%")

    salida = Path(args.salida or (dir_run / "analisis_conducta.json"))
    salida.write_text(json.dumps({
        "pesos": str(pesos),
        "episodios": args.episodios,
        "acciones_pct": reparto,
        "pct_disparo": disparo,
        "pct_movimiento": movimiento,
        "puntaje": {"media": float(r.mean()), "std": float(r.std()),
                    "min": float(r.min()), "max": float(r.max())},
        "correlacion_puntaje_duracion": float(np.corrcoef(r, l)[0, 1]),
        "posicion_nave": resumen_pos,
    }, indent=2), encoding="utf-8")
    print(f"\nGuardado en {salida}")


if __name__ == "__main__":
    main()
