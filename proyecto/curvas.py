"""
Curvas de entrenamiento para la seccion 2.3 del informe.

Grafica, para una o varias corridas:

  - el puntaje REAL de evaluacion (media +- std) frente a los pasos, que es la
    metrica comparable entre iteraciones;
  - la recompensa de entrenamiento que reporta SB3 (`rollout/ep_rew_mean`),
    que NO es el puntaje del juego -- esta recortada a {-1,0,+1} y el episodio
    termina con la primera vida perdida -- pero sirve para ver estabilidad;
  - las politicas de referencia como lineas horizontales, para saber de un
    vistazo si el agente ya supera el piso trivial.

Uso:
    python -m proyecto.curvas --runs runs/ppo-piloto-1M
    python -m proyecto.curvas --runs runs/ppo-base runs/dqn-base --salida figuras/comparacion.png
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def leer_evaluaciones(dir_run: Path) -> dict[str, np.ndarray]:
    """Lee evaluaciones.csv (puntaje real vs pasos de entrenamiento)."""
    ruta = dir_run / "evaluaciones.csv"
    if not ruta.exists():
        return {}
    with ruta.open(encoding="utf-8") as f:
        filas = list(csv.DictReader(f))
    if not filas:
        return {}
    return {
        "n_episodios": int(filas[0].get("episodios", 0) or 0),
        "pasos": np.array([int(r["pasos"]) for r in filas]),
        "media": np.array([float(r["media"]) for r in filas]),
        "std": np.array([float(r["std"]) for r in filas]),
        "maximo": np.array([float(r["maximo"]) for r in filas]),
    }


def leer_tensorboard(dir_run: Path, etiqueta: str = "rollout/ep_rew_mean"):
    """Extrae una serie escalar de los logs de TensorBoard, si existen."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        return None

    eventos = sorted((dir_run / "tb").rglob("events.out.tfevents.*"))
    if not eventos:
        return None

    pasos: list[int] = []
    valores: list[float] = []
    for ev in eventos:
        acc = EventAccumulator(str(ev), size_guidance={"scalars": 0})
        acc.Reload()
        if etiqueta not in acc.Tags().get("scalars", []):
            continue
        for s in acc.Scalars(etiqueta):
            pasos.append(s.step)
            valores.append(s.value)
    if not pasos:
        return None
    orden = np.argsort(pasos)
    return np.array(pasos)[orden], np.array(valores)[orden]


def leer_baselines(ruta: Path = Path("runs/baselines.csv")) -> dict[str, float]:
    """Politicas de referencia, para dibujarlas como piso."""
    if not ruta.exists():
        return {}
    with ruta.open(encoding="utf-8") as f:
        return {r["politica"]: float(r["media"]) for r in csv.DictReader(f)}


def graficar(dirs_run: list[Path], salida: Path) -> Path:
    base = leer_baselines()
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))

    # --- izquierda: puntaje real de evaluacion ---
    ax = axes[0]
    n_eps: set[int] = set()
    for d in dirs_run:
        ev = leer_evaluaciones(d)
        if not ev:
            continue
        n_eps.add(ev["n_episodios"])
        ax.plot(ev["pasos"], ev["media"], marker="o", lw=1.8, label=d.name)
        ax.fill_between(
            ev["pasos"], ev["media"] - ev["std"], ev["media"] + ev["std"], alpha=.18
        )

    for nombre, color, estilo in [
        ("constante:FIRE", "#c0392b", "--"),
        ("aleatoria", "#7f8c8d", ":"),
    ]:
        if nombre in base:
            ax.axhline(base[nombre], color=color, ls=estilo, lw=1.3,
                       label=f"{nombre} ({base[nombre]:.0f})")

    ax.set_xlabel("pasos de entrenamiento")
    ax.set_ylabel("puntaje real del juego")
    etiqueta_n = f"{n_eps.pop()} episodios" if len(n_eps) == 1 else "n episodios"
    ax.set_title(f"Evaluacion greedy (media $\\pm$ std, {etiqueta_n})")
    ax.grid(alpha=.3)
    ax.legend(fontsize=8)

    # --- derecha: recompensa de entrenamiento (SB3) ---
    ax = axes[1]
    hay_datos = False
    for d in dirs_run:
        serie = leer_tensorboard(d)
        if serie is None:
            continue
        pasos, valores = serie
        # Media movil para que la curva sea legible.
        k = max(1, len(valores) // 100)
        suavizado = np.convolve(valores, np.ones(k) / k, mode="valid")
        ax.plot(pasos[k - 1:], suavizado, lw=1.6, label=d.name)
        hay_datos = True

    ax.set_xlabel("pasos de entrenamiento")
    ax.set_ylabel("ep_rew_mean (recompensa recortada)")
    ax.set_title("Recompensa de entrenamiento\n(recortada y por vida: no comparable con el puntaje)")
    ax.grid(alpha=.3)
    if hay_datos:
        ax.legend(fontsize=8)
    else:
        ax.text(.5, .5, "sin logs de TensorBoard", ha="center", va="center",
                transform=ax.transAxes, color="#7f8c8d")

    fig.tight_layout()
    salida.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(salida, dpi=150, bbox_inches="tight")
    return salida


def main() -> None:
    p = argparse.ArgumentParser(description="Grafica curvas de entrenamiento")
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--salida", default="figuras/curvas.png")
    args = p.parse_args()

    ruta = graficar([Path(r) for r in args.runs], Path(args.salida))
    print(f"Figura guardada en {ruta}")


if __name__ == "__main__":
    main()
