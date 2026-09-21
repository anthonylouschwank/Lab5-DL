"""
Entrenamiento de agentes de RL sobre ALE/SpaceInvaders-v5.

Cada corrida escribe todo lo necesario para reproducirla y para documentarla en
el informe:

    runs/<id>/modelo.zip          pesos finales
    runs/<id>/mejor/best_model.zip mejores pesos segun evaluacion greedy
    runs/<id>/config_entorno.json  preprocesamiento exacto (para evaluar igual)
    runs/<id>/hiperparametros.json algoritmo e hiperparametros
    runs/<id>/tb/                  logs de TensorBoard
    runs/<id>/evaluaciones.csv     puntaje real vs. pasos de entrenamiento
    runs/registro_iteraciones.csv  una fila por corrida (tabla del informe)

Uso:
    python -m proyecto.train --algo ppo --id ppo-base --pasos 10_000_000
    python -m proyecto.train --algo dqn --id dqn-base --pasos 5_000_000
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from sb3_contrib import QRDQN
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy

from .config import CONFIG_BASE, ConfigEntorno
from .envs import crear_entorno_entrenamiento, crear_entorno_evaluacion

RAIZ_RUNS = Path("runs")


# ----------------------------------------------------------------------
# Hiperparametros por algoritmo
# ----------------------------------------------------------------------

def _decaimiento_lineal(valor_inicial: float) -> Callable[[float], float]:
    """Programa lineal de SB3: recibe el progreso restante (1 -> 0)."""

    def f(progreso_restante: float) -> float:
        return progreso_restante * valor_inicial

    return f


#: Configuraciones de referencia. PPO y DQN siguen los valores estandar de
#: Atari (Mnih et al. / rl-baselines3-zoo), que son el punto de partida
#: sensato antes de cualquier ajuste propio.
HIPERPARAMETROS: dict[str, dict[str, Any]] = {
    "ppo": dict(
        n_envs=8,
        n_steps=128,          # 128 x 8 envs = 1024 transiciones por actualizacion
        batch_size=256,
        n_epochs=4,
        learning_rate=2.5e-4,  # con decaimiento lineal
        clip_range=0.1,        # con decaimiento lineal
        ent_coef=0.01,         # exploracion: bonificacion de entropia
        vf_coef=0.5,
        gamma=0.99,
        gae_lambda=0.95,
        max_grad_norm=0.5,
    ),
    "dqn": dict(
        n_envs=1,
        buffer_size=100_000,   # 1M seria lo canonico, pero no cabe en RAM
        learning_rate=1e-4,
        batch_size=32,
        learning_starts=100_000,
        target_update_interval=1_000,
        train_freq=4,
        gradient_steps=1,
        exploration_fraction=0.1,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.01,
        gamma=0.99,
        optimize_memory_usage=False,
    ),
    # QR-DQN: DQN distribucional. En vez de estimar Q(s,a) como un escalar,
    # aprende `n_quantiles` cuantiles de la distribucion de retornos. Eso lo
    # hace mas robusto al ruido de la recompensa y suele superar a DQN y a PPO
    # en Atari.
    "qrdqn": dict(
        n_envs=1,
        n_quantiles=200,
        # El buffer domina el consumo de RAM: 100k transiciones de 84x84x4 en
        # uint8 son ~2.8 GB con optimize_memory_usage. Lo canonico serian 1M,
        # que no cabe en 16 GB.
        buffer_size=100_000,
        optimize_memory_usage=True,
        # SB3 no permite optimize_memory_usage junto con handle_timeout_termination.
        # Desactivar el segundo es inocuo aqui: solo afecta el bootstrap cuando un
        # episodio se TRUNCA por limite de tiempo, y en Atari ese limite son 108000
        # frames (27000 pasos de agente) frente a episodios de ~1500. Nunca se
        # alcanza. A cambio, el buffer ocupa la mitad: 2.8 GB en vez de 5.6 GB.
        replay_buffer_kwargs={"handle_timeout_termination": False},
        learning_rate=1e-4,
        batch_size=32,
        learning_starts=100_000,
        target_update_interval=1_000,
        train_freq=4,
        gradient_steps=1,
        exploration_fraction=0.025,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.01,
        gamma=0.99,
    ),
    "a2c": dict(
        n_envs=8,
        n_steps=5,
        learning_rate=7e-4,
        ent_coef=0.01,
        vf_coef=0.25,
        gamma=0.99,
        max_grad_norm=0.5,
    ),
}

CLASES = {"ppo": PPO, "dqn": DQN, "a2c": A2C, "qrdqn": QRDQN}


def construir_modelo(
    algo: str,
    env,
    hp: dict[str, Any],
    tensorboard_log: str,
    seed: int,
    device: str = "cuda",
):
    """Instancia el modelo de SB3 con los hiperparametros dados."""
    hp = dict(hp)
    hp.pop("n_envs", None)

    # n_quantiles no es un argumento del algoritmo sino de la politica.
    if "n_quantiles" in hp:
        hp.setdefault("policy_kwargs", {})
        hp["policy_kwargs"] = {**hp["policy_kwargs"], "n_quantiles": hp.pop("n_quantiles")}

    if algo == "ppo":
        # Los programas se construyen aqui para que el dict serializado a JSON
        # conserve el valor numerico inicial y no un objeto invisible.
        hp["learning_rate"] = _decaimiento_lineal(hp["learning_rate"])
        hp["clip_range"] = _decaimiento_lineal(hp["clip_range"])

    return CLASES[algo](
        "CnnPolicy",
        env,
        verbose=1,
        seed=seed,
        device=device,
        tensorboard_log=tensorboard_log,
        **hp,
    )


# ----------------------------------------------------------------------
# Evaluacion periodica con el puntaje REAL del juego
# ----------------------------------------------------------------------

class EvaluacionReal(BaseCallback):
    """Evalua con la configuracion de evaluacion y guarda el mejor modelo.

    Es indispensable tener esta metrica aparte: durante el entrenamiento la
    recompensa esta recortada a {-1,0,+1} y el episodio termina con la primera
    vida perdida, asi que ``ep_rew_mean`` de SB3 **no es** el puntaje del juego
    y no sirve ni para comparar iteraciones ni para el ranking.
    """

    def __init__(
        self,
        cfg: ConfigEntorno,
        dir_run: Path,
        cada_pasos: int = 250_000,
        n_episodios: int = 5,
        deterministico: bool = True,
        seed: int = 1000,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.cfg = cfg
        self.dir_run = Path(dir_run)
        self.cada_pasos = cada_pasos
        self.n_episodios = n_episodios
        self.deterministico = deterministico
        self.seed = seed
        self.mejor = -np.inf
        self._proximo = cada_pasos
        self._csv = self.dir_run / "evaluaciones.csv"

    def _on_training_start(self) -> None:
        self.dir_run.mkdir(parents=True, exist_ok=True)
        if not self._csv.exists():
            with self._csv.open("w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(
                    ["pasos", "media", "std", "maximo", "minimo", "episodios", "segundos"]
                )

        # Al reanudar, num_timesteps ya viene alto: la proxima evaluacion debe
        # programarse a partir de ahi y no en `cada_pasos` absolutos, o se
        # dispararia en el primer step.
        self._proximo = self.num_timesteps + self.cada_pasos

        # Y el mejor puntaje historico debe recuperarse del CSV. Si se dejara en
        # -inf, la primera evaluacion de la reanudacion sobrescribiria
        # best_model.zip aunque fuera peor que el mejor ya alcanzado.
        with self._csv.open(encoding="utf-8") as f:
            previas = [float(r["media"]) for r in csv.DictReader(f)]
        if previas:
            self.mejor = max(previas)
            if self.verbose:
                print(f"[eval real] reanudando; mejor media previa = {self.mejor:.1f}")

    def _on_step(self) -> bool:
        if self.num_timesteps < self._proximo:
            return True
        self._proximo += self.cada_pasos
        self._evaluar()
        return True

    def _evaluar(self) -> None:
        t0 = time.time()
        env = crear_entorno_evaluacion(self.cfg, seed=self.seed)
        try:
            recompensas, _ = evaluate_policy(
                self.model,
                env,
                n_eval_episodes=self.n_episodios,
                deterministic=self.deterministico,
                return_episode_rewards=True,
            )
        finally:
            env.close()

        r = np.asarray(recompensas, dtype=float)
        dt = time.time() - t0

        self.logger.record("eval_real/media", float(r.mean()))
        self.logger.record("eval_real/maximo", float(r.max()))
        self.logger.record("eval_real/std", float(r.std()))

        with self._csv.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                [self.num_timesteps, f"{r.mean():.1f}", f"{r.std():.1f}",
                 f"{r.max():.0f}", f"{r.min():.0f}", self.n_episodios, f"{dt:.0f}"]
            )

        if self.verbose:
            print(
                f"[eval real] pasos={self.num_timesteps:,} "
                f"media={r.mean():.1f}+-{r.std():.1f} max={r.max():.0f} ({dt:.0f}s)"
            )

        if r.mean() > self.mejor:
            self.mejor = float(r.mean())
            destino = self.dir_run / "mejor"
            destino.mkdir(parents=True, exist_ok=True)
            self.model.save(destino / "best_model")
            if self.verbose:
                print(f"[eval real] nuevo mejor modelo: {r.mean():.1f}")


# ----------------------------------------------------------------------
# Registro de iteraciones
# ----------------------------------------------------------------------

def registrar_iteracion(fila: dict[str, Any]) -> None:
    """Agrega una fila a la tabla de iteraciones que pide el informe (2.3)."""
    RAIZ_RUNS.mkdir(parents=True, exist_ok=True)
    ruta = RAIZ_RUNS / "registro_iteraciones.csv"
    columnas = [
        "id", "fecha", "algoritmo", "pasos", "n_envs", "eval_media", "eval_max",
        "eval_std", "minutos", "fps", "notas",
    ]
    nuevo = not ruta.exists()
    with ruta.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columnas)
        if nuevo:
            w.writeheader()
        w.writerow({c: fila.get(c, "") for c in columnas})


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Entrena un agente sobre ALE/SpaceInvaders-v5")
    p.add_argument("--algo", choices=sorted(CLASES), default="ppo")
    p.add_argument("--id", dest="run_id", default=None, help="identificador de la iteracion")
    p.add_argument("--pasos", type=int, default=10_000_000)
    p.add_argument("--n-envs", type=int, default=None, help="sobrescribe el valor del algoritmo")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-cada", type=int, default=250_000)
    p.add_argument("--eval-episodios", type=int, default=5,
                   help="episodios de cada evaluacion intermedia (barata, ruidosa)")
    # La competencia usa 5 episodios, pero con std ~100 el error estandar de la
    # media sobre 5 es ~45 puntos: insuficiente para comparar iteraciones entre
    # si. La evaluacion final usa mas episodios para que la tabla del informe
    # tenga barras de error utilizables.
    p.add_argument("--eval-final-episodios", type=int, default=30)
    p.add_argument("--checkpoint-cada", type=int, default=500_000)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--notas", default="", help="que cambio respecto a la iteracion anterior")
    p.add_argument(
        "--reanudar", default=None, metavar="CHECKPOINT.zip",
        help="continuar el entrenamiento desde un checkpoint en vez de empezar de cero. "
             "--pasos indica entonces los pasos ADICIONALES a correr.",
    )
    p.add_argument("--hp", default="{}", help='overrides JSON, ej: \'{"learning_rate": 1e-4}\'')
    args = p.parse_args()

    run_id = args.run_id or f"{args.algo}-{datetime.now():%m%d-%H%M}"
    dir_run = RAIZ_RUNS / run_id
    dir_run.mkdir(parents=True, exist_ok=True)

    hp = dict(HIPERPARAMETROS[args.algo])
    hp.update(json.loads(args.hp))
    n_envs = args.n_envs or hp.get("n_envs", 8)
    hp["n_envs"] = n_envs

    # Al reanudar se conserva la configuracion original de la corrida: cambiarla
    # invalidaria los pesos que se estan cargando.
    ruta_cfg = dir_run / "config_entorno.json"
    if args.reanudar and ruta_cfg.exists():
        cfg = ConfigEntorno.cargar(ruta_cfg)
    else:
        cfg = CONFIG_BASE
        cfg.guardar(ruta_cfg)

    if args.reanudar:
        (dir_run / "reanudacion.json").write_text(
            json.dumps({"checkpoint": str(args.reanudar), "pasos_adicionales": args.pasos,
                        "fecha": f"{datetime.now():%Y-%m-%d %H:%M}"}, indent=2),
            encoding="utf-8",
        )
    else:
        (dir_run / "hiperparametros.json").write_text(
            json.dumps({"algoritmo": args.algo, "pasos": args.pasos, "seed": args.seed, **hp},
                       indent=2, default=str),
            encoding="utf-8",
        )

    print("=" * 70)
    print(f"  Iteracion : {run_id}")
    print(f"  Algoritmo : {args.algo.upper()}   pasos: {args.pasos:,}   envs: {n_envs}")
    print(f"  Entorno   : {cfg.resumen()}")
    print(f"  Device    : {args.device}")
    if args.notas:
        print(f"  Notas     : {args.notas}")
    print("=" * 70)

    env = crear_entorno_entrenamiento(cfg, n_envs=n_envs, seed=args.seed)

    if args.reanudar:
        modelo = CLASES[args.algo].load(
            args.reanudar, env=env, device=args.device,
            tensorboard_log=str(dir_run / "tb"),
        )
        print(f"  Reanudado desde {args.reanudar} en {modelo.num_timesteps:,} pasos")
    else:
        modelo = construir_modelo(
            args.algo, env, hp, str(dir_run / "tb"), args.seed, args.device
        )

    callbacks = [
        EvaluacionReal(cfg, dir_run, args.eval_cada, args.eval_episodios, seed=1000),
        CheckpointCallback(
            save_freq=max(args.checkpoint_cada // n_envs, 1),
            save_path=str(dir_run / "checkpoints"),
            name_prefix="ckpt",
        ),
    ]

    t0 = time.time()
    try:
        # reset_num_timesteps=False hace dos cosas necesarias al reanudar: el
        # contador sigue desde donde iba, y SB3 suma los pasos ya hechos a
        # _total_timesteps, con lo que los programas lineales de learning rate y
        # clip_range continuan su descenso en vez de reiniciarse.
        modelo.learn(
            total_timesteps=args.pasos,
            callback=callbacks,
            progress_bar=True,
            reset_num_timesteps=not args.reanudar,
        )
    except KeyboardInterrupt:
        print("\nInterrumpido: se guarda el modelo con lo aprendido hasta ahora.")
    finally:
        minutos = (time.time() - t0) / 60
        modelo.save(dir_run / "modelo")
        env.close()

    # Evaluacion final con la configuracion oficial de la competencia.
    n_final = args.eval_final_episodios
    print(f"\nEvaluacion final ({n_final} episodios, politica greedy)...")
    env_eval = crear_entorno_evaluacion(cfg, seed=1000)
    try:
        recompensas, _ = evaluate_policy(
            modelo, env_eval, n_eval_episodes=n_final, deterministic=True,
            return_episode_rewards=True,
        )
    finally:
        env_eval.close()

    r = np.asarray(recompensas, dtype=float)
    err = r.std() / np.sqrt(len(r))
    fps = modelo.num_timesteps / max(time.time() - t0, 1e-9)
    print(f"  episodios : {[int(x) for x in r]}")
    print(f"  media     : {r.mean():.1f} +- {r.std():.1f}  (error estandar {err:.1f})")
    print(f"  maximo    : {r.max():.0f}   <- metrica del ranking")
    print(f"  tiempo    : {minutos:.1f} min   ({fps:.0f} pasos/s)")

    registrar_iteracion({
        "id": run_id,
        "fecha": f"{datetime.now():%Y-%m-%d %H:%M}",
        "algoritmo": args.algo,
        "pasos": modelo.num_timesteps,
        "n_envs": n_envs,
        "eval_media": f"{r.mean():.1f}",
        "eval_max": f"{r.max():.0f}",
        "eval_std": f"{r.std():.1f}",
        "minutos": f"{minutos:.1f}",
        "fps": f"{fps:.0f}",
        "notas": args.notas,
    })
    print(f"\nRegistrado en {RAIZ_RUNS / 'registro_iteraciones.csv'}")


if __name__ == "__main__":
    main()
