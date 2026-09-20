"""
Construccion de entornos para entrenamiento y evaluacion.

Ambos caminos parten del mismo ConfigEntorno, de modo que el preprocesamiento
de la observacion es identico por construccion y no por disciplina.

Orden de los wrappers (de adentro hacia afuera):

    gym.make(frameskip=1)      emulador crudo, 210x160x3, sticky actions
      -> RecordVideo           opcional; graba el RGB crudo a 60 fps de juego
      -> AtariWrapper          NOOP inicial, max+skip(4), FIRE, 84x84 gris,
                               [vidas como fin de episodio], [clip de reward]
      -> Monitor               registra return y longitud del episodio REAL
      -> VecEnv                paralelizacion
      -> VecFrameStack(4)      apila los ultimos 4 frames

RecordVideo va lo mas adentro posible, antes del salto de frames, para que el
video quede fluido y no a un cuarto de la tasa de refresco.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import gymnasium as gym
import ale_py
from gymnasium.wrappers import RecordVideo
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecEnv,
    VecFrameStack,
)

from .config import ConfigEntorno

gym.register_envs(ale_py)


def _constructor(
    cfg: ConfigEntorno,
    seed: int,
    idx: int,
    video_folder: str | Path | None = None,
    name_prefix: str = "agente",
    episode_trigger: Callable[[int], bool] | None = None,
) -> Callable[[], gym.Env]:
    """Retorna la fabrica de un entorno individual (lo que espera un VecEnv)."""

    def _init() -> gym.Env:
        env = gym.make(
            cfg.env_id,
            # El salto lo aplica AtariWrapper; aqui debe ser 1 para no duplicarlo.
            frameskip=1,
            repeat_action_probability=cfg.repeat_action_probability,
            full_action_space=cfg.full_action_space,
            render_mode="rgb_array" if video_folder is not None else None,
        )

        if video_folder is not None:
            Path(video_folder).mkdir(parents=True, exist_ok=True)
            env = RecordVideo(
                env,
                video_folder=str(video_folder),
                name_prefix=name_prefix,
                episode_trigger=episode_trigger or (lambda ep: True),
                disable_logger=True,
            )

        env = AtariWrapper(
            env,
            noop_max=cfg.noop_max,
            frame_skip=cfg.frame_skip,
            screen_size=cfg.screen_size,
            terminal_on_life_loss=cfg.terminal_on_life_loss,
            clip_reward=cfg.clip_reward,
            # El entorno v5 ya aplica sus propias sticky actions; agregar otra
            # capa aqui las compondria.
            action_repeat_probability=0.0,
        )

        env = Monitor(env)
        env.reset(seed=seed + idx)
        env.action_space.seed(seed + idx)
        return env

    return _init


def crear_vec_entorno(
    cfg: ConfigEntorno,
    n_envs: int = 8,
    seed: int = 0,
    subproceso: bool = True,
    video_folder: str | Path | None = None,
    name_prefix: str = "agente",
    episode_trigger: Callable[[int], bool] | None = None,
) -> VecEnv:
    """Crea el entorno vectorizado y apilado listo para SB3.

    Parameters
    ----------
    n_envs:
        Entornos en paralelo. Con 4 nucleos fisicos, 8 es un punto razonable
        para PPO; para evaluacion se usa 1.
    subproceso:
        ``SubprocVecEnv`` (procesos reales) frente a ``DummyVecEnv`` (secuencial).
        Con un solo entorno, o al grabar video, conviene ``DummyVecEnv``: los
        subprocesos no aportan nada y complican la escritura del archivo.
    video_folder:
        Si se especifica, cada entorno graba sus episodios ahi.
    """
    if video_folder is not None and n_envs != 1:
        raise ValueError(
            "La grabacion de video solo esta soportada con n_envs=1; "
            f"se recibio n_envs={n_envs}."
        )

    fabricas = [
        _constructor(cfg, seed, i, video_folder, name_prefix, episode_trigger)
        for i in range(n_envs)
    ]

    usar_subproceso = subproceso and n_envs > 1 and video_folder is None
    vec: VecEnv = SubprocVecEnv(fabricas) if usar_subproceso else DummyVecEnv(fabricas)

    if cfg.frame_stack > 1:
        vec = VecFrameStack(vec, n_stack=cfg.frame_stack)

    return vec


def crear_entorno_entrenamiento(
    cfg: ConfigEntorno, n_envs: int = 8, seed: int = 0
) -> VecEnv:
    """Entorno de entrenamiento: usa la configuracion tal cual viene."""
    return crear_vec_entorno(cfg, n_envs=n_envs, seed=seed, subproceso=True)


def crear_entorno_evaluacion(
    cfg: ConfigEntorno,
    seed: int = 1000,
    video_folder: str | Path | None = None,
    name_prefix: str = "agente-evaluacion",
) -> VecEnv:
    """Entorno de evaluacion: un solo entorno, sin atajos de entrenamiento.

    Aplica ``cfg.para_evaluacion()``, de modo que un episodio son las 3 vidas
    completas y la recompensa es el puntaje real del juego.
    """
    return crear_vec_entorno(
        cfg.para_evaluacion(),
        n_envs=1,
        seed=seed,
        subproceso=False,
        video_folder=video_folder,
        name_prefix=name_prefix,
    )


def describir_entorno(cfg: ConfigEntorno) -> str:
    """Texto con los espacios resultantes; util para el informe y para verificar."""
    vec = crear_vec_entorno(cfg, n_envs=1, seed=0, subproceso=False)
    try:
        base = vec.envs[0].unwrapped
        lineas = [
            f"config          : {cfg.resumen()}",
            f"observation     : {vec.observation_space}",
            f"action          : {vec.action_space}",
            f"acciones        : {base.get_action_meanings()}",
        ]
    finally:
        vec.close()
    return "\n".join(lineas)
