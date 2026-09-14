"""
Laboratorio #5 - CC3092 Deep Learning y Sistemas Inteligentes
Modulo de funciones reutilizables para interactuar con el Arcade Learning
Environment (ALE) a traves de Gymnasium.

Provee la infraestructura para crear entornos, ejecutar agentes (sin
entrenamiento) y grabar video de las partidas:

    crear_entorno          crea el entorno, opcionalmente con RecordVideo
    agente_aleatorio       baseline: accion muestreada del action_space
    agente_regla_simple    heuristica basada en pixeles para Space Invaders
    ejecutar_episodio      corre un episodio completo y reporta metricas
    generar_video_agente   funcion de alto nivel: entorno + episodios + video
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordVideo

# Registra el namespace "ALE/..." en Gymnasium. Sin esto, gym.make("ALE/...")
# falla con NamespaceNotFound.
import ale_py

gym.register_envs(ale_py)


# --------------------------------------------------------------------------
# Tipos y constantes
# --------------------------------------------------------------------------

#: Firma de una funcion de agente: (observacion, entorno) -> accion
FuncionAgente = Callable[[Any, gym.Env], Any]

#: Espacio de accion reducido (por defecto) de ALE/SpaceInvaders-v5.
ACCIONES_SPACE_INVADERS = {
    0: "NOOP",       # no hacer nada
    1: "FIRE",       # disparar sin moverse
    2: "RIGHT",      # mover la nave a la derecha
    3: "LEFT",       # mover la nave a la izquierda
    4: "RIGHTFIRE",  # mover a la derecha y disparar simultaneamente
    5: "LEFTFIRE",   # mover a la izquierda y disparar simultaneamente
}


# --------------------------------------------------------------------------
# 1. Creacion de entornos
# --------------------------------------------------------------------------

def crear_entorno(
    nombre_entorno: str,
    render_mode: str | None = "rgb_array",
    video_folder: str | os.PathLike | None = None,
    name_prefix: str = "rl-video",
    episode_trigger: Callable[[int], bool] | None = None,
    **kwargs: Any,
) -> gym.Env:
    """Crea y retorna un entorno de Gymnasium, con grabacion de video opcional.

    Parameters
    ----------
    nombre_entorno:
        Id del entorno, p. ej. "ALE/SpaceInvaders-v5" o "CartPole-v1".
        La funcion es generica: sirve para cualquier entorno de Gymnasium.
    render_mode:
        Modo de render. Debe ser "rgb_array" para poder grabar video.
    video_folder:
        Si se especifica, el entorno se envuelve con
        gymnasium.wrappers.RecordVideo y los .mp4 se escriben en esta carpeta
        (se crea si no existe). Si es None no se graba nada.
    name_prefix:
        Prefijo de los archivos de video generados.
    episode_trigger:
        Funcion (episode_id: int) -> bool que decide que episodios grabar.
        Por defecto graba todos los episodios.
    **kwargs:
        Argumentos extra pasados a gym.make (frameskip, obs_type,
        repeat_action_probability, full_action_space, ...).

    Returns
    -------
    gym.Env
        El entorno, envuelto en RecordVideo si se pidio grabacion.
    """
    if video_folder is not None and render_mode != "rgb_array":
        raise ValueError(
            "Para grabar video se requiere render_mode='rgb_array', "
            f"pero se recibio render_mode={render_mode!r}."
        )

    env = gym.make(nombre_entorno, render_mode=render_mode, **kwargs)

    if video_folder is not None:
        Path(video_folder).mkdir(parents=True, exist_ok=True)
        if episode_trigger is None:
            def episode_trigger(episode_id: int) -> bool:
                return True
        env = RecordVideo(
            env,
            video_folder=str(video_folder),
            episode_trigger=episode_trigger,
            name_prefix=name_prefix,
            disable_logger=True,
        )

    return env


# --------------------------------------------------------------------------
# 2. Agentes (sin entrenamiento)
# --------------------------------------------------------------------------

def agente_aleatorio(observation: Any, env: gym.Env) -> Any:
    """Agente baseline: ignora la observacion y muestrea del espacio de accion.

    Sirve como piso de comparacion (baseline) para cualquier agente entrenado
    posteriormente.
    """
    return env.action_space.sample()


# Colores RGB de los sprites de Space Invaders (constantes del emulador,
# obtenidos inspeccionando los frames del entorno).
_COLOR_NAVE = np.array([50, 132, 50], dtype=np.uint8)       # nave del jugador
_COLOR_ALIENS = np.array([134, 134, 29], dtype=np.uint8)    # bloque de invasores
_COLOR_PROYECTIL = np.array([142, 142, 142], dtype=np.uint8)  # laser y bombas

# Bandas de filas donde vive cada sprite en el frame de 210x160.
# La nave se aisla en las filas bajas para no confundirla con el marcador,
# que usa el mismo verde en la parte superior de la pantalla.
_FILAS_NAVE = slice(180, 196)
_FILAS_ALIENS = slice(25, 160)

# Zona de peligro: franja inmediatamente encima de la nave donde una bomba ya
# no da tiempo a nada mas que apartarse.
_FILAS_PELIGRO = slice(140, 178)
_MARGEN_ESQUIVE = 4   # columnas de tolerancia para considerar una bomba "encima"
_BORDE_IZQ, _BORDE_DER = 10, 150   # limites utiles del riel de la nave


def _columnas_de_color(frame: np.ndarray, filas: slice, color: np.ndarray) -> np.ndarray:
    """Retorna las columnas del frame donde aparece el color dentro de filas."""
    mascara = (frame[filas] == color).all(axis=-1)
    return np.flatnonzero(mascara.any(axis=0))


def _mapa_acciones(env: gym.Env) -> dict[str, int] | None:
    """Mapea nombre de accion -> indice, usando los action meanings del ALE."""
    try:
        nombres = env.unwrapped.get_action_meanings()
    except AttributeError:
        return None
    return {nombre: i for i, nombre in enumerate(nombres)}


def agente_regla_simple(observation: Any, env: gym.Env, tolerancia: float = 2.0) -> Any:
    """Heuristica sin entrenamiento para Space Invaders con observacion RGB.

    La regla tiene dos niveles, en orden de prioridad:

    1. **Esquivar.** Si hay un proyectil en la franja justo encima de la nave y
       a menos de ``_MARGEN_ESQUIVE`` columnas de ella, se aparta hacia el lado
       contrario (disparando, porque las acciones compuestas no cuestan nada).
    2. **Apuntar.** Si no hay peligro inmediato, se alinea con la columna del
       invasor vivo mas cercano y dispara.

    El paso 1 es lo que hace que la heuristica supere al agente aleatorio: sin
    el, alinearse con una columna de invasores deja a la nave justo en la
    trayectoria de las bombas que esa columna deja caer.

    Nota: el laser del jugador y las bombas de los invasores comparten color en
    el frame, asi que un disparo propio recien lanzado puede leerse como
    amenaza. El efecto es benigno -- provoca que la nave se mueva despues de
    disparar, que es justamente lo que conviene hacer.

    Si la observacion no es un frame RGB de 210x160x3 (p. ej. obs_type="ram"
    o un entorno que no es Atari), o si el espacio de accion no expone las
    acciones necesarias, degrada a agente_aleatorio.
    """
    frame = np.asarray(observation)
    if frame.ndim != 3 or frame.shape != (210, 160, 3):
        return agente_aleatorio(observation, env)

    acciones = _mapa_acciones(env)
    if acciones is None or not {"FIRE", "LEFTFIRE", "RIGHTFIRE"} <= acciones.keys():
        return agente_aleatorio(observation, env)

    cols_nave = _columnas_de_color(frame, _FILAS_NAVE, _COLOR_NAVE)
    if cols_nave.size == 0:
        # Nave no visible (explosion / respawn): mantener el gatillo apretado.
        return acciones["FIRE"]
    x_nave = float(cols_nave.mean())

    # --- 1. Esquivar proyectiles inminentes -------------------------------
    cols_balas = _columnas_de_color(frame, _FILAS_PELIGRO, _COLOR_PROYECTIL)
    if cols_balas.size:
        amenazas = cols_balas[np.abs(cols_balas - x_nave) <= _MARGEN_ESQUIVE]
        if amenazas.size:
            if x_nave < _BORDE_IZQ:      # contra la pared izquierda
                return acciones["RIGHTFIRE"]
            if x_nave > _BORDE_DER:      # contra la pared derecha
                return acciones["LEFTFIRE"]
            x_bala = float(amenazas[np.argmin(np.abs(amenazas - x_nave))])
            return acciones["RIGHTFIRE"] if x_bala < x_nave else acciones["LEFTFIRE"]

    # --- 2. Apuntar al invasor mas cercano --------------------------------
    cols_aliens = _columnas_de_color(frame, _FILAS_ALIENS, _COLOR_ALIENS)
    if cols_aliens.size == 0:
        return acciones["FIRE"]

    # La columna con invasor mas cercana a la nave minimiza el desplazamiento
    # necesario antes de poder acertar un disparo.
    x_objetivo = float(cols_aliens[np.argmin(np.abs(cols_aliens - x_nave))])

    if x_objetivo < x_nave - tolerancia:
        return acciones["LEFTFIRE"]
    if x_objetivo > x_nave + tolerancia:
        return acciones["RIGHTFIRE"]
    return acciones["FIRE"]


# --------------------------------------------------------------------------
# 3. Ejecucion de episodios
# --------------------------------------------------------------------------

def ejecutar_episodio(
    env: gym.Env,
    funcion_agente: FuncionAgente,
    max_steps: int = 10_000,
    seed: int | None = None,
) -> dict[str, Any]:
    """Ejecuta un episodio completo y retorna sus metricas.

    Corre hasta que terminated o truncated sea verdadero, o hasta alcanzar
    max_steps.

    Returns
    -------
    dict
        pasos: numero de pasos ejecutados.
        recompensa_total: return acumulado del episodio.
        terminated / truncated: como termino el episodio.
        limite_alcanzado: True si se corto por max_steps.
        info: ultimo diccionario info del entorno.

    Notes
    -----
    ``seed`` tambien siembra el ``action_space``. Sin eso, ``env.reset(seed=...)``
    solo fija el generador del entorno y ``action_space.sample()`` seguiria
    siendo aleatorio, con lo que una corrida de ``agente_aleatorio`` no seria
    reproducible.
    """
    observation, info = env.reset(seed=seed)
    if seed is not None:
        env.action_space.seed(seed)

    recompensa_total = 0.0
    pasos = 0
    terminated = False
    truncated = False

    while pasos < max_steps:
        accion = funcion_agente(observation, env)
        observation, recompensa, terminated, truncated, info = env.step(accion)
        recompensa_total += float(recompensa)
        pasos += 1
        if terminated or truncated:
            break

    return {
        "pasos": pasos,
        "recompensa_total": recompensa_total,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "limite_alcanzado": pasos >= max_steps and not (terminated or truncated),
        "info": info,
    }


# --------------------------------------------------------------------------
# 4. Generacion de video (funcion de alto nivel)
# --------------------------------------------------------------------------

def generar_video_agente(
    nombre_entorno: str,
    funcion_agente: FuncionAgente,
    video_folder: str | os.PathLike,
    name_prefix: str = "agente",
    n_episodios: int = 1,
    max_steps: int = 10_000,
    seed: int | None = None,
    **kwargs: Any,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Crea el entorno con grabacion, corre n_episodios y retorna videos + metricas.

    Combina crear_entorno, la funcion de agente recibida y ejecutar_episodio.
    Cierra el entorno con env.close() dentro de un finally, paso indispensable
    para que RecordVideo termine de escribir los .mp4 a disco.

    Parameters
    ----------
    seed:
        Semilla del primer episodio. Los siguientes usan seed + i para que la
        corrida sea reproducible sin repetir exactamente el mismo episodio.

    Returns
    -------
    (videos, metricas)
        videos: rutas de los .mp4 generados por esta llamada.
        metricas: un dict por episodio (ver ejecutar_episodio), con la clave
        adicional "episodio".
    """
    carpeta = Path(video_folder)
    carpeta.mkdir(parents=True, exist_ok=True)
    # Snapshot previo: permite reportar solo los videos nuevos aunque la
    # carpeta ya contenga corridas anteriores.
    previos = {p.resolve() for p in carpeta.glob("*.mp4")}

    env = crear_entorno(
        nombre_entorno,
        render_mode="rgb_array",
        video_folder=carpeta,
        name_prefix=name_prefix,
        **kwargs,
    )

    metricas: list[dict[str, Any]] = []
    try:
        for i in range(n_episodios):
            semilla = None if seed is None else seed + i
            resultado = ejecutar_episodio(
                env, funcion_agente, max_steps=max_steps, seed=semilla
            )
            resultado["episodio"] = i
            metricas.append(resultado)
    finally:
        # Indispensable: sin close() el ultimo video queda incompleto o vacio.
        env.close()

    videos = sorted(
        str(p) for p in carpeta.glob("*.mp4") if p.resolve() not in previos
    )
    return videos, metricas


# --------------------------------------------------------------------------
# 5. Utilidades de reporte
# --------------------------------------------------------------------------

def resumir_metricas(metricas: Sequence[dict[str, Any]]) -> str:
    """Formatea las metricas de una corrida como una tabla de texto."""
    lineas = [
        f"{'Episodio':>8} | {'Pasos':>7} | {'Recompensa':>10}",
        f"{'-' * 8}-+-{'-' * 7}-+-{'-' * 10}",
    ]
    for m in metricas:
        lineas.append(
            f"{m.get('episodio', 0):>8} | {m['pasos']:>7} | {m['recompensa_total']:>10.1f}"
        )
    if len(metricas) > 1:
        pasos = [m["pasos"] for m in metricas]
        recompensas = [m["recompensa_total"] for m in metricas]
        lineas.append(f"{'-' * 8}-+-{'-' * 7}-+-{'-' * 10}")
        lineas.append(
            f"{'media':>8} | {np.mean(pasos):>7.1f} | {np.mean(recompensas):>10.1f}"
        )
    return "\n".join(lineas)


if __name__ == "__main__":
    # Demo rapida: un episodio de agente aleatorio grabado en videos/
    videos, metricas = generar_video_agente(
        nombre_entorno="ALE/SpaceInvaders-v5",
        funcion_agente=agente_aleatorio,
        video_folder="videos",
        name_prefix="demo-aleatorio",
        n_episodios=1,
        seed=42,
    )
    print(resumir_metricas(metricas))
    print("Videos generados:", videos)
