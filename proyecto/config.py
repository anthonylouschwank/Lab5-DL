"""
Proyecto - CC3092: Agente de RL para ALE/SpaceInvaders-v5.

Configuracion del entorno. Este modulo es la **unica fuente de verdad** del
preprocesamiento: entrenamiento y evaluacion construyen sus entornos desde el
mismo objeto ConfigEntorno, y la configuracion se serializa junto al modelo.

Esto responde a la recomendacion 2 del enunciado: el preprocesamiento usado en
evaluacion debe ser identico al usado durante el entrenamiento. Si divergen, el
agente ve observaciones fuera de distribucion y el puntaje se desploma sin
ninguna senal de error.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ConfigEntorno:
    """Parametros del entorno y de su preprocesamiento.

    Los campos se dividen en tres grupos:

    - **Entorno base**: que ROM y con que estocasticidad se emula.
    - **Observacion**: como se transforma el frame antes de la red. Debe ser
      IDENTICO en entrenamiento y evaluacion.
    - **Solo entrenamiento**: atajos que ayudan a aprender pero falsean la
      metrica. Se desactivan siempre en evaluacion (ver `para_evaluacion`).
    """

    # --- entorno base ---
    env_id: str = "ALE/SpaceInvaders-v5"
    #: Sticky actions. 0.25 es el valor por defecto de v5 y el que se usara en
    #: la competencia; mantenerlo en entrenamiento evita que el agente aprenda
    #: una politica fragil que depende de control perfecto.
    repeat_action_probability: float = 0.25
    #: El subconjunto minimo (6 acciones) basta: el canon no se mueve en
    #: vertical, asi que las 18 del joystick solo agrandan el espacio de
    #: busqueda sin aportar nada.
    full_action_space: bool = False

    # --- observacion (identica en train y eval) ---
    #: El entorno base se crea con frameskip=1 porque el salto lo aplica
    #: AtariWrapper. Dejar el frameskip=4 nativo de v5 lo aplicaria dos veces.
    frame_skip: int = 4
    screen_size: int = 84
    frame_stack: int = 4
    #: Numero maximo de NOOP aleatorios al reiniciar, para que el agente no
    #: memorice un inicio fijo.
    noop_max: int = 30

    # --- solo entrenamiento ---
    #: Tratar la perdida de una vida como fin de episodio. Acorta el horizonte
    #: de asignacion de credito y acelera el aprendizaje, pero parte el
    #: episodio real de 3 vidas: jamas debe usarse al medir el puntaje.
    terminal_on_life_loss: bool = True
    #: Recortar la recompensa a {-1, 0, +1}. Estabiliza el gradiente, pero
    #: destruye la metrica real del juego.
    clip_reward: bool = True

    # ------------------------------------------------------------------
    def para_evaluacion(self) -> "ConfigEntorno":
        """Version de esta configuracion apta para medir el puntaje real.

        Conserva intacto el preprocesamiento de la observacion y desactiva los
        dos atajos de entrenamiento, de modo que un episodio sean las 3 vidas
        completas y la recompensa sea el puntaje del juego.
        """
        return replace(self, terminal_on_life_loss=False, clip_reward=False)

    # ------------------------------------------------------------------
    def guardar(self, ruta: str | Path) -> Path:
        """Serializa la configuracion a JSON junto al modelo."""
        ruta = Path(ruta)
        ruta.parent.mkdir(parents=True, exist_ok=True)
        ruta.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        return ruta

    @staticmethod
    def cargar(ruta: str | Path) -> "ConfigEntorno":
        """Reconstruye la configuracion guardada junto a un modelo entrenado."""
        datos: dict[str, Any] = json.loads(Path(ruta).read_text(encoding="utf-8"))
        return ConfigEntorno(**datos)

    def resumen(self) -> str:
        """Descripcion de una linea, util para nombrar corridas y para el log."""
        return (
            f"{self.env_id} | {self.screen_size}x{self.screen_size} gris "
            f"x{self.frame_stack} | skip={self.frame_skip} "
            f"sticky={self.repeat_action_probability} "
            f"| vidas_terminan={self.terminal_on_life_loss} "
            f"clip={self.clip_reward}"
        )


#: Configuracion por defecto del proyecto.
CONFIG_BASE = ConfigEntorno()
