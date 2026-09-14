# Laboratorio #5 — Agentes en el Arcade Learning Environment (ALE): Space Invaders

**CC3092 — Deep Learning y Sistemas Inteligentes**

Infraestructura de código para que un agente **sin entrenamiento** (aleatorio o de regla simple)
interactúe con un entorno de Atari a través de ALE y genere un video de la partida.

## Contenido

| Archivo | Descripción |
|---|---|
| `Informe_Lab5_ALE_SpaceInvaders.pdf` | Informe de investigación (3 páginas) |
| `ale_utils.py` | Módulo de funciones reutilizables (el entregable de código) |
| `Laboratorio5_ALE_SpaceInvaders.ipynb` | Notebook con la investigación, las pruebas y la generación de videos |
| `videos/` | Videos `.mp4` generados |
| `informe/` | Fuente HTML del informe y su figura |
| `requirements.txt` | Dependencias con versiones fijadas |

El PDF se regenera desde el HTML con Chrome headless:

```bash
chrome --headless=new --no-pdf-header-footer --print-to-pdf=Informe_Lab5_ALE_SpaceInvaders.pdf informe/informe.html
```

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows;  source .venv/bin/activate en Linux/macOS
pip install -r requirements.txt
python -m ipykernel install --user --name lab5-dl --display-name "Python (Lab5-DL)"
```

El notebook usa el kernel **Python (Lab5-DL)**. `ale-py >= 0.10` incluye las ROMs, no hace falta AutoROM.

## Uso rápido

```python
from ale_utils import generar_video_agente, agente_aleatorio, resumir_metricas

videos, metricas = generar_video_agente(
    nombre_entorno="ALE/SpaceInvaders-v5",
    funcion_agente=agente_aleatorio,
    video_folder="videos",
    name_prefix="aleatorio-spaceinvaders",
    n_episodios=1,
    seed=42,
)
print(resumir_metricas(metricas), videos)
```

O directamente:

```bash
python ale_utils.py
```

## API del módulo

| Función | Rol |
|---|---|
| `crear_entorno(nombre_entorno, render_mode, video_folder, name_prefix, episode_trigger, **kwargs)` | Crea el entorno; lo envuelve con `RecordVideo` si se pasa `video_folder`. Genérica para cualquier entorno de Gymnasium. |
| `agente_aleatorio(observation, env)` | Baseline: `env.action_space.sample()`. |
| `agente_regla_simple(observation, env)` | Heurística sobre píxeles: esquiva proyectiles y se alinea con el invasor más cercano. |
| `ejecutar_episodio(env, funcion_agente, max_steps=10000, seed=None)` | Corre un episodio; retorna pasos y recompensa total. |
| `generar_video_agente(nombre_entorno, funcion_agente, video_folder, name_prefix, n_episodios=1)` | Alto nivel: entorno + episodios + `env.close()` + rutas de video y métricas. |
| `resumir_metricas(metricas)` | Tabla de texto con los resultados. |

## Resultados (10 semillas, `ALE/SpaceInvaders-v5`)

| Agente | Return medio | Desv. est. | Pasos medios |
|---|---:|---:|---:|
| Aleatorio | 115.0 | 49.8 | 445.8 |
| Regla simple | 239.5 | 56.8 | 522.7 |

Ninguno de los dos aprende: la regla simple tiene el conocimiento del juego cableado a mano. Sirve
como prueba de que el pipeline observación → acción → recompensa está bien conectado.
