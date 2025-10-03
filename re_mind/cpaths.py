from pathlib import Path

PROJ_HOME = Path.home().joinpath('.local/opt/re-mind')
QDRANT_DATA_PATH = PROJ_HOME / 'qdrant'

CONFIG_PATH = PROJ_HOME / 'config.json'
