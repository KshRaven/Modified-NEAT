from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from game import EnvironmentConfig, Game


if __name__ == "__main__":
    env = Game(config=EnvironmentConfig(params={
        "agent": {"discrete_actions": True},
    }))
    env.run(total=10)