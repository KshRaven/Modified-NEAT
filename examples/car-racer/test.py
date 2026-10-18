import torch
import os, sys
import warnings

from numba.core.errors import NumbaPerformanceWarning

EXMP_DIR = os.path.dirname(os.path.abspath(__file__))
if EXMP_DIR not in sys.path: sys.path.insert(0, EXMP_DIR)

from game import Game, EnvironmentConfig

warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DTYPE  = torch.float32

ENV: Game | None = None

def run():
    global ENV
    
    # Create environment configuration from file
    config_file = os.path.join(EXMP_DIR, '.config', 'test.json')
    env_config = EnvironmentConfig(file=None) #config_file)
    
    # Create environment with configuration
    ENV = Game(render_mode="human", config=env_config)
    # ENV.reset(keys=10)
    # print(f"{ENV.cars.get_state().shape}")
    # print(ENV.observation_space.sample().shape)
    # print(ENV.action_space.sample().shape)
    # sys.exit()

    with ENV.race_mode():
        ENV.run()
    
    
if __name__ == "__main__":
    run()
