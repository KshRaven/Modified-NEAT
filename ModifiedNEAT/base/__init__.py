
from .execution import (
    # Device management
    cuda_is_available,
    set_device,  
    get_device, 
    set_device_env,
    get_device_env,
    DEVICE_ENV_NAME,
    # Threads per block management
    get_tpb,
    set_tpb,
    get_tpb_env,
    set_tpb_env,
    TPB_ENV_NAME,
    # Core execution functions
    initialize, 
    speciate, 
    reproduce,
)