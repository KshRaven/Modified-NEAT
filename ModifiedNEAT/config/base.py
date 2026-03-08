from ModifiedNEAT.util.fancy_text import CM, Fore
from ModifiedNEAT.util.qol import manage_params

from typing import Union, Any
import json
import os


class Configuration:
    def __init__(self, name: Union[str, None]):
        self._name = name

    def exists(self, file_path: str):
        try:
            if not os.path.exists(file_path):
                return False
            with open(file_path, 'r') as file:
                data = json.load(file)
                return self._name in data
        except (FileNotFoundError, json.JSONDecodeError) as error:
            print(CM(f"Error reading configuration file", Fore.LIGHTRED_EX) + f"; {error}")
        return False

    def create(self, path: str, **ex):
        debug = manage_params(ex, 'debug', True)
        variables = {attr: val for attr, val in vars(self).items() if attr[0] != '_'}
        
        try:
            # Load existing data or create empty dict
            data = {}
            if os.path.exists(path):
                try:
                    with open(path, 'r') as file:
                        data = json.load(file)
                except json.JSONDecodeError:
                    data = {}
            
            # Add or update this configuration section
            data[self._name] = variables
            
            # Write to file
            with open(path, 'w') as file:
                json.dump(data, file, indent=2)
            
            if debug:
                name, extension = os.path.splitext(os.path.basename(path))
                print(f"Created configuration '{CM(self._name.upper(), Fore.LIGHTCYAN_EX)}' "
                      f"in NEAT Configuration {CM(f'{name}{extension}', Fore.LIGHTMAGENTA_EX)}")
            return True
        except Exception as error:
            print(CM(f"Failed to create configuration", Fore.LIGHTRED_EX) + f"; {error}")
        return False

    def update(self, path: str, **ex):
        verbose = manage_params(ex, 'verbose', None)
        variables = {attr: val for attr, val in vars(self).items() if attr[0] != '_'}
        
        try:
            # Load existing data
            with open(path, 'r') as file:
                data = json.load(file)
            
            # Update this configuration section
            data[self._name] = variables
            
            # Write back to file
            with open(path, 'w') as file:
                json.dump(data, file, indent=2)
            
            if verbose:
                name, extension = os.path.splitext(os.path.basename(path))
                print(f"Updated configuration '{CM(self._name.upper(), Fore.LIGHTCYAN_EX)}' "
                      f"in NEAT Configuration {CM(f'{name}{extension}', Fore.LIGHTMAGENTA_EX)}")
            return True
        except (FileNotFoundError, json.JSONDecodeError) as error:
            print(CM(f"The NEAT configuration update failed", Fore.LIGHTRED_EX) + f"; {error}")
            self.create(path, **ex)
        return False

    def load(self, path: str, **ex):
        """
        :param path:
        :keyword verbose:
        :return:
        """
        verbose: Union[int, None] = manage_params(ex, 'verbose', None)
        try:
            with open(path, 'r') as file:
                data = json.load(file)
            
            if self._name in data:
                config_data = data[self._name]
                for label, value in config_data.items():
                    setattr(self, label, value)
                    if verbose and verbose >= 2:
                        print(f"{label:<25} = {value}")
                
                if verbose and verbose >= 1:
                    name, extension = os.path.splitext(os.path.basename(path))
                    print(f"Loaded configuration '{CM(self._name.upper(), Fore.LIGHTCYAN_EX)}' "
                          f"in NEAT Configuration {CM(f'{name}{extension}', Fore.LIGHTMAGENTA_EX)}")
                return True
            else:
                print(CM(f"Configuration '{self._name}' not found in file", Fore.LIGHTRED_EX))
        except FileNotFoundError as error:
            print(CM(f"The NEAT configuration file does not exist", Fore.LIGHTRED_EX) + f"; {error}")
            self.create(path, **ex)
        except json.JSONDecodeError as error:
            print(CM(f"Invalid JSON in configuration file", Fore.LIGHTRED_EX) + f"; {error}")
        return False

    def set(self, **parameters):
        for attr in vars(self).keys():
            for param, value in parameters.items():
                if attr in param:
                    setattr(self, attr, value)
                    break

    def get(self, var: str, default: Any = None):
        if var in vars(self):
            return getattr(self, var)
        else:
            return default
