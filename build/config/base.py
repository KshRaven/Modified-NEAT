
from build.util.fancy_text import CM, Fore
from build.util.qol import manage_params

from typing import Union, Any

import os


class Configuration:
    spacing = 40

    def _handle_spacing(self, text: str):
        text = text + max(0, self.spacing - len(text)) * " "
        return text

    @staticmethod
    def _set_value(var: Any):
        if isinstance(var, (str, float, int, bool)):
            var_type = str(type(var)).split("'")[1]
        else:
            var_type = f"NULL"
        return f"{var} ~ {var_type}"

    @staticmethod
    def _parse_variable(text: str):
        words = text.split()
        if len(words) >= 5:
            label = words[0]
            value = words[2]
            var_type  = words[4]
            if var_type == "str":
                value = str(value)
            elif var_type == "int":
                value = int(value)
            elif var_type == "float":
                value = float(value)
            elif var_type == "bool":
                value = True if value == "True" else False
            elif var_type == 'NULL':
                pass
            return label, value, var_type
        else:
            return None

    def __init__(self, name: Union[str, None]):
        self._name = name

    def exists(self, file_path: str):
        try:
            with open(file_path, 'r') as file:
                lines = file.readlines()
                if "<<NEAT CONFIGURATION>>" in lines[0]:
                    for line in lines[1:]:
                        if self._name.upper() in line:
                            return True
                else:
                    print(CM(f"The file is not a NEAT configuration file", Fore.LIGHTRED_EX))
        except FileNotFoundError as error:
            print(CM(f"The NEAT configuration file does not exist", Fore.LIGHTRED_EX) + f"; {error}")
        return False

    def create(self, path: str, **ex):
        debug = manage_params(ex, 'debug', True)
        # os.makedirs(path, exist_ok=True)
        variables = {attr: val for attr, val in vars(self).items() if attr[0] != '_'}
        try:
            # Create file
            if not os.path.exists(path):
                with open(path, 'a') as file:
                    file.close()
            # Get lines
            lines = []
            with open(path, 'r') as file:
                for sen in file.readlines():
                    lines.append(sen)
                file.close()
            # Write to each line
            if not any(self._name.upper() in line for line in lines):
                with open(path, 'w') as file:
                    file.writelines(lines)
                    text = []
                    if not any("NEAT" in line for line in lines):
                        # print(True)
                        text.append("<<NEAT CONFIGURATION>>\n")
                    text.append(f"\n[{self._name.upper()}]\n")
                    for label, value in variables.items():
                        text.append(f"{self._handle_spacing(label)} = {self._set_value(value)}\n")
                    file.writelines(text)
                    file.close()
            # Update each line
            else:
                self.update(path, debug=False)
            # Debugging
            if debug:
                name, extension = os.path.splitext(os.path.basename(path))
                print(f"Created configuration '{CM(self._name.upper(), Fore.LIGHTCYAN_EX)}' "
                      f"in NEAT Configuration {CM(f'{name}{extension}', Fore.LIGHTMAGENTA_EX)}")
            return True
        except FileNotFoundError as error:
            print(CM(f"The NEAT configuration file does not exist", Fore.LIGHTRED_EX) + f"; {error}")
        return False

    def update(self, path: str, **ex):
        debug = manage_params(ex, 'debug', True)
        variables = {attr: val for attr, val in vars(self).items() if attr[0] != '_'}
        try:
            # Fetch lines
            lines = []
            with open(path, 'r') as file:
                for sen in file.readlines():
                    lines.append(sen)
                file.close()
            # Check file
            if any("NEAT" in line for line in lines):
                # Update lines
                with open(path, 'w') as file:
                    for num, line in enumerate(lines):
                        words = line.split()
                        label = None if len(words) <= 0 else words[0]
                        if label is not None and label in variables.keys():
                            value = variables[label]
                            text = f"{self._handle_spacing(label)} = {self._set_value(value)}\n"
                            file.write(text)
                        else:
                            file.write(line)
                    file.close()
                # Debugging
                if debug:
                    name, extension = os.path.splitext(os.path.basename(path))
                    print(f"Updated configuration '{CM(self._name.upper(), Fore.LIGHTCYAN_EX)}' "
                          f"in NEAT Configuration {CM(f'{name}{extension}', Fore.LIGHTMAGENTA_EX)}")
                return True
            else:
                print(CM(f"File is not a NEAT configuration file", Fore.LIGHTRED_EX))
                file.close()
        except FileNotFoundError as error:
            print(CM(f"The NEAT configuration file does not exist", Fore.LIGHTRED_EX) + f"; {error}")
        return False

    def load(self, path: str, **ex):
        """
        :param path:
        :keyword debug:
        :return:
        """
        verbose: Union[int, None] = manage_params(ex, 'verbose', None)
        try:
            # Fetch lines
            with open(path, 'r') as file:
                lines = file.readlines()
                file.close()
            # Check file
            if any("NEAT" in line for line in lines):
                # Update variables
                fetch = False
                for line in lines:
                    # Start fetching where Config name encountered
                    if self._name.upper() in line:
                        fetch = True
                        continue
                    if fetch:
                        # Stop fetching on next Config
                        if '[' in line:
                            break
                        variable = self._parse_variable(line)
                        if variable is not None:
                            label, value, var_type = variable
                            if verbose and verbose >= 2:
                                print(f"{text_fill(label, 25)} = {value}")
                            setattr(self, label, value)
                # Debugging
                if verbose and verbose >= 1:
                    name, extension = os.path.splitext(os.path.basename(path))
                    print(f"Loaded configuration '{CM(self._name.upper(), Fore.LIGHTCYAN_EX)}' "
                          f"in NEAT Configuration {CM(f'{name}{extension}', Fore.LIGHTMAGENTA_EX)}")
                return True
            else:
                print(CM(f"File is not a NEAT configuration file", Fore.LIGHTRED_EX))
                file.close()
        except FileNotFoundError as error:
            print(CM(f"The NEAT configuration file does not exist", Fore.LIGHTRED_EX) + f"; {error}")
        return False

    def set(self, **parameters):
        for attr in vars(self).keys():
            for param, value in parameters.items():
                if attr in param:
                    setattr(self, attr, value)
                    break

    def get(self, var: str, default: Any = None):
        if var in vars(self):
            return var
        else:
            return default


def text_fill(text: str, fill=20):
    return text + (" " * max(0, fill - len(text)))
