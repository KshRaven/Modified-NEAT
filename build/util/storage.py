
from build.util.fancy_text import *

from typing import Any, Union

import os
import shutil
import pickle
import json
import time as clock
import warnings


PRINT_COLOUR = Fore.GREEN
CURRENT_DIR = os.path.abspath(__file__)
PROJECT_DIR = CURRENT_DIR
while not PROJECT_DIR.endswith("Modified-NEAT"):
    PROJECT_DIR = os.path.dirname(PROJECT_DIR)
STORAGE_DIR = PROJECT_DIR + "\\storage\\"


def save(items: dict[str, Any], filename: str, directory: str, file_no: int = None, replace=False,
         subdirectory: str = None, save_location: str = None, extension: str = None,
         items_name: str = None, time: int = None, debug=True) -> bool:
    # Use default directory and name as subdirectory
    if save_location is None:
        save_location = STORAGE_DIR
    directory = save_location + f"{directory}\\"
    if subdirectory is not None:
        directory = directory + f"{subdirectory}\\"

    # Get  file_path
    if extension is None:
        extension = '.pkl'
    if file_no is None:
        filepath = directory + filename + extension
        # Get the latest filepath if replace
        if replace:
            counter = 0
            while True:
                counter += 1
                filepath_to_check = f'{directory+filename}-{counter}{extension}'
                if os.path.exists(filepath_to_check):
                    filepath = filepath_to_check
                else:
                    break
        # Else get the next name
        else:
            counter = 0
            while os.path.exists(filepath):
                counter += 1
                filepath = f'{directory + filename}-{counter}{extension}'
    else:
        # Get specific file
        filepath = f'{directory + filename}-{file_no}{extension}'

    # Serialize and save genome info

    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        save_data = {
            'items': items,
            'time': time if time is not None else int(clock.time())
        }
        with open(filepath, 'wb') as file:
            if extension == '.json':
                json.dump(save_data, file)
            else:
                pickle.dump(save_data, file)
        if debug:
            print(CM(f"\nSuccessfully dumped {items_name if items_name is not None else ''} "
                     f"save file to '{filepath}'.", PRINT_COLOUR))

        return True
    except Exception as e:
        raise RuntimeError(f"\nUtilityError: Failed to save the items: {items}; \n{e}.")


def load(filename: str, directory: str, file_no: int = None,
         subdirectory: str = None, save_location: str = None, extension: str = None,
         items_name: str = None, time: int = None, cooldown: int = None, debug=True):
    try:
        # Use default directory and name as subdirectory
        if save_location is None:
            save_location = STORAGE_DIR
        directory = save_location + f"{directory}\\"
        if subdirectory is not None:
            directory = directory + f"{subdirectory}\\"

        # Check if Save folder exists
        if os.path.exists(directory) is False:
            raise NotADirectoryError(f"Failed to load save folder; '{directory}' does no exist")

        # Get filepath
        if extension is None:
            extension = '.pkl'
        filepath = directory + filename + extension

        if file_no == 0:
            filepath = directory + filename + extension
        elif file_no is None:
            counter = 0
            while True:
                counter += 1
                filepath_to_check = f'{directory+filename}-{counter}{extension}'
                if os.path.exists(filepath_to_check):
                    filepath = filepath_to_check
                else:
                    break
        elif file_no > 0:
            filepath = f'{directory+filename}-{file_no}{extension}'
        else:
            raise NotADirectoryError(f"Failed to load save folder; invalid 'file_no'")

        # Check if Save file exists
        if os.path.exists(filepath) is False:
            raise NotADirectoryError(f"Failed to load save file; '{filepath}' does no exist")

        with open(filepath, 'rb') as file:
            if extension == '.json':
                save_data: tuple[dict[str, Any], int] = json.load(file)
            else:
                save_data: tuple[dict[str, Any], int] = pickle.load(file)
            items = save_data['items']
            time_of_save = save_data['time']
        if time is not None and time_of_save is not None:
            if cooldown is None:
                cooldown = 0
            if time < time_of_save + cooldown:
                return None
        if items is None:
            raise ValueError(f"No items found in save file {filepath}")
        if debug:
            print(CM(f"\nSuccessfully loaded {items_name if items_name is not None else ''} "
                     f"save file from '{filepath}'.", PRINT_COLOUR))
        return items
    except Exception as e:
        print(f"\n{CM(f'UtilityError: {e}', Fore.LIGHTRED_EX)}")
        return None


def delete(filename: str, directory: str, file_no: Union[int, None],
           subdirectory: str = None, save_location: str = None, extension: str = None, disable_warn=False, debug=True):
    try:
        # Use default directory and name as subdirectory
        if save_location is None:
            save_location = STORAGE_DIR
        directory = save_location + f"{directory}"
        if subdirectory is not None:
            directory = directory + f"{subdirectory}"

        # Check if Save folder exists
        if os.path.exists(directory) is False:
            raise NotADirectoryError(f"Failed to delete save folder; '{directory}' does no exist")

        # Get filepath
        if extension is None:
            extension = '.pkl'

        entire_dir = False
        if file_no == 0:
            filepath = directory + '\\' + filename + extension
        elif file_no is None:
            filepath = directory
            if not disable_warn:
                warnings.warn(f"\nThe entire directory {filepath} will be deleted")
                confirm = input(f"Are you sure you want to delete (yes / no): ")
                if confirm not in ['yes', 'y', 'True', '1']:
                    return None
            entire_dir = True
        elif file_no > 0:
            filepath = directory + f"\\{filename}-{file_no}{extension}"
        else:
            raise NotADirectoryError(f"Failed to delete save folder; invalid 'file_no'")

        # Check if Save file exists
        if os.path.exists(filepath) is False:
            raise NotADirectoryError(f"Failed to delete save file; '{filepath}' does no exist")

        # Delete file
        if not entire_dir:
            os.remove(filepath)
            if debug:
                print(CM(f"\nSuccessfully deleted {filename + extension} in {directory}.", PRINT_COLOUR))
        else:
            shutil.rmtree(filepath)
            if debug:
                print(CM(f"\nSuccessfully deleted {directory}.", PRINT_COLOUR))
    except NotADirectoryError as e:
        if not disable_warn:
            print(f"\n{CM(f'UtilityError: {e}', Fore.LIGHTRED_EX)}")
        raise e
    except OSError as e:
        print(f"Error deleting file '{filename}': {e}")
        raise e
