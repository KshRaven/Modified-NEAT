from ModifiedNEAT.util.fancy_text import *

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
TRIES = 10
errored = False
for trial_idx in range(TRIES):
    try:
        # print(f"{trial_idx} = {PROJECT_DIR}")
        if PROJECT_DIR.endswith("ModifiedNEAT"):
            PROJECT_DIR = os.path.dirname(PROJECT_DIR)
            break
        elif PROJECT_DIR.endswith("Modified-NEAT"):
            PROJECT_DIR = PROJECT_DIR
            break
        PROJECT_DIR = os.path.dirname(PROJECT_DIR)
    except Exception as e:
        print(CM(e, Fore.LIGHTRED_EX))
        errored = True
    if trial_idx == TRIES-1 or errored:
        # Define possible gpu directories
        directories = ["./home", "C:/Users/Default/AppData/Local"]
        # Check which directory exists
        PROJECT_DIR = "./root"
        for base_dir in directories:
            if os.path.exists(base_dir):
                PROJECT_DIR = base_dir + '/PythonProjectData'
                break
STORAGE_DIR = PROJECT_DIR + "/storage/"
FILE_INDEX_PADDING = 3  # Zero-padding for file indices (e.g., '003', '004')


def set_storage_location(directory: str = STORAGE_DIR):
    global STORAGE_DIR
    STORAGE_DIR = directory


def get_last_index(filename: str, directory: str, extension: str = '.pkl', 
                   save_location: str = None) -> int:
    """Get the latest index available for a filename in a directory.
    
    Searches for all numbered files matching the pattern: filename-N<extension>
    Returns the highest N found, or 0 if no numbered files exist.
    
    Args:
        filename: Base filename (without extension or number)
        directory: Directory to search
        extension: File extension to search for (default: '.pkl')
        save_location: Base save location (default: STORAGE_DIR)
    
    Returns:
        int: The highest index found (0 if no numbered files exist)
    """
    if save_location is None:
        save_location = STORAGE_DIR
    
    full_dir = save_location + f"{directory}/"
    
    if not os.path.exists(full_dir):
        return 0
    
    max_index = 0
    try:
        for file in os.listdir(full_dir):
            # Match pattern: filename-<number><extension>
            if file.startswith(filename + "-") and file.endswith(extension):
                # Extract the number part
                number_part = file[len(filename) + 1:-len(extension)]
                try:
                    index = int(number_part)
                    max_index = max(max_index, index)
                except ValueError:
                    continue
    except (OSError, PermissionError):
        pass
    
    return max_index


def get_next_index(filename: str, directory: str, extension: str = '.pkl', 
                   save_location: str = None) -> int:
    """Get the next available index for a filename in a directory.
    
    Returns the next index to use for saving. If only one or zero files exist,
    starts at 100; otherwise increments the latest found index.
    
    Args:
        filename: Base filename (without extension or number)
        directory: Directory to search
        extension: File extension (default: '.pkl')
        save_location: Base save location (default: STORAGE_DIR)
    
    Returns:
        int: The next available index, formatted with FILE_INDEX_PADDING
    """
    last_idx = get_last_index(filename, directory, extension, save_location)
    
    if last_idx == 0:
        # If no files exist, start at 100
        return 1 # 100
    else:
        # Increment the last index
        return last_idx + 1


def save(items: dict[str, Any], filename: str = '', directory: str = '', file_no: int = None, replace=False,
         subdirectory: str = None, save_location: str = None, extension: str = None,
         items_name: str = None, time: int = None, debug=True, filepath: str = None):
    if filepath is None:
        if isinstance(file_no, int) and file_no == 0:
            file_no = None
        # Use default directory and name as subdirectory
        if save_location is None:
            save_location = STORAGE_DIR
        directory = save_location + f"{directory}/"
        if subdirectory is not None:
            directory = directory + f"{subdirectory}/"

        # Get  file_path
        if extension is None:
            extension = '.pkl'
        if file_no is None:
            filepath = directory + filename + extension
            # Get the latest filepath if replace
            if replace:
                file_no = get_last_index(filename, directory.replace(save_location, '').rstrip('/'), 
                                        extension, save_location)
                if file_no == 0: _file_no = ''
                else: _file_no = f"-{file_no:0{FILE_INDEX_PADDING}d}"
                filepath = f'{directory}{filename}{_file_no}{extension}'
            # Else get the next name
            else:
                file_no = get_next_index(filename, directory.replace(save_location, '').rstrip('/'),
                                        extension, save_location)
                if file_no == 0: _file_no = ''
                else: _file_no = f"-{file_no:0{FILE_INDEX_PADDING}d}"
                filepath = f'{directory}{filename}{_file_no}{extension}'
        else:
            # Get specific file - use zero-padding if file_no is provided
            if file_no == 0: _file_no = ''
            else: _file_no = f"-{file_no:0{FILE_INDEX_PADDING}d}"
            filepath = f'{directory}{filename}{_file_no}{extension}'
    else:
        assert isinstance(filepath, str)

    # Serialize and save info
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        save_data = {
            'items': items,
            'time': time if time is not None else int(clock.time())
        }
        with open(filepath, 'wb') as file:
            if extension is not None and '.json' in extension:
                json.dump(save_data, file)
            else:
                pickle.dump(save_data, file)
        if debug:
            print(CM(f"Successfully dumped {items_name if items_name is not None else ''} "
                     f"save file to '{filepath}'.", PRINT_COLOUR))

        return True, file_no
    except Exception as e:
        print(CM(f"\nUtilityError: Failed to save the items: {items}; \n{e}."), Fore.LIGHTRED_EX)
        return False, file_no


def load(filename: str = '', directory: str = '', file_no: int = None,
         subdirectory: str = None, save_location: str = None, extension: str = None,
         items_name: str = None, time: int = None, cooldown: int = None, debug=True,
         filepath: str = None):
    try:
        if filepath is None:
            if isinstance(file_no, int) and file_no == 0:
                file_no = None
            # Use default directory and name as subdirectory
            if save_location is None:
                save_location = STORAGE_DIR
            directory = save_location + f"{directory}/"
            if subdirectory is not None:
                directory = directory + f"{subdirectory}/"

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
                # Get the latest file number
                last_idx = get_last_index(filename, directory.replace(save_location, '').rstrip('/'), 
                                        extension, save_location)
                if last_idx > 0:
                    filepath = f'{directory}{filename}-{last_idx:0{FILE_INDEX_PADDING}d}{extension}'
                else:
                    # No numbered files found, use base filename
                    filepath = directory + filename + extension
            elif file_no > 0:
                filepath = f'{directory}{filename}-{file_no:0{FILE_INDEX_PADDING}d}{extension}'
            else:
                raise NotADirectoryError(f"Failed to load save folder; invalid 'file_no'")
        else:
            assert isinstance(filepath, str)

        # Check if Save file exists
        if os.path.exists(filepath) is False:
            raise NotADirectoryError(f"Failed to load save file; '{filepath}' does no exist")

        with open(filepath, 'rb') as file:
            if extension is not None and '.json' in extension:
                save_data: tuple[dict[str, Any], int] = json.load(file)
            else:
                # Use standard pickle.load() - the new save/load methods in NeatModule
                # and Population use proper serialization with neat_dict()/load_neat_dict()
                # that don't require custom unpickling for class resolution
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
            print(CM(f"Successfully loaded {items_name if items_name is not None else ''} "
                     f"save file from '{filepath}'.", PRINT_COLOUR))
        return items
    except Exception as e:
        import traceback
        print(f"\n{CM(f'UtilityError: {str(e)}\n{traceback.format_exc()}', Fore.LIGHTRED_EX)}")
        return None


def delete(filename: str, directory: str, file_no: Union[int, None],
           subdirectory: str = None, save_location: str = None, extension: str = None, disable_warn=False, debug=True):
    try:
        if isinstance(file_no, int) and file_no == 0:
            file_no = None
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
            filepath = directory + '/' + filename + extension
        elif file_no is None:
            filepath = directory
            if not disable_warn:
                warnings.warn(f"\nThe entire directory {filepath} will be deleted")
                confirm = input(f"Are you sure you want to delete (yes / no): ")
                if confirm not in ['yes', 'y', 'True', '1']:
                    return None
            entire_dir = True
        elif file_no > 0:
            filepath = directory + f"/{filename}-{file_no:0{FILE_INDEX_PADDING}d}{extension}"
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
