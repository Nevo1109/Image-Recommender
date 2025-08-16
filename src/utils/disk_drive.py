import win32api
import imagesize


def get_drive_letter() -> str:
    """Search for image storage drive and return it's letter (e.g. 'D', 'E', 'F')"""
    
    drives = win32api.GetLogicalDriveStrings()
    drives = drives.split("\000")[:-1]
    for drive in drives:
        if "BigData" in win32api.GetVolumeInformation(drive)[0]:
            return drive[0]  # only letter
    raise Exception("Drive could not be found automaticly.")

def get_data_folder():
    """Get default recommender data on image storage drive"""
    drive = get_drive_letter()
    return drive + ":\\data\\recommender_data"

def get_db_path():
    return get_data_folder() + "\\images.db"

def get_images_folder():
    """Get default image folder path on image storage drive"""
    drive = get_drive_letter()
    return f"{drive}:\\data\\image_data"

def get_image_size(path: str) -> tuple[int, int, int, str]:
    """Get pixel dimensions of an image

    Args:
        path (str): absolute path to image

    Returns:
        tuple: (width, height, pixelcount, path)
    """
    
    if path[0] == ":":
        path = get_drive_letter() + path
        
    w, h = imagesize.get(path)
    return w, h, w * h, path
