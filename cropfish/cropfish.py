import cv2
import numpy as np
from typing import Tuple, Union

def scale_corners(coord1: np.ndarray, coord2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Scale the coordinates of a rectangle from a 6x6 checkerboard to an 8x8 checkerboard.

    Args:
        coord1 (np.ndarray): Coordinates of the first corner.
        coord2 (np.ndarray): Coordinates of the second corner.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Scaled coordinates for the 8x8 checkerboard rectangle.
    """
    center = (coord1 + coord2) / 2
    scaling_factor = 8 / 6
    scaled_coord1 = center + (coord1 - center) * scaling_factor
    scaled_coord2 = center + (coord2 - center) * scaling_factor
    return scaled_coord1, scaled_coord2

def find_corners(image: np.ndarray, resize_factor: float) -> Tuple[Union[np.ndarray, None], Union[np.ndarray, None]]:
    """
    Find the corners of a checkerboard in the resized image.

    Args:
        image (np.ndarray): Resized image in grayscale.
        resize_factor (float): Factor by which the image is resized.

    Returns:
        Tuple[Union[np.ndarray, None], Union[np.ndarray, None]]: Coordinates of the found corners.
    """
    resized_image = cv2.resize(image, None, fx=resize_factor, fy=resize_factor)
    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    for threshold_value in (160, 192):
        print(f"Trying threshold {threshold_value}")
        binary = (gray < threshold_value).astype(np.uint8) * 255
        pattern_size = (7, 7)
        ret, corners = cv2.findChessboardCorners(binary, pattern_size, None)
        if ret:
            corners *= 1 / resize_factor
            return corners[0][0], corners[-1][0]
    print("Checkerboard corners not found.")
    return None, None

def crop_image(image: np.ndarray, coord1: np.ndarray, coord2: np.ndarray) -> np.ndarray:
    """
    Crop the image based on the provided coordinates.

    Args:
        image (np.ndarray): Original image.
        coord1 (np.ndarray): Coordinates of the first corner.
        coord2 (np.ndarray): Coordinates of the second corner.

    Returns:
        np.ndarray: Cropped image.
    """
    min_x, max_x = int(min(coord1[0], coord2[0])), int(max(coord1[0], coord2[0]))
    min_y, max_y = int(min(coord1[1], coord2[1])), int(max(coord1[1], coord2[1]))
    cropped_image = image[min_y:max_y, min_x:max_x]
    return cropped_image

@click.command(
    context_settings=dict(help_option_names=["-h", "--help"]),
    help=__doc__,
)
@click.argument("image_path", type=click.Path(exists=True), nargs=1)
def cropfish_cli(image_path) -> None:
    """
    Main function to find checkerboard corners, scale them, and crop the image.
    """
    image = cv2.imread(image_path)
    resize_factor = 0.5
    coord1, coord2 = find_corners(image, resize_factor)
    if coord1 is not None and coord2 is not None:
        scaled_coord1, scaled_coord2 = scale_corners(coord1, coord2)
        print(f"Coordinates for the 8x8 checkerboard rectangle: {scaled_coord1} {scaled_coord2}")
        cropped_image = crop_image(image, scaled_coord1, scaled_coord2)
        cv2.imwrite("output.png", cropped_image)

if __name__ == "__main__":
    cropfish_cli()
