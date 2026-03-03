import cv2
import cv2.aruco as aruco

def generate_charuco_board(output_path, squares_x, squares_y, square_length, marker_length, dictionary_id):
    """
    Generates a ChArUco board image and saves it to a file.
    
    Arguments:
    - output_path (str): The filename to save the board (e.g., 'charuco.png').
    - squares_x (int): Number of squares in the X direction.
    - squares_y (int): Number of squares in the Y direction.
    - square_length (float): The length of one side of the chessboard square (usually in mm or m).
    - marker_length (float): The length of one side of the ArUco marker (must be smaller than square_length).
    - dictionary_id (cv2.aruco.Dictionary): The ArUco dictionary to use.
    """
    
    # 1. Define the ArUco dictionary
    aruco_dict = aruco.getPredefinedDictionary(dictionary_id)
    
    # 2. Create the CharucoBoard object
    # Parameters: (squares_x, squares_y, square_length, marker_length, dictionary)
    board = aruco.CharucoBoard((squares_x, squares_y), square_length, marker_length, aruco_dict)
    
    # 3. Draw the board to an image
    # Note: The size (px) should be high enough for a crisp print. 
    # For an A4 sheet at 300 DPI, roughly 3500x2500 is good.
    img_size = (2000, 2000) 
    board_img = board.generateImage(img_size)
    
    # 4. Save the result
    cv2.imwrite(output_path, board_img)
    print(f"Board saved successfully to {output_path}")

# Example Usage:
# Using DICT_4X4_50 for 50 markers with a 4x4 grid resolution
generate_charuco_board(
    output_path="my_charuco_board.png",
    squares_x=5,
    squares_y=7,
    square_length=0.04, # e.g., 40mm
    marker_length=0.03, # e.g., 20mm
    dictionary_id=aruco.DICT_4X4_50
)