# PROJECT — IMAGE COLOR CHANNEL EXPLORER
  - How images are stored as NumPy arrays
  - What RGB and HSV color spaces mean
  - How to split and visualize individual channels
  - How to plot per-channel histograms
  - What a histogram tells you about brightness/contrast

# Requirements:
    pip install opencv-python numpy matplotlib

# Run:
    python color_channel_explorer.py
    (uses a built-in generated image if no image path is given)


# PART 1 — LOAD OR CREATE A TEST IMAGE


    """
    Load an image from disk, or create a colorful test image
    if no path is provided.

    OpenCV reads images in BGR order (not RGB), so we convert
    to RGB immediately so everything downstream is consistent.

    Args:
        path: filepath string, or None to use test image

    Returns:
        img_rgb: H x W x 3 uint8 NumPy array in RGB order
    """
