import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import os




def load_image(path=None):

    if path and os.path.exists(path):
        # cv2.imread returns a BGR image as a NumPy array
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            print(f"Could not load '{path}'. Using test image instead.")
            return create_test_image()
        # Convert BGR → RGB so matplotlib shows correct colors
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        print(f"Loaded: {path}  shape={img_rgb.shape}  dtype={img_rgb.dtype}")
        return img_rgb
    else:
        print("No image path provided. Generating a colorful test image.")
        return create_test_image()


def create_test_image():
    """
    Build a 400×600 synthetic image with:
      - A red rectangle
      - A green circle
      - A blue triangle
      - A yellow gradient background

    This ensures you always have something interesting to explore
    even without a real photo.

    Returns:
        img: 400 x 600 x 3 uint8 RGB image
    """
    h, w = 400, 600
    img = np.zeros((h, w, 3), dtype=np.uint8)

    # Yellow gradient background (red + green channels)
    for x in range(w):
        intensity = int(255 * x / w)          # ramps 0 → 255 left to right
        img[:, x, 0] = intensity               # Red channel
        img[:, x, 1] = intensity // 2 + 64    # Green channel (offset)

    # Red filled rectangle — top-left region
    img[50:200, 50:250, 0] = 220   # Red channel high
    img[50:200, 50:250, 1] = 30    # Green channel low
    img[50:200, 50:250, 2] = 30    # Blue channel low

    # Green filled circle — center
    center = (w // 2, h // 2)
    for y in range(h):
        for x in range(w):
            dist = np.sqrt((x - center[0])**2 + (y - center[1])**2)
            if dist < 80:
                img[y, x] = [30, 200, 30]      # Green pixel

    # Blue filled triangle — bottom-right (using fillPoly)
    pts = np.array([[400, 350], [580, 350], [490, 220]], dtype=np.int32)
    # cv2 expects BGR but we'll work in our RGB array directly
    # We draw blue: R=30, G=30, B=220
    cv2.fillPoly(img, [pts], color=(30, 30, 220))

    # White text label so we know which image this is
    cv2.putText(img, "Test Image - CV Stage 1",
                (10, 390), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2)

    print(f"Test image created: shape={img.shape}  dtype={img.dtype}")
    return img


# =============================================================
# PART 2 — SPLIT INTO RGB CHANNELS
# =============================================================

def split_rgb_channels(img_rgb):
    """
    Separate a color image into its Red, Green, and Blue planes.

    Each channel is a 2D (H x W) array with values 0–255.
    We also build 'colorized' versions: a 3-channel image where
    only one channel is active (the others are set to 0).
    This lets you see the channel contribution in its actual color.

    Args:
        img_rgb: H x W x 3 RGB image

    Returns:
        channels: dict with keys 'R', 'G', 'B'
                  each value is a 2D uint8 array
        colorized: dict with keys 'R', 'G', 'B'
                   each value is a 3-channel image showing only that color
    """
    # Split returns three 2D arrays
    r_channel = img_rgb[:, :, 0]   # shape: (H, W)
    g_channel = img_rgb[:, :, 1]
    b_channel = img_rgb[:, :, 2]

    channels = {'R': r_channel, 'G': g_channel, 'B': b_channel}

    # Build colorized versions by zeroing out other channels
    colorized = {}

    r_colored = np.zeros_like(img_rgb)
    r_colored[:, :, 0] = r_channel           # only red channel active
    colorized['R'] = r_colored

    g_colored = np.zeros_like(img_rgb)
    g_colored[:, :, 1] = g_channel           # only green channel active
    colorized['G'] = g_colored

    b_colored = np.zeros_like(img_rgb)
    b_colored[:, :, 2] = b_channel           # only blue channel active
    colorized['B'] = b_colored

    # Print some stats about each channel
    for name, ch in channels.items():
        print(f"  Channel {name}: min={ch.min():3d}  max={ch.max():3d}  "
              f"mean={ch.mean():.1f}  std={ch.std():.1f}")

    return channels, colorized


# =============================================================
# PART 3 — CONVERT TO HSV AND SPLIT
# =============================================================

def split_hsv_channels(img_rgb):
    """
    Convert RGB image to HSV (Hue, Saturation, Value) color space
    and split into its three components.

    HSV is more intuitive for color-based operations:
      - H (Hue):        0–179  → the color type (red, green, blue…)
      - S (Saturation): 0–255  → color purity (0 = gray, 255 = vivid)
      - V (Value):      0–255  → brightness

    Args:
        img_rgb: H x W x 3 RGB image

    Returns:
        hsv_img:  H x W x 3 HSV image (OpenCV convention)
        h, s, v:  individual 2D channel arrays
    """
    # OpenCV HSV conversion expects BGR input
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    hsv_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    h = hsv_img[:, :, 0]   # Hue:        0–179  (OpenCV uses half range)
    s = hsv_img[:, :, 1]   # Saturation: 0–255
    v = hsv_img[:, :, 2]   # Value:      0–255

    print(f"\n  HSV Hue:        min={h.min():3d}  max={h.max():3d}  mean={h.mean():.1f}")
    print(f"  HSV Saturation: min={s.min():3d}  max={s.max():3d}  mean={s.mean():.1f}")
    print(f"  HSV Value:      min={v.min():3d}  max={v.max():3d}  mean={v.mean():.1f}")

    return hsv_img, h, s, v


# =============================================================
# PART 4 — COMPUTE HISTOGRAMS
# =============================================================

def compute_histogram(channel, num_bins=256):
    """
    Compute a pixel value histogram for one channel.

    A histogram counts how many pixels have each intensity value (0–255).
    Peaks tell you the dominant intensities; spread tells you contrast.

    Args:
        channel:  2D uint8 array (one image channel)
        num_bins: number of histogram bins (default 256 = one per value)

    Returns:
        hist:   array of length num_bins, each entry = pixel count
        bins:   array of bin edge values (length num_bins+1)
    """
    hist, bins = np.histogram(channel.ravel(),    # .ravel() flattens to 1D
                              bins=num_bins,
                              range=(0, 256))
    return hist, bins


# =============================================================
# PART 5 — VISUALIZATION
# =============================================================

def plot_rgb_analysis(img_rgb, channels, colorized):
    """
    Create a figure showing:
      Row 1: original image + three colorized channel views
      Row 2: three per-channel histogram plots

    Args:
        img_rgb:    original RGB image
        channels:   dict of 2D grayscale channel arrays
        colorized:  dict of 3-channel colorized images
    """
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle("RGB Channel Explorer", fontsize=16, fontweight='bold')

    # --- Row 1: images ---

    # Column 0: original image
    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title("Original (RGB)", fontsize=12)
    axes[0, 0].axis('off')

    # Columns 1-3: colorized channels
    channel_names = ['R', 'G', 'B']
    titles = ['Red Channel', 'Green Channel', 'Blue Channel']
    for i, (name, title) in enumerate(zip(channel_names, titles)):
        axes[0, i+1].imshow(colorized[name])
        axes[0, i+1].set_title(title, fontsize=12)
        axes[0, i+1].axis('off')

    # --- Row 2: histograms ---

    # Column 0: all three channels overlaid
    colors = ['red', 'green', 'blue']
    for name, color in zip(channel_names, colors):
        hist, bins = compute_histogram(channels[name])
        axes[1, 0].plot(bins[:-1], hist, color=color, alpha=0.7, linewidth=1.5)
    axes[1, 0].set_title("All Channels Overlay", fontsize=12)
    axes[1, 0].set_xlabel("Pixel Intensity (0–255)")
    axes[1, 0].set_ylabel("Pixel Count")
    axes[1, 0].legend(['Red', 'Green', 'Blue'])
    axes[1, 0].set_xlim([0, 255])

    # Columns 1-3: individual histograms
    for i, (name, color, title) in enumerate(zip(channel_names, colors, titles)):
        hist, bins = compute_histogram(channels[name])
        axes[1, i+1].fill_between(bins[:-1], hist, color=color, alpha=0.6)
        axes[1, i+1].plot(bins[:-1], hist, color=color, linewidth=1)
        mean_val = channels[name].mean()
        axes[1, i+1].axvline(mean_val, color='black', linestyle='--',
                             linewidth=1.5, label=f'Mean={mean_val:.1f}')
        axes[1, i+1].set_title(f"{title} Histogram", fontsize=12)
        axes[1, i+1].set_xlabel("Pixel Intensity (0–255)")
        axes[1, i+1].set_ylabel("Pixel Count")
        axes[1, i+1].legend(fontsize=9)
        axes[1, i+1].set_xlim([0, 255])

    plt.tight_layout()
    plt.savefig("rgb_analysis.png", dpi=120, bbox_inches='tight')
    print("\nSaved: rgb_analysis.png")
    plt.show()


def plot_hsv_analysis(img_rgb, h, s, v):
    """
    Create a figure showing the HSV decomposition.

    Args:
        img_rgb: original RGB image
        h, s, v: individual HSV channel arrays (2D)
    """
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle("HSV Channel Explorer", fontsize=16, fontweight='bold')

    # Row 1: images

    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title("Original (RGB)", fontsize=12)
    axes[0, 0].axis('off')

    # Hue shown with a cyclic colormap (hsv map wraps around)
    axes[0, 1].imshow(h, cmap='hsv', vmin=0, vmax=179)
    axes[0, 1].set_title("Hue (0=Red, 60=Yellow,\n90=Green, 120=Cyan, 150=Blue)", fontsize=10)
    axes[0, 1].axis('off')
    plt.colorbar(axes[0, 1].images[0], ax=axes[0, 1], fraction=0.04)

    # Saturation: gray = unsaturated, white = vivid
    axes[0, 2].imshow(s, cmap='Greys_r', vmin=0, vmax=255)
    axes[0, 2].set_title("Saturation\n(bright=vivid, dark=gray)", fontsize=10)
    axes[0, 2].axis('off')
    plt.colorbar(axes[0, 2].images[0], ax=axes[0, 2], fraction=0.04)

    # Value: brightness map
    axes[0, 3].imshow(v, cmap='Greys_r', vmin=0, vmax=255)
    axes[0, 3].set_title("Value (Brightness)\n(bright=light, dark=dark)", fontsize=10)
    axes[0, 3].axis('off')
    plt.colorbar(axes[0, 3].images[0], ax=axes[0, 3], fraction=0.04)

    # Row 2: histograms

    axes[1, 0].axis('off')   # placeholder

    # Hue histogram — use a rainbow colormap for the bars
    hist_h, bins_h = compute_histogram(h, num_bins=180)
    bar_colors = [plt.cm.hsv(i / 180) for i in range(180)]
    axes[1, 1].bar(range(180), hist_h, color=bar_colors, width=1)
    axes[1, 1].set_title("Hue Histogram", fontsize=12)
    axes[1, 1].set_xlabel("Hue (0–179)")
    axes[1, 1].set_ylabel("Pixel Count")

    # Saturation histogram
    hist_s, bins_s = compute_histogram(s)
    axes[1, 2].fill_between(bins_s[:-1], hist_s, color='purple', alpha=0.6)
    axes[1, 2].set_title("Saturation Histogram", fontsize=12)
    axes[1, 2].set_xlabel("Saturation (0–255)")
    axes[1, 2].set_ylabel("Pixel Count")
    axes[1, 2].set_xlim([0, 255])

    # Value histogram
    hist_v, bins_v = compute_histogram(v)
    axes[1, 3].fill_between(bins_v[:-1], hist_v, color='gray', alpha=0.6)
    axes[1, 3].set_title("Value (Brightness) Histogram", fontsize=12)
    axes[1, 3].set_xlabel("Value (0–255)")
    axes[1, 3].set_ylabel("Pixel Count")
    axes[1, 3].set_xlim([0, 255])

    plt.tight_layout()
    plt.savefig("hsv_analysis.png", dpi=120, bbox_inches='tight')
    print("Saved: hsv_analysis.png")
    plt.show()


def show_channel_stats(img_rgb, channels_rgb, h, s, v):
    """
    Print a clean stats table for all channels in both color spaces.
    """
    print("\n" + "="*60)
    print("  IMAGE CHANNEL STATISTICS")
    print("="*60)
    print(f"  Image shape: {img_rgb.shape}")
    print(f"  Total pixels: {img_rgb.shape[0] * img_rgb.shape[1]:,}")
    print()
    print(f"  {'Channel':<14} {'Min':>5} {'Max':>5} {'Mean':>8} {'Std':>8} {'Median':>8}")
    print("  " + "-"*54)

    for name, ch in channels_rgb.items():
        print(f"  RGB-{name:<10} {ch.min():>5} {ch.max():>5} "
              f"{ch.mean():>8.1f} {ch.std():>8.1f} "
              f"{np.median(ch):>8.1f}")

    for name, ch in zip(['HSV-H', 'HSV-S', 'HSV-V'], [h, s, v]):
        print(f"  {name:<14} {ch.min():>5} {ch.max():>5} "
              f"{ch.mean():>8.1f} {ch.std():>8.1f} "
              f"{np.median(ch):>8.1f}")
    print("="*60)


# =============================================================
# MAIN — Run everything
# =============================================================

def main():
    # Accept optional image path from command line
    image_path = sys.argv[1] if len(sys.argv) > 1 else None

    print("="*60)
    print("  PROJECT 1 — IMAGE COLOR CHANNEL EXPLORER")
    print("="*60)

    # Step 1: Load image
    print("\n[1] Loading image...")
    img_rgb = load_image(image_path)

    # Step 2: RGB channel analysis
    print("\n[2] Splitting RGB channels...")
    rgb_channels, rgb_colorized = split_rgb_channels(img_rgb)

    # Step 3: HSV channel analysis
    print("\n[3] Converting to HSV and splitting...")
    hsv_img, h, s, v = split_hsv_channels(img_rgb)

    # Step 4: Print statistics table
    show_channel_stats(img_rgb, rgb_channels, h, s, v)

    # Step 5: Plot RGB analysis
    print("\n[4] Plotting RGB channel analysis...")
    plot_rgb_analysis(img_rgb, rgb_channels, rgb_colorized)

    # Step 6: Plot HSV analysis
    print("\n[5] Plotting HSV channel analysis...")
    plot_hsv_analysis(img_rgb, h, s, v)

    print("\nDone! Check rgb_analysis.png and hsv_analysis.png")


if __name__ == "__main__":
    main()
