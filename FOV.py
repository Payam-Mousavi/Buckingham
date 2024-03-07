import math

# Given values
H_pixels = 4032  # Height in pixels
W_pixels = 3024  # Width in pixels
D_pixels_per_inch2 = 9.707e3  # Pixel density in pixels^2 per inch^2
FOV_degrees = 77  # Field of view in degrees

# Calculate PPI (Pixels Per Inch)
PPI = math.sqrt(D_pixels_per_inch2)

# Calculate the physical dimensions of the image in inches
W_inch = W_pixels / PPI
H_inch = H_pixels / PPI

# Calculate the diagonal of the image in inches
D_inch = math.sqrt(W_inch**2 + H_inch**2)

# Calculate the height of the camera
# Convert FOV to radians for the tan function
FOV_radians = math.radians(FOV_degrees)
camera_height = (D_inch / 2) / math.tan(FOV_radians / 2)

camera_height
