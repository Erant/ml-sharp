# Orbital Rendering Script

This script renders gaussian splats from arbitrary orbital viewpoints with full camera control.

## Features

- **Center of Mass Calculation**: Automatically computes and rotates around the splat center
- **Y-axis Rotation**: Horizontal orbit (left/right around object)
- **X-axis Rotation**: Vertical orbit (up/down around object)
- **Z-axis Translation**: Move closer/farther along camera-to-center vector
- **Look-at Constraint**: Camera always points at center of mass
- **RGBA Output**: Full transparency support in output PNG

## Coordinate System

- **X-axis**: Left/Right (side-to-side)
- **Y-axis**: Up/Down (vertical)
- **Z-axis**: Depth (distance, forward/backward)

## Usage

### Basic Usage

```bash
python render_orbit.py -i input.ply -o output.png
```

### Rotate Horizontally (Y-axis)

```bash
# Rotate 45 degrees around Y-axis (horizontal orbit)
python render_orbit.py -i input.ply -o output.png --rotate-y 45
```

### Rotate Vertically (X-axis)

```bash
# Tilt camera 30 degrees up
python render_orbit.py -i input.ply -o output.png --rotate-x 30

# Tilt camera 30 degrees down
python render_orbit.py -i input.ply -o output.png --rotate-x -30
```

### Move Closer/Farther (Z-axis)

```bash
# Move 2 units closer to object
python render_orbit.py -i input.ply -o output.png --translate-z -2

# Move 5 units farther from object
python render_orbit.py -i input.ply -o output.png --translate-z 5
```

### Combined Transformations

```bash
# Orbit 45° horizontally, tilt 15° up, zoom in by 3 units
python render_orbit.py -i input.ply -o output.png \
    --rotate-y 45 \
    --rotate-x 15 \
    --translate-z -3
```

### Custom Resolution

```bash
# Render at 2048x2048
python render_orbit.py -i input.ply -o output.png \
    --width 2048 \
    --height 2048
```

### Generate 360° Turntable

```bash
# Generate frames for a full rotation
for angle in {0..360..10}; do
    python render_orbit.py -i input.ply \
        -o turntable/frame_$(printf "%03d" $angle).png \
        --rotate-y $angle
done

# Convert to video with ffmpeg
ffmpeg -framerate 30 -i turntable/frame_%03d.png -c:v libx264 -pix_fmt yuv420p turntable.mp4
```

## Command-Line Options

```
Options:
  -i, --input-ply PATH      Path to input PLY file [required]
  -o, --output-png PATH     Path to output PNG file [required]
  --rotate-y FLOAT          Rotation around Y-axis (horizontal) in degrees [default: 0.0]
  --rotate-x FLOAT          Rotation around X-axis (vertical) in degrees [default: 0.0]
  --translate-z FLOAT       Translation along camera-to-center vector [default: 0.0]
                            Positive = move away, Negative = move closer
  --width INTEGER           Output image width in pixels [default: 1024]
  --height INTEGER          Output image height in pixels [default: 1024]
  --device TEXT             Device to use for rendering (cuda/cpu) [default: cuda]
  --help                    Show this message and exit
```

## How It Works

1. **Load PLY**: Loads gaussian splats from file
2. **Compute Center**: Calculates center of mass of all splats
3. **Original Viewpoint**: Camera starts at origin (0, 0, 0) looking down +Z axis
4. **Apply Rotations**: Rotates camera around center of mass
   - Y-axis rotation applied first (horizontal orbit)
   - X-axis rotation applied second (vertical tilt)
5. **Apply Translation**: Moves camera along the camera-to-center vector
6. **Build View Matrix**: Creates look-at matrix pointing at center of mass
7. **Render**: Uses gsplat to rasterize splats from camera viewpoint
8. **Save RGBA**: Outputs PNG with full alpha channel support

## Technical Details

### Transformation Order

The transformations are applied in this order:
1. Translate center of mass to origin
2. Apply Y-axis rotation (horizontal orbit)
3. Apply X-axis rotation (vertical tilt)
4. Translate back to original center
5. Apply Z-axis translation along camera-to-center vector
6. Build look-at matrix to point at center

### Camera Convention

Uses OpenCV convention:
- **Right-handed coordinate system**
- **+X**: Right
- **+Y**: Down
- **+Z**: Forward (away from camera)

### Rendering

- Uses gsplat rasterizer (CUDA accelerated)
- Supports linearRGB and sRGB color spaces
- Alpha channel represents splat opacity
- Background color: black (fully transparent)

## Requirements

- CUDA-capable GPU (recommended, CPU fallback available)
- Python packages: torch, PIL, click, gsplat
- All dependencies from ml-sharp requirements.txt

## Examples

### Create orbit visualization
```bash
# 8 views around object at 45° increments
for i in {0..7}; do
    angle=$((i * 45))
    python render_orbit.py -i object.ply \
        -o views/orbit_${angle}.png \
        --rotate-y $angle
done
```

### Elevation sweep
```bash
# Views from below to above
for elevation in {-45..45..15}; do
    python render_orbit.py -i object.ply \
        -o views/elevation_${elevation}.png \
        --rotate-x $elevation
done
```

### Zoom sequence
```bash
# Zoom in from far to close
for dist in {10..0..-2}; do
    python render_orbit.py -i object.ply \
        -o views/zoom_${dist}.png \
        --translate-z $dist
done
```
