# Camera Calibration Tool for football courts

A tool that helps estimate camera parameters for a given image of football court. The tool simulates the perspective view of a camera, given a set of camera parameters. 
Current camera parameters include:
- focal length
- camera location x, y, z
- tilt, pan and roll angles
- barrel distortion parameters (first 3)

### How it works
- Should you wish to calibrate an image, first uncomment related code block in update_image() function.
- Execute ```$ python camera_tool.py``` from the directory of camera_tool.py.
- Use sliders to adapt court template on the image.
- Current camera parameters are shown at the upper left window corner. Respective homography matrix for hawk-eye perspective transformation is shown at the upper right window corner.
- Record camera parameter set using 1st slider. Parameters sets are autosaved in 'saved_camera_param_data/' (make dir first!) when you quit the tool with Q

### Keyboard controls
- Change tilt angle: W-S
- Change pan angle: A-D
- Quit tool: Q

### Notes
Current implementation misses tangential distortion simulation.