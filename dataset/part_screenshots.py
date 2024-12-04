import os
from OCC.Extend.DataExchange import read_step_file
from OCC.Display.SimpleGui import init_display
from OCC.Core.V3d import V3d_XposYposZneg


def generate_screenshot_png(display, png_folder, filename, perspective='ISO'):
    """
    Generates a PNG screenshot of the current model view from the given perspective.
    """

    # Map perspectives to corresponding display methods
    perspective_methods = {
        'ISO': display.View_Iso,
        'ISOAlt': lambda: display.View.SetProj(V3d_XposYposZneg),
        'TOP': display.View_Top,
        'BOTTOM': display.View_Bottom,
        'LEFT': display.View_Left,
        'RIGHT': display.View_Right,
        'FRONT': display.View_Front,
        'REAR': display.View_Rear
    }

    # Select the method based on the given perspective
    method = perspective_methods.get(perspective)
    if not method:
        print(f"Invalid perspective: {perspective}. Skipping screenshot.")
        return

    # Set view and take the screenshot
    method()
    display.FitAll()
    image_path = os.path.join(png_folder, f'{filename}_{perspective}.png')
    display.ExportToImage(image_path)
    print(f'Screenshot saved: {image_path}')


if __name__ == '__main__':
    # INPUT:
    step_folder_path = r"insert_STEP_folder_path_here"
    # OUTPUT:
    png_folder = r"insert_png_folder_path_here"

    # Ensure PNG folder exists
    os.makedirs(png_folder, exist_ok=True)

    # Initialize display
    resolution = (1024, 768)
    display, start_display, add_menu, add_function_to_menu = init_display(
        None, resolution, True, [255, 255, 255], [255, 255, 255])
    
    # Configure display settings
    display.SetModeShaded()
    display.SetRenderingParams(
        IsShadowEnabled=False,
        IsReflectionEnabled=False,
        IsAntialiasingEnabled=False)
    display.hide_triedron()

    # List already taken screenshots
    existing_pngs = {f for f in os.listdir(png_folder) if f.endswith('.png')}

    # Process STEP files
    for step_file in os.listdir(step_folder_path):
        name, ext = os.path.splitext(step_file)

        if ext.lower() in ('.step', '.stp') and f'{name}_ISO.png' not in existing_pngs:
            step_file_path = os.path.join(step_folder_path, step_file)

            if os.path.isfile(step_file_path):
                try:
                    # Load STEP file
                    shape = read_step_file(step_file_path, as_compound=False)

                    # Display the shape and capture screenshots
                    display.DisplayShape(shape, update=True)
                    generate_screenshot_png(display, png_folder, name, perspective='ISO')
                    generate_screenshot_png(display, png_folder, name, perspective='ISOAlt')
                    # ... add more perspectives if needed ...
                    display.EraseAll()

                except Exception as e:
                    print(f"Error processing file {name}: {e}")
