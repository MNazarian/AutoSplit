"""
STEP File Metadata Generator

This script processes STEP files to extract metadata including:
- Bounding box dimensions (x, y, z)
- Volume
- Surface area

Valid models must consist of exactly one solid body. Metadata is saved as JSON files
in the specified output folder.

Usage:
1. Set `step_folder_path` to the folder containing STEP files.
2. Set `info_folder_path` to the folder for saving JSON files.
3. Run the script.

Notes:
- Skips unreadable files or models without exactly one solid.
- Creates the output folder if it does not exist.

Dependencies:
- Python 3.x
- Open CASCADE Technology (OCC)
"""

import os
import json
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Extend.TopologyUtils import TopologyExplorer
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.GProp import GProp_GProps
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib


def read_step_file(step_path):
    """
    Reads a STEP file and returns the shape object.
    """
    step_reader = STEPControl_Reader()
    status = step_reader.ReadFile(step_path)
    if status == IFSelect_RetDone:
        step_reader.TransferRoots()
        return step_reader.Shape()
    print(f"Error reading STEP file: {step_path}")
    return None
    

def read_bounding_box_dimensions(shape):
    """
    Calculates and returns the bounding box dimensions of the shape.
    """
    bbox = Bnd_Box()
    brepbndlib.Add(shape, bbox)
    x_min, y_min, z_min, x_max, y_max, z_max = bbox.Get()
    x_dim = round(x_max - x_min, 2)
    y_dim = round(y_max - y_min, 2)
    z_dim = round(z_max - z_min, 2)
    return x_dim, y_dim, z_dim


def read_volume(shape):
    """
    Calculates and returns the volume of the shape.
    """
    props = GProp_GProps()
    brepgprop.VolumeProperties(shape, props)
    volume = round(props.Mass(), 2)
    return volume


def read_surface_area(shape):
    """
    Calculates and returns the surface area of the shape.
    """
    props = GProp_GProps()
    brepgprop.SurfaceProperties(shape, props)
    surface_area = round(props.Mass(), 2)
    return surface_area


if __name__ == '__main__':

    step_folder_path = r"insert_path_to_STEP_folder_here"
    info_folder_path = r"insert_path_to_info_folder_here"

    # Ensure STEP folder exists
    if not os.path.isdir(step_folder_path):
        print(f"STEP folder not found: {step_folder_path}")
        exit(1)

    # Ensure INFO folder exists, create it if not
    if not os.path.isdir(info_folder_path):
        print(f"INFO folder not found: {info_folder_path}. Creating it...")
        os.makedirs(info_folder_path)

    for i, step_file in enumerate(os.listdir(step_folder_path)):
        filename, ext = os.path.splitext(step_file)

        # Process only STEP files
        if ext.lower() in (".stp", ".step"):
            print(f"Processing file #{i}: {filename}")
            
            # Load the STEP file
            step_path = os.path.join(step_folder_path, step_file)
            shape = read_step_file(step_path)
            if not shape:
                continue  # Skip files that cannot be read

            # Check topology for solid bodies
            topology = TopologyExplorer(shape)
            if topology.number_of_solids() != 1:
                print(f"Skipping {filename}: Model does not consist of exactly one solid.")
                continue  # Skip models that don't consist of only one solid

            # Calculate geometry properties
            x_dim, y_dim, z_dim = read_bounding_box_dimensions(shape)
            volume = read_volume(shape)
            area = read_surface_area(shape)

            # Prepare metadata for output
            json_data = {
                "Filename": filename,
                "Valid": True,
                "xSize": x_dim,
                "ySize": y_dim,
                "zSize": z_dim,
                "Volume": volume,
                "Area": area,
            }

            # Write metadata to JSON file
            output_json_path = os.path.join(info_folder_path, f"{filename}.json")
            with open(output_json_path, "w") as json_file:
                json.dump(json_data, json_file, indent=4)
            print(f"Analysis complete. JSON output written to {output_json_path}")
