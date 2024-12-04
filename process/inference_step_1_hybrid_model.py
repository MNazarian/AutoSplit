"""
Hybrid CNN and Random Forest STEP File Analysis

This script processes STEP files to extract geometric features, generate 
screenshots, and classify parts using a hybrid model that combines numerical 
features and image data.

Features:
- **Feature Extraction**: Calculates dimensions, volume, surface area, edge/surface types, 
  and other geometric/topological attributes from STEP files.
- **Screenshot Generation**: Captures ISO and alternate perspective screenshots 
  of STEP models for image-based predictions.
- **Hybrid Model Prediction**: Uses a CNN model combined with numerical features to 
  classify parts as "Standard" or "Non-standard."
- **CSV Export**: Saves predictions, classifications, and probabilities to a CSV file.

Usage:
1. Update the paths for:
   - `step_folder_path`: Directory containing STEP files to process.
   - `model_path`: Path to the pre-trained hybrid model (`.h5` file).
   - `output_csv_path`: Destination for saving predictions in CSV format.
2. Run the script to process the STEP files.
3. Check the console for progress updates and the CSV file for results.

Dependencies:
- TensorFlow
- NumPy
- Pandas
- PythonOCC (OpenCASCADE)
- Scikit-learn
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from OCC.Extend.DataExchange import read_step_file
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop, brepgprop_LinearProperties
from OCC.Extend.TopologyUtils import TopologyExplorer
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface, BRepAdaptor_Curve
from OCC.Core.GeomAbs import (GeomAbs_Line, GeomAbs_Circle, GeomAbs_Ellipse, GeomAbs_Hyperbola, 
                              GeomAbs_Parabola, GeomAbs_BezierCurve, GeomAbs_BSplineCurve,
                              GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone, GeomAbs_Sphere, 
                              GeomAbs_Torus, GeomAbs_BezierSurface, GeomAbs_BSplineSurface, 
                              GeomAbs_SurfaceOfRevolution, GeomAbs_SurfaceOfExtrusion)
from OCC.Core.gp import gp_Vec, gp_Pnt
from OCC.Display.SimpleGui import init_display
from OCC.Core.V3d import V3d_XposYposZneg

def extract_step_info(shape):
    try:
        bbox = Bnd_Box()
        brepbndlib.Add(shape, bbox)
        xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()

        props = GProp_GProps()
        brepgprop.VolumeProperties(shape, props)
        volume = props.Mass()

        brepgprop.SurfaceProperties(shape, props)
        surface_area = props.Mass()

        x_dim, y_dim, z_dim = xmax - xmin, ymax - ymin, zmax - zmin
        
        if x_dim == 0 or y_dim == 0 or z_dim == 0:
            print("Warning: One or more dimensions are 0.")
            return None

        t = TopologyExplorer(shape)

        num_solids = t.number_of_solids()
        num_faces = t.number_of_faces()
        num_edges = t.number_of_edges()

        edge_types = {'line': 0, 'circle': 0, 'ellipse': 0, 'hyperbola': 0, 'parabola': 0, 'bezier': 0, 'bspline': 0, 'other': 0}
        for edge in t.edges():
            curve = BRepAdaptor_Curve(edge)
            curve_type = curve.GetType()
            if curve_type == GeomAbs_Line: edge_types['line'] += 1
            elif curve_type == GeomAbs_Circle: edge_types['circle'] += 1
            elif curve_type == GeomAbs_Ellipse: edge_types['ellipse'] += 1
            elif curve_type == GeomAbs_Hyperbola: edge_types['hyperbola'] += 1
            elif curve_type == GeomAbs_Parabola: edge_types['parabola'] += 1
            elif curve_type == GeomAbs_BezierCurve: edge_types['bezier'] += 1
            elif curve_type == GeomAbs_BSplineCurve: edge_types['bspline'] += 1
            else: edge_types['other'] += 1

        surface_types = {'plane': 0, 'cylinder': 0, 'cone': 0, 'sphere': 0, 'torus': 0, 'bezier': 0, 'bspline': 0, 'revolution': 0, 'extrusion': 0, 'other': 0}
        
        unique_normals = set()
        
        for face in t.faces():
            surface = BRepAdaptor_Surface(face)
            surface_type = surface.GetType()
            if surface_type == GeomAbs_Plane: surface_types['plane'] += 1
            elif surface_type == GeomAbs_Cylinder: surface_types['cylinder'] += 1
            elif surface_type == GeomAbs_Cone: surface_types['cone'] += 1
            elif surface_type == GeomAbs_Sphere: surface_types['sphere'] += 1
            elif surface_type == GeomAbs_Torus: surface_types['torus'] += 1
            elif surface_type == GeomAbs_BezierSurface: surface_types['bezier'] += 1
            elif surface_type == GeomAbs_BSplineSurface: surface_types['bspline'] += 1
            elif surface_type == GeomAbs_SurfaceOfRevolution: surface_types['revolution'] += 1
            elif surface_type == GeomAbs_SurfaceOfExtrusion: surface_types['extrusion'] += 1
            else: surface_types['other'] += 1
            
            u = (surface.LastUParameter() + surface.FirstUParameter()) / 2
            v = (surface.LastVParameter() + surface.FirstVParameter()) / 2
            
            pnt = gp_Pnt()
            du = gp_Vec()
            dv = gp_Vec()
            
            surface.D1(u, v, pnt, du, dv)
            normal = du.Crossed(dv)
            normal.Normalize()
            
            unique_normals.add((round(normal.X(), 5), round(normal.Y(), 5), round(normal.Z(), 5)))

        center_of_mass = props.CentreOfMass()
        x_center = (center_of_mass.X() - xmin) / x_dim if x_dim != 0 else 0
        y_center = (center_of_mass.Y() - ymin) / y_dim if y_dim != 0 else 0
        z_center = (center_of_mass.Z() - zmin) / z_dim if z_dim != 0 else 0
        
        linear_props = GProp_GProps()
        brepgprop_LinearProperties(shape, linear_props)
        wall_thickness = 2 * volume / surface_area  # This is a rough approximation

        # aspect_ratio = max(x_dim, y_dim, z_dim) / min(x_dim, y_dim, z_dim) if min(x_dim, y_dim, z_dim) > 0 else 1.0

        return {
            'num_solids': num_solids,
            'num_faces': num_faces,
            'num_edges': num_edges,
            'volume': volume,
            'surface_area': surface_area,
            'x_dim': x_dim,
            'y_dim': y_dim,
            'z_dim': z_dim,
            'num_line_edges': edge_types['line'],
            'num_circle_edges': edge_types['circle'],
            'num_ellipse_edges': edge_types['ellipse'],
            'num_hyperbola_edges': edge_types['hyperbola'],
            'num_parabola_edges': edge_types['parabola'],
            'num_beziercurve_edges': edge_types['bezier'],
            'num_bsplinecurve_edges': edge_types['bspline'],
            'num_othercurve_edges': edge_types['other'],
            'num_plane_surfaces': surface_types['plane'],
            'num_cylinder_surfaces': surface_types['cylinder'],
            'num_cone_surfaces': surface_types['cone'],
            'num_sphere_surfaces': surface_types['sphere'],
            'num_torus_surfaces': surface_types['torus'],
            'num_bezier_surfaces': surface_types['bezier'],
            'num_bspline_surfaces': surface_types['bspline'],
            'num_revolution_surfaces': surface_types['revolution'],
            'num_extrusion_surfaces': surface_types['extrusion'],
            'num_other_surfaces': surface_types['other'],
            'num_unique_normals': len(unique_normals),
            'wall_thickness': wall_thickness,
            'x_center_mass_relative': x_center,
            'y_center_mass_relative': y_center,
            'z_center_mass_relative': z_center}
    
    except Exception as e:
        print(f"Error extracting STEP information: {e}")
        return None

def generate_screenshot_png(display, png_path, perspective='ISO'):
    try:
        if perspective == 'ISO':
            display.View_Iso()
        elif perspective == 'ISOAlt':
            display.View.SetProj(V3d_XposYposZneg)
        else:
            raise ValueError(f"Invalid perspective: {perspective}")
        
        display.FitAll()
        display.ExportToImage(png_path)
        print(f'Screenshot created: {png_path}')
    except Exception as e:
        print(f"Error creating screenshot: {e}")

def process_step_file(step_file_path):
    try:
        shape = read_step_file(step_file_path)
        info = extract_step_info(shape)
        if info is None:
            raise ValueError("Error extracting STEP information")
        
        # Display the shape
        display.DisplayShape(shape, update=True)
        
        # Generate screenshots
        png_folder = os.path.join(os.path.dirname(step_file_path), "screenshots")
        os.makedirs(png_folder, exist_ok=True)
        iso_png_path = os.path.join(png_folder, f"{os.path.splitext(os.path.basename(step_file_path))[0]}_ISO.png")
        isoalt_png_path = os.path.join(png_folder, f"{os.path.splitext(os.path.basename(step_file_path))[0]}_ISOAlt.png")
        
        generate_screenshot_png(display, iso_png_path, perspective='ISO')
        generate_screenshot_png(display, isoalt_png_path, perspective='ISOAlt')
        
        return info, iso_png_path, isoalt_png_path
    except Exception as e:
        print(f"Error processing {os.path.basename(step_file_path)}: {str(e)}")
        return None, None, None

def prepare_features(features):
    numeric_columns = [
        'num_solids', 'num_faces', 'num_edges', 'volume', 'surface_area',
        'x_dim', 'y_dim', 'z_dim', 'num_line_edges', 'num_circle_edges',
        'num_ellipse_edges', 'num_hyperbola_edges', 'num_parabola_edges',
        'num_beziercurve_edges', 'num_bsplinecurve_edges', 'num_othercurve_edges',
        'num_plane_surfaces', 'num_cylinder_surfaces', 'num_cone_surfaces',
        'num_sphere_surfaces', 'num_torus_surfaces', 'num_bezier_surfaces',
        'num_bspline_surfaces', 'num_revolution_surfaces', 'num_extrusion_surfaces',
        'num_other_surfaces', 'num_unique_normals', 'wall_thickness',
        'x_center_mass_relative', 'y_center_mass_relative', 'z_center_mass_relative'#, 'aspect_ratio'
    ]
    return np.array([[features[col] for col in numeric_columns]])

def preprocess_image(file_path):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, [224, 224])
    img = tf.cast(img, tf.float32) / 255.0
    return img

def load_and_predict(model_path, features, iso_png_path, isoalt_png_path):
    model = load_model(model_path)
    
    # Vorbereiten der numerischen Eingabe
    numeric_input = prepare_features(features)
    
    # Normalisieren der numerischen Eingabe
    scaler = StandardScaler()
    numeric_input_scaled = scaler.fit_transform(numeric_input)
    
    # Laden und Vorverarbeiten der Bilder
    iso_img = preprocess_image(iso_png_path)
    isoalt_img = preprocess_image(isoalt_png_path)
    
    # Vorhersagen f�r beide Perspektiven machen
    iso_prediction = model.predict([tf.expand_dims(iso_img, 0), numeric_input_scaled])
    isoalt_prediction = model.predict([tf.expand_dims(isoalt_img, 0), numeric_input_scaled])
    
    # Durchschnittliche Vorhersage berechnen
    avg_prediction = (iso_prediction[0][0] + isoalt_prediction[0][0]) / 2
    
    return avg_prediction

if __name__ == '__main__':
    # INPUT:
    step_folder_path = r"insert_STEP_folder_path_here"
    model_path = r"insert_model_h5_path_here"

    # OUTPUT:
    output_csv_path = r"insert_output_csv_path_here"

    threshold = 0.5  # threshold to classify a prediction as correct. Change as desired

    # Initialize display
    display, start_display, add_menu, add_function_to_menu = init_display()

    # Initialize a DataFrame for predictions
    predictions = []

    # Process each STEP file in the folder
    step_files = [f for f in os.listdir(step_folder_path) if f.lower().endswith(('.step', '.stp'))]
    print(f"Found {len(step_files)} STEP files for processing.\n")

    # Cycle through STEPs
    for step_file in step_files:
        step_path = os.path.join(step_folder_path, step_file)
    
        # Process the STEP file and generate features and screenshots
        features, iso_png_path, isoalt_png_path = process_step_file(step_path)

        if features and iso_png_path and isoalt_png_path:

            # Run AI prediction
            prediction = load_and_predict(model_path, features, iso_png_path, isoalt_png_path)
            classification = 'Standard part' if prediction > threshold else 'Non-standard part'

            # Log results
            print(f"File: {step_file}")
            print(f"Prediction: {prediction:.2f}")
            print(f"Class: {classification}\n")

            # Append the result
            predictions.append({
                "step_filename": step_file,
                "final_prediction": classification,
                "final_probability": round(prediction, 2)
            })

        else:
            print(f"Failed to process STEP file: {step_file}")
            
        # Clear display for the next file
        display.EraseAll()

    # Save predictions to CSV
    if predictions:
        prediction_df = pd.DataFrame(predictions)
        prediction_df.to_csv(output_csv_path, index=False)
        print(f"\nPredictions saved to {output_csv_path}")
    else:
        print("No predictions were made. Ensure files are valid and try again.")
