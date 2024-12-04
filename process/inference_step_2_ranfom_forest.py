"""
STEP File Analysis and Feature Extraction Script

This script processes STEP files to extract geometric and topological features, 
predicts their classifications using a pre-trained Random Forest model, and 
saves the predictions to a CSV file. 

Key Features:
- Extracts metadata (e.g., dimensions, volume, edge types) from STEP files.
- Predicts class labels for the STEP models using a machine learning model.
- Saves the results and probabilities in a structured CSV format.
- Provides visualization of STEP files using OCC.

Usage:
- Set the input STEP folder path and output CSV file path.
- Update the paths to the Random Forest model, scaler, and feature list.
- Run the script to process all STEP files in the specified folder.

Dependencies:
- OCC (OpenCascade)
- NumPy, pandas, joblib
"""

import os
import pandas as pd
import warnings
import numpy as np
import joblib
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.GProp import GProp_GProps
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Extend.TopologyUtils import TopologyExplorer
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface, BRepAdaptor_Curve
from OCC.Core.GeomLProp import GeomLProp_SLProps
from OCC.Core.TopoDS import TopoDS_Face
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.GeomAbs import GeomAbs_Plane
from OCC.Core.GCPnts import GCPnts_AbscissaPoint
from OCC.Core.gp import gp_Dir
from OCC.Display.SimpleGui import init_display

warnings.filterwarnings("ignore")

def load_model_components(model_path, scaler_path, features_path):
    """
    Load the model, scaler, and selected features from the given paths.
    """
    rf = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    selected_features = joblib.load(features_path)
    return rf, scaler, selected_features


def read_step_file(step_path):
    """
    Reads a STEP file and returns the shape object, or logs an error if unsuccessful.
    """
    step_reader = STEPControl_Reader()
    status = step_reader.ReadFile(step_path)
    if status == IFSelect_RetDone:
        step_reader.TransferRoots()
        return step_reader.Shape()
    print(f"Error reading STEP file: {step_path}")
    return None


def safe_divide(a, b):
    """
    Safely divide two numbers.
    """
    return a / b if b != 0 else 0


def get_edge_length(edge):
    """
    Calculate the length of an edge.
    """
    curve_adaptor = BRepAdaptor_Curve(edge)
    first = curve_adaptor.FirstParameter()
    last = curve_adaptor.LastParameter()
    return GCPnts_AbscissaPoint().Length(curve_adaptor, first, last, 1.0e-6)


def get_all_faces(shape):
    """
    Retrieve all faces from the given shape.
    """
    faces = []
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        faces.append(explorer.Current())
        explorer.Next()
    return faces


def calculate_face_normal_direction(face):
    """
    Computes the normal direction of a given face at its midpoint.
    """
    if not isinstance(face, TopoDS_Face):
        raise TypeError("Input must be a TopoDS_Face")
    
    # Retrieve the surface and parameter ranges of the face
    surface = BRep_Tool.Surface(face)
    adaptor = BRepAdaptor_Surface(face)
    u_mid = (adaptor.FirstUParameter() + adaptor.LastUParameter()) / 2.0
    v_mid = (adaptor.FirstVParameter() + adaptor.LastVParameter()) / 2.0

    # Calculate the normal direction at the midpoint
    sl_props = GeomLProp_SLProps(surface, u_mid, v_mid, 1, 0.01)
    if sl_props.IsNormalDefined():
        return sl_props.Normal()

    raise ValueError("Normal is not defined for this face")


def compute_unique_normal_vectors(faces):
    """
    Computes unique normal vectors from a list of planar faces, 
    ignoring reversed normals (i.e., only one of a vector and its reverse is kept).
    Normal vectors are rounded to 3 decimals.
    """
    unique_normals = []

    for face in faces:
        # Get the surface and calculate the normal vector
        surface_adaptor = BRepAdaptor_Surface(face)
        normal_vector = surface_adaptor.Plane().Axis().Direction()

        # Round components of the normal vector
        rounded_normal = gp_Dir(
            round(normal_vector.X(), 3),
            round(normal_vector.Y(), 3),
            round(normal_vector.Z(), 3))
        
        def is_normal_unique(normal, unique_normals):
            """
            Checks if a given normal vector is unique, considering reversed vectors as duplicates.
            """
            for existing_normal in unique_normals:
                dot_product = (
                    normal.X() * existing_normal.X() +
                    normal.Y() * existing_normal.Y() +
                    normal.Z() * existing_normal.Z()
                )
                if abs(dot_product) == 1:  # Parallel or anti-parallel vectors
                    return False
            return True
        
        # Check if the normal is unique
        if is_normal_unique(rounded_normal, unique_normals):
            unique_normals.append(rounded_normal)

        return unique_normals


def read_num_normalvectors(t):
    """
    Counts the number of unique normal vectors for planar faces in the given topology.
    """
    unique_normals = set()

    for face in t.faces():
        # Check if face is planar:
        surface = BRepAdaptor_Surface(face, True)
        if surface.GetType() == 0:  # 0 corresponds to planar surfaces
            try:
                # Calculate and round the normal vector
                normal = calculate_face_normal_direction(face)
                normal_tuple = (round(normal.X(), 2), round(normal.Y(), 2), round(normal.Z(), 2))
                unique_normals.add(normal_tuple)
            except ValueError:
                pass  # Skip faces where the normal vector cannot be calculated
        else:
            continue
    
    return len(unique_normals)


def get_face_centroid(face):
        """
        Computes the centroid of a given face.
        """
        surface_adaptor = BRepAdaptor_Surface(face)
        u_mid = (surface_adaptor.FirstUParameter() + surface_adaptor.LastUParameter()) / 2
        v_mid = (surface_adaptor.FirstVParameter() + surface_adaptor.LastVParameter()) / 2
        return surface_adaptor.Value(u_mid, v_mid)


def gather_faces_perpendicular_to_vector(faces, normal):
    """
    Gather all faces that are parallel or anti-parallel to the given normal vector using the dot product.
    """
    perpendicular_faces = []

    def get_face_normal(face):
        """
        Computes and returns the normal vector of a given face.
        """
        surface_adaptor = BRepAdaptor_Surface(face)
        return surface_adaptor.Plane().Axis().Direction()

    def is_parallel_or_antiparallel(v1, v2):
        """
        Checks if two vectors are parallel or anti-parallel using the dot product.
        """
        dot_product = round(
            v1.X() * v2.X() +
            v1.Y() * v2.Y() +
            v1.Z() * v2.Z(), 3
        )
        return abs(dot_product) >= 1

    for face in faces:
        face_normal = get_face_normal(face)
        if is_parallel_or_antiparallel(face_normal, normal):
            perpendicular_faces.append(face)

    return perpendicular_faces


def sort_faces_along_normal(faces, normal):
    """
    Sorts faces based on their centroid projection along the given normal vector.
    """
    
    def project_centroid_along_normal(centroid, normal):
        """
        Projects a centroid onto the given normal vector.
        """
        return (
            centroid.X() * normal.X() +
            centroid.Y() * normal.Y() +
            centroid.Z() * normal.Z())

    # Sort faces based on their projection along the normal vector
    return sorted(faces,
                  key=lambda face: project_centroid_along_normal(get_face_centroid(face), normal))


def calculate_face_to_face_thicknesses(faces, normal, lower_limit, upper_limit):
    """
    Calculate thicknesses between consecutive faces along the normal direction.
    """

    def calculate_distance_along_normal(centroid1, centroid2, normal):
        """
        Calculates the distance between two centroids along a normal vector.
        """
        return round(abs(
                (centroid2.X() - centroid1.X()) * normal.X() +
                (centroid2.Y() - centroid1.Y()) * normal.Y() +
                (centroid2.Z() - centroid1.Z()) * normal.Z()), 3)
    
    thicknesses = []
    
    for i in range(len(faces) - 1):
        centroid1 = get_face_centroid(faces[i])
        centroid2 = get_face_centroid(faces[i + 1])
        distance = calculate_distance_along_normal(centroid1, centroid2, normal)

        if lower_limit <= distance <= upper_limit:
            thicknesses.append(distance)

    return thicknesses


def calculate_minimal_wall_thickness(shape, lower_limit, upper_limit):
    """
    Calculates the minimal wall thickness of a part, considering a lower and upper limit for valid thicknesses.
    Only considers the largest planar faces based on their areas.
    """

    # Get all faces and filter planar ones
    all_faces = get_all_faces(shape)
    if not all_faces:
        return 0  # No faces, no wall thickness to calculate
    
    props = GProp_GProps()

    face_areas = [
        (face, round(brepgprop.SurfaceProperties(face, props) or props.Mass(), 3))
        for face in all_faces]
    
    planar_faces = [
        face for face, _ in face_areas
        if BRepAdaptor_Surface(face).GetType() == GeomAbs_Plane]
    
    # Get the largest planar faces by area
    largest_planar_faces = sorted(
        planar_faces,
        key=lambda face: next(area for f, area in face_areas if f == face),
        reverse=True)[:2]  # Considering only the top two largest faces
    
    if not largest_planar_faces:
        return 0  # No valid planar faces among the largest
    
    # Compute unique normal vectors for the largest planar faces
    unique_normals = compute_unique_normal_vectors(largest_planar_faces)

    # Calculate thicknesses along each normal direction
    all_thicknesses = []
    for normal in unique_normals:
        perpendicular_faces = gather_faces_perpendicular_to_vector(largest_planar_faces, normal)
        if len(perpendicular_faces) < 2:
            continue  # Not enough faces to calculate thickness

        sorted_faces = sort_faces_along_normal(perpendicular_faces, normal)
        thicknesses = calculate_face_to_face_thicknesses(
            sorted_faces, normal, lower_limit, upper_limit
        )
        all_thicknesses.extend(thicknesses)

    return min(all_thicknesses, default=0)


def extract_features_from_step(shape):
    """Extract all geometric features from STEP file."""
    # Initialize topology explorer
    topology = TopologyExplorer(shape)
    
    # Basic counts
    num_solids = topology.number_of_solids()
    num_faces = topology.number_of_faces()
    num_edges = topology.number_of_edges()
    
    # Volume and surface area
    props = GProp_GProps()
    brepgprop.VolumeProperties(shape, props)
    volume = round(props.Mass(), 2)

    brepgprop.SurfaceProperties(shape, props)
    surface_area = round(props.Mass(), 2)
    
    # Bounding box and dimensions
    bbox = Bnd_Box()
    brepbndlib.Add(shape, bbox)
    xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
    x_dim = round(xmax - xmin, 2)
    y_dim = round(ymax - ymin, 2)
    z_dim = round(zmax - zmin, 2)
    
    # Center of mass (mit Rundung auf 2 Dezimalstellen)
    cogx, cogy, cogz = props.CentreOfMass().Coord()
    x_center_mass_relative = round(safe_divide(cogx - xmin, x_dim), 2)
    y_center_mass_relative = round(safe_divide(cogy - ymin, y_dim), 2)
    z_center_mass_relative = round(safe_divide(cogz - zmin, z_dim), 2)
    
    # Wall thickness
    wall_thickness = calculate_minimal_wall_thickness(shape, lower_limit=0.1, upper_limit=10)
    
    # Edge types
    edge_type = []
    total_edge_length = 0
    for e in topology.edges():
        edge = BRepAdaptor_Curve(e)
        edge_type.append(edge.GetType())

        # Get edge cumulative edge length
        try:
            total_edge_length += get_edge_length(e)
        except:
            pass
    edge_counts = {
        'line': edge_type.count(0), 'circle': edge_type.count(1), 'ellipse': edge_type.count(2),
        'hyperbola': edge_type.count(3), 'parabola': edge_type.count(4), 'bezier': edge_type.count(5),
        'bspline': edge_type.count(6), 'other': edge_type.count(7)}
    
    # Surface types
    surface_type = []
    for f in topology.faces():
        surface = BRepAdaptor_Surface(f, True)
        surface_type.append(surface.GetType())
    surface_counts = {
        'plane': surface_type.count(0), 'cylinder': surface_type.count(1), 'cone': surface_type.count(2),
        'sphere': surface_type.count(3), 'torus': surface_type.count(4), 'bezier': surface_type.count(5), 
        'bspline': surface_type.count(6), 'revolution': surface_type.count(7), 'extrusion': surface_type.count(8),
        'other': surface_type.count(9)}

    # Unique normals
    num_unique_normals = read_num_normalvectors(topology)
    
    # Calculate base features
    features = {
        'x_dim': x_dim,
        'y_dim': y_dim,
        'z_dim': z_dim,
        'volume': volume,
        'surface_area': surface_area,
        'wall_thickness': wall_thickness,
        'x_center_mass_relative': x_center_mass_relative,
        'y_center_mass_relative': y_center_mass_relative,
        'z_center_mass_relative': z_center_mass_relative,
        'num_unique_normals': num_unique_normals,
        'num_solids': num_solids,
        'num_faces': num_faces,
        'num_edges': num_edges,
        
        # Edge counts
        'num_line_edges': edge_counts['line'],
        'num_circle_edges': edge_counts['circle'],
        'num_ellipse_edges': edge_counts['ellipse'],
        'num_hyperbola_edges': edge_counts['hyperbola'],
        'num_parabola_edges': edge_counts['parabola'],
        'num_beziercurve_edges': edge_counts['bezier'],
        'num_bsplinecurve_edges': edge_counts['bspline'],
        'num_othercurve_edges': edge_counts['other'],
        
        # Surface counts
        'num_plane_surfaces': surface_counts['plane'],
        'num_cylinder_surfaces': surface_counts['cylinder'],
        'num_cone_surfaces': surface_counts['cone'],
        'num_sphere_surfaces': surface_counts['sphere'],
        'num_torus_surfaces': surface_counts['torus'],
        'num_bezier_surfaces': surface_counts['bezier'],
        'num_bspline_surfaces': surface_counts['bspline'],
        'num_revolution_surfaces': surface_counts['revolution'],
        'num_extrusion_surfaces': surface_counts['extrusion'],
        'num_other_surfaces': surface_counts['other'],
    }
    
    # Calculate edge ratios
    total_edges = sum(edge_counts.values())
    features['num_line_edges_ratio'] = round(safe_divide(edge_counts['line'], total_edges), 2)
    features['num_circle_edges_ratio'] = round(safe_divide(edge_counts['circle'], total_edges), 2)
    features['num_ellipse_edges_ratio'] = round(safe_divide(edge_counts['ellipse'], total_edges), 2)
    features['num_bsplinecurve_edges_ratio'] = round(safe_divide(edge_counts['bspline'], total_edges), 2)
    
    # Calculate surface ratios
    total_surfaces = sum(surface_counts.values())
    features['num_plane_surfaces_ratio'] = round(safe_divide(surface_counts['plane'], total_surfaces), 2)
    features['num_cylinder_surfaces_ratio'] = round(safe_divide(surface_counts['cylinder'], total_surfaces), 2)
    features['num_cone_surfaces_ratio'] = round(safe_divide(surface_counts['cone'], total_surfaces), 2)
    features['num_sphere_surfaces_ratio'] = round(safe_divide(surface_counts['sphere'], total_surfaces), 2)
    features['num_torus_surfaces_ratio'] = round(safe_divide(surface_counts['torus'], total_surfaces), 2)
    features['num_bspline_surfaces_ratio'] = round(safe_divide(surface_counts['bspline'], total_surfaces), 2)
    
    # Calculate derived features
    # Aspect ratios
    features['aspect_ratio_xy'] = round(safe_divide(x_dim, y_dim), 2)
    features['aspect_ratio_xz'] = round(safe_divide(x_dim, z_dim), 2)
    features['aspect_ratio_yz'] = round(safe_divide(y_dim, z_dim), 2)
    
    # Volume ratios
    xyz_product = x_dim * y_dim * z_dim
    features['volume_to_xyz'] = round(safe_divide(volume, xyz_product), 2)
    features['surface_to_xyz'] = round(safe_divide(surface_area, xyz_product), 2)
    
    # Center mass distance
    center_coords = np.array([x_center_mass_relative, y_center_mass_relative, z_center_mass_relative]) - 0.5
    features['center_mass_distance'] = round(np.linalg.norm(center_coords), 2)
    
    # Basic ratios and complexities
    features['volume_to_surface_ratio'] = round(safe_divide(volume, surface_area), 2)
    features['edge_to_face_ratio'] = round(safe_divide(num_edges, num_faces), 2)
    features['average_face_area'] = round(safe_divide(surface_area, num_faces), 2)
    features['edge_complexity'] = round(safe_divide(num_edges, num_faces), 2)
    features['surface_complexity'] = round(safe_divide(num_unique_normals, num_faces), 2)
    features['thickness_complexity'] = round(safe_divide(wall_thickness, num_faces), 2)
    features['average_edge_length'] = round(safe_divide(total_edge_length, num_edges), 2)
    features['shape_factor'] = round(safe_divide(surface_area, np.power(volume, 2/3)) if volume > 0 else 0, 2)
    
    # Log transforms
    for col in ['volume', 'surface_area', 'num_faces', 'num_edges']:
        min_positive = max(features[col], 1e-10)
        features[f'log_{col}'] = round(np.log1p(min_positive), 2)
    
    # Overall complexity
    complexity_cols = ['edge_complexity', 'surface_complexity', 'thickness_complexity']
    complexity_values = [features[col] for col in complexity_cols]
    features['overall_complexity'] = round(np.mean([v for v in complexity_values if v >= 0]), 2)
    
    return features


def predict_step_file(file_path):
    """Load a STEP file, extract features, and make a prediction."""
    # Load the model components
    rf = joblib.load('models/new_model_11.joblib')
    scaler = joblib.load('models/new_scaler_11.joblib')
    selected_features = joblib.load('models/selected_features_11.joblib')
    
    # Load and process the STEP file
    print("Loading STEP file...")
    shape = read_step_file(file_path)
    
    print("Extracting features...")
    features = extract_features_from_step(shape)
    
    # Print all extracted features for debugging
    print("\nExtracted features:")
    for key, value in sorted(features.items()):
        print(f"{key}: {value}")
    
    # Prepare features for prediction
    feature_vector = []
    print("\nPreparing feature vector...")
    for feature in selected_features:
        if feature not in features:
            print(f"Warning: Feature '{feature}' not found. Setting to 0.")
            features[feature] = 0
        feature_vector.append(features[feature])
    
    # Scale features
    X = np.array([feature_vector])
    X_scaled = scaler.transform(X)
    
    # Make prediction
    prediction = rf.predict(X_scaled)[0]
    probabilities = rf.predict_proba(X_scaled)[0]
    
    print(f"\nPrediction: Class {prediction}")
    print("Class probabilities:")
    for i, prob in enumerate(probabilities):
        print(f"Class {i}: {prob:.3f}")
    
    # Visualize the STEP file
    display, start_display, _, _ = init_display()
    display.DisplayShape(shape, update=True)
    display.FitAll()
    start_display()
    
    return prediction, probabilities, features


def analyze_step_files_in_folder(folder_path, csv_path):
    """Analyze all STEP files in the given folder."""
    results = []

    # Initialize an empty DataFrame to store predictions
    prediction_df = pd.DataFrame(columns=["filename", "prediction"])
    
    # Get all STEP files in the folder
    step_files = [f for f in os.listdir(folder_path) 
                 if f.lower().endswith(('.step', '.stp'))]
    
    print(f"Found {len(step_files)} STEP files in folder.")
    
    # Process each file
    for filename in step_files:
        try:
            file_path = os.path.join(folder_path, filename)
            
            # Load and process STEP file
            shape = read_step_file(file_path)
            features = extract_features_from_step(shape)
            
            # Prepare feature vector
            feature_vector = []
            for feature in selected_features:
                if feature not in features:
                    features[feature] = 0
                feature_vector.append(features[feature])
            
            # Scale features and predict
            X = np.array([feature_vector])
            X_scaled = scaler.transform(X)
            prediction = rf.predict(X_scaled)[0]
            probabilities = rf.predict_proba(X_scaled)[0]
            
            # Store results
            results.append({
                'filename': filename,
                'prediction': prediction,
                'probabilities': probabilities
            })

            # Add prediction and file name to DataFrame
            prediction_df = pd.concat([prediction_df, pd.DataFrame([{
                "filename": filename,
                "prediction": prediction + 1,
            }])], ignore_index=True)
            
            # Print prediction for this file
            print(f"\n{filename}:")
            print(f"Prediction: Class {prediction}")
            print("Probabilities:", end=" ")
            for i, prob in enumerate(probabilities):
                print(f"Class {i}: {prob:.3f}", end=" ")
            print()
            
        except Exception as e:
            print(f"\nError processing {filename}: {str(e)}")
    
    # Save the predictions DataFrame to a CSV file
    prediction_df.to_csv(csv_path, index=False)
    print(f"Predictions saved to {csv_path}")

    return results


if __name__ == "__main__":
    # INPUT:
    STEP_folder_path = r"insert_STEP_folder_path_here"
    # Paths to model components
    model_path = r"insert_model_joblib_path_here"
    scaler_path = r"insert_scaler_joblib_path_here"
    features_path = r"insert_features_joblib_path_here"
    
    # OUTPUT:
    output_prediction_csv_path = os.path.join(STEP_folder_path, "TEST.csv")

    try:
        print("Starting analysis of all STEP files...\n")

        # Load model components
        rf, scaler, selected_features = load_model_components(model_path, scaler_path, features_path)

        # Analyze STEP files in the folder
        results = analyze_step_files_in_folder(STEP_folder_path, output_prediction_csv_path)
        
        # Show summary
        print("\nAnalysis Summary:\n", "-" * 50)
        class_counts = {}
        for r in results:
            prediction = r['prediction']
            class_counts[prediction] = class_counts.get(prediction, 0) + 1
        
        print(f"Total files processed: {len(results)}")
        print("\nPredicted class distribution:")
        for class_label, count in sorted(class_counts.items()):
            print(f"Class {class_label+1}: {count} files ({count/len(results)*100:.1f}%)")
        
    except Exception as e:
        print(f"\nError during analysis: {str(e)}")
        raise
