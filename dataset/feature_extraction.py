"""
STEP File Feature Extraction Script

This script processes STEP files to extract geometric and topological features, which are then used to generate labeled datasets for machine learning applications.
Features are saved in a CSV file, and files that don't meet specific criteria (e.g., more than one solid) are removed from the dataset.

Key Features:
- Extracts geometry (volume, area, bounding box dimensions) and topology (number of faces, edges, etc.).
- Identifies and processes only planar faces.
- Computes unique normal vectors and wall thickness.
- Saves extracted features to a CSV file and updates the labels CSV accordingly.

Notes:
- STEP files with errors or more than one solid are skipped and removed from the dataset.
- Requires Open CASCADE Technology (OCC) and pandas.

Usage:
1. Set the paths for the STEP files, labels CSV, and features CSV.
2. Run the script to process STEP files and generate features.
"""

import os
import gc
import pandas as pd
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
from OCC.Core.TopoDS import TopoDS_Face, TopoDS_Iterator, topods
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.GeomAbs import GeomAbs_Plane
from OCC.Core.gp import gp_Dir


# ---------- HELP FUNTIONS  ----------
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


def get_label_majority(filename, df):
    """
    Retrieves the majority label for a given file.
    """
    row = df.loc[df['filename'] == filename]
    return row['label_majority'].values[0] if not row.empty else None


def iterate_shapes(shape, faces):
    """
    Recursively iterates through the given shape and collects all faces.
    """
    for sub_shape in TopoDS_Iterator(shape):
        if sub_shape.ShapeType() == TopAbs_FACE:
            faces.append(sub_shape)
        else:
            iterate_shapes(sub_shape, faces)


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


def get_planar_faces(shape):
    """
    Extracts all planar faces from the given shape.
    """
    planar_faces = []

    for face in TopExp_Explorer(shape, TopAbs_FACE):
        # Convert to TopoDS_Face
        topo_face = topods.Face(face)

        # Check if the face is planar
        surface_adaptor = BRepAdaptor_Surface(topo_face)
        if surface_adaptor.GetType() == GeomAbs_Plane:
            planar_faces.append(topo_face)

    return planar_faces


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


def get_face_centroid(face):
        """
        Computes the centroid of a given face.
        """
        surface_adaptor = BRepAdaptor_Surface(face)
        u_mid = (surface_adaptor.FirstUParameter() + surface_adaptor.LastUParameter()) / 2
        v_mid = (surface_adaptor.FirstVParameter() + surface_adaptor.LastVParameter()) / 2
        return surface_adaptor.Value(u_mid, v_mid)


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

# ---------- EXTRACT FEATURES  ----------
def extract_STEP_features(step_path, filename, verbose=True):
    """
    Extracts geometric and topological features from a STEP file.
    """
    features = {'filename': filename}

    # Read STEP file shape and topology
    shape = read_step_file(step_path)
    topology = TopologyExplorer(shape)
    
    # Number of solids, faces, edges
    features.update(dict(zip(['num_solids', 'num_faces', 'num_edges'], num_solids_faces_edges(topology))))
    if verbose:
        print(f"Solids = {features['num_solids']}, Faces = {features['num_faces']}, Edges = {features['num_edges']}")

    # Volume and surface area
    features.update({'volume': read_volume(shape),'surface_area': read_surface_area(shape)})
    if verbose:
        print(f"Volume = {features['volume']}, Surface Area = {features['surface_area']}")

    # Bounding box dimensions
    x_min, y_min, z_min, x_max, y_max, z_max, x_dim, y_dim, z_dim = read_bounding_box_dimensions(shape)
    features.update({'x_dim': x_dim, 'y_dim': y_dim, 'z_dim': z_dim})
    if verbose:
        print(f"Bounding Box: X={x_dim}, Y={y_dim}, Z={z_dim}")

    # Classify edges and count
    features.update(dict(zip(
        ['num_line_edges', 'num_circle_edges', 'num_ellipse_edges', 'num_hyperbola_edges',
         'num_parabola_edges', 'num_beziercurve_edges', 'num_bsplinecurve_edges', 'num_othercurve_edges'],
        read_edge_types(topology)
    )))
    if verbose:
        print("\n".join([f"{key}: {features[key]}" for key in features if 'edges' in key]))
    
    # Classify surfaces and count
    features.update(dict(zip(
        ['num_plane_surfaces', 'num_cylinder_surfaces', 'num_cone_surfaces', 'num_sphere_surfaces',
         'num_torus_surfaces', 'num_bezier_surfaces', 'num_bspline_surfaces', 'num_revolution_surfaces',
         'num_extrusion_surfaces', 'num_other_surfaces'],
        read_surface_types(topology)
    )))
    if verbose:
        print("\n".join([f"{key}: {features[key]}" for key in features if 'surface' in key]))

    # Number of unique face normals
    num_unique_normals = read_num_normalvectors(topology)
    features.update({'num_unique_normals': num_unique_normals})
    num_planes = features.get('num_plane_surfaces', 0)
    if verbose:
        percentage = round((num_unique_normals / num_planes) * 100, 2) if num_planes else 0
        print(f"Num unique normals: {num_unique_normals} ({percentage}%)")

    # Minimal wall thickness
    features.update({'wall_thickness': calculate_minimal_wall_thickness(shape, lower_limit=0.1, upper_limit=10)})
    if verbose:
        print(f"Wall thickness: {features['wall_thickness']}")

    # Relative center of mass
    x_center, y_center, z_center = read_center_of_mass(shape)
    features.update({
    'x_center_mass_relative': round((x_center - x_min) / (x_max - x_min), 2),
    'y_center_mass_relative': round((y_center - y_min) / (y_max - y_min), 2),
    'z_center_mass_relative': round((z_center - z_min) / (z_max - z_min), 2)})
    if verbose:
        print(f"Relative center of mass X={features['x_center_mass_relative']}, "
            f"Y={features['y_center_mass_relative']}, Z={features['z_center_mass_relative']}")
        
    del shape, topology
    gc.collect()

    return features


def num_solids_faces_edges(topology):
    """
    Counts the number of solids, faces, and edges in a topology.
    """
    return topology.number_of_solids(), topology.number_of_faces(), topology.number_of_edges()


def read_volume(shape):
    """
    Calculates and returns the volume of a shape.
    """
    props = GProp_GProps()
    brepgprop.VolumeProperties(shape, props)
    return round(props.Mass(), 2)


def read_surface_area(shape):
    """
    Calculates and returns the surface area of a shape.
    """
    props = GProp_GProps()
    brepgprop.SurfaceProperties(shape, props)
    return round(props.Mass(), 2)


def read_bounding_box_dimensions(shape):
    """
    Calculates and returns the bounding box dimensions of a shape.
    """
    bbox = Bnd_Box()
    brepbndlib.Add(shape, bbox)
    x_min, y_min, z_min, x_max, y_max, z_max = bbox.Get()
    x_dim = round(x_max - x_min, 2)
    y_dim = round(y_max - y_min, 2)
    z_dim = round(z_max - z_min, 2)
    return x_min, y_min, z_min, x_max, y_max, z_max, x_dim, y_dim, z_dim


def read_edge_types(topology):
    """
    Classifies and counts edge types in the given topology.

    Edge type mapping:
    - 0: Line
    - 1: Circle
    - 2: Ellipse
    - 3: Hyperbola
    - 4: Parabola
    - 5: Bezier curve
    - 6: BSpline curve
    - 7: Other curve

    Returns:
        A tuple of counts for each edge type, in the following order:
        (num_line_edges, num_circle_edges, num_ellipse_edges, num_hyperbola_edges,
         num_parabola_edges, num_beziercurve_edges, num_bsplinecurve_edges, num_othercurve_edges)
    """
    edge_type = []
    for e in topology.edges():
        edge = BRepAdaptor_Curve(e)
        edge_type.append(edge.GetType())

    num_line_edges = edge_type.count(0)
    num_circle_edges = edge_type.count(1) 
    num_ellipse_edges = edge_type.count(2) 
    num_hyperbola_edges = edge_type.count(3) 
    num_parabola_edges = edge_type.count(4) 
    num_beziercurve_edges = edge_type.count(5) 
    num_bsplinecurve_edges = edge_type.count(6) 
    num_othercurve_edges = edge_type.count(7)

    return (num_line_edges, num_circle_edges, num_ellipse_edges, num_hyperbola_edges,
            num_parabola_edges, num_beziercurve_edges, num_bsplinecurve_edges, num_othercurve_edges)


def read_surface_types(t):
    """
    Classifies and counts surface types in the given topology.

    Surface type mapping:
    - 0: Plane
    - 1: Cylinder
    - 2: Cone
    - 3: Sphere
    - 4: Torus
    - 5: Bezier surface
    - 6: BSpline surface
    - 7: Revolution surface
    - 8: Extrusion surface
    - 9: Other surface

    Returns:
        A tuple of counts for each surface type, in the following order:
        (num_plane_surfaces, num_cylinder_surfaces, num_cone_surfaces, num_sphere_surfaces, 
         num_torus_surfaces, num_bezier_surfaces, num_bspline_surfaces, num_revolution_surfaces, 
         num_extrusion_surfaces, num_other_surfaces)
    """
    surface_type = []
    for f in t.faces():
        surface = BRepAdaptor_Surface(f, True)
        surface_type.append(surface.GetType())

    return (surface_type.count(0),  # Plane
            surface_type.count(1),  # Cylinder
            surface_type.count(2),  # Cone
            surface_type.count(3),  # Sphere
            surface_type.count(4),  # Torus
            surface_type.count(5),  # Bezier surface
            surface_type.count(6),  # BSpline surface
            surface_type.count(7),  # Revolution surface
            surface_type.count(8),  # Extrusion surface
            surface_type.count(9)   # Other surface
    )


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


def read_center_of_mass(shape):
    """
    Calculates and returns the center of mass of the given shape, rounded to 2 decimal places.
    """
    props = GProp_GProps()
    brepgprop.VolumeProperties(shape, props)
    center = props.CentreOfMass()

    return round(center.X(), 2), round(center.Y(), 2), round(center.Z(), 2)


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
    

if __name__ == '__main__':
    # INPUT:
    step_folder_path = r"insert_STEP_folder_path_here"
    df_labels_path = r"insert_label_csv_path_here"    
    # OUTPUT:
    feature_csv_path = r"C:\Users\rafae\Documents\10_AutoSplit\11_TESTDATA\test_AM.csv" 

    # Initialize feature list and read labels
    features_list = []
    try:
        df_labels = pd.read_csv(df_labels_path)
    except FileNotFoundError:
        print(f"Error: Labels CSV file not found at {df_labels_path}")
        exit(1)

    # Process STEP files
    for i, step_file in enumerate(filter(lambda f: f.lower().endswith(('.step', '.stp')), os.listdir(step_folder_path))):
        step_file_path = os.path.join(step_folder_path, step_file)
        filename = os.path.splitext(step_file)[0]
        print(f"\nProcessing file #{i + 1}: {filename}")

        try:
            # Extract features for current STEP
            features = extract_STEP_features(step_file_path, filename, verbose=True)

            # Skip parts with more than one solid
            if features['num_solids'] != 1:
                print(f"Skipping file: {filename} (more than 1 solid)")
                continue

            # Get the label and skip if not found
            label_majority = get_label_majority(filename, df_labels)
            if label_majority is None:
                print(f"Skipping file: {filename} (label not found in CSV)")
                continue

            # Append features and label to the list
            features['label_majority'] = label_majority
            features_list.append(features)

        except Exception as e:
            print(f"Error processing file: {filename}. Error: {e}")

    # Create a DataFrame and save to CSV
    if features_list:
        columns = [
            'filename', 'label_majority', 'num_solids', 'num_faces', 'num_edges', 'volume', 'surface_area',
            'x_dim', 'y_dim', 'z_dim', 'num_line_edges', 'num_circle_edges', 'num_ellipse_edges',
            'num_hyperbola_edges', 'num_parabola_edges', 'num_beziercurve_edges', 'num_bsplinecurve_edges',
            'num_othercurve_edges', 'num_plane_surfaces', 'num_cylinder_surfaces', 'num_cone_surfaces',
            'num_sphere_surfaces', 'num_torus_surfaces', 'num_bezier_surfaces', 'num_bspline_surfaces',
            'num_revolution_surfaces', 'num_extrusion_surfaces', 'num_other_surfaces', 'num_unique_normals',
            'wall_thickness', 'x_center_mass_relative', 'y_center_mass_relative', 'z_center_mass_relative']
        df = pd.DataFrame(features_list)[columns]
        df.to_csv(feature_csv_path, index=False)
        print(f"\nFeatures successfully written to {feature_csv_path}")
    else:
        print("\nNo valid STEP files were processed.")
