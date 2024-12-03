import warnings
import numpy as np
from sklearn.preprocessing import StandardScaler
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
from OCC.Core.TopoDS import TopoDS_Face, TopoDS_Iterator, topods
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_REVERSED, TopAbs_EDGE, TopAbs_SOLID
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.GeomAbs import (GeomAbs_Line, GeomAbs_Circle, GeomAbs_Ellipse, 
                             GeomAbs_BSplineCurve, GeomAbs_Hyperbola, GeomAbs_Parabola,
                             GeomAbs_BezierCurve, GeomAbs_Plane, GeomAbs_Cylinder, 
                             GeomAbs_Cone, GeomAbs_Sphere, GeomAbs_Torus, 
                             GeomAbs_BSplineSurface, GeomAbs_BezierSurface,
                             GeomAbs_SurfaceOfRevolution, GeomAbs_SurfaceOfExtrusion)
from OCC.Core.GCPnts import GCPnts_AbscissaPoint
from OCC.Core.gp import gp_Vec, gp_Dir
from OCC.Display.SimpleGui import init_display

warnings.filterwarnings("ignore")

def read_step_file(step_path):
    #Read a STEP file 
    step_reader = STEPControl_Reader()
    status = step_reader.ReadFile(step_path)
    if status == IFSelect_RetDone:
        step_reader.TransferRoots()
        shape = step_reader.Shape()
        return shape
    else:
        raise Exception("Error reading STEP file.")

def safe_divide(a, b):
    # Safely divide 
    return a / b if b != 0 else 0

def get_edge_length(edge):
    
    curve_adaptor = BRepAdaptor_Curve(edge)
    first = curve_adaptor.FirstParameter()
    last = curve_adaptor.LastParameter()
    return GCPnts_AbscissaPoint().Length(curve_adaptor, first, last, 1.0e-6)

def get_all_faces(shp):
    faces = []
    explorer = TopExp_Explorer(shp, TopAbs_FACE)
    while explorer.More():
        faces.append(explorer.Current())
        explorer.Next()
    return faces

def calculate_face_normal_direction(face):
    if not isinstance(face, TopoDS_Face):
        raise TypeError("Input must be a TopoDS_Face")
    
    surface = BRep_Tool.Surface(face)
    adaptor = BRepAdaptor_Surface(face)
    u_min, u_max = adaptor.FirstUParameter(), adaptor.LastUParameter()
    v_min, v_max = adaptor.FirstVParameter(), adaptor.LastVParameter()
    u_mid = (u_min + u_max) / 2.0
    v_mid = (v_min + v_max) / 2.0
    
    sl_props = GeomLProp_SLProps(surface, u_mid, v_mid, 1, 0.01)
    if sl_props.IsNormalDefined():
        return sl_props.Normal()
    else:
        raise ValueError("Normal is not defined for this face")

def compute_unique_normal_vectors(faces: list) -> list:
    unique_normals = []
    for face in faces:
        surface_adaptor = BRepAdaptor_Surface(face)
        normal_vector = surface_adaptor.Plane().Axis().Direction()
        x, y, z = round(normal_vector.X(), 3), round(normal_vector.Y(), 3), round(normal_vector.Z(), 3)
        cleaned_normal = gp_Dir(x, y, z)
        
        is_unique = True
        for existing_normal in unique_normals:
            dot_product = (cleaned_normal.X() * existing_normal.X() +
                         cleaned_normal.Y() * existing_normal.Y() +
                         cleaned_normal.Z() * existing_normal.Z())
            if abs(dot_product) == 1:
                is_unique = False
                break
        if is_unique:
            unique_normals.append(cleaned_normal)
    return unique_normals

def gather_faces_perpendicular_to_vector(faces: list, normal: gp_Dir) -> list:
    perpendicular_faces = []
    for face in faces:
        surface_adaptor = BRepAdaptor_Surface(face)
        face_normal = surface_adaptor.Plane().Axis().Direction()
        dot_product = (face_normal.X() * normal.X() +
                      face_normal.Y() * normal.Y() +
                      face_normal.Z() * normal.Z())
        dot_product = round(dot_product, 3)
        if abs(dot_product) >= 1:
            perpendicular_faces.append(face)
    return perpendicular_faces

def sort_faces_along_normal(faces: list, normal: gp_Dir) -> list:
    def project_face_centroid(face):
        surface_adaptor = BRepAdaptor_Surface(face)
        u_min, u_max = surface_adaptor.FirstUParameter(), surface_adaptor.LastUParameter()
        v_min, v_max = surface_adaptor.FirstVParameter(), surface_adaptor.LastVParameter()
        centroid = surface_adaptor.Value((u_min + u_max) / 2, (v_min + v_max) / 2)
        return normal.X() * centroid.X() + normal.Y() * centroid.Y() + normal.Z() * centroid.Z()
    return sorted(faces, key=project_face_centroid)

def calculate_face_to_face_thicknesses(faces: list, normal: gp_Dir, lower_limit: float, upper_limit: float) -> list:
    # Calculate thicknesses between consecutive faces along normal direction.
    thicknesses = []
    for i in range(len(faces) - 1):
        face1 = faces[i]
        face2 = faces[i + 1]
        
        surface1 = BRepAdaptor_Surface(face1)
        surface2 = BRepAdaptor_Surface(face2)
        
        u_min1, u_max1 = surface1.FirstUParameter(), surface1.LastUParameter()
        v_min1, v_max1 = surface1.FirstVParameter(), surface1.LastVParameter()
        u_min2, u_max2 = surface2.FirstUParameter(), surface2.LastUParameter()
        v_min2, v_max2 = surface2.FirstVParameter(), surface2.LastVParameter()
        
        centroid1 = surface1.Value((u_min1 + u_max1) / 2, (v_min1 + v_max1) / 2)
        centroid2 = surface2.Value((u_min2 + u_max2) / 2, (v_min2 + v_max2) / 2)
        
        distance = round(abs((centroid2.X() - centroid1.X()) * normal.X() +
                           (centroid2.Y() - centroid1.Y()) * normal.Y() +
                           (centroid2.Z() - centroid1.Z()) * normal.Z()), 3)
        
        if lower_limit <= distance <= upper_limit:
            thicknesses.append(distance)
    return thicknesses

def calculate_minimal_wall_thickness(shp, lower_limit=0.1, upper_limit=10):
    #Calculate minimal wall thickness of the part
    # Get all faces
    all_faces = get_all_faces(shp)
    if not all_faces:
        return 0
    
    # Calculate face areas 
    face_areas = []
    planar_faces = []
    props = GProp_GProps()
    for face in all_faces:
        # Area
        brepgprop.SurfaceProperties(face, props)
        area = round(props.Mass(), 3)
        face_areas.append((face, area))

        # Planarity
        surface_adaptor = BRepAdaptor_Surface(face)
        if surface_adaptor.GetType() == GeomAbs_Plane:
            planar_faces.append(face)

    # Sort by size
    face_areas.sort(key=lambda x: x[1], reverse=True)
    largest_faces = [item[0] for item in face_areas[:2]]

    # Get planar faces
    largest_planar_faces = []
    for face in largest_faces:
        if face in planar_faces:
            largest_planar_faces.append(face)

    if not largest_planar_faces:
        return 0
        
    # Compute unique normal vector
    unique_normals = compute_unique_normal_vectors(largest_planar_faces)

    # Initialize a list to hold all valid thicknesses
    all_thicknesses = []

    # For each unique normal, gather faces, sort them, and calculate thicknesses
    for normal in unique_normals:
        perpendicular_faces = gather_faces_perpendicular_to_vector(largest_planar_faces, normal)
        if len(perpendicular_faces) < 2:
            continue

        sorted_faces = sort_faces_along_normal(perpendicular_faces, normal)
        thicknesses = calculate_face_to_face_thicknesses(sorted_faces, normal, lower_limit, upper_limit)
        all_thicknesses.extend(thicknesses)
    
    if not all_thicknesses:
        return 0

    return min(all_thicknesses)

def extract_features_from_step(shape):
    # Extract all geometric featur
    # Initialize topology explorer
    t = TopologyExplorer(shape)
    
    # Basic counts
    num_solids = t.number_of_solids()
    num_faces = t.number_of_faces()
    num_edges = t.number_of_edges()
    
    # Volume and surface area
    props = GProp_GProps()
    brepgprop.VolumeProperties(shape, props)
    volume = props.Mass()
    brepgprop.SurfaceProperties(shape, props)
    surface_area = props.Mass()
    
    # Bounding box and dimensions
    bbox = Bnd_Box()
    brepbndlib.Add(shape, bbox)
    xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
    x_dim = xmax - xmin
    y_dim = ymax - ymin
    z_dim = zmax - zmin
    
    # Center of mass (mit Rundung auf 2 Dezimalstellen)
    cogx, cogy, cogz = props.CentreOfMass().Coord()
    x_center_mass_relative = round(safe_divide(cogx - xmin, x_dim), 2)
    y_center_mass_relative = round(safe_divide(cogy - ymin, y_dim), 2)
    z_center_mass_relative = round(safe_divide(cogz - zmin, z_dim), 2)
    
    # Wall thickness
    wall_thickness = calculate_minimal_wall_thickness(shape, lower_limit=0.1, upper_limit=10)
    
    # Edge types
    edge_counts = {
        'line': 0, 'circle': 0, 'ellipse': 0, 'hyperbola': 0,
        'parabola': 0, 'bezier': 0, 'bspline': 0, 'other': 0
    }
    
    total_edge_length = 0
    for edge in t.edges():
        curve = BRepAdaptor_Curve(edge)
        curve_type = curve.GetType()
        
        try:
            total_edge_length += get_edge_length(edge)
        except:
            pass
        
        if curve_type == GeomAbs_Line:
            edge_counts['line'] += 1
        elif curve_type == GeomAbs_Circle:
            edge_counts['circle'] += 1
        elif curve_type == GeomAbs_Ellipse:
            edge_counts['ellipse'] += 1
        elif curve_type == GeomAbs_Hyperbola:
            edge_counts['hyperbola'] += 1
        elif curve_type == GeomAbs_Parabola:
            edge_counts['parabola'] += 1
        elif curve_type == GeomAbs_BezierCurve:
            edge_counts['bezier'] += 1
        elif curve_type == GeomAbs_BSplineCurve:
            edge_counts['bspline'] += 1
        else:
            edge_counts['other'] += 1
    
    # Surface types
    surface_counts = {
        'plane': 0, 'cylinder': 0, 'cone': 0, 'sphere': 0,
        'torus': 0, 'bezier': 0, 'bspline': 0, 'revolution': 0,
        'extrusion': 0, 'other': 0
    }
    
    unique_normals = set()
    for face in t.faces():
        surface = BRepAdaptor_Surface(face)
        surface_type = surface.GetType()
        
        if surface_type == GeomAbs_Plane:
            surface_counts['plane'] += 1
            try:
                normal = calculate_face_normal_direction(topods.Face(face))
                normal_tuple = (round(normal.X(), 2), round(normal.Y(), 2), round(normal.Z(), 2))
                unique_normals.add(normal_tuple)
            except:
                pass
        elif surface_type == GeomAbs_Cylinder:
            surface_counts['cylinder'] += 1
        elif surface_type == GeomAbs_Cone:
            surface_counts['cone'] += 1
        elif surface_type == GeomAbs_Sphere:
            surface_counts['sphere'] += 1
        elif surface_type == GeomAbs_Torus:
            surface_counts['torus'] += 1
        elif surface_type == GeomAbs_BezierSurface:
            surface_counts['bezier'] += 1
        elif surface_type == GeomAbs_BSplineSurface:
            surface_counts['bspline'] += 1
        elif surface_type == GeomAbs_SurfaceOfRevolution:
            surface_counts['revolution'] += 1
        elif surface_type == GeomAbs_SurfaceOfExtrusion:
            surface_counts['extrusion'] += 1
        else:
            surface_counts['other'] += 1
    
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
        'num_unique_normals': len(unique_normals),
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
    features['num_line_edges_ratio'] = safe_divide(edge_counts['line'], total_edges)
    features['num_circle_edges_ratio'] = safe_divide(edge_counts['circle'], total_edges)
    features['num_ellipse_edges_ratio'] = safe_divide(edge_counts['ellipse'], total_edges)
    features['num_bsplinecurve_edges_ratio'] = safe_divide(edge_counts['bspline'], total_edges)
    
    # Calculate surface ratios
    total_surfaces = sum(surface_counts.values())
    features['num_plane_surfaces_ratio'] = safe_divide(surface_counts['plane'], total_surfaces)
    features['num_cylinder_surfaces_ratio'] = safe_divide(surface_counts['cylinder'], total_surfaces)
    features['num_cone_surfaces_ratio'] = safe_divide(surface_counts['cone'], total_surfaces)
    features['num_sphere_surfaces_ratio'] = safe_divide(surface_counts['sphere'], total_surfaces)
    features['num_torus_surfaces_ratio'] = safe_divide(surface_counts['torus'], total_surfaces)
    features['num_bspline_surfaces_ratio'] = safe_divide(surface_counts['bspline'], total_surfaces)
    
    # Calculate derived features
    # Aspect ratios
    features['aspect_ratio_xy'] = safe_divide(x_dim, y_dim)
    features['aspect_ratio_xz'] = safe_divide(x_dim, z_dim)
    features['aspect_ratio_yz'] = safe_divide(y_dim, z_dim)
    
    # Volume ratios
    xyz_product = x_dim * y_dim * z_dim
    features['volume_to_xyz'] = safe_divide(volume, xyz_product)
    features['surface_to_xyz'] = safe_divide(surface_area, xyz_product)


    # Center mass distance
    center_coords = np.array([x_center_mass_relative, y_center_mass_relative, z_center_mass_relative]) - 0.5
    features['center_mass_distance'] = np.linalg.norm(center_coords)
    
    # Basic ratios and complexities
    features['volume_to_surface_ratio'] = safe_divide(volume, surface_area)
    features['edge_to_face_ratio'] = safe_divide(num_edges, num_faces)
    features['average_face_area'] = safe_divide(surface_area, num_faces)
    features['edge_complexity'] = safe_divide(num_edges, num_faces)
    features['surface_complexity'] = safe_divide(len(unique_normals), num_faces)
    features['thickness_complexity'] = safe_divide(wall_thickness, num_faces)
    features['average_edge_length'] = safe_divide(total_edge_length, num_edges)
    features['shape_factor'] = safe_divide(surface_area, np.power(volume, 2/3)) if volume > 0 else 0


    # Log transforms
    for col in ['volume', 'surface_area', 'num_faces', 'num_edges']:
        min_positive = max(features[col], 1e-10)
        features[f'log_{col}'] = np.log1p(min_positive)
    
    # Overall complexity
    complexity_cols = ['edge_complexity', 'surface_complexity', 'thickness_complexity']
    complexity_values = [features[col] for col in complexity_cols]
    features['overall_complexity'] = np.mean([v for v in complexity_values if v >= 0])


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


def analyze_step_files_in_folder(folder_path):
    """Analyze all STEP files in the given folder."""
    results = []
    
    # Get all STEP files in the folder
    step_files = [f for f in os.listdir(folder_path) 
                 if f.lower().endswith(('.step', '.stp'))]
    
    print(f"Found {len(step_files)} STEP files in folder.")
    
    # Load model components once
    rf = joblib.load('models/new_model_11.joblib')
    scaler = joblib.load('models/new_scaler_11.joblib')
    selected_features = joblib.load('models/selected_features_11.joblib')
    
    # Process each file
    for filename in step_files:
        try:
            file_path = os.path.join(folder_path, filename)
            
            # Load STEP file
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
            
            # Print prediction for this file
            print(f"\n{filename}:")
            print(f"Prediction: Class {prediction}")
            print("Probabilities:", end=" ")
            for i, prob in enumerate(probabilities):
                print(f"Class {i}: {prob:.3f}", end=" ")
            print()
            
        except Exception as e:
            print(f"\nError processing {filename}: {str(e)}")
    
    return results

if __name__ == "__main__":
    folder_path = "path_to_your_step_folder"
    
    try:
        print("Starting analysis of all STEP files...\n")
        results = analyze_step_files_in_folder(folder_path)
        
        # Show summary
        print("\nAnalysis Summary:")
        print("-" * 50)
        class_counts = {}
        for r in results:
            prediction = r['prediction']
            class_counts[prediction] = class_counts.get(prediction, 0) + 1
        
        print(f"Total files processed: {len(results)}")
        print("\nPredicted class distribution:")
        for class_label, count in sorted(class_counts.items()):
            print(f"Class {class_label}: {count} files ({count/len(results)*100:.1f}%)")
        
    except Exception as e:
        print(f"\nError during analysis: {str(e)}")
        raise
