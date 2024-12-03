import os
import numpy as np
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
from OCC.Core.gp import gp_Dir, gp_Vec, gp_Pnt
from OCC.Display.SimpleGui import init_display
from OCC.Core.V3d import V3d_XposYposZneg
import pandas as pd

# extract features
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

        edge_types = {'line': 0, 'circle': 0, 'ellipse': 0, 'hyperbola': 0, 
                     'parabola': 0, 'bezier': 0, 'bspline': 0, 'other': 0}
        
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

        surface_types = {'plane': 0, 'cylinder': 0, 'cone': 0, 'sphere': 0, 
                        'torus': 0, 'bezier': 0, 'bspline': 0, 'revolution': 0, 
                        'extrusion': 0, 'other': 0}
        
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
        wall_thickness = 2 * volume / surface_area if surface_area > 0 else 0

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
            'z_center_mass_relative': z_center
        }
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
        
        display, start_display, add_menu, add_function_to_menu = init_display()
        display.DisplayShape(shape, update=True)
        
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
        'x_center_mass_relative', 'y_center_mass_relative', 'z_center_mass_relative'
    ]
    return np.array([[features[col] for col in numeric_columns]])

def preprocess_image(file_path):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, [224, 224])
    img = tf.cast(img, tf.float32) / 255.0
    return img

def load_and_predict(model_path, features, iso_png_path, isoalt_png_path):
    try:
        model = load_model(model_path)
        numeric_input = prepare_features(features)
        
        scaler = StandardScaler()
        numeric_input_scaled = scaler.fit_transform(numeric_input)
        
        iso_img = preprocess_image(iso_png_path)
        isoalt_img = preprocess_image(isoalt_png_path)
        
        iso_prediction = model.predict([tf.expand_dims(iso_img, 0), numeric_input_scaled])
        isoalt_prediction = model.predict([tf.expand_dims(isoalt_img, 0), numeric_input_scaled])
        
        avg_prediction = (iso_prediction[0][0] + isoalt_prediction[0][0]) / 2
        
        return avg_prediction
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return None

def process_step_directory(directory_path, model_path):
    """Process all STEP files in the given directory."""
    step_extensions = {'.step', '.stp', '.STEP', '.STP'}
    
    step_files = [f for f in os.listdir(directory_path) 
                 if os.path.isfile(os.path.join(directory_path, f)) 
                 and os.path.splitext(f)[1] in step_extensions]
    
    if not step_files:
        print(f"No STEP files found in {directory_path}")
        return
    
    results = []
    
    for step_file in step_files:
        full_path = os.path.join(directory_path, step_file)
        print(f"\nProcessing: {step_file}")
        
        try:
            features, iso_png_path, isoalt_png_path = process_step_file(full_path)
            
            if features and iso_png_path and isoalt_png_path:
                prediction = load_and_predict(model_path, features, iso_png_path, isoalt_png_path)
                
                if prediction is not None:
                    result = {
                        'filename': step_file,
                        'prediction': float(prediction),
                        'classification': 'Standard part' if prediction > 0.5 else 'Non-standard part',
                        'confidence': float(max(prediction, 1-prediction) * 100)
                    }
                    result.update(features)  # Add all features to the result
                    results.append(result)
                    
                    print(f"Successfully processed {step_file}")
                    print(f"Prediction Score: {prediction:.4f}")
                    print(f"Classification: {result['classification']}")
                    print(f"Confidence: {result['confidence']:.2f}%")
            else:
                print(f"Failed to process {step_file}")
                
        except Exception as e:
            print(f"Error processing {step_file}: {str(e)}")
    
    return results

if __name__ == '__main__':
    # GPU Configuration
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
        except RuntimeError as e:
            print(f"GPU error: {e}")

    # Define paths
    step_directory = "path_to_your_step_folder"
  
    model_path = "hybrid_model_optimized_final_new.h5"
    

    print("Starting batch processing of STEP files...")
    print(f"Directory: {step_directory}")
    print(f"Model: {model_path}")
    print("-" * 50)
    
    try:
        # Process all STEP files in directory
        results = process_step_directory(step_directory, model_path)
        
        if results:
            # Create DataFrame from results
            df = pd.DataFrame(results)
            
            # Calculate summary statistics
            total_files = len(results)
            standard_parts = sum(1 for r in results if r['classification'] == 'Standard part')
            non_standard_parts = sum(1 for r in results if r['classification'] == 'Non-standard part')
            
            # Print summary
            print("\nAnalysis Summary:")
            print("-" * 50)
            print(f"Total files processed: {total_files}")
            print(f"Standard parts found: {standard_parts}")
            print(f"Non-standard parts found: {non_standard_parts}")
            print(f"Average confidence: {df['confidence'].mean():.2f}%")
            
            # Save  results to CSV
            csv_path = os.path.join(step_directory, 'step_analysis_results.csv')
            df.to_csv(csv_path, index=False)
            print(f"\nResults saved to CSV: {csv_path}")
            
            # Save Excel report
            excel_path = os.path.join(step_directory, 'step_analysis_detailed.xlsx')
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                
                df.to_excel(writer, sheet_name='Complete Results', index=False)
                
                # Summary sheet
                summary_data = pd.DataFrame({
                    'Metric': ['Total Files', 'Standard Parts', 'Non-Standard Parts',
                             'Average Confidence', 'Min Confidence', 'Max Confidence'],
                    'Value': [total_files, standard_parts, non_standard_parts,
                             f"{df['confidence'].mean():.2f}%",
                             f"{df['confidence'].min():.2f}%",
                             f"{df['confidence'].max():.2f}%"]
                })
                summary_data.to_excel(writer, sheet_name='Summary', index=False)
                
                # Standard parts sheet
                standard_parts_df = df[df['classification'] == 'Standard part']
                if not standard_parts_df.empty:
                    standard_parts_df.to_excel(writer, sheet_name='Standard Parts', index=False)
                
                # Non-standard parts sheet
                non_standard_parts_df = df[df['classification'] == 'Non-standard part']
                if not non_standard_parts_df.empty:
                    non_standard_parts_df.to_excel(writer, sheet_name='Non-Standard Parts', index=False)
                
                # Feature statistics sheet
                feature_stats = df.describe()
                feature_stats.to_excel(writer, sheet_name='Feature Statistics')
            
            print(f"Detailed Excel report saved to: {excel_path}")
            
            # Print detailed results
            print("\nDetailed Results:")
            print("-" * 50)
            for result in results:
                print(f"\nFile: {result['filename']}")
                print(f"Classification: {result['classification']}")
                print(f"Confidence: {result['confidence']:.2f}%")
                print(f"Key Features:")
                print(f"  - Solids: {result['num_solids']}")
                print(f"  - Faces: {result['num_faces']}")
                print(f"  - Edges: {result['num_edges']}")
                print(f"  - Volume: {result['volume']:.2f}")
                print(f"  - Surface Area: {result['surface_area']:.2f}")
            
        else:
            print("\nNo results to display. Check if there were errors during processing.")
    
    except Exception as e:
        print(f"\nMain execution error: {str(e)}")
        print("Detailed error information:")
        import traceback
        traceback.print_exc()
    
    finally:
        print("\nBatch processing completed.")
        try:
            from OCC.Display.SimpleGui import close_display
            close_display()
        except:
            pass
