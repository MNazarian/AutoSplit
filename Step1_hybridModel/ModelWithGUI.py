                                                ''' läuft ohne Bilder '''


import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
import os
import threading
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from OCC.Extend.DataExchange import read_step_file
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop
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
from OCC.Display.OCCViewer import Viewer3d
import gc
import logging

# Configure logging
logging.basicConfig(level=logging.WARNING)
logging.getLogger('OCC.Display.backend').setLevel(logging.WARNING)
logging.getLogger('tensorflow').setLevel(logging.WARNING)


# Pattern for the Display
class DisplayManager:
    _instance = None
    _display = None
    
    @classmethod
    def get_display(cls):
        if cls._display is None:
            cls._display, start_display, _, _ = init_display()
        return cls._display
    
    @classmethod
    def cleanup(cls):
        if cls._display:
            cls._display.EraseAll()
            cls._display = None

class StepAnalyzerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("STEP File Analyzer")
        self.root.geometry("1000x800")
        
        # Disable TensorFlow warnings
        tf.get_logger().setLevel('ERROR')
        
        # Configure GPU with memory growth
        self.configure_gpu()
        
        # Cache für das Model
        self.model_cache = None
        
        # Main container
        main_container = ttk.Frame(root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Input section
        self.setup_input_section(main_container)
        
        # Progress section
        self.setup_progress_section(main_container)
        
        # Results section
        self.setup_results_section(main_container)
        
        # Control buttons
        self.setup_control_buttons(main_container)
        
        self.processing = False
        self.results = []

    
    def configure_gpu(self):
        try:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.set_soft_device_placement(True)
        except Exception as e:
            print(f"GPU configuration error: {e}")

    def setup_input_section(self, container):
        input_frame = ttk.LabelFrame(container, text="Input Settings")
        input_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Directory selection
        ttk.Label(input_frame, text="STEP Files Directory:").grid(row=0, column=0, padx=5, pady=5)
        self.dir_path = tk.StringVar()
        dir_entry = ttk.Entry(input_frame, textvariable=self.dir_path, width=50)
        dir_entry.grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(input_frame, text="Browse", command=self.browse_directory).grid(row=0, column=2, padx=5, pady=5)
        
        # Model selection
        ttk.Label(input_frame, text="Model File:").grid(row=1, column=0, padx=5, pady=5)
        self.model_path = tk.StringVar(value="hybrid_model_optimized_final_new.h5")
        model_entry = ttk.Entry(input_frame, textvariable=self.model_path, width=50)
        model_entry.grid(row=1, column=1, padx=5, pady=5)
        ttk.Button(input_frame, text="Browse", command=self.browse_model).grid(row=1, column=2, padx=5, pady=5)

    def setup_progress_section(self, container):
        progress_frame = ttk.LabelFrame(container, text="Progress")
        progress_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, padx=5, pady=5)
        
        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(progress_frame, textvariable=self.status_var)
        status_label.pack(padx=5, pady=5)

    def setup_results_section(self, container):
        results_frame = ttk.LabelFrame(container, text="Results")
        results_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.notebook = ttk.Notebook(results_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Log tab
        self.log_text = scrolledtext.ScrolledText(self.notebook, height=20)
        self.notebook.add(self.log_text, text="Log")
        
        # Summary tab
        self.summary_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.summary_frame, text="Summary")
        
        # Details tab
        self.details_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.details_frame, text="Details")
        
        # Create treeview for details
        self.setup_treeview()

    def setup_treeview(self):
        columns = ("File", "Classification", "Confidence")
        self.tree = ttk.Treeview(self.details_frame, columns=columns, show="headings")
        
        # Configure columns
        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=150)
        
        # Add scrollbars
        yscrollbar = ttk.Scrollbar(self.details_frame, orient="vertical", command=self.tree.yview)
        xscrollbar = ttk.Scrollbar(self.details_frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=yscrollbar.set, xscrollcommand=xscrollbar.set)
        
        # Pack everything
        self.tree.grid(row=0, column=0, sticky="nsew")
        yscrollbar.grid(row=0, column=1, sticky="ns")
        xscrollbar.grid(row=1, column=0, sticky="ew")
        
        # Configure grid weights
        self.details_frame.grid_rowconfigure(0, weight=1)
        self.details_frame.grid_columnconfigure(0, weight=1)

    def setup_control_buttons(self, container):
        button_frame = ttk.Frame(container)
        button_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(button_frame, text="Start Analysis", command=self.start_analysis).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Stop", command=self.stop_analysis).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Export Results", command=self.export_results).pack(side=tk.LEFT, padx=5)

    def generate_screenshot(self, shape, output_path, perspective='ISO'):
        try:
            display = DisplayManager.get_display()
            display.EraseAll()
            display.DisplayShape(shape, update=True)
            
            if perspective == 'ISO':
                display.View_Iso()
            elif perspective == 'ISOAlt':
                display.View.SetProj(V3d_XposYposZneg)
            
            display.FitAll()
            display.ExportToImage(output_path)
            return True
        except Exception as e:
            self.log(f"Error generating screenshot: {e}")
            return False
        finally:
            gc.collect()

    def process_step_file(self, file_path):
        try:
            shape = read_step_file(file_path)
            info = self.extract_step_info(shape)
            if info is None:
                return None
            
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            screenshots_dir = os.path.join(os.path.dirname(file_path), "screenshots")
            os.makedirs(screenshots_dir, exist_ok=True)
            
            iso_path = os.path.join(screenshots_dir, f"{base_name}_ISO.png")
            isoalt_path = os.path.join(screenshots_dir, f"{base_name}_ISOAlt.png")
            
            if not (self.generate_screenshot(shape, iso_path, 'ISO') and 
                   self.generate_screenshot(shape, isoalt_path, 'ISOAlt')):
                return None
            
            prediction = self.predict_with_model(info, iso_path, isoalt_path)
            if prediction is None:
                return None
            
            return {
                'filename': os.path.basename(file_path),
                'prediction': float(prediction),
                'classification': 'Standard part' if prediction > 0.5 else 'Non-standard part',
                'confidence': float(max(prediction, 1-prediction) * 100),
                **info
            }
            
        except Exception as e:
            self.log(f"Error processing file: {e}")
            return None
        finally:
            gc.collect()

    def extract_step_info(self, shape):
        try:
            # Berechnet Bounding Box
            bbox = Bnd_Box()
            brepbndlib.Add(shape, bbox)
            xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()

            # Volumeneigenschaften
            props = GProp_GProps()
            brepgprop.VolumeProperties(shape, props)
            volume = props.Mass()
            
            brepgprop.SurfaceProperties(shape, props)
            surface_area = props.Mass()
            
            x_dim = xmax - xmin
            y_dim = ymax - ymin
            z_dim = zmax - zmin
            
            if min(x_dim, y_dim, z_dim) <= 1e-6:
                self.log("Warning: Very small or zero dimension detected")
                return None
            
            t = TopologyExplorer(shape)
            edge_types = self.analyze_edges(t)
            surface_types, unique_normals = self.analyze_surfaces(t)
            
            center_of_mass = props.CentreOfMass()
            relative_coords = self.calculate_relative_coordinates(
                center_of_mass, xmin, ymin, zmin, x_dim, y_dim, z_dim)
            
            brepgprop.LinearProperties(shape, props)
            wall_thickness = 2 * volume / surface_area if surface_area > 0 else 0
            
            return {
                'num_solids': t.number_of_solids(),
                'num_faces': t.number_of_faces(),
                'num_edges': t.number_of_edges(),
                'volume': volume,
                'surface_area': surface_area,
                'x_dim': x_dim,
                'y_dim': y_dim,
                'z_dim': z_dim,
                **edge_types,
                **surface_types,
                'num_unique_normals': len(unique_normals),
                'wall_thickness': wall_thickness,
                **relative_coords
            }
        except Exception as e:
            self.log(f"Error extracting STEP info: {e}")
            return None

    def analyze_edges(self, topology):
        edge_types = {
            'num_line_edges': 0, 'num_circle_edges': 0,
            'num_ellipse_edges': 0, 'num_hyperbola_edges': 0,
            'num_parabola_edges': 0, 'num_beziercurve_edges': 0,
            'num_bsplinecurve_edges': 0, 'num_othercurve_edges': 0
        }
        
        type_map = {
            GeomAbs_Line: 'num_line_edges',
            GeomAbs_Circle: 'num_circle_edges',
            GeomAbs_Ellipse: 'num_ellipse_edges',
            GeomAbs_Hyperbola: 'num_hyperbola_edges',
            GeomAbs_Parabola: 'num_parabola_edges',
            GeomAbs_BezierCurve: 'num_beziercurve_edges',
            GeomAbs_BSplineCurve: 'num_bsplinecurve_edges'
        }
        
        for edge in topology.edges():
            curve_type = BRepAdaptor_Curve(edge).GetType()
            edge_types[type_map.get(curve_type, 'num_othercurve_edges')] += 1
            
        return edge_types

    def analyze_surfaces(self, topology):
        surface_types = {
            'num_plane_surfaces': 0, 'num_cylinder_surfaces': 0,
            'num_cone_surfaces': 0, 'num_sphere_surfaces': 0,
            'num_torus_surfaces': 0, 'num_bezier_surfaces': 0,
            'num_bspline_surfaces': 0, 'num_revolution_surfaces': 0,
            'num_extrusion_surfaces': 0, 'num_other_surfaces': 0
        }
        
        type_map = {
            GeomAbs_Plane: 'num_plane_surfaces',
            GeomAbs_Cylinder: 'num_cylinder_surfaces',
            GeomAbs_Cone: 'num_cone_surfaces',
            GeomAbs_Sphere: 'num_sphere_surfaces',
            GeomAbs_Torus: 'num_torus_surfaces',
            GeomAbs_BezierSurface: 'num_bezier_surfaces',
            GeomAbs_BSplineSurface: 'num_bspline_surfaces',
            GeomAbs_SurfaceOfRevolution: 'num_revolution_surfaces',
            GeomAbs_SurfaceOfExtrusion: 'num_extrusion_surfaces'
        }
        
        unique_normals = set()
        
        for face in topology.faces():
            surface = BRepAdaptor_Surface(face)
            surface_type = surface.GetType()
            surface_types[type_map.get(surface_type, 'num_other_surfaces')] += 1
            
            try:
                u = (surface.LastUParameter() + surface.FirstUParameter()) / 2
                v = (surface.LastVParameter() + surface.FirstVParameter()) / 2
                
                pnt = gp_Pnt()
                du = gp_Vec()
                dv = gp_Vec()
                
                surface.D1(u, v, pnt, du, dv)
                normal = du.Crossed(dv)
                if normal.Magnitude() > 1e-7:
                    normal.Normalize()
                    unique_normals.add((
                        round(normal.X(), 5),
                        round(normal.Y(), 5),
                        round(normal.Z(), 5)
                    ))
            except:
                pass
                
        return surface_types, unique_normals

    def calculate_relative_coordinates(self, point, xmin, ymin, zmin, x_dim, y_dim, z_dim):
        return {
            'x_center_mass_relative': (point.X() - xmin) / x_dim if x_dim > 0 else 0,
            'y_center_mass_relative': (point.Y() - ymin) / y_dim if y_dim > 0 else 0,
            'z_center_mass_relative': (point.Z() - zmin) / z_dim if z_dim > 0 else 0
        }

    def load_cached_model(self):
        if self.model_cache is None:
            try:
                self.model_cache = load_model(self.model_path.get(), compile=False)
                self.model_cache.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            except Exception as e:
                self.log(f"Error loading model: {e}")
                return None
        return self.model_cache

    @tf.function(experimental_relax_shapes=True)
    def preprocess_image(self, file_path):
        try:
            img = tf.io.read_file(file_path)
            img = tf.image.decode_png(img, channels=3)
            img = tf.image.resize(img, [224, 224])
            img = tf.cast(img, tf.float32) / 255.0
            return img
        except Exception as e:
            self.log(f"Error preprocessing image: {e}")
            return None

    def prepare_features(self, features):
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

    @tf.function(experimental_relax_shapes=True)
    def predict_single(self, model, img, features):
        return model([tf.expand_dims(img, 0), features], training=False)

    def predict_with_model(self, features, iso_path, isoalt_path):
        try:
            model = self.load_cached_model()
            if model is None:
                return None

            # Combines geometric features with image analysis
            numeric_input = self.prepare_features(features)
            scaler = StandardScaler()
            numeric_input_scaled = scaler.fit_transform(numeric_input)
            
            iso_img = self.preprocess_image(iso_path)
            isoalt_img = self.preprocess_image(isoalt_path)
            
            if iso_img is None or isoalt_img is None:
                return None
            
            with tf.device('/CPU:0'):
                iso_prediction = self.predict_single(
                    model, iso_img, numeric_input_scaled).numpy()
                isoalt_prediction = self.predict_single(
                    model, isoalt_img, numeric_input_scaled).numpy()
            
            return (float(iso_prediction[0][0]) + float(isoalt_prediction[0][0])) / 2
            
        except Exception as e:
            self.log(f"Error in prediction: {e}")
            return None
        finally:
            gc.collect()

    def browse_directory(self):
        directory = filedialog.askdirectory()
        if directory:
            self.dir_path.set(directory)

    def browse_model(self):
        model_file = filedialog.askopenfilename(filetypes=[("H5 files", "*.h5")])
        if model_file:
            self.model_path.set(model_file)
            self.model_cache = None

    def log(self, message):
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()

    def start_analysis(self):
        if not self.validate_inputs():
            return
            
        self.processing = True
        self.results = []
        self.log_text.delete(1.0, tk.END)
        self.tree.delete(*self.tree.get_children())
        self.status_var.set("Processing...")
        self.progress_var.set(0)
        
        thread = threading.Thread(target=self.process_files)
        thread.daemon = True
        thread.start()

    def validate_inputs(self):
        if not self.dir_path.get() or not self.model_path.get():
            messagebox.showerror("Error", "Please select both directory and model file.")
            return False
            
        if not os.path.exists(self.dir_path.get()):
            messagebox.showerror("Error", "Selected directory does not exist.")
            return False
            
        if not os.path.exists(self.model_path.get()):
            messagebox.showerror("Error", "Selected model file does not exist.")
            return False
            
        return True

    def process_files(self):
        try:
            step_files = self.get_step_files()
            if not step_files:
                self.log("No STEP files found in the selected directory")
                return
                
            total_files = len(step_files)
            processed_files = 0
            
            for step_file in step_files:
                if not self.processing:
                    break
                    
                full_path = os.path.join(self.dir_path.get(), step_file)
                self.log(f"\nProcessing: {step_file}")
                
                result = self.process_step_file(full_path)
                if result:
                    self.results.append(result)
                    self.log(f"Successfully processed {step_file}")
                    self.log(f"Classification: {result['classification']}")
                    self.log(f"Confidence: {result['confidence']:.2f}%")
                    
                processed_files += 1
                progress = (processed_files / total_files) * 100
                self.progress_var.set(progress)
                self.root.update_idletasks()
                
                # Periodically cleanup memory
                if processed_files % 5 == 0:
                    gc.collect()
                
            self.update_results_display()
            
        except Exception as e:
            self.log(f"Error during processing: {str(e)}")
        finally:
            self.processing = False
            self.status_var.set("Processing completed")
            self.progress_var.set(100)
            DisplayManager.cleanup()

    def get_step_files(self):
        step_extensions = {'.step', '.stp', '.STEP', '.STP'}
        return [f for f in os.listdir(self.dir_path.get()) 
                if os.path.isfile(os.path.join(self.dir_path.get(), f)) 
                and os.path.splitext(f)[1] in step_extensions]

    def stop_analysis(self):
        self.processing = False
        self.status_var.set("Analysis stopped by user")
        DisplayManager.cleanup()

    def export_results(self):
        if not self.results:
            messagebox.showwarning("Warning", "No results to export.")
            return
            
        try:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".xlsx",
                filetypes=[("Excel files", "*.xlsx"), ("CSV files", "*.csv")]
            )
            
            if file_path:
                df = pd.DataFrame(self.results)
                if file_path.endswith('.xlsx'):
                    df.to_excel(file_path, index=False)
                else:
                    df.to_csv(file_path, index=False)
                    
                messagebox.showinfo("Success", f"Results exported to {file_path}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error exporting results:\n{str(e)}")

    def update_results_display(self):
        self.tree.delete(*self.tree.get_children())
        
        for result in self.results:
            self.tree.insert("", tk.END, values=(
                result['filename'],
                result['classification'],
                f"{result['confidence']:.2f}%"
            ))
        
        for widget in self.summary_frame.winfo_children():
            widget.destroy()
        
        if self.results:
            total_files = len(self.results)
            standard_parts = sum(1 for r in self.results if r['classification'] == 'Standard part')
            non_standard_parts = total_files - standard_parts
            avg_confidence = sum(r['confidence'] for r in self.results) / total_files
            
            summary_text = f"""
            Total files processed: {total_files}
            Standard parts found: {standard_parts}
            Non-standard parts found: {non_standard_parts}
            Average confidence: {avg_confidence:.2f}%
            """
            
            ttk.Label(self.summary_frame, text=summary_text, justify=tk.LEFT).pack(padx=10, pady=10)

if __name__ == '__main__':
    # Configure logging levels
    logging.basicConfig(level=logging.WARNING)
    tf.get_logger().setLevel('WARNING')
    
    root = tk.Tk()
    app = StepAnalyzerGUI(root)
    root.mainloop()