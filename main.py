#!/usr/bin/env python3
# main.py - Aplicación Principal MVC con PyQt6 + Climate Data + Export
import sys
import os
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QDir

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from view.main_window import MainWindow
from controller.app_controller import AppController
from model.excel_model import ExcelModel
from model.climate_model import ClimateModel  


def clean_pycache(root_dir=None):
    """
    Elimina todas las carpetas __pycache__ del proyecto
    """
    if root_dir is None:
        root_dir = os.path.dirname(os.path.abspath(__file__))
    
    deleted_count = 0
    
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if '__pycache__' in dirnames:
            try:
                import shutil
                cache_dir = os.path.join(dirpath, '__pycache__')
                shutil.rmtree(cache_dir)
                deleted_count += 1
            except Exception:
                pass
    
    if deleted_count > 0:
        print(f"\n✓ Total de carpetas __pycache__ eliminadas: {deleted_count}")
    else:
        print("✓ No se encontraron carpetas __pycache__")
    
    return deleted_count


class SAIDIApplication:
    """Aplicación principal SAIDI con arquitectura MVC"""
    
    def __init__(self):
        self.app = QApplication(sys.argv)
        self.setup_mvc()
    
    def setup_mvc(self):
        """Configurar arquitectura MVC"""
        # Modelos
        self.excel_model = ExcelModel()
        self.climate_model = ClimateModel() 
        
        # Vista
        self.main_window = MainWindow()
        
        # Controlador
        self.controller = AppController(self.main_window, self.excel_model, self.climate_model)
        
        # Hacer referencia cruzada para que la vista pueda intentar conectar botones durante construcción
        self.main_window.controller = self.controller
        
        # Conectar señales básicas
        self.connect_signals()
    
    def connect_signals(self):
        """Conectar señales entre vista y controlador"""
        try:
            # Botones principales
            if hasattr(self.main_window, 'load_excel_button'):
                self.main_window.load_excel_button.clicked.connect(self.controller.load_excel_file)
            
            if hasattr(self.main_window, 'predict_button'):
                self.main_window.predict_button.clicked.connect(self.controller.run_prediction)
            
            if hasattr(self.main_window, 'optimize_button'):
                if getattr(self.controller, 'run_optimization', None):
                    self.main_window.optimize_button.clicked.connect(self.controller.run_optimization)
            
            # Selector de regional
            self.main_window.regional_selected.connect(self.controller.on_regional_selected)
            
            # Carga de datos climáticos desde botones
            self.main_window.climate_load_requested.connect(self.controller.load_climate_file)
            
            # ✅ CONEXIÓN CRÍTICA: Exportar a Excel
            if hasattr(self.main_window, 'export_requested'):
                self.main_window.export_requested.connect(self.controller.export_predictions_to_excel)
                print("✓ Señal de exportación conectada correctamente")
            
            # Señales del modelo -> vista
            if getattr(self.excel_model, 'data_loaded', None):
                self.excel_model.data_loaded.connect(self.main_window.on_excel_loaded)
            
            if getattr(self.climate_model, 'climate_data_loaded', None):
                self.climate_model.climate_data_loaded.connect(self.controller.on_climate_data_loaded)
            if getattr(self.climate_model, 'all_climate_loaded', None):
                self.climate_model.all_climate_loaded.connect(self.controller.on_all_climate_loaded)
        
        except Exception as e:
            print(f"[WARN] Error conectando señales: {e}")
    
    def run(self):
        """Mostrar la ventana principal y ejecutar la app"""
        self.main_window.show()
        return self.app.exec()


if __name__ == '__main__':
    # Limpiar carpetas __pycache__ antes de iniciar
    clean_pycache()
    
    app = SAIDIApplication()
    sys.exit(app.run())