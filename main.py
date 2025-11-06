#!/usr/bin/env python3
# main.py - Aplicación Principal MVC 
import sys
import os
from PyQt6.QtWidgets import QApplication
#from PyQt6.QtCore import QDir

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
            # ========== BOTONES PRINCIPALES ==========
            
            # Carga de Excel
            if hasattr(self.main_window, 'load_excel_button'):
                self.main_window.load_excel_button.clicked.connect(self.controller.load_excel_file)
                print(" Botón 'Cargar Excel' conectado")
            
            # Predicción
            if hasattr(self.main_window, 'predict_button'):
                self.main_window.predict_button.clicked.connect(self.controller.run_prediction)
                print(" Botón 'Predicción' conectado")
            
            # Optimización
            if hasattr(self.main_window, 'optimize_button'):
                if hasattr(self.controller, 'run_optimization'):
                    self.main_window.optimize_button.clicked.connect(self.controller.run_optimization)
                    print(" Botón 'Optimización' conectado")
            
            # Validación
            if hasattr(self.main_window, 'validate_button'):
                if hasattr(self.controller, 'run_validation'):
                    self.main_window.validate_button.clicked.connect(self.controller.run_validation)
                    print("Botón 'Validación' conectado")
                else:
                    print("Controlador no tiene método 'run_validation'")
            else:
                print("Vista no tiene 'validate_button'")

            # Generación de Informe (reemplaza Rolling Validation)
            if hasattr(self.main_window, 'report_button'):
                if hasattr(self.controller, 'generate_validation_report'):
                    self.main_window.report_button.clicked.connect(self.controller.generate_validation_report)
                    print("Botón 'Generar Informe' conectado")
                else:
                    print("Controlador no tiene método 'generate_validation_report'")
            else:
                print("Vista no tiene 'report_button'")
            
            # ========== SELECTOR DE REGIONAL ==========
            
            if hasattr(self.main_window, 'regional_selected'):
                self.main_window.regional_selected.connect(self.controller.on_regional_selected)
                print(" Selector de regional conectado")
            
            # ========== CARGA DE DATOS CLIMÁTICOS ==========
            
            if hasattr(self.main_window, 'climate_load_requested'):
                self.main_window.climate_load_requested.connect(self.controller.load_climate_file)
                print(" Carga de datos climáticos conectada")
            
            # ========== EXPORTACIÓN A EXCEL ==========
            
            if hasattr(self.main_window, 'export_requested'):
                if hasattr(self.controller, 'export_predictions_to_excel'):
                    self.main_window.export_requested.connect(self.controller.export_predictions_to_excel)
                    print(" Señal de exportación conectada")
                else:
                    print("Controlador no tiene método 'export_predictions_to_excel'")
            
            # ========== SEÑALES DEL MODELO -> VISTA ==========
            
            # Excel Model
            if hasattr(self.excel_model, 'data_loaded'):
                self.excel_model.data_loaded.connect(self.main_window.on_excel_loaded)
                print(" Modelo Excel -> Vista conectado")
            
            # Climate Model
            if hasattr(self.climate_model, 'climate_data_loaded'):
                self.climate_model.climate_data_loaded.connect(self.controller.on_climate_data_loaded)
                print(" Modelo Clima -> Controlador conectado (data_loaded)")
            
            if hasattr(self.climate_model, 'all_climate_loaded'):
                self.climate_model.all_climate_loaded.connect(self.controller.on_all_climate_loaded)
                print(" Modelo Clima -> Controlador conectado (all_loaded)")
            
            print("\n Todas las conexiones MVC establecidas correctamente\n")
        
        except Exception as e:
            print(f"\n ERROR conectando señales: {e}\n")
            import traceback
            traceback.print_exc()
    
    def run(self):
        """Mostrar la ventana principal y ejecutar la app"""
        self.main_window.show()
        return self.app.exec()


if __name__ == '__main__':
    print("=" * 60)
    print("SAIDI Analysis Tool - Iniciando aplicación")
    print("=" * 60)
    
    # Limpiar carpetas __pycache__ antes de iniciar
    clean_pycache()
    
    print("\nConfigurando arquitectura MVC...")
    app = SAIDIApplication()
    
    print("\nAplicación lista para usar")
    print("=" * 60)
    print()
    
    sys.exit(app.run())