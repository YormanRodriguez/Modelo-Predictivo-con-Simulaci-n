#!/usr/bin/env python3
# main.py - Aplicación Principal MVC
#for /d /r . %d in (__pycache__) do @if exist "%d" rd /s /q "%d"

import shutil
import sys
import traceback
from pathlib import Path

from PyQt6.QtWidgets import QApplication

from controller.app_controller import AppController
from model.climate_model import ClimateModel
from model.excel_model import ExcelModel
from view.main_window import MainWindow

current_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(current_dir))

def clean_pycache(root_dir=None):
    """Elimina todas las carpetas __pycache__ del proyecto."""
    if root_dir is None:
        root_dir = Path(__file__).resolve().parent

    deleted_count = 0
    failed_count = 0

    for cache_dir in root_dir.rglob("__pycache__"):
        try:
            shutil.rmtree(cache_dir)
            deleted_count += 1
        except PermissionError:
            print(f"No se pudo eliminar {cache_dir}: sinpermisos suficientes.")
            failed_count += 1
        except OSError as exc:
            print(f"No se pudo eliminar {cache_dir}: {exc}.")
            failed_count += 1

    if deleted_count > 0:
        print(f"\nTotal de carpetas __pycache__ eliminadas: {deleted_count}")
    else:
        print("No se encontraron carpetas __pycache__")

    return deleted_count

class SAIDIApplication:
    """Aplicación principal SAIDI con arquitectura MVC."""

    def __init__(self):
        self.app = QApplication(sys.argv)
        self.setup_mvc()

    def setup_mvc(self):
        """Configurar arquitectura MVC."""
        # Modelos
        self.excel_model = ExcelModel()
        self.climate_model = ClimateModel()

        # Vista
        self.main_window = MainWindow()

        # Controlador
        self.controller = AppController(
            self.main_window, self.excel_model, self.climate_model,
        )

        # Hacer referencia cruzada para que la vista pueda intentar conectar botones durante construcción
        self.main_window.controller = self.controller

        # Conectar señales básicas
        self.connect_signals()

    def connect_signals(self):
        """Conectar señales entre vista y controlador."""
        try:
            self._connect_main_buttons()
            self._connect_regional_selector()
            self._connect_climate_loader()
            self._connect_export_signal()
            self._connect_model_signals()
            print("\nTodas las conexiones MVC establecidas correctamente\n")
        except (AttributeError, TypeError, RuntimeError) as e:
            print(f"\nERROR conectando señales: {e}\n")
            traceback.print_exc()

    def _connect_main_buttons(self):
        """Conectar botones principales de la interfaz."""
        self._connect_load_excel_button()
        self._connect_predict_button()
        self._connect_optimize_button()
        self._connect_validate_button()
        self._connect_report_button()

    def _connect_load_excel_button(self):
        """Conectar botón de carga de Excel."""
        if hasattr(self.main_window, "load_excel_button"):
            self.main_window.load_excel_button.clicked.connect(
                self.controller.load_excel_file,
            )
            print("Botón 'Cargar Excel' conectado")

    def _connect_predict_button(self):
        """Conectar botón de predicción."""
        if hasattr(self.main_window, "predict_button"):
            self.main_window.predict_button.clicked.connect(
                self.controller.run_prediction,
            )
            print("Botón 'Predicción' conectado")

    def _connect_optimize_button(self):
        """Conectar botón de optimización."""
        if hasattr(self.main_window, "optimize_button") and hasattr(
            self.controller, "run_optimization",
        ):
            self.main_window.optimize_button.clicked.connect(
                self.controller.run_optimization,
            )
            print("Botón 'Optimización' conectado")

    def _connect_validate_button(self):
        """Conectar botón de validación."""
        if not hasattr(self.main_window, "validate_button"):
            print("Vista no tiene 'validate_button'")
            return

        if hasattr(self.controller, "run_validation"):
            self.main_window.validate_button.clicked.connect(
                self.controller.run_validation,
            )
            print("Botón 'Validación' conectado")
        else:
            print("Controlador no tiene método 'run_validation'")

    def _connect_report_button(self):
        """Conectar botón de generación de informe."""
        if not hasattr(self.main_window, "report_button"):
            print("Vista no tiene 'report_button'")
            return

        if hasattr(self.controller, "generate_validation_report"):
            self.main_window.report_button.clicked.connect(
                self.controller.generate_validation_report,
            )
            print("Botón 'Generar Informe' conectado")
        else:
            print("Controlador no tiene método 'generate_validation_report'")

    def _connect_regional_selector(self):
        """Conectar selector de regional."""
        if hasattr(self.main_window, "regional_selected"):
            self.main_window.regional_selected.connect(
                self.controller.on_regional_selected,
            )
            print("Selector de regional conectado")

    def _connect_climate_loader(self):
        """Conectar carga de datos climáticos."""
        if hasattr(self.main_window, "climate_load_requested"):
            self.main_window.climate_load_requested.connect(
                self.controller.load_climate_file,
            )
            print("Carga de datos climáticos conectada")

    def _connect_export_signal(self):
        """Conectar señal de exportación a Excel."""
        if not hasattr(self.main_window, "export_requested"):
            return

        if hasattr(self.controller, "export_predictions_to_excel"):
            self.main_window.export_requested.connect(
                self.controller.export_predictions_to_excel,
            )
            print("Señal de exportación conectada")
        else:
            print("Controlador no tiene método 'export_predictions_to_excel'")

    def _connect_model_signals(self):
        """Conectar señales del modelo a la vista."""
        self._connect_excel_model_signals()
        self._connect_climate_model_signals()

    def _connect_excel_model_signals(self):
        """Conectar señales del modelo Excel."""
        if hasattr(self.excel_model, "data_loaded"):
            self.excel_model.data_loaded.connect(self.main_window.on_excel_loaded)
            print("Modelo Excel -> Vista conectado")

    def _connect_climate_model_signals(self):
        """Conectar señales del modelo de clima."""
        if hasattr(self.climate_model, "climate_data_loaded"):
            self.climate_model.climate_data_loaded.connect(
                self.controller.on_climate_data_loaded,
            )
            print("Modelo Clima -> Controlador conectado (data_loaded)")

        if hasattr(self.climate_model, "all_climate_loaded"):
            self.climate_model.all_climate_loaded.connect(
                self.controller.on_all_climate_loaded,
            )
            print("Modelo Clima -> Controlador conectado (all_loaded)")

    def run(self):
        """Mostrar la ventana principal y ejecutar la app."""
        self.main_window.show()
        return self.app.exec()

if __name__ == "__main__":
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
