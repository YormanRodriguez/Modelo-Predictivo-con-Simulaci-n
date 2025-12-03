# controller/app_controller.py - Controlador principal MVC
import gc
import os
import traceback
from datetime import datetime

import pandas as pd
from PyQt6.QtCore import QObject, QThread, QTimer, pyqtSignal
from PyQt6.QtWidgets import QFileDialog, QMessageBox

from services.optimization_service import OptimizationService

# Importar servicios de backend
from services.prediction_service import PredictionConfig, PredictionService
from services.report_generation_service import ValidationReportService
from services.rolling_validation_service import RollingValidationService
from services.validation_service import ValidationService
from view.climate_simulation_dialog import ClimateSimulationDialog

# Importar el visor de gráficas
from view.main_window import PlotViewerDialog


class AppController(QObject):
    """Controlador principal de la aplicación."""

    def __init__(self, view, model, climate_model):
        super().__init__()
        self.view = view
        self.model = model
        self.climate_model = climate_model

        # Servicios de backend
        self.prediction_service = PredictionService()
        self.optimization_service = OptimizationService()
        self.validation_service = ValidationService()
        self.rolling_validation_service = RollingValidationService()
        self.report_service = ValidationReportService()

        # Referencias a diálogos de gráficas para limpieza
        self.plot_dialogs = []

        # Timer para limpieza automática de archivos temporales
        self.cleanup_timer = QTimer()
        self.cleanup_timer.timeout.connect(self.cleanup_temp_files)
        self.cleanup_timer.start(300000)  # Limpiar cada 5 minutos

        # Configurar conexiones de modelo
        self.setup_model_connections()
        self.setup_climate_connections()


    def setup_climate_connections(self):
        """Configurar conexiones del modelo climático."""
        self.climate_model.climate_data_loaded.connect(self.on_climate_data_loaded)
        self.climate_model.all_climate_loaded.connect(self.on_all_climate_loaded)
        self.climate_model.status_changed.connect(self.view.update_status)
        self.climate_model.error_occurred.connect(self.show_error)

    def load_climate_file(self, regional_code: str):
        """Cargar archivo climático para una regional específica."""
        try:
            regional_nombre = self.climate_model.REGIONALES.get(regional_code, regional_code)
            file_path, _ = QFileDialog.getOpenFileName(
                self.view,
                f"Seleccionar datos climáticos - {regional_nombre}",
                "",
                "Archivos Excel (*.xlsx *.xls);;Archivos CSV (*.csv);;Todos los archivos (*.*)",
            )
            if not file_path:
                self.view.log_message(f"Carga cancelada para {regional_nombre}")
                return

            self.view.update_climate_status(regional_code, "loading")
            self.view.log_message(f"Cargando datos climáticos de {regional_nombre}...")

            success = self.climate_model.load_climate_file(regional_code, file_path)

            if success:
                self.view.update_climate_status(regional_code, "success")
                self.view.log_success(f"Datos climáticos de {regional_nombre} cargados correctamente")
                loaded = len(self.climate_model.get_loaded_regionales())
                total = len(self.climate_model.REGIONALES)
                self.view.update_climate_progress_summary(loaded, total)
            else:
                self.view.update_climate_status(regional_code, "error")
                self.view.log_error(f"Error cargando datos climáticos de {regional_nombre}")

        except Exception as e:
            self.view.log_error(f"Error inesperado cargando clima: {e!s}")
            self.view.update_climate_status(regional_code, "error")
            self.show_error(f"Error al cargar datos climáticos: {e!s}")

    def on_climate_data_loaded(self, climate_info: dict):
        """Callback cuando se cargan datos climáticos de una regional."""
        regional_name = climate_info.get("regional_name", "Desconocida")
        total_records = climate_info.get("total_records", 0)
        completeness = climate_info.get("avg_completeness", 0)

        self.view.log_message(f" {regional_name}:")
        self.view.log_message(f"  • Registros: {total_records:,}")
        self.view.log_message(f"  • Completitud: {completeness:.1f}%")
        self.view.log_message(
            f"  • Período: {climate_info['date_range']['start'].strftime('%Y-%m-%d')} "
            f"a {climate_info['date_range']['end'].strftime('%Y-%m-%d')}",
        )

        self.update_climate_details_panel()

    def on_all_climate_loaded(self, summary: dict):
        """Callback cuando todas las regionales tienen datos climáticos."""
        self.view.log_success("=" * 60)
        self.view.log_success("TODOS LOS DATOS CLIMÁTICOS CARGADOS")
        self.view.log_success("=" * 60)

        total = summary.get("total_regionales", 0)
        self.view.log_message(f"Total de regionales con datos: {total}")

        climate_info = summary.get("climate_info", {})
        for regional_code, info in climate_info.items():
            regional_name = info.get("regional_name", regional_code)
            records = info.get("total_records", 0)
            self.view.log_message(f"  • {regional_name}: {records:,} registros")

        self.view.log_message("\nLos datos climáticos están listos para análisis")
        # Actualizar panel de detalles en la vista
        self.update_climate_details_panel()

    def update_climate_details_panel(self):
        """Actualizar panel de detalles de datos climáticos en la vista."""
        all_info = {}

        for regional_code in self.climate_model.REGIONALES.keys():
            info = self.climate_model.get_climate_info(regional_code)
            if info:
                all_info[regional_code] = info

        if all_info:
            # La vista espera un resumen para mostrar
            self.view.update_climate_details(all_info)

    def get_climate_data_for_regional(self, regional_code: str):
        """Obtener datos climáticos para una regional específica."""
        return self.climate_model.get_climate_data(regional_code)

    def are_climate_data_available(self, regional_code: str) -> bool:
        """Verificar si hay datos climáticos disponibles para una regional."""
        return self.climate_model.is_regional_loaded(regional_code)

    def cleanup_temp_files(self):
        """Limpiar archivos temporales de gráficas."""
        try:
            # Servicios que generan plots implementan cleanup_plot_file
            try:
                self.prediction_service.cleanup_plot_file()
            except Exception:
                pass
            try:
                self.validation_service.cleanup_plot_file()
            except Exception:
                pass
            try:
                self.rolling_validation_service.cleanup_plot_file()
            except Exception:
                pass
            gc.collect()
        except Exception as e:
            print(f"Error durante limpieza automática: {e}")

    def show_plot(self, plot_file_path, title="Gráfica SAIDI"):
        """Mostrar gráfica en un diálogo separado."""
        if not plot_file_path or not os.path.exists(plot_file_path):
            self.view.log_error("No se encontró el archivo de gráfica")
            return

        try:
            dialog = PlotViewerDialog(plot_file_path, title, self.view)
            self.plot_dialogs.append(dialog)
            dialog.show()
            self.view.log_success(f"Gráfica mostrada: {title}")
        except Exception as e:
            self.view.log_error(f"Error mostrando gráfica: {e!s}")

    def setup_model_connections(self):
        """Configurar conexiones del modelo."""
        self.model.error_occurred.connect(self.show_error)

    def on_regional_selected(self, regional_codigo: str):
        """Callback cuando el usuario selecciona una regional."""
        try:
            if self.model.set_selected_regional(regional_codigo):
                nombre = self.model.REGIONAL_MAPPING.get(regional_codigo, regional_codigo)
                self.view.log_success(f"Regional '{nombre}' seleccionada correctamente")
                self.view.update_status(f"Trabajando con: {nombre}")

                if self.are_climate_data_available(regional_codigo):
                    self.view.log_message(f"✓ Datos climáticos disponibles para {nombre}")
                else:
                    self.view.log_message(f"Cargue datos climáticos para {nombre} (opcional)")
            else:
                self.view.log_error(f"Error al seleccionar regional: {regional_codigo}")
        except Exception as e:
            self.view.log_error(f"Error al seleccionar regional: {e!s}")

    def load_excel_file(self):
        """Cargar archivo Excel mediante diálogo."""
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self.view,
                "Seleccionar archivo Excel - SAIDI Analysis",
                "",
                "Archivos Excel (*.xlsx *.xls);;Todos los archivos (*.*)",
            )

            if not file_path:
                return

            success = self.model.load_excel_file(file_path)

            if success:
                self.view.log_success(f"Archivo cargado exitosamente: {os.path.basename(file_path)}")
            else:
                self.view.log_error("Error al cargar el archivo Excel")

        except Exception as e:
            self.view.log_error(f"Error inesperado: {e!s}")
            self.show_error(f"Error al cargar archivo: {e!s}")

    def run_prediction(self):
        """Ejecutar predicción SAIDI CON VARIABLES EXÓGENAS Y SIMULACIÓN."""
        if not self.model.is_excel_loaded():
            self.show_warning("Debe cargar un archivo Excel primero")
            return

        regional_code = None
        climate_data = None

        if self.model.is_regional_format():
            if not self.model.get_selected_regional():
                self.show_warning("Debe seleccionar una regional primero")
                return
            regional_code = self.model.get_selected_regional()
            regional_nombre = self.model.REGIONAL_MAPPING.get(regional_code, regional_code)

            # Obtener datos climáticos si existen
            if self.are_climate_data_available(regional_code):
                climate_data = self.get_climate_data_for_regional(regional_code)
                self.view.log_message("Datos climáticos disponibles - Se incluirán en la predicción")
            else:
                self.view.log_message("Sin datos climáticos - Predicción sin variables exógenas")

            # Si la simulación está habilitada en la UI, abrir diálogo
            if getattr(self.view, "enable_simulation_checkbox", None) and self.view.enable_simulation_checkbox.isChecked():
                # Crear configuración para el diálogo
                from dataclasses import dataclass

                @dataclass
                class SimulationConfig:
                    climate_data: any
                    mes_prediccion: int
                    regional_code: str
                    regional_nombre: str
                    mode: str = "prediction"

                # Obtener mes de predicción
                mes_prediccion = 1
                if self.model.get_excel_data_for_analysis() is not None:
                    mes_prediccion = pd.to_datetime(
                        self.model.get_excel_data_for_analysis()["Fecha"].iloc[-1],
                    ).month

                # Crear configuración
                config = SimulationConfig(
                    climate_data=climate_data,
                    mes_prediccion=mes_prediccion,
                    regional_code=regional_code,
                    regional_nombre=regional_nombre,
                    mode="prediction",
                )

                # Abrir diálogo con configuración
                dialog = ClimateSimulationDialog(config, parent=self.view)
                dialog.simulation_accepted.connect(
                    lambda cfg: self._on_simulation_configured(cfg, regional_code, climate_data),
                )
                dialog.simulation_cancelled.connect(
                    lambda: self._on_simulation_configured({"enabled": False}, regional_code, climate_data),
                )
                dialog.exec()
                return

        # Si llegamos aquí, ejecutar predicción normal (sin simulación)
        self._execute_prediction(regional_code, climate_data, None)
    def export_predictions_to_excel(self):
        """Exportar predicciones a Excel con validación completa."""
        try:
            if not hasattr(self, "last_prediction_result") or not self.last_prediction_result:
                self.view.log_error("No hay predicciones disponibles para exportar")
                self.show_warning("Debe ejecutar una predicción primero antes de exportar")
                return

            result = self.last_prediction_result
            predictions = result.get("predictions")

            if not predictions:
                self.view.log_error("Las predicciones están vacías")
                self.show_warning("No hay predicciones para exportar")
                return

            model_params = result.get("model_params", {})
            regional_code = model_params.get("regional_code")

            if not regional_code:
                self.view.log_error("No se pudo determinar la regional de las predicciones")
                self.show_warning("No se pudo determinar la regional de las predicciones")
                return

            regional_nombre = self.model.REGIONAL_MAPPING.get(regional_code, regional_code)

            self.view.log_message("=" * 60)
            self.view.log_message("INICIANDO EXPORTACIÓN A EXCEL")
            self.view.log_message("=" * 60)
            self.view.log_message(f"Regional: {regional_nombre} ({regional_code})")
            self.view.log_message(f"Número de predicciones: {len(predictions)}")

            default_name = f"Predicciones_SAIDI_{regional_nombre}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            default_path = os.path.join(os.path.expanduser("~/Desktop"), default_name)

            filepath, _ = QFileDialog.getSaveFileName(
                self.view,
                "Guardar Predicciones SAIDI",
                default_path,
                "Archivos Excel (*.xlsx);;Todos los archivos (*.*)",
            )

            if not filepath:
                self.view.log_message("Exportación cancelada por el usuario")
                return

            if not filepath.endswith(".xlsx"):
                filepath += ".xlsx"

            self.view.log_message(f"Ubicación seleccionada: {filepath}")

            export_service = result.get("export_service")

            if not export_service:
                self.view.log_message("Creando servicio de exportación...")
                from services.export_service import ExportService
                export_service = ExportService()

            model_info = {
                "order": model_params.get("order"),
                "seasonal_order": model_params.get("seasonal_order"),
                "transformation": model_params.get("transformation"),
                "with_exogenous": model_params.get("with_exogenous", False),
                "with_simulation": model_params.get("with_simulation", False),
                "confidence_level": model_params.get("confidence_level", 0.95),
                "uncertainty_method": model_params.get("uncertainty_method", "parametric"),
                "regional_code": regional_code,
                "metrics": result.get("metrics"),
            }

            self.view.show_progress(visible=True)
            self.view.update_progress(30, "Preparando datos para exportación...")
            self.view.log_message("Generando archivo Excel con formato profesional...")

            self.view.update_progress(60, "Escribiendo archivo Excel...")

            # IMPORTAR RegionalInfo y crear objeto
            from services.export_service import RegionalInfo
            regional_info = RegionalInfo(code=regional_code, nombre=regional_nombre)

            saved_path = export_service.export_to_custom_location(
                predictions_dict=predictions,
                regional_info=regional_info,  # Usar objeto RegionalInfo
                custom_path=filepath,
                include_confidence_intervals=True,
                model_info=model_info,
            )

            self.view.update_progress(100, "Exportación completada")
            self.view.show_progress(visible=False)

            if saved_path and os.path.exists(saved_path):
                file_size = os.path.getsize(saved_path) / 1024  # KB

                self.view.log_success("=" * 60)
                self.view.log_success("EXPORTACIÓN COMPLETADA EXITOSAMENTE")
                self.view.log_success("=" * 60)
                self.view.log_message(f"Archivo: {os.path.basename(saved_path)}")
                self.view.log_message(f"Ubicación: {os.path.dirname(saved_path)}")
                self.view.log_message(f"Tamaño: {file_size:.2f} KB")
                self.view.log_message(f"Predicciones exportadas: {len(predictions)}")

                # Mostrar características exportadas
                first_pred = next(iter(predictions.values()))
                has_intervals = isinstance(first_pred, dict) and "limite_inferior" in first_pred

                self.view.log_message("\nContenido del archivo:")
                self.view.log_message("  • Hoja 1: Predicciones SAIDI")
                self.view.log_message("    - Fecha y valores predichos")
                if has_intervals:
                    self.view.log_message("    - Intervalos de confianza (95%)")
                    self.view.log_message("    - Márgenes de error")
                self.view.log_message("  • Hoja 2: Información del Modelo")
                self.view.log_message("    - Parámetros SARIMAX")
                self.view.log_message("    - Transformación aplicada")
                self.view.log_message("    - Métricas de precisión")

                if model_params.get("with_exogenous"):
                    self.view.log_message("    - Variables exógenas utilizadas")
                if model_params.get("with_simulation"):
                    self.view.log_message("    - Simulación climática aplicada")
                self.view.log_message("=" * 60)

                # Mensaje de éxito con opción de abrir
                from PyQt6.QtWidgets import QMessageBox
                msg = QMessageBox(self.view)
                msg.setIcon(QMessageBox.Icon.Information)
                msg.setWindowTitle("Exportación Exitosa")
                msg.setText("Las predicciones se han exportado correctamente")
                msg.setInformativeText(f"Archivo guardado en:\n{saved_path}")
                msg.setDetailedText(
                    f"Regional: {regional_nombre}\n"
                    f"Predicciones: {len(predictions)}\n"
                    f"Intervalos de confianza: {'Sí' if has_intervals else 'No'}\n"
                    f"Tamaño: {file_size:.2f} KB",
                )

                # Botones
                open_btn = msg.addButton("Abrir Archivo", QMessageBox.ButtonRole.AcceptRole)
                open_folder_btn = msg.addButton("Abrir Carpeta", QMessageBox.ButtonRole.ActionRole)
                msg.addButton(QMessageBox.StandardButton.Close)

                msg.exec()

                # Abrir archivo o carpeta según la elección
                clicked_button = msg.clickedButton()
                if clicked_button == open_btn:
                    self._open_file(saved_path)
                elif clicked_button == open_folder_btn:
                    self._open_folder(os.path.dirname(saved_path))

                self.view.update_status(f"Predicciones exportadas: {os.path.basename(saved_path)}")

            else:
                self.view.log_error("Error: No se pudo exportar el archivo")
                self.show_error("No se pudo exportar el archivo Excel")

        except Exception as e:
            self.view.show_progress(visible=False)
            error_msg = str(e)
            self.view.log_error("=" * 60)
            self.view.log_error("ERROR EN EXPORTACIÓN")
            self.view.log_error("=" * 60)
            self.view.log_error(f"Error: {error_msg}")

            # Mostrar traceback completo en log para debugging
            self.view.log_error("\nTraceback completo:")
            for line in traceback.format_exc().split("\n"):
                if line.strip():
                    self.view.log_error(f"  {line}")

            self.show_error(f"Error al exportar predicciones:\n\n{error_msg}")

    def _open_file(self, filepath):
        """Abrir archivo con la aplicación predeterminada del sistema."""
        try:
            import platform
            import subprocess

            system = platform.system()

            if system == "Windows":
                os.startfile(filepath)
            elif system == "Darwin":  # macOS
                subprocess.call(["open", filepath])
            else:  # Linux
                subprocess.call(["xdg-open", filepath])

            self.view.log_message(f"Abriendo archivo: {os.path.basename(filepath)}")
        except Exception as e:
            self.view.log_error(f"No se pudo abrir el archivo: {e!s}")

    def _open_folder(self, folder_path):
        """Abrir carpeta con el explorador de archivos del sistema."""
        try:
            import platform
            import subprocess

            system = platform.system()

            if system == "Windows":
                os.startfile(folder_path)
            elif system == "Darwin":  # macOS
                subprocess.call(["open", folder_path])
            else:  # Linux
                subprocess.call(["xdg-open", folder_path])

            self.view.log_message(f"Abriendo carpeta: {folder_path}")
        except Exception as e:
            self.view.log_error(f"No se pudo abrir la carpeta: {e!s}")

    def _on_simulation_configured(self, simulation_config, regional_code, climate_data):
        """Callback cuando el usuario configura la simulación."""
        # VERIFICAR que simulation_config tenga la estructura correcta:
        # {
        #     'enabled': True,
        #     'scenario_name': 'calor_extremo',
        #     'intensity_adjustment': 1.2,
        #     'alcance_meses': 3,
        #     'percentiles': {...},
        #     'regional_code': 'SAIDI_O',
        #     'summary': {...}
        # }

        # Si el diálogo aún envía la estructura antigua, convertirla:
        if "escenario" in simulation_config and "scenario_name" not in simulation_config:
            # CONVERSIÓN DE FORMATO ANTIGUO A NUEVO
            simulation_config["scenario_name"] = simulation_config.pop("escenario")

        if "slider_adjustment" in simulation_config and "dias_base" in simulation_config:
            # Calcular intensity_adjustment desde parámetros antiguos
            slider_adj = simulation_config.pop("slider_adjustment")
            dias_base = simulation_config.pop("dias_base")

            dias_simulados = dias_base + slider_adj
            simulation_config["intensity_adjustment"] = dias_simulados / dias_base if dias_base > 0 else 1.0

        # Ejecutar predicción con configuración actualizada
        self._execute_prediction(regional_code, climate_data, simulation_config)

    def _execute_prediction(self, regional_code, climate_data, simulation_config):
        """Ejecutar la predicción con o sin simulación."""
        try:
            self.view.set_buttons_enabled(False)
            self.view.show_progress(visible=True)
            self.view.update_status("Generando predicción...")

            df_prepared = self.model.get_excel_data_for_analysis()
            if df_prepared is None:
                raise Exception("No se pudieron preparar los datos SAIDI")

            # Crear config
            config = PredictionConfig(
                file_path=self.model.get_file_path(),
                df_prepared=df_prepared,
                order=None,
                seasonal_order=None,
                regional_code=regional_code,
                climate_data=climate_data,
                simulation_config=simulation_config,
                progress_callback=self.view.update_progress,
                log_callback=self.view.log_message,
            )

            # Llamar al método
            result = self.prediction_service.run_prediction(config)

            # Llamar al manejador de finalización
            self.on_prediction_finished(result)

        except Exception as e:
            self.on_prediction_error(str(e))

    def run_optimization(self):
        """Ejecutar optimización de parámetros CON VARIABLES EXÓGENAS."""
        if not self.model.is_excel_loaded():
            self.show_warning("Debe cargar un archivo Excel primero")
            return

        regional_code = None
        climate_data = None

        if self.model.is_regional_format():
            if not self.model.get_selected_regional():
                self.show_warning("Debe seleccionar una regional primero")
                return

            regional_code = self.model.get_selected_regional()
            regional_nombre = self.model.REGIONAL_MAPPING.get(regional_code, regional_code)

            self.view.log_message(f"Ejecutando optimización para: {regional_nombre}")
            self.view.log_message("Se evaluarán TODAS las transformaciones disponibles")
            self.view.log_message(f"Transformaciones: {', '.join(self.optimization_service.AVAILABLE_TRANSFORMATIONS)}")

            # CRÍTICO: Obtener datos climáticos
            if self.are_climate_data_available(regional_code):
                climate_data = self.get_climate_data_for_regional(regional_code)

                # VERIFICAR QUE SE OBTUVIERON CORRECTAMENTE
                if climate_data is not None and not climate_data.empty:
                    self.view.log_message(f"✓ Datos climáticos disponibles: {len(climate_data)} registros")
                    self.view.log_message(f"   Periodo: {climate_data.index[0]} a {climate_data.index[-1]}")
                    self.view.log_message(f"   Columnas: {len(climate_data.columns)}")

                    # DEBUG: Mostrar primeras columnas
                    self.view.log_message(f"   Primeras 5 columnas: {list(climate_data.columns[:5])}")
                else:
                    self.view.log_message("⚠ Datos climáticos vacíos o inválidos")
                    climate_data = None
            else:
                self.view.log_message(" Sin datos climáticos - Optimización sin variables exógenas")

        try:
            self.view.log_message("Iniciando optimización de parámetros...")
            self.view.log_message("NOTA: Este proceso puede tardar varios minutos")
            self.view.set_buttons_enabled(False)
            self.view.update_status("Optimizando parámetros SARIMAX...")

            df_prepared = self.model.get_excel_data_for_analysis()

            if df_prepared is None:
                self.show_error("No se pudieron preparar los datos. Verifique la regional seleccionada.")
                self.view.set_buttons_enabled(True)
                return

            #  CREAR THREAD CON climate_data
            self.optimization_thread = OptimizationThread(
                optimization_service=self.optimization_service,
                df_prepared=df_prepared,
                regional_code=regional_code,
                climate_data=climate_data,  #  PASAR climate_data AL THREAD
            )

            # Conectar señales
            self.optimization_thread.progress_updated.connect(self.view.update_progress)
            self.optimization_thread.message_logged.connect(self.view.log_message)
            self.optimization_thread.iteration_logged.connect(self.view.log_message)
            self.optimization_thread.finished.connect(self.on_optimization_finished)
            self.optimization_thread.error_occurred.connect(self.on_optimization_error)

            self.view.show_progress(visible=True)
            self.optimization_thread.start()

        except Exception as e:
            self.view.log_error(f"Error iniciando optimización: {e!s}")
            self.view.set_buttons_enabled(True)
            self.view.show_progress(visible=False)

    def run_validation(self):
        """Ejecutar validación del modelo CON VARIABLES EXÓGENAS, SIMULACIÓN E INTERVALOS."""
        if not self.model.is_excel_loaded():
            self.show_warning("Debe cargar un archivo Excel primero")
            return

        regional_code = None
        climate_data = None

        if self.model.is_regional_format():
            if not self.model.get_selected_regional():
                self.show_warning("Debe seleccionar una regional primero")
                return

            regional_code = self.model.get_selected_regional()
            regional_nombre = self.model.REGIONAL_MAPPING.get(regional_code, regional_code)

            # ========== NUEVO: Intentar cargar configuración optimizada ==========
            optimized_config = self.validation_service.load_optimized_config(regional_code)

            if optimized_config:
                self.view.log_message("=" * 60)
                self.view.log_message("DETECTADA CONFIGURACIÓN OPTIMIZADA")
                self.view.log_message("=" * 60)
                self.view.log_message(f"Regional: {regional_nombre}")
                self.view.log_message(f"Transformación optimizada: {optimized_config['transformation'].upper()}")
                self.view.log_message(f"Precisión documentada: {optimized_config['precision_final']:.1f}%")
                self.view.log_message(f"Optimizado en: {optimized_config['optimization_date']}")
                self.view.log_message("Estos parámetros se usarán automáticamente en la validación")
                self.view.log_message("=" * 60)
            else:
                # Si no hay optimización, mostrar transformación default
                transformation = self.validation_service.REGIONAL_TRANSFORMATIONS.get(
                    regional_code, "original",
                )
                self.view.log_message(f"Ejecutando validación para: {regional_nombre}")
                self.view.log_message(f"Transformación default: {transformation.upper()}")

            # Obtener datos climáticos si están disponibles
            if self.are_climate_data_available(regional_code):
                climate_data = self.get_climate_data_for_regional(regional_code)
                self.view.log_message(f"✓ Datos climáticos disponibles para {regional_nombre}")

                # Si la simulación está habilitada, abrir diálogo
                if getattr(self.view, "enable_simulation_checkbox", None) and \
                self.view.enable_simulation_checkbox.isChecked():

                    # Obtener mes de validación (último mes histórico)
                    df_prepared = self.model.get_excel_data_for_analysis()
                    if df_prepared is not None and "Fecha" in df_prepared.columns:
                        try:
                            ultimo_mes = pd.to_datetime(df_prepared["Fecha"].iloc[-1]).month
                        except Exception:
                            ultimo_mes = datetime.now().month
                    else:
                        ultimo_mes = datetime.now().month

                    self.view.log_message("Simulación climática habilitada para validación")
                    self.view.log_message("Abriendo configurador de escenarios...")

                    # Usar el mismo formato que run_prediction
                    from dataclasses import dataclass

                    @dataclass
                    class SimulationConfig:
                        climate_data: any
                        mes_prediccion: int
                        regional_code: str
                        regional_nombre: str
                        mode: str = "validation"

                    # Crear configuración
                    config = SimulationConfig(
                        climate_data=climate_data,
                        mes_prediccion=ultimo_mes,
                        regional_code=regional_code,
                        regional_nombre=regional_nombre,
                        mode="validation",
                    )

                    # Abrir diálogo
                    dialog = ClimateSimulationDialog(config, parent=self.view)

                    # Conectar señales
                    dialog.simulation_accepted.connect(
                        lambda cfg: self._execute_validation(regional_code, climate_data, cfg),
                    )
                    dialog.simulation_cancelled.connect(
                        lambda: self._execute_validation(regional_code, climate_data, {"enabled": False}),
                    )

                    dialog.exec()
                    return
            else:
                self.view.log_message(f"Sin datos climáticos para {regional_nombre}")

        # Si llegamos aquí, ejecutar validación normal (sin simulación)
        self._execute_validation(regional_code, climate_data, None)

    def _execute_validation(self, regional_code, climate_data, simulation_config):
        """Ejecutar validación con o sin simulación."""
        try:
            self.view.set_buttons_enabled(False)
            self.view.update_status("Validando modelo SARIMAX...")

            df_prepared = self.model.get_excel_data_for_analysis()

            if df_prepared is None:
                self.show_error("No se pudieron preparar los datos. Verifique la regional seleccionada.")
                self.view.set_buttons_enabled(True)
                return

            # Log según tipo de validación
            if simulation_config and simulation_config.get("enabled", False):
                summary = simulation_config.get("summary", {})
                self.view.log_message("=" * 60)
                self.view.log_message("VALIDACIÓN CON SIMULACIÓN CLIMÁTICA")
                self.view.log_message("=" * 60)
                self.view.log_message(f"Escenario: {summary.get('escenario', 'N/A')}")
                self.view.log_message(f"Días simulados: {summary.get('dias_simulados', 'N/A')}")
                self.view.log_message(f"Alcance: {summary.get('alcance_meses', 'N/A')} meses")
            else:
                self.view.log_message("Iniciando validación estándar...")

            # Crear y ejecutar thread
            self.validation_thread = ValidationThread(
                df_prepared=df_prepared,
                validation_service=self.validation_service,
                regional_code=regional_code,
                climate_data=climate_data,
                simulation_config=simulation_config,  # NUEVO PARÁMETRO
            )

            self.validation_thread.progress_updated.connect(self.view.update_progress)
            self.validation_thread.message_logged.connect(self.view.log_message)
            self.validation_thread.finished.connect(self.on_validation_finished)
            self.validation_thread.error_occurred.connect(self.on_validation_error)

            self.view.show_progress(visible=True)
            self.validation_thread.start()

        except Exception as e:
            self.view.log_error(f"Error ejecutando validación: {e!s}")
            self.view.set_buttons_enabled(True)
            self.view.show_progress(visible=False)

    def generate_validation_report(self):
        """Generar informe PDF completo de validación temporal."""
        if not self.model.is_excel_loaded():
            self.show_warning("Debe cargar un archivo Excel primero")
            return

        regional_code = None
        climate_data = None

        if self.model.is_regional_format():
            if not self.model.get_selected_regional():
                self.show_warning("Debe seleccionar una regional primero")
                return

            regional_code = self.model.get_selected_regional()
            regional_nombre = self.model.REGIONAL_MAPPING.get(regional_code, regional_code)

            transformation = self.rolling_validation_service.REGIONAL_TRANSFORMATIONS.get(
                regional_code, "original",
            )

            self.view.log_message("=" * 60)
            self.view.log_message("GENERANDO INFORME DE VALIDACIÓN DEL MODELO")
            self.view.log_message("=" * 60)
            self.view.log_message(f"Regional: {regional_nombre}")
            self.view.log_message(f"Transformación: {transformation.upper()}")

            # Obtener datos climáticos si están disponibles
            if self.are_climate_data_available(regional_code):
                climate_data = self.get_climate_data_for_regional(regional_code)
                self.view.log_message("Datos climáticos incluidos en el análisis")
            else:
                self.view.log_message(" Sin datos climáticos - Análisis básico")

        try:
            self.view.set_buttons_enabled(False)
            self.view.update_status("Ejecutando validación temporal completa...")

            df_prepared = self.model.get_excel_data_for_analysis()

            if df_prepared is None:
                self.show_error("No se pudieron preparar los datos. Verifique la regional seleccionada.")
                self.view.set_buttons_enabled(True)
                return

            # Crear thread de validación (mismo que antes)
            self.rolling_validation_thread = RollingValidationThread(
                rolling_validation_service=self.rolling_validation_service,
                df_prepared=df_prepared,
                regional_code=regional_code,
                climate_data=climate_data,
            )
            self.rolling_validation_thread.progress_updated.connect(self.view.update_progress)
            self.rolling_validation_thread.message_logged.connect(self.view.log_message)
            self.rolling_validation_thread.finished.connect(self.on_report_validation_finished)
            self.rolling_validation_thread.error_occurred.connect(self.on_report_validation_error)

            self.view.show_progress(visible=True)
            self.rolling_validation_thread.start()

        except Exception as e:
            self.view.log_error(f"Error iniciando generación de informe: {e!s}")
            self.view.set_buttons_enabled(True)
            self.view.show_progress(visible=False)

    def on_prediction_finished(self, result):
        """Callback cuando termina la predicción."""
        self.last_prediction_result = result

        try:
            # Restaurar UI
            try:
                self.view.set_buttons_enabled(True)
                self.view.show_progress(visible=False)
                self.view.update_status("Predicción completada")
            except Exception:
                pass

            self.view.log_success("Predicción SAIDI completada exitosamente")

            if not result:
                self.view.log_error("Resultado vacío de la predicción")
                return

            preds = result.get("predictions")

            if preds:
                self.view.enable_export_button(enabled=True)
                self.view.log_message(f"✓ Botón de exportación habilitado ({len(preds)} predicciones disponibles)")

                self.view.log_message(f"Se generaron {len(preds)} predicciones")

                # Logging detallado de predicciones
                for fecha, entry in preds.items():
                    if isinstance(entry, dict):
                        valor = entry.get("valor_predicho", None)
                        inferior = entry.get("limite_inferior", None)
                        superior = entry.get("limite_superior", None)
                        margen = entry.get("margen_error", None)

                        if isinstance(valor, (int, float)):
                            if inferior is not None and superior is not None:
                                try:
                                    val_f = float(valor)
                                    inf_f = float(inferior)
                                    sup_f = float(superior)
                                    margen_sup = sup_f - val_f
                                    margen_inf = val_f - inf_f
                                    margen_pct = (float(margen) / val_f * 100) if val_f > 0 else 0

                                    self.view.log_message(
                                        f"  • {fecha}: {val_f:.2f} min "
                                        f"[IC: {inf_f:.2f} - {sup_f:.2f}] "
                                        f"(+{margen_sup:.2f}/-{margen_inf:.2f} | ±{margen_pct:.0f}%)",
                                    )
                                except Exception:
                                    self.view.log_message(f"  • {fecha}: {valor:.2f} minutos")
                                    self.view.log_message(f"     Intervalo: [{inferior} - {superior}]")
                            else:
                                self.view.log_message(f"  • {fecha}: {valor:.2f} minutos")
                        else:
                            self.view.log_message(f"  • {fecha}: {valor}")
                    else:
                        try:
                            val = float(entry)
                            self.view.log_message(f"  • {fecha}: {val:.2f} minutos")
                        except Exception:
                            self.view.log_message(f"  • {fecha}: {entry}")

                # Mostrar gráfica si fue generada
                plot_path = result.get("plot_file") if isinstance(result, dict) else None
                if plot_path:
                    self.show_plot(plot_path, "Predicción SAIDI")

                self.view.log_message("")
                self.view.log_message("═" * 60)
                self.view.log_message("Para guardar estas predicciones en Excel, use el botón 'Exportar a Excel'")
                self.view.log_message("═" * 60)

        except Exception as e:
            # Manejo seguro de errores en el callback
            try:
                self.view.log_error(f"Error en on_prediction_finished: {e!s}")
            except Exception:
                pass
            print(f"[DEBUG] on_prediction_finished error: {e}")

    def on_optimization_finished(self, result):
        """Callback cuando termina la optimización."""
        self.view.set_buttons_enabled(True)
        self.view.show_progress(visible=False)
        self.view.update_status("Optimización completada")
        self.view.log_success("Optimización de parámetros completada")

        if result and "top_models" in result:
            self.view.log_message(f"Se evaluaron {result.get('total_evaluated', 0)} modelos")

            if result.get("transformation_stats"):
                self.view.log_message("\n ESTADÍSTICAS POR TRANSFORMACIÓN:")
                for transform, stats in result["transformation_stats"].items():
                    self.view.log_message(
                        f"  {transform.upper():12s} | Modelos válidos: {stats['count']:4d} | "
                        f"Mejor precisión: {stats['best_precision']:.1f}%",
                    )

            self.view.log_message("\n Top 5 mejores modelos (todas las transformaciones):")
            for i, model in enumerate(result["top_models"][:5], 1):
                precision = model.get("precision_final", 0)
                order = model.get("order", "N/A")
                seasonal_order = model.get("seasonal_order", "N/A")
                rmse = model.get("rmse", 0)
                transformation = model.get("transformation", "unknown")

                top_1 = 1
                top_2 = 2
                top_3 = 3
                medal = "1" if i == top_1 else "2" if i == top_2 else "3" if i == top_3 else f"#{i}"
                self.view.log_message(
                    f"  {medal} [{transformation.upper():8s}] Precisión: {precision:.1f}% | RMSE: {rmse:.4f} | "
                    f"order={order}, seasonal={seasonal_order}",
                )

            if result["top_models"]:
                best = result["top_models"][0]
                precision = best["precision_final"]
                best_transformation = best.get("transformation", "unknown")

                self.view.log_message("\nN1 MODELO ÓPTIMO SELECCIONADO:")
                self.view.log_message(f"  Transformación: {best_transformation.upper()}")
                self.view.log_message(f"  Parámetros: order={best['order']}, seasonal={best['seasonal_order']}")

                precision_mayor = 90
                precision_media = 80
                precision_menor = 70

                if precision >= precision_mayor:
                    interpretacion = "EXCELENTE - Predicciones muy confiables "
                elif precision >= precision_media:
                    interpretacion = "BUENO - Predicciones confiables "
                elif precision >= precision_menor:
                    interpretacion = "ACEPTABLE - Predicciones moderadamente confiables "
                else:
                    interpretacion = "REGULAR - Usar con precaución "

                self.view.log_message(f"  Interpretación: {interpretacion}")
                self.view.log_message("  Métricas calculadas en escala original")

                self.view.log_message("\n MEJOR MODELO POR TRANSFORMACIÓN:")
                transformations_shown = set()
                for model in result["top_models"][:20]:
                    trans = model.get("transformation", "unknown")
                    if trans not in transformations_shown:
                        transformations_shown.add(trans)
                        self.view.log_message(
                            f"  {trans.upper():12s} | Precisión: {model['precision_final']:.1f}% | "
                            f"RMSE: {model['rmse']:.4f}",
                        )
                    if len(transformations_shown) >= 5:
                        break

    def on_validation_finished(self, result):
        """Callback cuando termina la validacion."""
        self.view.set_buttons_enabled(True)
        self.view.show_progress(visible=False)
        self.view.update_status("Validacion completada")

        # Detectar si hubo simulación
        model_params = result.get("model_params", {}) if result else {}
        simulation_applied = model_params.get("with_simulation", False)

        regional_code = model_params.get("regional_code")
        used_optimized = False

        if regional_code:
            optimized_config = self.validation_service.load_optimized_config(regional_code)
            if optimized_config:
                used_optimized = True

                # Verificar que los parámetros coincidan
                order_match = tuple(optimized_config["order"]) == model_params.get("order")
                seasonal_match = tuple(optimized_config["seasonal_order"]) == model_params.get("seasonal_order")
                transform_match = optimized_config["transformation"] == model_params.get("transformation")

                if order_match and seasonal_match and transform_match:
                    regional_nombre = self.model.REGIONAL_MAPPING.get(regional_code, regional_code)

                    self.view.log_success("=" * 60)
                    self.view.log_success("VALIDACIÓN CON CONFIGURACIÓN OPTIMIZADA")
                    self.view.log_success("=" * 60)
                    self.view.log_message(f"Regional: {regional_nombre}")
                    self.view.log_message("Parámetros utilizados: OPTIMIZADOS")
                    self.view.log_message(f"Transformación: {optimized_config['transformation'].upper()}")
                    self.view.log_message(f"Order: {optimized_config['order']}")
                    self.view.log_message(f"Seasonal: {optimized_config['seasonal_order']}")
                    self.view.log_message(f"Precisión esperada (según optimización): {optimized_config['precision_final']:.1f}%")

                    if simulation_applied:
                        self.view.log_message("")
                        self.view.log_message("NOTA ADICIONAL: Validación bajo simulación climática")
                        sim_config = result.get("simulation_config", {}) if result else {}
                        if sim_config:
                            summary = sim_config.get("summary", {})
                            self.view.log_message(f"Escenario simulado: {summary.get('escenario', 'N/A')}")

                    self.view.log_message("=" * 60)

        if simulation_applied and not used_optimized:
            self.view.log_success("=" * 60)
            self.view.log_success("VALIDACIÓN CON SIMULACIÓN CLIMÁTICA COMPLETADA")
            self.view.log_success("=" * 60)

            # Mostrar resumen de simulación
            sim_config = result.get("simulation_config", {}) if result else {}
            if sim_config:
                summary = sim_config.get("summary", {})
                self.view.log_message(f"Escenario simulado: {summary.get('escenario', 'N/A')}")
                self.view.log_message(f"Alcance: {summary.get('alcance_meses', 'N/A')} meses")
                self.view.log_message(f"Días simulados: {summary.get('dias_simulados', 'N/A')}")

                # Mostrar cambios en variables
                changes = summary.get("percentage_changes", {})
                if changes:
                    self.view.log_message("\nCambios aplicados a variables:")
                    var_names = {
                        "temp_max": "Temperatura máxima",
                        "humedad_avg": "Humedad relativa",
                        "precip_total": "Precipitación total",
                    }
                    for var, change_pct in changes.items():
                        var_name = var_names.get(var, var)
                        arrow = "↑" if change_pct > 0 else "↓" if change_pct < 0 else "→"
                        self.view.log_message(f"   {arrow} {var_name}: {change_pct:+.1f}%")

                self.view.log_message("")
                self.view.log_message("IMPORTANTE:")
                self.view.log_message("Las métricas reflejan el desempeño del modelo bajo condiciones")
                self.view.log_message("climáticas HIPOTÉTICAS del escenario simulado.")
                self.view.log_message("Los resultados reales dependerán de las condiciones climáticas efectivas.")
                self.view.log_message("")
        elif not used_optimized and not simulation_applied:
            self.view.log_success("Validación del modelo completada")

        if result and "metrics" in result:
            metrics = result["metrics"]
            model_params = result.get("model_params", {})

            self.view.log_message("=" * 60)
            self.view.log_message("RESULTADOS DE VALIDACION")
            self.view.log_message("=" * 60)

            # Informacion del modelo
            self.view.log_message(f"Transformacion: {model_params.get('transformation', 'N/A').upper()}")
            self.view.log_message(f"Parametros: order={model_params.get('order')}, seasonal={model_params.get('seasonal_order')}")

            if model_params.get("with_exogenous"):
                exog_info = result.get("exogenous_vars", {})
                self.view.log_message(f"Variables exogenas: {len(exog_info)}")
                for var_code, var_data in exog_info.items():
                    self.view.log_message(f"  - {var_data['nombre']}")

            self.view.log_message("")
            self.view.log_message("Metricas de validacion (escala original):")
            self.view.log_message(f"  - Precision Final: {metrics.get('precision_final', 0):.1f}%")
            self.view.log_message(f"  - RMSE: {metrics.get('rmse', 0):.4f} minutos")
            self.view.log_message(f"  - MAE: {metrics.get('mae', 0):.4f} minutos")
            self.view.log_message(f"  - MAPE: {metrics.get('mape', 0):.1f}%")
            self.view.log_message(f"  - R2: {metrics.get('r2_score', 0):.3f}")

            if used_optimized:
                optimized_config = self.validation_service.load_optimized_config(regional_code)
                if optimized_config:
                    expected_precision = optimized_config["precision_final"]
                    actual_precision = metrics.get("precision_final", 0)
                    difference = actual_precision - expected_precision

                    self.view.log_message("")
                    self.view.log_message("COMPARACIÓN CON OPTIMIZACIÓN:")
                    self.view.log_message(f"Precisión esperada: {expected_precision:.1f}%")
                    self.view.log_message(f"Precisión obtenida: {actual_precision:.1f}%")

                    valor_abs = 5

                    if abs(difference) <= valor_abs:
                        self.view.log_message(f"Diferencia: {difference:+.1f}% (CONSISTENTE)")
                    elif difference > 0:
                        self.view.log_message(f"Diferencia: {difference:+.1f}% (MEJOR de lo esperado)")
                    else:
                        self.view.log_message(f"Diferencia: {difference:+.1f}% (Menor de lo esperado)")
                        self.view.log_message("Posible causa: Datos de validación diferentes a optimización")

            # Informacion de validacion
            self.view.log_message("")
            self.view.log_message(f"Datos de entrenamiento: {result.get('training_count', 0)} observaciones")
            self.view.log_message(f"Datos de validacion: {result.get('validation_count', 0)} observaciones")
            self.view.log_message(f"Porcentaje validacion: {result.get('validation_percentage', 0):.0f}%")

            # Predicciones con intervalos (SOLO para referencia visual)
            if "predictions" in result:
                predictions = result["predictions"]
                mean_preds = predictions.get("mean", {})
                lower_bounds = predictions.get("lower_bound", {})
                upper_bounds = predictions.get("upper_bound", {})

                if mean_preds and lower_bounds and upper_bounds:
                    self.view.log_message("")
                    self.view.log_message("Predicciones de validacion:")
                    self.view.log_message("(Intervalos de confianza 95% solo en grafica)")

                    for fecha in sorted(mean_preds.keys()):
                        mean_val = mean_preds[fecha]
                        lower_val = lower_bounds.get(fecha, mean_val)
                        upper_val = upper_bounds.get(fecha, mean_val)

                        self.view.log_message(
                            f"  - {fecha}: {mean_val:.2f} min "
                            f"[IC: {lower_val:.2f} - {upper_val:.2f}]",
                        )

            # Interpretacion de calidad basada en PRECISION
            precision = metrics.get("precision_final", 0)
            self.view.log_message("")

            precision_mayor = 90
            precision_mayor_media = 80
            precision_media = 70
            precision_baja = 60

            if precision >= precision_mayor:
                self.view.log_success("Calidad: EXCELENTE - Predicciones muy confiable")
            elif precision >= precision_mayor_media:
                self.view.log_success("Calidad: BUENO - Predicciones confiables")
            elif precision >= precision_media:
                self.view.log_message("Calidad: ACEPTABLE - Predicciones moderadamente confiables")
            elif precision >= precision_baja:
                self.view.log_message("Calidad: REGULAR - Usar con precaución")
            else:
                self.view.log_error("Calidad: BAJO - Modelo poco confiable")

            # Nota sobre intervalos (solo para referencia)
            if model_params.get("confidence_level"):
                conf_level = model_params["confidence_level"] * 100
                self.view.log_message("")
                self.view.log_message(f"Nota: Intervalos de confianza al {conf_level:.0f}% disponibles en grafica")
                self.view.log_message("(solo para referencia visual, no afectan la precision del modelo)")

            self.view.log_message("=" * 60)

        # Mostrar grafica
        if result and "plot_file" in result and result["plot_file"]:
            self.show_plot(result["plot_file"], "Validacion del Modelo SAIDI")

    def on_report_validation_finished(self, result):
        """Callback cuando termina validación - Genera PDF."""
        try:
            # Preguntar al usuario dónde guardar el PDF

            model_params = result.get("model_params", {})
            regional_code = model_params.get("regional_code", "SAIDI")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            default_name = f"Informe_Validacion_{regional_code}_{timestamp}.pdf"
            default_path = os.path.join(os.path.expanduser("~/Desktop"), default_name)

            filepath, _ = QFileDialog.getSaveFileName(
                self.view,
                "Guardar Informe de Validación SAIDI",
                default_path,
                "Archivos PDF (*.pdf);;Todos los archivos (*.*)",
            )

            if not filepath:
                self.view.set_buttons_enabled(True)
                self.view.show_progress(visible=False)
                self.view.log_message("Generación de informe cancelada por el usuario")
                return

            if not filepath.endswith(".pdf"):
                filepath += ".pdf"

            self.view.update_progress(95, "Generando informe PDF...")
            self.view.log_message("=" * 60)
            self.view.log_message("GENERANDO INFORME PDF PROFESIONAL")
            self.view.log_message("=" * 60)

            # Generar informe PDF en la ubicación elegida
            pdf_path = self.report_service.generate_validation_report(
                result=result,
                output_path=filepath,
                log_callback=self.view.log_message,
            )

            self.view.set_buttons_enabled(True)
            self.view.show_progress(visible=False)
            self.view.update_status("Informe de validación generado")

            # Log resumen rápido
            final_diagnosis = result.get("validation_analysis", {}).get("final_diagnosis", {})
            quality = final_diagnosis.get("model_quality", "N/A")
            confidence = final_diagnosis.get("confidence_level", 0)

            self.view.log_success("=" * 60)
            self.view.log_success("INFORME PDF GENERADO EXITOSAMENTE")
            self.view.log_success("=" * 60)
            self.view.log_message(f" Calidad del Modelo: {quality}")
            self.view.log_message(f" Nivel de Confianza: {confidence:.1f}%")
            self.view.log_message(f" Ubicación: {pdf_path}")
            self.view.log_message("")
            self.view.log_message("El informe incluye:")
            self.view.log_message("  1. Rolling Forecast - Walk-Forward Validation")
            self.view.log_message("  2. Time Series Cross-Validation")
            self.view.log_message("  3. Análisis de Estabilidad de Parámetros")
            self.view.log_message("  4. Backtesting Multi-Horizonte")
            self.view.log_message("  5. Diagnóstico Final Integrado")
            self.view.log_message("  6. Visualización Integrada de Resultados")
            self.view.log_message("=" * 60)

            # Ofrecer abrir PDF
            msg = QMessageBox(self.view)
            msg.setIcon(QMessageBox.Icon.Information)
            msg.setWindowTitle("Informe Generado")
            msg.setText("El informe de validación se ha generado correctamente")
            msg.setInformativeText(f"Archivo: {os.path.basename(pdf_path)}")

            open_btn = msg.addButton("Abrir PDF", QMessageBox.ButtonRole.AcceptRole)
            open_folder_btn = msg.addButton("Abrir Carpeta", QMessageBox.ButtonRole.ActionRole)

            msg.exec()

            clicked_button = msg.clickedButton()
            if clicked_button == open_btn:
                self._open_file(pdf_path)
            elif clicked_button == open_folder_btn:
                self._open_folder(os.path.dirname(pdf_path))

        except Exception as e:
            self.view.set_buttons_enabled(True)
            self.view.show_progress(visible=False)
            self.view.log_error(f"Error generando informe PDF: {e!s}")
            self.view.log_error(traceback.format_exc())
            self.show_error(f"Error al generar informe PDF:\n\n{e!s}")

    def on_report_validation_error(self, error_msg):
        """Callback cuando hay error en generación de informe."""
        self.view.set_buttons_enabled(True)
        self.view.show_progress(visible=False)
        self.view.update_status("Error en generación de informe")
        self.view.log_error(f"Error: {error_msg}")
        self.show_error(f"Error durante la generación del informe: {error_msg}")

    def on_prediction_error(self, error_msg):
        self.view.set_buttons_enabled(True)
        self.view.show_progress(visible=False)
        self.view.update_status("Error en predicción")
        self.view.log_error(f"Error en predicción: {error_msg}")
        self.show_error(f"Error durante la predicción: {error_msg}")

    def on_optimization_error(self, error_msg):
        """Callback cuando hay error en optimización."""
        self.view.set_buttons_enabled(True)
        self.view.show_progress(visible=False)
        self.view.update_status("Error en optimización")
        self.view.log_error(f"Error en optimización: {error_msg}")
        self.show_error(f"Error durante la optimización: {error_msg}")

    def on_validation_error(self, error_msg):
        """Callback cuando hay error en validación."""
        self.view.set_buttons_enabled(True)
        self.view.show_progress(visible=False)
        self.view.update_status("Error en validacion")
        self.view.log_error(f"Error en validacion: {error_msg}")
        self.show_error(f"Error durante la validacion: {error_msg}")

    def show_error(self, message):
        """Mostrar mensaje de error."""
        QMessageBox.critical(self.view, "Error", message)

    def show_warning(self, message):
        """Mostrar mensaje de advertencia."""
        QMessageBox.warning(self.view, "Advertencia", message)

    def show_info(self, message):
        """Mostrar mensaje informativo."""
        QMessageBox.information(self.view, "Información", message)

class PredictionThread(QThread):
    """Hilo para ejecutar predicción en background."""

    progress_updated = pyqtSignal(int, str)
    message_logged = pyqtSignal(str)
    finished = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

    def __init__(self, prediction_service, file_path=None, df_prepared=None,
                 regional_code=None, climate_data=None, simulation_config=None):
        super().__init__()
        self.prediction_service = prediction_service
        self.file_path = file_path
        self.df_prepared = df_prepared
        self.regional_code = regional_code
        self.climate_data = climate_data
        self.simulation_config = simulation_config

    def run(self):
        try:
            self.message_logged.emit("Ejecutando predicción SAIDI...")
            result = self.prediction_service.run_prediction(
                file_path=self.file_path,
                df_prepared=self.df_prepared,
                regional_code=self.regional_code,
                climate_data=self.climate_data,
                simulation_config=self.simulation_config,
                progress_callback=self.progress_updated.emit,
                log_callback=self.message_logged.emit,
            )
            self.finished.emit(result)
        except Exception as e:
            self.error_occurred.emit(str(e))


class OptimizationThread(QThread):
    """Hilo para ejecutar optimización en background CON CLIMA."""

    progress_updated = pyqtSignal(int, str)
    message_logged = pyqtSignal(str)
    iteration_logged = pyqtSignal(str)
    finished = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

    def __init__(self, optimization_service, file_path=None, df_prepared=None,
                 regional_code=None, climate_data=None):
        super().__init__()
        self.optimization_service = optimization_service
        self.file_path = file_path
        self.df_prepared = df_prepared
        self.regional_code = regional_code
        self.climate_data = climate_data  # ASEGURAR QUE SE GUARDA

    def run(self):
        try:
            self.message_logged.emit("Ejecutando optimización de parámetros...")

            # VERIFICAR QUE climate_data SE PASA CORRECTAMENTE
            if self.climate_data is not None:
                self.message_logged.emit(f"Climate data disponible: {len(self.climate_data)} registros")
            else:
                self.message_logged.emit("Sin climate data - Optimización sin exógenas")

            result = self.optimization_service.run_optimization(
                file_path=self.file_path,
                df_prepared=self.df_prepared,
                regional_code=self.regional_code,
                climate_data=self.climate_data,  # CRÍTICO: Pasar climate_data
                progress_callback=self.progress_updated.emit,
                log_callback=self.message_logged.emit,
                iteration_callback=self.iteration_logged.emit,
            )
            self.finished.emit(result)
        except Exception as e:
            import traceback
            error_detail = traceback.format_exc()
            self.message_logged.emit(f"Error detallado:\n{error_detail}")
            self.error_occurred.emit(str(e))


class ValidationThread(QThread):
    """Hilo para ejecutar validación en background."""

    progress_updated = pyqtSignal(int, str)
    message_logged = pyqtSignal(str)
    finished = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

    def __init__(self, validation_service, file_path=None, df_prepared=None,
                 regional_code=None, climate_data=None, simulation_config=None):
        super().__init__()
        self.validation_service = validation_service
        self.file_path = file_path
        self.df_prepared = df_prepared
        self.regional_code = regional_code
        self.climate_data = climate_data
        self.simulation_config = simulation_config

    def run(self):
        try:
            self.message_logged.emit("Ejecutando validación del modelo...")
            result = self.validation_service.run_validation(
                file_path=self.file_path,
                df_prepared=self.df_prepared,
                regional_code=self.regional_code,
                climate_data=self.climate_data,
                simulation_config=self.simulation_config,
                progress_callback=self.progress_updated.emit,
                log_callback=self.message_logged.emit,
            )
            self.finished.emit(result)
        except Exception as e:
            self.error_occurred.emit(str(e))


class RollingValidationThread(QThread):
    """Hilo para ejecutar validación temporal completa en background."""

    progress_updated = pyqtSignal(int, str)
    message_logged = pyqtSignal(str)
    finished = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

    def __init__(self, rolling_validation_service, file_path=None, df_prepared=None,
                 regional_code=None, climate_data=None, validation_months=6):
        super().__init__()
        self.rolling_validation_service = rolling_validation_service
        self.file_path = file_path
        self.df_prepared = df_prepared
        self.regional_code = regional_code
        self.climate_data = climate_data
        self.validation_months = validation_months

    def run(self):
        try:
            self.message_logged.emit("Ejecutando validación temporal completa...")
            result = self.rolling_validation_service.run_comprehensive_validation(
                file_path=self.file_path,
                df_prepared=self.df_prepared,
                regional_code=self.regional_code,
                climate_data=self.climate_data,
                validation_months=self.validation_months,
                progress_callback=self.progress_updated.emit,
                log_callback=self.message_logged.emit,
            )
            self.finished.emit(result)
        except Exception as e:
            self.error_occurred.emit(str(e))
