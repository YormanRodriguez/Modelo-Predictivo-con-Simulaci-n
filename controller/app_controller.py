# controller/app_controller.py - Controlador principal MVC (COMPLETO + CLIMA)
import os
import gc
import pandas as pd
import numpy as np
from PyQt6.QtWidgets import QFileDialog, QMessageBox
from PyQt6.QtCore import QObject, QThread, pyqtSignal, QTimer
from datetime import datetime

# Importar servicios de backend
from services.prediction_service import PredictionService
from services.optimization_service import OptimizationService
from services.validation_service import ValidationService
from services.overfitting_detection_service import OverfittingDetectionService
from services.improved_model_service import ImprovedModelService

# Importar el visor de gráficas
from view.main_window import PlotViewerDialog
from view.climate_simulation_dialog import ClimateSimulationDialog

class AppController(QObject):
    """Controlador principal de la aplicación"""
    
    def __init__(self, view, model, climate_model): 
        super().__init__()
        self.view = view
        self.model = model
        self.climate_model = climate_model  
        
        # Servicios de backend
        self.prediction_service = PredictionService()
        self.optimization_service = OptimizationService()
        self.validation_service = ValidationService()
        self.overfitting_service = OverfittingDetectionService()
        self.improved_model_service = ImprovedModelService()
        
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
        """Configurar conexiones del modelo climático"""
        self.climate_model.climate_data_loaded.connect(self.on_climate_data_loaded)
        self.climate_model.all_climate_loaded.connect(self.on_all_climate_loaded)
        self.climate_model.status_changed.connect(self.view.update_status)
        self.climate_model.error_occurred.connect(self.show_error)
    
    def load_climate_file(self, regional_code: str):
        """Cargar archivo climático para una regional específica"""
        try:
            regional_nombre = self.climate_model.REGIONALES.get(regional_code, regional_code)
            file_path, _ = QFileDialog.getOpenFileName(
                self.view,
                f"Seleccionar datos climáticos - {regional_nombre}",
                "",
                "Archivos Excel (*.xlsx *.xls);;Archivos CSV (*.csv);;Todos los archivos (*.*)"
            )
            if not file_path:
                self.view.log_message(f"Carga cancelada para {regional_nombre}")
                return

            self.view.update_climate_status(regional_code, 'loading')
            self.view.log_message(f"Cargando datos climáticos de {regional_nombre}...")

            success = self.climate_model.load_climate_file(regional_code, file_path)

            if success:
                self.view.update_climate_status(regional_code, 'success')
                self.view.log_success(f"Datos climáticos de {regional_nombre} cargados correctamente")
                loaded = len(self.climate_model.get_loaded_regionales())
                total = len(self.climate_model.REGIONALES)
                self.view.update_climate_progress_summary(loaded, total)
            else:
                self.view.update_climate_status(regional_code, 'error')
                self.view.log_error(f"Error cargando datos climáticos de {regional_nombre}")

        except Exception as e:
            self.view.log_error(f"Error inesperado cargando clima: {str(e)}")
            self.view.update_climate_status(regional_code, 'error')
            self.show_error(f"Error al cargar datos climáticos: {str(e)}")
    
    def on_climate_data_loaded(self, climate_info: dict):
        """Callback cuando se cargan datos climáticos de una regional"""
        regional_name = climate_info.get('regional_name', 'Desconocida')
        total_records = climate_info.get('total_records', 0)
        completeness = climate_info.get('avg_completeness', 0)
        
        self.view.log_message(f" {regional_name}:")
        self.view.log_message(f"  • Registros: {total_records:,}")
        self.view.log_message(f"  • Completitud: {completeness:.1f}%")
        self.view.log_message(
            f"  • Período: {climate_info['date_range']['start'].strftime('%Y-%m-%d')} "
            f"a {climate_info['date_range']['end'].strftime('%Y-%m-%d')}"
        )
        
        self.update_climate_details_panel()
    
    def on_all_climate_loaded(self, summary: dict):
        """Callback cuando todas las regionales tienen datos climáticos"""
        self.view.log_success("=" * 60)
        self.view.log_success("TODOS LOS DATOS CLIMÁTICOS CARGADOS")
        self.view.log_success("=" * 60)

        total = summary.get('total_regionales', 0)
        self.view.log_message(f"Total de regionales con datos: {total}")

        climate_info = summary.get('climate_info', {})
        for regional_code, info in climate_info.items():
            regional_name = info.get('regional_name', regional_code)
            records = info.get('total_records', 0)
            self.view.log_message(f"  • {regional_name}: {records:,} registros")

        self.view.log_message("\nLos datos climáticos están listos para análisis")
        # Actualizar panel de detalles en la vista
        self.update_climate_details_panel()

    def update_climate_details_panel(self):
        """Actualizar panel de detalles de datos climáticos en la vista"""
        all_info = {}

        for regional_code in self.climate_model.REGIONALES.keys():
            info = self.climate_model.get_climate_info(regional_code)
            if info:
                all_info[regional_code] = info

        if all_info:
            # La vista espera un resumen para mostrar
            self.view.update_climate_details(all_info)
    
    def get_climate_data_for_regional(self, regional_code: str):
        """Obtener datos climáticos para una regional específica"""
        return self.climate_model.get_climate_data(regional_code)
    
    def are_climate_data_available(self, regional_code: str) -> bool:
        """Verificar si hay datos climáticos disponibles para una regional"""
        return self.climate_model.is_regional_loaded(regional_code)

    def run_regularized_prediction(self):
        """Ejecutar predicción con regularización"""
        if not self.model.is_excel_loaded():
            self.show_warning("Debe cargar un archivo Excel primero")
            return

        # Diálogo para seleccionar variables y parámetros
        from PyQt6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel
        from PyQt6.QtWidgets import QListWidget, QComboBox, QDoubleSpinBox, QPushButton

        dialog = QDialog(self.view)
        dialog.setWindowTitle("Configurar Regularización")
        dialog.resize(500, 400)

        layout = QVBoxLayout(dialog)

        # Selector de variables exógenas
        layout.addWidget(QLabel("Seleccione variables exógenas:"))
        var_list = QListWidget()
        var_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)

        # Obtener columnas del Excel (excepto Fecha y SAIDI)
        df = self.model.get_excel_data()
        available_cols = []
        if df is not None:
            available_cols = [col for col in df.columns
                              if 'SAIDI' not in str(col).upper() and 'FECHA' not in str(col).upper()]
        var_list.addItems(available_cols)
        layout.addWidget(var_list)

        # Método de regularización
        layout.addWidget(QLabel("\nMétodo de regularización:"))
        method_combo = QComboBox()
        method_combo.addItems(['elastic', 'lasso', 'ridge'])
        layout.addWidget(method_combo)

        # Alpha (fuerza)
        alpha_layout = QHBoxLayout()
        alpha_layout.addWidget(QLabel("Alpha (fuerza):"))
        alpha_spin = QDoubleSpinBox()
        alpha_spin.setRange(0.01, 10.0)
        alpha_spin.setValue(1.0)
        alpha_spin.setSingleStep(0.1)
        alpha_layout.addWidget(alpha_spin)
        layout.addLayout(alpha_layout)

        # L1 ratio (solo para ElasticNet)
        l1_layout = QHBoxLayout()
        l1_layout.addWidget(QLabel("L1 ratio (ElasticNet):"))
        l1_spin = QDoubleSpinBox()
        l1_spin.setRange(0.0, 1.0)
        l1_spin.setValue(0.5)
        l1_spin.setSingleStep(0.1)
        l1_layout.addWidget(l1_spin)
        layout.addLayout(l1_layout)

        # Botones
        button_layout = QHBoxLayout()
        ok_button = QPushButton("Ejecutar")
        cancel_button = QPushButton("Cancelar")
        ok_button.clicked.connect(dialog.accept)
        cancel_button.clicked.connect(dialog.reject)
        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            selected_vars = [item.text() for item in var_list.selectedItems()]
            method = method_combo.currentText()
            alpha = alpha_spin.value()
            l1_ratio = l1_spin.value()

            # Ejecutar regularización (sin hilo para simplificar)
            try:
                self.view.set_buttons_enabled(False)
                self.view.show_progress(True)
                self.view.log_message("Ejecutando regularización...")
                result = self.improved_model_service.fit_with_regularization(
                    self.model.get_file_path(),
                    (1, 1, 2),
                    (1, 0, 1, 12),
                    selected_vars,
                    method,
                    alpha,
                    l1_ratio,
                    progress_callback=self.view.update_progress,
                    log_callback=self.view.log_message
                )
                self.on_regularization_finished(result)
            except Exception as e:
                self.on_regularization_error(str(e))

    def on_regularization_finished(self, result):
        """Callback cuando termina regularización"""
        self.view.set_buttons_enabled(True)
        self.view.show_progress(False)
        self.view.update_status("Regularización completada")
        
        if result['regularization_applied']:
            self.view.log_success("Regularización aplicada exitosamente")
            self.view.log_message(f"Variables seleccionadas: {result['n_selected']}/{result['n_total']}")
            
            for var in result['selected_vars']:
                self.view.log_message(f"  ✓ {var}")
            
            if result.get('eliminated_vars'):
                self.view.log_message(f"\nVariables eliminadas:")
                for var in result['eliminated_vars']:
                    self.view.log_message(f"  ✗ {var}")
            
            self.view.log_message(f"\nScore de overfitting ajustado: {result['adjusted_score']:.2f}")
            self.view.log_message(f"Recomendación: {result['recommendation']}")
            
            if result['adjusted_score'] < 15:
                self.view.log_success(" Modelo bien regularizado - seguro usar")
            elif result['adjusted_score'] < 25:
                self.view.log_message(" Regularización aceptable - monitorear")
            else:
                self.view.log_error("X Considerar aumentar alpha o reducir variables")
        
        if result.get('plot_file'):
            self.show_plot(result['plot_file'], "Análisis de Regularización")

    def on_regularization_error(self, error_msg):
        """Callback cuando hay error en regularización"""
        self.view.set_buttons_enabled(True)
        self.view.show_progress(False)
        self.view.update_status("Error en regularización")
        self.view.log_error(f"Error: {error_msg}")
        self.show_error(f"Error durante regularización: {error_msg}")

    def run_cross_validation(self, n_splits=5):
        """Ejecutar cross-validation temporal con transformación por regional"""
        if not self.model.is_excel_loaded():
            self.show_warning("Debe cargar un archivo Excel primero")
            return
        
        regional_code = None
        if self.model.is_regional_format():
            if not self.model.get_selected_regional():
                self.show_warning("Debe seleccionar una regional primero")
                return
            
            regional_code = self.model.get_selected_regional()
            regional_nombre = self.model.REGIONAL_MAPPING.get(regional_code, regional_code)
            
            transformation = self.improved_model_service.REGIONAL_TRANSFORMATIONS.get(
                regional_code, 'original'
            )
            
            self.view.log_message(f"Ejecutando CV para: {regional_nombre}")
            self.view.log_message(f"Transformación asignada: {transformation.UPPER()}")
            
            # NUEVO: Verificar datos climáticos
            if self.are_climate_data_available(regional_code):
                self.view.log_message(f"Datos climáticos disponibles para {regional_nombre}")
            else:
                self.view.log_message(f"Sin datos climáticos para {regional_nombre}")
            
        try:
            self.view.log_message(f"Iniciando Cross-Validation con {n_splits} folds...")
            self.view.set_buttons_enabled(False)
            self.view.update_status("Ejecutando CV temporal...")
            
            df_prepared = self.model.get_excel_data_for_analysis()
            
            if df_prepared is None:
                self.show_error("No se pudieron preparar los datos. Verifique la regional seleccionada.")
                self.view.set_buttons_enabled(True)
                return
            
            order = (3, 1, 3)
            seasonal_order = (3, 1, 0, 12)
            
            self.cv_thread = CrossValidationThread(
                df_prepared=df_prepared,
                improved_service=self.improved_model_service, 
                order=order, 
                seasonal_order=seasonal_order, 
                n_splits=n_splits,
                regional_code=regional_code
            )
            self.cv_thread.progress_updated.connect(self.view.update_progress)
            self.cv_thread.message_logged.connect(self.view.log_message)
            self.cv_thread.finished.connect(self.on_cv_finished)
            self.cv_thread.error_occurred.connect(self.on_cv_error)
            
            self.view.show_progress(True)
            self.cv_thread.start()
            
        except Exception as e:
            self.view.log_error(f"Error iniciando CV: {str(e)}")
            self.view.set_buttons_enabled(True)
            self.view.show_progress(False)
    
    def run_find_best_simple_model(self):
        """Buscar modelo simple con mejor balance (con transformación por regional)"""
        if not self.model.is_excel_loaded():
            self.show_warning("Debe cargar un archivo Excel primero")
            return
        
        regional_code = None
        if self.model.is_regional_format():
            if not self.model.get_selected_regional():
                self.show_warning("Debe seleccionar una regional primero")
                return
            
            regional_code = self.model.get_selected_regional()
            regional_nombre = self.model.REGIONAL_MAPPING.get(regional_code, regional_code)
            
            transformation = self.improved_model_service.REGIONAL_TRANSFORMATIONS.get(
                regional_code, 'original'
            )
            
            self.view.log_message(f"Ejecutando búsqueda para: {regional_nombre}")
            self.view.log_message(f"Transformación asignada: {transformation.upper()}")
            
        try:
            self.view.log_message("Iniciando búsqueda de modelo simple óptimo...")
            self.view.log_message("Buscando mejor balance: AIC/BIC, complejidad y overfitting")
            self.view.set_buttons_enabled(False)
            self.view.update_status("Buscando modelo simple...")
            
            df_prepared = self.model.get_excel_data_for_analysis()
            
            if df_prepared is None:
                self.show_error("No se pudieron preparar los datos. Verifique la regional seleccionada.")
                self.view.set_buttons_enabled(True)
                return
            
            self.simple_model_thread = SimpleModelThread(
                df_prepared=df_prepared,
                improved_service=self.improved_model_service,
                regional_code=regional_code
            )
            self.simple_model_thread.progress_updated.connect(self.view.update_progress)
            self.simple_model_thread.message_logged.connect(self.view.log_message)
            self.simple_model_thread.finished.connect(self.on_simple_model_finished)
            self.simple_model_thread.error_occurred.connect(self.on_simple_model_error)
            
            self.view.show_progress(True)
            self.simple_model_thread.start()
            
        except Exception as e:
            self.view.log_error(f"Error iniciando búsqueda: {str(e)}")
            self.view.set_buttons_enabled(True)
            self.view.show_progress(False)
     
    def run_compare_transformations(self):
        """Comparar transformaciones de datos con transformación por regional"""
        if not self.model.is_excel_loaded():
            self.show_warning("Debe cargar un archivo Excel primero")
            return
        
        regional_code = None
        if self.model.is_regional_format():
            if not self.model.get_selected_regional():
                self.show_warning("Debe seleccionar una regional primero")
                return
            
            regional_code = self.model.get_selected_regional()
            regional_nombre = self.model.REGIONAL_MAPPING.get(regional_code, regional_code)
            
            transformation = self.improved_model_service.REGIONAL_TRANSFORMATIONS.get(
                regional_code, 'original'
            )
            
            self.view.log_message(f"Ejecutando comparación para: {regional_nombre}")
            self.view.log_message(f"Transformación asignada actual: {transformation.upper()}")
            
        try:
            self.view.log_message("Iniciando comparación de transformaciones...")
            self.view.log_message("Evaluando: Original, StandardScaler, Log, Box-Cox")
            self.view.set_buttons_enabled(False)
            self.view.update_status("Comparando transformaciones...")
            
            df_prepared = self.model.get_excel_data_for_analysis()
            
            if df_prepared is None:
                self.show_error("No se pudieron preparar los datos. Verifique la regional seleccionada.")
                self.view.set_buttons_enabled(True)
                return
            
            order = (3, 1, 3)
            seasonal_order = (3, 1, 0, 12)
            
            self.transformation_thread = TransformationThread(
                df_prepared=df_prepared,
                improved_service=self.improved_model_service, 
                order=order, 
                seasonal_order=seasonal_order,
                regional_code=regional_code
            )
            self.transformation_thread.progress_updated.connect(self.view.update_progress)
            self.transformation_thread.message_logged.connect(self.view.log_message)
            self.transformation_thread.finished.connect(self.on_transformation_finished)
            self.transformation_thread.error_occurred.connect(self.on_transformation_error)
            
            self.view.show_progress(True)
            self.transformation_thread.start()
            
        except Exception as e:
            self.view.log_error(f"Error iniciando comparación: {str(e)}")
            self.view.set_buttons_enabled(True)
            self.view.show_progress(False)
      
    def on_cv_finished(self, result):
        """Callback cuando termina CV (con información de transformación)"""
        self.view.set_buttons_enabled(True)
        self.view.show_progress(False)
        self.view.update_status("Cross-validation completada")
        self.view.log_success("Cross-validation temporal completada")
        
        if result and 'cv_results' in result:
            cv = result['cv_results']
            self.view.log_message(f"Folds completados: {len(cv['fold_scores'])}")
            self.view.log_message(f"RMSE promedio: {cv['mean_rmse']:.4f} ± {cv['std_rmse']:.4f} min")
            self.view.log_message(f"Score de estabilidad: {cv['stability_score']:.2f}/100")
            
            if cv.get('transformation'):
                self.view.log_message(f"Transformación: {cv['transformation'].upper()}")
            
            if cv['stability_score'] >= 80:
                self.view.log_success(" Modelo MUY ESTABLE")
            elif cv['stability_score'] >= 60:
                self.view.log_success(" Modelo ESTABLE")
            else:
                self.view.log_message(" Modelo con estabilidad limitada")
        
        if result and 'plot_files' in result and result['plot_files'].get('cv_plot'):
            self.show_plot(result['plot_files']['cv_plot'], "Cross-Validation Temporal")

    def on_cv_error(self, error_msg):
        """Callback cuando hay error en CV"""
        self.view.set_buttons_enabled(True)
        self.view.show_progress(False)
        self.view.update_status("Error en CV")
        self.view.log_error(f"Error en CV: {error_msg}")
        self.show_error(f"Error durante cross-validation: {error_msg}")
    
    def on_simple_model_finished(self, result):
        """Callback cuando termina búsqueda de modelo simple"""
        self.view.set_buttons_enabled(True)
        self.view.show_progress(False)
        self.view.update_status("Búsqueda completada")
        self.view.log_success("Búsqueda de modelo simple completada")
        
        if result and 'best_model' in result:
            best = result['best_model']
            self.view.log_message("=" * 60)
            self.view.log_message("MEJOR MODELO SIMPLE ENCONTRADO:")
            self.view.log_message(f"  order={best['order']}, seasonal={best['seasonal_order']}")
            self.view.log_message(f"  Complejidad: {best['complexity']} parámetros")
            self.view.log_message(f"  AIC: {best['aic']:.1f} | BIC: {best['bic']:.1f}")
            self.view.log_message(f"  Overfitting Score: {best['overfitting_score']:.2f}/100")
            self.view.log_message(f"  R² Train: {best['r2_train']:.3f} | Test: {best['r2_test']:.3f}")
            
            if result.get('transformation'):
                self.view.log_message(f"  Transformación: {result['transformation'].upper()}")
            
            self.view.log_message("=" * 60)
            
            if best['overfitting_score'] < 10:
                self.view.log_success(" Excelente balance - Sin overfitting")
            elif best['overfitting_score'] < 20:
                self.view.log_success(" Buen balance - Overfitting mínimo")
            
            if 'top_models' in result:
                self.view.log_message("\nTop 3 alternativas:")
                for i, m in enumerate(result['top_models'][:3], 1):
                    self.view.log_message(f"  {i}. order={m['order']}, seasonal={m['seasonal_order']} "
                                        f"| Overfitting: {m['overfitting_score']:.2f}")
        
        if result and 'plot_files' in result and result['plot_files'].get('model_comparison_plot'):
            self.show_plot(result['plot_files']['model_comparison_plot'], 
                        "Comparación de Modelos Simples")
    
    def on_simple_model_error(self, error_msg):
        """Callback cuando hay error en búsqueda"""
        self.view.set_buttons_enabled(True)
        self.view.show_progress(False)
        self.view.update_status("Error en búsqueda")
        self.view.log_error(f"Error en búsqueda: {error_msg}")
        self.show_error(f"Error durante búsqueda de modelo: {error_msg}")
    
    def on_transformation_finished(self, result):
        """Callback cuando termina comparación de transformaciones"""
        self.view.set_buttons_enabled(True)
        self.view.show_progress(False)
        self.view.update_status("Comparación completada")
        self.view.log_success("Comparación de transformaciones completada")
        
        if result and 'best_transformation' in result:
            best = result['best_transformation']
            assigned = result.get('assigned_transformation', 'unknown')
            
            self.view.log_message("=" * 60)
            self.view.log_message("RESULTADOS DE COMPARACIÓN:")
            self.view.log_message(f"  Transformación asignada: {assigned.upper()}")
            self.view.log_message(f"  Mejor encontrada: {best['method'].upper()}")
            self.view.log_message(f"  Overfitting Score: {best['final_overfitting_score']:.2f}/100")
            self.view.log_message(f"  Mejora: {best['overfitting_improvement']:+.1f}%")
            self.view.log_message(f"  R² Train: {best['r2_train']:.3f} | Test: {best['r2_test']:.3f}")
            self.view.log_message("=" * 60)
            
            if best['method'] != assigned:
                self.view.log_message(f" La transformación óptima ({best['method'].upper()}) difiere de la asignada ({assigned.upper()})")
                self.view.log_message(f"   Considere actualizar REGIONAL_TRANSFORMATIONS en los servicios")
            else:
                self.view.log_success(f" La transformación asignada ({assigned.upper()}) es óptima")
            
            if best['overfitting_improvement'] > 10:
                self.view.log_success(f" Mejora significativa de {best['overfitting_improvement']:.1f}%")
            elif best['overfitting_improvement'] > 0:
                self.view.log_message(f"Mejora moderada de {best['overfitting_improvement']:.1f}%")
            else:
                self.view.log_message("Los datos originales son la mejor opción")
            
            if 'all_transformations' in result:
                self.view.log_message("\nTodas las transformaciones evaluadas:")
                for method, trans in result['all_transformations'].items():
                    marker = "★" if method == best['method'] else "✓" if method == assigned else "○"
                    self.view.log_message(f"  {marker} {method.upper()}: "
                                        f"Overfitting={trans['overfitting_score']:.2f}")
        
        if result and 'plot_files' in result and result['plot_files'].get('transformation_plot'):
            self.show_plot(result['plot_files']['transformation_plot'], 
                        "Comparación de Transformaciones")
    
    def on_transformation_error(self, error_msg):
        """Callback cuando hay error en transformaciones"""
        self.view.set_buttons_enabled(True)
        self.view.show_progress(False)
        self.view.update_status("Error en transformaciones")
        self.view.log_error(f"Error en transformaciones: {error_msg}")
        self.show_error(f"Error durante comparación de transformaciones: {error_msg}")

    def cleanup_temp_files(self):
        """Limpiar archivos temporales de gráficas"""
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
                self.overfitting_service.cleanup_plot_file()
            except Exception:
                pass
            gc.collect()
        except Exception as e:
            print(f"Error durante limpieza automática: {e}")

    def show_plot(self, plot_file_path, title="Gráfica SAIDI"):
        """Mostrar gráfica en un diálogo separado"""
        if not plot_file_path or not os.path.exists(plot_file_path):
            self.view.log_error("No se encontró el archivo de gráfica")
            return

        try:
            dialog = PlotViewerDialog(plot_file_path, title, self.view)
            self.plot_dialogs.append(dialog)
            dialog.show()
            self.view.log_success(f"Gráfica mostrada: {title}")
        except Exception as e:
            self.view.log_error(f"Error mostrando gráfica: {str(e)}")

    def setup_model_connections(self):
        """Configurar conexiones del modelo"""
        self.model.error_occurred.connect(self.show_error)

    def on_regional_selected(self, regional_codigo: str):
        """Callback cuando el usuario selecciona una regional"""
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
            self.view.log_error(f"Error al seleccionar regional: {str(e)}")

    def load_excel_file(self):
        """Cargar archivo Excel mediante diálogo"""
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self.view,
                "Seleccionar archivo Excel - SAIDI Analysis",
                "",
                "Archivos Excel (*.xlsx *.xls);;Todos los archivos (*.*)"
            )

            if not file_path:
                return

            success = self.model.load_excel_file(file_path)

            if success:
                self.view.log_success(f"Archivo cargado exitosamente: {os.path.basename(file_path)}")
            else:
                self.view.log_error("Error al cargar el archivo Excel")

        except Exception as e:
            self.view.log_error(f"Error inesperado: {str(e)}")
            self.show_error(f"Error al cargar archivo: {str(e)}")

    def run_prediction(self):
        """Ejecutar predicción SAIDI CON VARIABLES EXÓGENAS Y SIMULACIÓN"""
        if not self.model.is_excel_loaded():
            self.show_warning("Debe cargar un archivo Excel primero")
            return

        regional_code = None
        climate_data = None
        simulation_config = None

        if self.model.is_regional_format():
            if not self.model.get_selected_regional():
                self.show_warning("Debe seleccionar una regional primero")
                return
            regional_code = self.model.get_selected_regional()
            regional_nombre = self.model.REGIONAL_MAPPING.get(regional_code, regional_code)

            # Obtener datos climáticos si existen
            if self.are_climate_data_available(regional_code):
                climate_data = self.get_climate_data_for_regional(regional_code)
                self.view.log_message(f"✓ Datos climáticos disponibles - Se incluirán en la predicción")
            else:
                self.view.log_message(f"Sin datos climáticos - Predicción sin variables exógenas")

            # Si la simulación está habilitada en la UI, abrir diálogo
            if getattr(self.view, 'enable_simulation_checkbox', None) and self.view.enable_simulation_checkbox.isChecked():
                # abrir diálogo de configuración de simulación
                dialog = ClimateSimulationDialog(
                    climate_data=climate_data,
                    mes_prediccion=pd.to_datetime(self.model.get_excel_data_for_analysis()['Fecha'].iloc[-1]).month
                                    if self.model.get_excel_data_for_analysis() is not None else 1,
                    regional_code=regional_code,
                    regional_nombre=regional_nombre,
                    parent=self.view
                )
                dialog.simulation_accepted.connect(lambda cfg: self._on_simulation_configured(cfg, regional_code, climate_data))
                dialog.simulation_cancelled.connect(lambda: self._on_simulation_configured({'enabled': False}, regional_code, climate_data))
                dialog.exec()
                return

        # Si llegamos aquí, ejecutar predicción normal (sin simulación)
        self._execute_prediction(regional_code, climate_data, None)

    def export_predictions_to_excel(self):
        """Exportar predicciones a Excel con validación completa"""
        try:
            if not hasattr(self, 'last_prediction_result') or not self.last_prediction_result:
                self.view.log_error("No hay predicciones disponibles para exportar")
                self.show_warning("Debe ejecutar una predicción primero antes de exportar")
                return
            
            result = self.last_prediction_result
            predictions = result.get('predictions')
            
            if not predictions:
                self.view.log_error("Las predicciones están vacías")
                self.show_warning("No hay predicciones para exportar")
                return
            
            model_params = result.get('model_params', {})
            regional_code = model_params.get('regional_code')
            
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
            
            from PyQt6.QtWidgets import QFileDialog
            
            default_name = f"Predicciones_SAIDI_{regional_nombre}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            default_path = os.path.join(os.path.expanduser("~/Desktop"), default_name)
            
            filepath, _ = QFileDialog.getSaveFileName(
                self.view,
                "Guardar Predicciones SAIDI",
                default_path,
                "Archivos Excel (*.xlsx);;Todos los archivos (*.*)"
            )
            
            if not filepath:
                self.view.log_message("Exportación cancelada por el usuario")
                return

            if not filepath.endswith('.xlsx'):
                filepath += '.xlsx'
            
            self.view.log_message(f"Ubicación seleccionada: {filepath}")

            export_service = result.get('export_service')
            
            if not export_service:
                self.view.log_message("Creando servicio de exportación...")
                from services.export_service import ExportService
                export_service = ExportService()

            model_info = {
                'order': model_params.get('order'),
                'seasonal_order': model_params.get('seasonal_order'),
                'transformation': model_params.get('transformation'),
                'with_exogenous': model_params.get('with_exogenous', False),
                'with_simulation': model_params.get('with_simulation', False),
                'confidence_level': model_params.get('confidence_level', 0.95),
                'uncertainty_method': model_params.get('uncertainty_method', 'parametric'),
                'regional_code': regional_code,
                'metrics': result.get('metrics')
            }
            
            self.view.show_progress(True)
            self.view.update_progress(30, "Preparando datos para exportación...")
            self.view.log_message("Generando archivo Excel con formato profesional...")
            
            self.view.update_progress(60, "Escribiendo archivo Excel...")
            
            saved_path = export_service.export_to_custom_location(
                predictions_dict=predictions,
                regional_code=regional_code,
                regional_nombre=regional_nombre,
                custom_path=filepath,
                include_confidence_intervals=True,
                model_info=model_info
            )
            
            self.view.update_progress(100, "Exportación completada")
            self.view.show_progress(False)
            
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
                has_intervals = isinstance(first_pred, dict) and 'limite_inferior' in first_pred
                
                self.view.log_message("\nContenido del archivo:")
                self.view.log_message("  • Hoja 1: Predicciones SAIDI")
                self.view.log_message(f"    - Fecha y valores predichos")
                if has_intervals:
                    self.view.log_message(f"    - Intervalos de confianza (95%)")
                    self.view.log_message(f"    - Márgenes de error")
                self.view.log_message("  • Hoja 2: Información del Modelo")
                self.view.log_message(f"    - Parámetros SARIMAX")
                self.view.log_message(f"    - Transformación aplicada")
                self.view.log_message(f"    - Métricas de precisión")
                
                if model_params.get('with_exogenous'):
                    self.view.log_message(f"    - Variables exógenas utilizadas")
                if model_params.get('with_simulation'):
                    self.view.log_message(f"    - Simulación climática aplicada")
                
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
                    f"Tamaño: {file_size:.2f} KB"
                )
                
                # Botones
                open_btn = msg.addButton("Abrir Archivo", QMessageBox.ButtonRole.AcceptRole)
                open_folder_btn = msg.addButton("Abrir Carpeta", QMessageBox.ButtonRole.ActionRole)
                close_btn = msg.addButton("Cerrar", QMessageBox.ButtonRole.RejectRole)
                
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
            self.view.show_progress(False)
            error_msg = str(e)
            self.view.log_error("=" * 60)
            self.view.log_error("ERROR EN EXPORTACIÓN")
            self.view.log_error("=" * 60)
            self.view.log_error(f"Error: {error_msg}")
            
            # Mostrar traceback completo en log para debugging
            import traceback
            self.view.log_error("\nTraceback completo:")
            for line in traceback.format_exc().split('\n'):
                if line.strip():
                    self.view.log_error(f"  {line}")
            
            self.show_error(f"Error al exportar predicciones:\n\n{error_msg}")

    def _open_file(self, filepath):
        """Abrir archivo con la aplicación predeterminada del sistema"""
        try:
            import platform
            import subprocess
            
            system = platform.system()
            
            if system == 'Windows':
                os.startfile(filepath)
            elif system == 'Darwin':  # macOS
                subprocess.call(['open', filepath])
            else:  # Linux
                subprocess.call(['xdg-open', filepath])
            
            self.view.log_message(f"Abriendo archivo: {os.path.basename(filepath)}")
        except Exception as e:
            self.view.log_error(f"No se pudo abrir el archivo: {str(e)}")

    def _open_folder(self, folder_path):
        """Abrir carpeta con el explorador de archivos del sistema"""
        try:
            import platform
            import subprocess
            
            system = platform.system()
            
            if system == 'Windows':
                os.startfile(folder_path)
            elif system == 'Darwin':  # macOS
                subprocess.call(['open', folder_path])
            else:  # Linux
                subprocess.call(['xdg-open', folder_path])
            
            self.view.log_message(f"Abriendo carpeta: {folder_path}")
        except Exception as e:
            self.view.log_error(f"No se pudo abrir la carpeta: {str(e)}")

    def _on_simulation_configured(self, simulation_config, regional_code, climate_data):
        """Callback cuando el usuario configura la simulación"""
        # Ejecutar predicción con la configuración
        self._execute_prediction(regional_code, climate_data, simulation_config)

    def _execute_prediction(self, regional_code, climate_data, simulation_config):
        """Ejecutar la predicción con o sin simulación"""
        try:
            self.view.set_buttons_enabled(False)
            self.view.show_progress(True)
            self.view.update_status("Generando predicción...")

            df_prepared = self.model.get_excel_data_for_analysis()
            if df_prepared is None:
                raise Exception("No se pudieron preparar los datos SAIDI")

            result = self.prediction_service.run_prediction(
                file_path=self.model.get_file_path(),
                df_prepared=df_prepared,
                order=None,
                seasonal_order=None,
                regional_code=regional_code,
                climate_data=climate_data,
                simulation_config=simulation_config,
                progress_callback=self.view.update_progress,
                log_callback=self.view.log_message
            )

            # Llamar al manejador de finalización
            self.on_prediction_finished(result)

        except Exception as e:
            self.on_prediction_error(str(e))

    def run_optimization(self):
        """Ejecutar optimización de parámetros CON VARIABLES EXÓGENAS"""
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
            self.view.log_message(f"Se evaluarán TODAS las transformaciones disponibles")
            self.view.log_message(f"Transformaciones: {', '.join(self.optimization_service.AVAILABLE_TRANSFORMATIONS)}")
            
            # NUEVO: Obtener datos climáticos si están disponibles
            if self.are_climate_data_available(regional_code):
                climate_data = self.get_climate_data_for_regional(regional_code)
                self.view.log_message(f"✓ Datos climáticos disponibles - Se incluirán en la optimización")
            else:
                self.view.log_message(f" Sin datos climáticos - Optimización sin variables exógenas")
        
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
            
            # MODIFICADO: Pasar climate_data al thread
            self.optimization_thread = OptimizationThread(
                optimization_service=self.optimization_service,
                df_prepared=df_prepared,
                regional_code=regional_code,
                climate_data=climate_data  # NUEVO
            )
            self.optimization_thread.progress_updated.connect(self.view.update_progress)
            self.optimization_thread.message_logged.connect(self.view.log_message)
            self.optimization_thread.iteration_logged.connect(self.view.log_message)
            self.optimization_thread.finished.connect(self.on_optimization_finished)
            self.optimization_thread.error_occurred.connect(self.on_optimization_error)
            
            self.view.show_progress(True)
            self.optimization_thread.start()
            
        except Exception as e:
            self.view.log_error(f"Error iniciando optimización: {str(e)}")
            self.view.set_buttons_enabled(True)
            self.view.show_progress(False)
    
    def run_validation(self):
        """Ejecutar validación del modelo CON VARIABLES EXÓGENAS, SIMULACIÓN E INTERVALOS"""
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
            
            transformation = self.validation_service.REGIONAL_TRANSFORMATIONS.get(
                regional_code, 'original'
            )
            
            self.view.log_message(f"Ejecutando validación para: {regional_nombre}")
            self.view.log_message(f"Transformación asignada: {transformation.upper()}")
            
            # Obtener datos climáticos si están disponibles
            if self.are_climate_data_available(regional_code):
                climate_data = self.get_climate_data_for_regional(regional_code)
                self.view.log_message(f"✓ Datos climáticos disponibles para {regional_nombre}")
                
                # NUEVO: Si la simulación está habilitada, abrir diálogo
                if getattr(self.view, 'enable_simulation_checkbox', None) and \
                self.view.enable_simulation_checkbox.isChecked():
                    
                    # Obtener mes de validación (último mes histórico)
                    df_prepared = self.model.get_excel_data_for_analysis()
                    if df_prepared is not None and 'Fecha' in df_prepared.columns:
                        ultimo_mes = pd.to_datetime(df_prepared['Fecha'].iloc[-1]).month
                    else:
                        ultimo_mes = datetime.now().month
                    
                    self.view.log_message("🌦️ Simulación climática habilitada para validación")
                    self.view.log_message("   Abriendo configurador de escenarios...")
                    
                    # Abrir diálogo de configuración
                    dialog = ClimateSimulationDialog(
                        climate_data=climate_data,
                        mes_prediccion=ultimo_mes,
                        regional_code=regional_code,
                        regional_nombre=regional_nombre,
                        parent=self.view
                    )
                    
                    # Conectar señales
                    dialog.simulation_accepted.connect(
                        lambda cfg: self._execute_validation(regional_code, climate_data, cfg)
                    )
                    dialog.simulation_cancelled.connect(
                        lambda: self._execute_validation(regional_code, climate_data, {'enabled': False})
                    )
                    
                    dialog.exec()
                    return
            else:
                self.view.log_message(f"⚠ Sin datos climáticos para {regional_nombre}")
        
        # Si llegamos aquí, ejecutar validación normal (sin simulación)
        self._execute_validation(regional_code, climate_data, None)

    def _execute_validation(self, regional_code, climate_data, simulation_config):
        """
        Ejecutar validación con o sin simulación
        NUEVO MÉTODO - Agregar después de run_validation()
        """
        try:
            self.view.set_buttons_enabled(False)
            self.view.update_status("Validando modelo SARIMAX...")
            
            df_prepared = self.model.get_excel_data_for_analysis()
            
            if df_prepared is None:
                self.show_error("No se pudieron preparar los datos. Verifique la regional seleccionada.")
                self.view.set_buttons_enabled(True)
                return
            
            # Log según tipo de validación
            if simulation_config and simulation_config.get('enabled', False):
                summary = simulation_config.get('summary', {})
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
                simulation_config=simulation_config  # NUEVO PARÁMETRO
            )
            
            self.validation_thread.progress_updated.connect(self.view.update_progress)
            self.validation_thread.message_logged.connect(self.view.log_message)
            self.validation_thread.finished.connect(self.on_validation_finished)
            self.validation_thread.error_occurred.connect(self.on_validation_error)
            
            self.view.show_progress(True)
            self.validation_thread.start()
            
        except Exception as e:
            self.view.log_error(f"Error ejecutando validación: {str(e)}")
            self.view.set_buttons_enabled(True)
            self.view.show_progress(False)
    
    def run_overfitting_detection(self):
        """Ejecutar detección de overfitting CON VARIABLES EXÓGENAS (sin simulación)"""
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
            
            transformation = self.overfitting_service.REGIONAL_TRANSFORMATIONS.get(
                regional_code, 'original'
            )
            
            self.view.log_message(f"Ejecutando detección de overfitting para: {regional_nombre}")
            self.view.log_message(f"Transformación asignada: {transformation.upper()}")
            
            # NUEVO: Obtener datos climáticos si están disponibles
            if self.are_climate_data_available(regional_code):
                climate_data = self.get_climate_data_for_regional(regional_code)
                self.view.log_message(f"✓ Datos climáticos disponibles - Se incluirán en el análisis")
            else:
                self.view.log_message(f"⚠ Sin datos climáticos - Análisis sin variables exógenas")
        
        try:
            self.view.log_message("Iniciando detección de overfitting...")
            self.view.log_message("División: 70% Training, 15% Validation, 15% Test")
            self.view.set_buttons_enabled(False)
            self.view.update_status("Analizando overfitting del modelo...")
            
            df_prepared = self.model.get_excel_data_for_analysis()
            
            if df_prepared is None:
                self.show_error("No se pudieron preparar los datos. Verifique la regional seleccionada.")
                self.view.set_buttons_enabled(True)
                return
            
            # MODIFICADO: Pasar climate_data al thread
            self.overfitting_thread = OverfittingThread(
                overfitting_service=self.overfitting_service,
                df_prepared=df_prepared,
                regional_code=regional_code,
                climate_data=climate_data  # NUEVO PARÁMETRO
            )
            self.overfitting_thread.progress_updated.connect(self.view.update_progress)
            self.overfitting_thread.message_logged.connect(self.view.log_message)
            self.overfitting_thread.finished.connect(self.on_overfitting_finished)
            self.overfitting_thread.error_occurred.connect(self.on_overfitting_error)
            
            self.view.show_progress(True)
            self.overfitting_thread.start()
            
        except Exception as e:
            self.view.log_error(f"Error iniciando detección de overfitting: {str(e)}")
            self.view.set_buttons_enabled(True)
            self.view.show_progress(False)
    
    def on_prediction_finished(self, result):
        """Callback cuando termina la predicción - ACTUALIZADO CON EXPORTACIÓN"""
        
        self.last_prediction_result = result 
        
        try:
            # Restaurar UI
            try:
                self.view.set_buttons_enabled(True)
                self.view.show_progress(False)
                self.view.update_status("Predicción completada")
            except Exception:
                pass

            self.view.log_success("Predicción SAIDI completada exitosamente")

            if not result:
                self.view.log_error("Resultado vacío de la predicción")
                return

            preds = result.get('predictions')
            
            if preds:
                self.view.enable_export_button(True)
                self.view.log_message(f"✓ Botón de exportación habilitado ({len(preds)} predicciones disponibles)")
                
                self.view.log_message(f"Se generaron {len(preds)} predicciones")
                
                # Logging detallado de predicciones
                for fecha, entry in preds.items():
                    if isinstance(entry, dict):
                        valor = entry.get('valor_predicho', None)
                        inferior = entry.get('limite_inferior', None)
                        superior = entry.get('limite_superior', None)
                        margen = entry.get('margen_error', None)

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
                                        f"(+{margen_sup:.2f}/-{margen_inf:.2f} | ±{margen_pct:.0f}%)"
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
                plot_path = result.get('plot_file') if isinstance(result, dict) else None
                if plot_path:
                    self.show_plot(plot_path, "Predicción SAIDI")
                
                self.view.log_message("")
                self.view.log_message("═" * 60)
                self.view.log_message("Para guardar estas predicciones en Excel, use el botón 'Exportar a Excel'")
                self.view.log_message("═" * 60)

        except Exception as e:
            # Manejo seguro de errores en el callback
            try:
                self.view.log_error(f"Error en on_prediction_finished: {str(e)}")
            except Exception:
                pass
            print(f"[DEBUG] on_prediction_finished error: {e}")

    def on_optimization_finished(self, result):
        """Callback cuando termina la optimización"""
        self.view.set_buttons_enabled(True)
        self.view.show_progress(False)
        self.view.update_status("Optimización completada")
        self.view.log_success("Optimización de parámetros completada")
        
        if result and 'top_models' in result:
            self.view.log_message(f"Se evaluaron {result.get('total_evaluated', 0)} modelos")
            
            if result.get('transformation_stats'):
                self.view.log_message("\n ESTADÍSTICAS POR TRANSFORMACIÓN:")
                for transform, stats in result['transformation_stats'].items():
                    self.view.log_message(
                        f"  {transform.upper():12s} | Modelos válidos: {stats['count']:4d} | "
                        f"Mejor precisión: {stats['best_precision']:.1f}%"
                    )
            
            self.view.log_message("\n Top 5 mejores modelos (todas las transformaciones):")
            for i, model in enumerate(result['top_models'][:5], 1):
                precision = model.get('precision_final', 0)
                order = model.get('order', 'N/A')
                seasonal_order = model.get('seasonal_order', 'N/A')
                rmse = model.get('rmse', 0)
                transformation = model.get('transformation', 'unknown')
                
                medal = "1" if i == 1 else "2" if i == 2 else "3" if i == 3 else f"#{i}"
                self.view.log_message(
                    f"  {medal} [{transformation.upper():8s}] Precisión: {precision:.1f}% | RMSE: {rmse:.4f} | "
                    f"order={order}, seasonal={seasonal_order}"
                )
            
            if result['top_models']:
                best = result['top_models'][0]
                precision = best['precision_final']
                best_transformation = best.get('transformation', 'unknown')
                
                self.view.log_message(f"\nN1 MODELO ÓPTIMO SELECCIONADO:")
                self.view.log_message(f"  Transformación: {best_transformation.upper()}")
                self.view.log_message(f"  Parámetros: order={best['order']}, seasonal={best['seasonal_order']}")
                
                if precision >= 90:
                    interpretacion = "EXCELENTE - Predicciones muy confiables "
                elif precision >= 80:
                    interpretacion = "BUENO - Predicciones confiables "
                elif precision >= 70:
                    interpretacion = "ACEPTABLE - Predicciones moderadamente confiables "
                else:
                    interpretacion = "REGULAR - Usar con precaución "
                
                self.view.log_message(f"  Interpretación: {interpretacion}")
                self.view.log_message("  Métricas calculadas en escala original")
                
                self.view.log_message("\n MEJOR MODELO POR TRANSFORMACIÓN:")
                transformations_shown = set()
                for model in result['top_models'][:20]:
                    trans = model.get('transformation', 'unknown')
                    if trans not in transformations_shown:
                        transformations_shown.add(trans)
                        self.view.log_message(
                            f"  {trans.upper():12s} | Precisión: {model['precision_final']:.1f}% | "
                            f"RMSE: {model['rmse']:.4f}"
                        )
                    if len(transformations_shown) >= 5:
                        break
    
    def on_validation_finished(self, result):
        """Callback cuando termina la validacion"""
        self.view.set_buttons_enabled(True)
        self.view.show_progress(False)
        self.view.update_status("Validacion completada")
        
        # NUEVO: Detectar si hubo simulación
        model_params = result.get('model_params', {}) if result else {}
        simulation_applied = model_params.get('with_simulation', False)
        
        if simulation_applied:
            self.view.log_success("=" * 60)
            self.view.log_success("VALIDACIÓN CON SIMULACIÓN CLIMÁTICA COMPLETADA")
            self.view.log_success("=" * 60)
            
            # Mostrar resumen de simulación
            sim_config = result.get('simulation_config', {}) if result else {}
            if sim_config:
                summary = sim_config.get('summary', {})
                self.view.log_message(f"📊 Escenario simulado: {summary.get('escenario', 'N/A')}")
                self.view.log_message(f"📅 Alcance: {summary.get('alcance_meses', 'N/A')} meses")
                self.view.log_message(f"🌡️ Días simulados: {summary.get('dias_simulados', 'N/A')}")
                
                # Mostrar cambios en variables
                changes = summary.get('percentage_changes', {})
                if changes:
                    self.view.log_message("\n🔄 Cambios aplicados a variables:")
                    var_names = {
                        'temp_max': 'Temperatura máxima',
                        'humedad_avg': 'Humedad relativa',
                        'precip_total': 'Precipitación total'
                    }
                    for var, change_pct in changes.items():
                        var_name = var_names.get(var, var)
                        arrow = "↑" if change_pct > 0 else "↓" if change_pct < 0 else "→"
                        self.view.log_message(f"   {arrow} {var_name}: {change_pct:+.1f}%")
                
                self.view.log_message("")
                self.view.log_message("⚠️  IMPORTANTE:")
                self.view.log_message("   Las métricas reflejan el desempeño del modelo bajo condiciones")
                self.view.log_message("   climáticas HIPOTÉTICAS del escenario simulado.")
                self.view.log_message("   Los resultados reales dependerán de las condiciones climáticas efectivas.")
                self.view.log_message("")
        else:
            self.view.log_success("Validación del modelo completada")
        
        if result and 'metrics' in result:
            metrics = result['metrics']
            model_params = result.get('model_params', {})
            
            self.view.log_message("=" * 60)
            self.view.log_message("RESULTADOS DE VALIDACION")
            self.view.log_message("=" * 60)
            
            # Informacion del modelo
            self.view.log_message(f"Transformacion: {model_params.get('transformation', 'N/A').upper()}")
            self.view.log_message(f"Parametros: order={model_params.get('order')}, seasonal={model_params.get('seasonal_order')}")
            
            if model_params.get('with_exogenous'):
                exog_info = result.get('exogenous_vars', {})
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
            
            # Informacion de validacion
            self.view.log_message("")
            self.view.log_message(f"Datos de entrenamiento: {result.get('training_count', 0)} observaciones")
            self.view.log_message(f"Datos de validacion: {result.get('validation_count', 0)} observaciones")
            self.view.log_message(f"Porcentaje validacion: {result.get('validation_percentage', 0):.0f}%")
            
            # Predicciones con intervalos (SOLO para referencia visual)
            if 'predictions' in result:
                predictions = result['predictions']
                mean_preds = predictions.get('mean', {})
                lower_bounds = predictions.get('lower_bound', {})
                upper_bounds = predictions.get('upper_bound', {})
                
                if mean_preds and lower_bounds and upper_bounds:
                    self.view.log_message("")
                    self.view.log_message("Predicciones de validacion:")
                    self.view.log_message("(Intervalos de confianza 95% solo en grafica)")
                    
                    for fecha in sorted(mean_preds.keys()):
                        mean_val = mean_preds[fecha]
                        lower_val = lower_bounds.get(fecha, mean_val)
                        upper_val = upper_bounds.get(fecha, mean_val)
                        
                        # Mostrar valor predicho y rango del intervalo (sin margenes individuales)
                        ancho_intervalo = upper_val - lower_val
                        
                        self.view.log_message(
                            f"  - {fecha}: {mean_val:.2f} min "
                            f"[IC: {lower_val:.2f} - {upper_val:.2f}]"
                        )
            
            # Interpretacion de calidad basada en PRECISION
            precision = metrics.get('precision_final', 0)
            self.view.log_message("")
            if precision >= 90:
                self.view.log_success("Calidad: EXCELENTE - Predicciones muy confiables")
            elif precision >= 80:
                self.view.log_success("Calidad: BUENO - Predicciones confiables")
            elif precision >= 70:
                self.view.log_message("Calidad: ACEPTABLE - Predicciones moderadamente confiables")
            elif precision >= 60:
                self.view.log_message("Calidad: REGULAR - Usar con precaución")
            else:
                self.view.log_error("Calidad: BAJO - Modelo poco confiable")
            
            # Nota sobre intervalos (solo para referencia)
            if model_params.get('confidence_level'):
                conf_level = model_params['confidence_level'] * 100
                self.view.log_message(f"")
                self.view.log_message(f"Nota: Intervalos de confianza al {conf_level:.0f}% disponibles en grafica")
                self.view.log_message(f"      (solo para referencia visual, no afectan la precision del modelo)")
            
            self.view.log_message("=" * 60)
        
        # Mostrar grafica
        if result and 'plot_file' in result and result['plot_file']:
            self.show_plot(result['plot_file'], "Validacion del Modelo SAIDI")
    
    def on_overfitting_finished(self, result):
        """Callback cuando termina detección de overfitting - MODERNIZADO"""
        self.view.set_buttons_enabled(True)
        self.view.show_progress(False)
        self.view.update_status("Análisis de overfitting completado")
        self.view.log_success("Detección de overfitting completada")
        
        if result and 'overfitting_analysis' in result:
            analysis = result['overfitting_analysis']
            model_params = result.get('model_params', {})
            
            self.view.log_message("=" * 60)
            self.view.log_message("ANÁLISIS DE OVERFITTING")
            self.view.log_message("=" * 60)
            
            # Información del modelo
            self.view.log_message(f"Transformación: {model_params.get('transformation', 'N/A').upper()}")
            self.view.log_message(f"Parámetros: order={model_params.get('order')}, seasonal={model_params.get('seasonal_order')}")
            
            if model_params.get('with_exogenous'):
                exog_info = result.get('exogenous_vars', {})
                self.view.log_message(f"Variables exógenas: {len(exog_info)}")
                for var_code, var_data in exog_info.items():
                    self.view.log_message(f"  • {var_data['nombre']}")
            
            self.view.log_message("")
            self.view.log_message(f"Estado: {analysis['status']}")
            self.view.log_message(f"Nivel de Overfitting: {analysis['overfitting_level']}")
            self.view.log_message(f"Score: {analysis['overfitting_score']:.2f}/100")
            
            # Mostrar degradaciones
            self.view.log_message("")
            self.view.log_message("DEGRADACIONES Train→Test:")
            self.view.log_message(f"  • Precisión: {analysis['precision_degradation']:.1f}%")
            self.view.log_message(f"  • R²: {analysis['r2_degradation']:.1f}%")
            self.view.log_message(f"  • RMSE: {analysis['rmse_increase']:.1f}%")
            
            # Métricas por conjunto
            metrics = result.get('metrics', {})
            self.view.log_message("")
            self.view.log_message("MÉTRICAS POR CONJUNTO:")
            
            for set_name, set_label in [('train', 'TRAINING'), ('validation', 'VALIDATION'), ('test', 'TEST')]:
                if set_name in metrics:
                    m = metrics[set_name]
                    self.view.log_message(f"\n{set_label}:")
                    self.view.log_message(f"  • Precisión: {m['precision_final']:.1f}%")
                    self.view.log_message(f"  • RMSE: {m['rmse']:.4f} min")
                    self.view.log_message(f"  • R²: {m['r2']:.4f}")
            
            self.view.log_message("")
            
            if analysis.get('has_overfitting', False):
                self.view.log_message("⚠️ OVERFITTING DETECTADO")
                self.view.log_message("Recomendaciones:")
                for rec in analysis['recommendations']:
                    self.view.log_message(f"  • {rec}")
            else:
                self.view.log_success("✅ NO SE DETECTÓ OVERFITTING SIGNIFICATIVO")
                self.view.log_success("El modelo generaliza adecuadamente")
            
            self.view.log_message("=" * 60)
        
        if result and 'plot_file' in result and result['plot_file']:
            self.show_plot(result['plot_file'], "Análisis de Overfitting")
    
    def on_prediction_error(self, error_msg):
        self.view.set_buttons_enabled(True)
        self.view.show_progress(False)
        self.view.update_status("Error en predicción")
        self.view.log_error(f"Error en predicción: {error_msg}")
        self.show_error(f"Error durante la predicción: {error_msg}")

    def on_optimization_error(self, error_msg):
        """Callback cuando hay error en optimización"""
        self.view.set_buttons_enabled(True)
        self.view.show_progress(False)
        self.view.update_status("Error en optimización")
        self.view.log_error(f"Error en optimización: {error_msg}")
        self.show_error(f"Error durante la optimización: {error_msg}")
    
    def on_validation_error(self, error_msg):
        """Callback cuando hay error en validación"""
        self.view.set_buttons_enabled(True)
        self.view.show_progress(False)
        self.view.update_status("Error en validacion")
        self.view.log_error(f"Error en validacion: {error_msg}")
        self.show_error(f"Error durante la validacion: {error_msg}")
    
    def on_overfitting_error(self, error_msg):
        """Callback cuando hay error en detección de overfitting"""
        self.view.set_buttons_enabled(True)
        self.view.show_progress(False)
        self.view.update_status("Error en análisis de overfitting")
        self.view.log_error(f"Error en overfitting: {error_msg}")
        self.show_error(f"Error durante la detección de overfitting: {error_msg}")
    
    def show_error(self, message):
        """Mostrar mensaje de error"""
        QMessageBox.critical(self.view, "Error", message)

    def show_warning(self, message):
        """Mostrar mensaje de advertencia"""
        QMessageBox.warning(self.view, "Advertencia", message)

    def show_info(self, message):
        """Mostrar mensaje informativo"""
        QMessageBox.information(self.view, "Información", message)


class TransformationThread(QThread):
    """Hilo para comparar transformaciones en background"""
    
    progress_updated = pyqtSignal(int, str)
    message_logged = pyqtSignal(str)
    finished = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, improved_service, order, seasonal_order, 
                 file_path=None, df_prepared=None, regional_code=None):
        super().__init__()
        self.improved_service = improved_service
        self.order = order if order else (3, 1, 3)
        self.seasonal_order = seasonal_order if seasonal_order else (3, 1, 3, 12)
        self.file_path = file_path
        self.df_prepared = df_prepared
        self.regional_code = regional_code
    
    def run(self):
        try:
            self.message_logged.emit("Comparando transformaciones de datos...")
            result = self.improved_service.compare_transformations(
                file_path=self.file_path,
                df_prepared=self.df_prepared,
                order=self.order,
                seasonal_order=self.seasonal_order,
                regional_code=self.regional_code,
                progress_callback=self.progress_updated.emit,
                log_callback=self.message_logged.emit
            )
            
            if result:
                self.message_logged.emit("Comparación de transformaciones completada")
                self.finished.emit(result)
            else:
                raise Exception("No se obtuvieron resultados de la comparación")
                
        except Exception as e:
            self.error_occurred.emit(str(e))


class SimpleModelThread(QThread):
    """Hilo para buscar modelo simple óptimo en background"""
    
    progress_updated = pyqtSignal(int, str)
    message_logged = pyqtSignal(str)
    finished = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, improved_service, file_path=None, df_prepared=None, regional_code=None):
        super().__init__()
        self.improved_service = improved_service
        self.file_path = file_path
        self.df_prepared = df_prepared
        self.regional_code = regional_code
    
    def run(self):
        try:
            self.message_logged.emit("Buscando modelo simple óptimo...")
            result = self.improved_service.find_best_simple_model(
                file_path=self.file_path,
                df_prepared=self.df_prepared,
                regional_code=self.regional_code,
                progress_callback=self.progress_updated.emit,
                log_callback=self.message_logged.emit
            )
            self.finished.emit(result)
        except Exception as e:
            self.error_occurred.emit(str(e))


class RegularizationThread(QThread):
    """Hilo para ejecutar regularización en background"""
    
    progress_updated = pyqtSignal(int, str)
    message_logged = pyqtSignal(str)
    finished = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, file_path, service, order, seasonal_order, 
                 exog_columns, method, alpha, l1_ratio):
        super().__init__()
        self.file_path = file_path
        self.service = service
        self.order = order
        self.seasonal_order = seasonal_order
        self.exog_columns = exog_columns
        self.method = method
        self.alpha = alpha
        self.l1_ratio = l1_ratio
    
    def run(self):
        try:
            self.message_logged.emit("Ejecutando regularización...")
            result = self.service.fit_with_regularization(
                self.file_path,
                self.order,
                self.seasonal_order,
                self.exog_columns,
                self.method,
                self.alpha,
                self.l1_ratio,
                progress_callback=self.progress_updated.emit,
                log_callback=self.message_logged.emit
            )
            self.finished.emit(result)
        except Exception as e:
            self.error_occurred.emit(str(e))


class CrossValidationThread(QThread):
    """Hilo para ejecutar cross-validation en background"""
    
    progress_updated = pyqtSignal(int, str)
    message_logged = pyqtSignal(str)
    finished = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, improved_service, order, seasonal_order, n_splits, 
                 file_path=None, df_prepared=None, regional_code=None):
        super().__init__()
        self.improved_service = improved_service
        self.order = order
        self.seasonal_order = seasonal_order
        self.n_splits = n_splits
        self.file_path = file_path
        self.df_prepared = df_prepared
        self.regional_code = regional_code
    
    def run(self):
        try:
            self.message_logged.emit("Ejecutando cross-validation temporal...")
            result = self.improved_service.run_cross_validation(
                file_path=self.file_path,
                df_prepared=self.df_prepared,
                order=self.order,
                seasonal_order=self.seasonal_order,
                n_splits=self.n_splits,
                regional_code=self.regional_code,
                progress_callback=self.progress_updated.emit,
                log_callback=self.message_logged.emit
            )
            self.finished.emit(result)
        except Exception as e:
            self.error_occurred.emit(str(e))


class PredictionThread(QThread):
    """Hilo para ejecutar predicción en background"""
    
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
                log_callback=self.message_logged.emit
            )
            self.finished.emit(result)
        except Exception as e:
            self.error_occurred.emit(str(e))


class OptimizationThread(QThread):
    """Hilo para ejecutar optimización en background"""
    
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
        self.climate_data = climate_data  
    
    def run(self):
        try:
            self.message_logged.emit("Ejecutando optimización de parámetros...")
            result = self.optimization_service.run_optimization(
                file_path=self.file_path,
                df_prepared=self.df_prepared,
                regional_code=self.regional_code,
                climate_data=self.climate_data, 
                progress_callback=self.progress_updated.emit,
                log_callback=self.message_logged.emit,
                iteration_callback=self.iteration_logged.emit
            )
            self.finished.emit(result)
        except Exception as e:
            self.error_occurred.emit(str(e))


class ValidationThread(QThread):
    """Hilo para ejecutar validación en background"""
    
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
                log_callback=self.message_logged.emit
            )
            self.finished.emit(result)
        except Exception as e:
            self.error_occurred.emit(str(e))


class OverfittingThread(QThread):
    """Hilo para ejecutar detección de overfitting en background"""
    
    progress_updated = pyqtSignal(int, str)
    message_logged = pyqtSignal(str)
    finished = pyqtSignal(dict)
   
    error_occurred = pyqtSignal(str)
    
    def __init__(self, overfitting_service, file_path=None, df_prepared=None, 
                 regional_code=None, climate_data=None):  
        super().__init__()
        self.overfitting_service = overfitting_service
        self.file_path = file_path
        self.df_prepared = df_prepared
        self.regional_code = regional_code
        self.climate_data = climate_data 
    
    def run(self):
        try:
            self.message_logged.emit("Ejecutando detección de overfitting...")
            result = self.overfitting_service.run_overfitting_detection(
                file_path=self.file_path,
                df_prepared=self.df_prepared,
                regional_code=self.regional_code,
                climate_data=self.climate_data, 
                progress_callback=self.progress_updated.emit,
                log_callback=self.message_logged.emit
            )
            self.finished.emit(result)
        except Exception as e:
            self.error_occurred.emit(str(e))
