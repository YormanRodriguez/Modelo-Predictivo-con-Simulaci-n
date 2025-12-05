# services/optimization_service.py
import gc
import json
import logging
import multiprocessing as mp
import traceback
import warnings
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import linregress
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

class OptimizationError(Exception):
    """Excepción personalizada para errores durante la optimización de modelos SARIMAX."""

@dataclass
class OptimizationCallbacks:
    """Contenedor para callbacks de optimización."""

    progress: Callable[[int, str], None] | None = None
    iteration: Callable | None = None
    log: Callable[[str], None] | None = None

@dataclass
class ClimateProjectionContext:
    """Contexto para proyección de variables climáticas."""

    forecast_dates: pd.DatetimeIndex
    hist_start: pd.Timestamp
    hist_end: pd.Timestamp
    log_callback: Callable | None = None

class OptimizationService:
    """
    Servicio de optimización de parámetros SARIMAX.

    Funcionalidades:
    - Evaluación exhaustiva de combinaciones de parámetros
    - Soporte para múltiples transformaciones de datos
    - Procesamiento paralelo para eficiencia
    - Integración con variables exógenas climáticas
    - Validación consistente con PredictionService
    """

    # Transformaciones disponibles para evaluar
    AVAILABLE_TRANSFORMATIONS: ClassVar[list[str]] = ["original", "log", "boxcox", "standard", "sqrt"]

    # Máximo de modelos a retornar
    MAX_TOP_MODELS = 50

    # Variables exógenas por regional
    REGIONAL_EXOG_VARS: ClassVar[dict[str, dict[str, str]]] = {
        "SAIDI_O": {
            "realfeel_min": "Temperatura aparente mínima",
            "windchill_avg": "Sensación térmica promedio",
            "dewpoint_avg": "Punto de rocío promedio",
            "windchill_max": "Sensación térmica máxima",
            "dewpoint_min": "Punto de rocío mínimo",
            "precipitation_max_daily": "Precipitación máxima diaria",
            "precipitation_avg_daily": "Precipitación promedio diaria",
        },
        "SAIDI_C": {
            "realfeel_avg": "Temperatura aparente promedio",
            "pressure_rel_avg": "Presión relativa promedio",
            "wind_speed_max": "Velocidad máxima del viento",
            "pressure_abs_avg": "Presión absoluta promedio",
        },
        "SAIDI_T": {
            "realfeel_avg": "Temperatura aparente promedio",
            "wind_dir_avg": "Dirección promedio del viento",
            "uv_index_avg": "Índice UV promedio",
            "heat_index_avg": "Índice de calor promedio",
            "temperature_min": "Temperatura mínima",
            "windchill_min": "Sensación térmica mínima",
            "temperature_avg": "Temperatura promedio",
            "pressure_rel_avg": "Presión relativa promedio",
        },
        "SAIDI_A": {
            "uv_index_max": "Índice UV máximo",
            "days_with_rain": "Días con lluvia",
        },
        "SAIDI_P": {
            "precipitation_total": "Precipitación total",
            "precipitation_avg_daily": "Precipitación promedio diaria",
            "realfeel_min": "Temperatura aparente mínima",
        },
    }

    def __init__(self):
        """Inicializar servicio de optimización."""
        # Parámetros por defecto
        self.default_order = (3, 0, 3)
        self.default_seasonal_order = (3, 1, 3, 12)

        # Almacenamiento de resultados
        self.all_models = []  # Lista simple en lugar de heap
        self.best_by_transformation = {}  # Mejor modelo por cada transformación

        # Control de progreso
        self.total_iterations = 0
        self.current_iteration = 0

        # Mejores métricas globales
        self.best_precision = 0.0
        self.best_rmse = float("inf")

        # Transformación y escalado
        self.scaler = None
        self.exog_scaler = None
        self.transformation_params = {}

        # Estadísticas de transformaciones
        self.transformation_stats = {}

        # DEBUG: Contador para verificar modelos procesados
        self._debug_models_evaluated = 0
        self._debug_models_added = 0

    def run_optimization(self,
                    file_path: str | None = None,
                    df_prepared: pd.DataFrame | None = None,
                    regional_code: str | None = None,
                    climate_data: pd.DataFrame | None = None,
                    progress_callback = None,
                    log_callback = None,
                    iteration_callback = None,
                    *,
                    use_parallel: bool = True,
                    max_workers: int | None = None) -> dict[str, Any]:
        """
        Ejecutar optimización de parámetros SARIMAX.

        Args:
            file_path: Ruta del archivo Excel SAIDI (opcional)
            df_prepared: DataFrame SAIDI preparado (opcional)
            regional_code: Código de la regional
            climate_data: DataFrame con datos climáticos mensuales
            progress_callback: Función callback para actualizar progreso
            log_callback: Función callback para logging en UI
            iteration_callback: Función callback para actualizar iteración actual
            use_parallel: Usar procesamiento paralelo (keyword-only)
            max_workers: Número de workers paralelos (None = automático)

        Returns:
            Diccionario con resultados de optimización

        """
        try:
            # Reiniciar estado
            self._reset_state()

            # Logging inicial
            if log_callback:
                log_callback("=" * 80)
                log_callback("OPTIMIZACION DE PARAMETROS SARIMAX")
                log_callback(f"Regional: {regional_code or 'No especificada'}")
                log_callback(f"CPU Cores disponibles: {mp.cpu_count()}")
                log_callback("=" * 80)

            print(f"[DEBUG_OPT] Iniciando optimizacion para regional: {regional_code}")

            if progress_callback:
                progress_callback(5, "Cargando datos SAIDI...")

            # Paso 1: Cargar y validar datos SAIDI
            _, col_saidi, historico = self._load_and_validate_data(
                file_path, df_prepared, log_callback,
            )

            print(f"[DEBUG_OPT] Datos SAIDI cargados: {len(historico)} observaciones")
            print(f"[DEBUG_OPT] Periodo: {historico.index[0]} a {historico.index[-1]}")

            if log_callback:
                log_callback(f"Datos historicos: {len(historico)} observaciones")
                log_callback(f"Periodo: {historico.index[0].strftime('%Y-%m')} a {historico.index[-1].strftime('%Y-%m')}")

            if progress_callback:
                progress_callback(10, "Preparando variables exogenas...")

            # Paso 2: Preparar variables exógenas (si están disponibles)
            exog_df, exog_info, _ = self._prepare_exogenous_adaptive(
                climate_data, regional_code, historico, log_callback,
            )

            # Diagnóstico: Validar cobertura ANTES de continuar
            if exog_df is not None:
                print("[DEBUG_OPT] Ejecutando diagnóstico de cobertura exógena...")

                if not self.diagnose_exog_coverage(historico[col_saidi], exog_df, log_callback):
                    print("[DEBUG_OPT] ⚠ Cobertura exógena insuficiente - Desactivando variables")
                    if log_callback:
                        log_callback("ADVERTENCIA: Variables exógenas desactivadas por cobertura insuficiente")
                        log_callback("La optimización continuará SIN variables climáticas")
                    exog_df = None
                    exog_info = None
                else:
                    print("[DEBUG_OPT] ✓ Diagnóstico de cobertura exógena OK")

            # Log final del estado de exógenas
            if exog_df is not None:
                print(f"[DEBUG_OPT] ✓ Variables exógenas activas: {len(exog_df.columns)} variables")
                if log_callback:
                    log_callback(f"Variables exogenas disponibles: {len(exog_df.columns)}")
            else:
                print("[DEBUG_OPT] Optimización SIN variables exógenas")
                if log_callback:
                    log_callback("Optimizacion sin variables exogenas")

            if progress_callback:
                progress_callback(15, "Configurando espacio de busqueda...")

            # Paso 3: Configurar rangos de parámetros
            param_combinations = self._configure_parameter_space(log_callback)

            # Calcular total de evaluaciones
            self.total_iterations = len(param_combinations) * len(self.AVAILABLE_TRANSFORMATIONS)

            print(f"[DEBUG_OPT] Combinaciones de parametros: {len(param_combinations)}")
            print(f"[DEBUG_OPT] Transformaciones: {len(self.AVAILABLE_TRANSFORMATIONS)}")
            print(f"[DEBUG_OPT] Total evaluaciones: {self.total_iterations}")

            if log_callback:
                log_callback("=" * 80)
                log_callback(f"Combinaciones de parametros: {len(param_combinations)}")
                log_callback(f"Transformaciones: {len(self.AVAILABLE_TRANSFORMATIONS)}")
                log_callback(f"Total evaluaciones: {self.total_iterations}")
                log_callback(f"Metodo: {'Paralelo' if use_parallel else 'Secuencial'}")
                log_callback("Validacion: Consistente con PredictionService (20-30% test)")
                log_callback("=" * 80)

            if progress_callback:
                progress_callback(20, f"Iniciando evaluacion de {self.total_iterations} modelos")

            # Paso 4: Ejecutar evaluación (paralela o secuencial)
            combinacion_de_parametros = 50
            if use_parallel and len(param_combinations) > combinacion_de_parametros:
                print("[DEBUG_OPT] Usando procesamiento PARALELO")
                callbacks = OptimizationCallbacks(
                    progress=progress_callback,
                    iteration=iteration_callback,
                    log=log_callback,
                )
                self._run_parallel_optimization(
                    historico[col_saidi],
                    param_combinations,
                    exog_df,
                    callbacks,
                    max_workers,
                )
            else:
                print("[DEBUG_OPT] Usando procesamiento SECUENCIAL")
                self._run_sequential_optimization(
                    historico[col_saidi], param_combinations, exog_df,
                    progress_callback, iteration_callback, log_callback,
                )

            print(f"[DEBUG_OPT] Evaluacion completada: {self._debug_models_evaluated} modelos evaluados")
            print(f"[DEBUG_OPT] Modelos agregados: {self._debug_models_added}")

            if progress_callback:
                progress_callback(90, "Seleccionando mejores modelos...")

            # Paso 5: Seleccionar y clasificar mejores modelos
            top_models, quality_level, quality_counts = self._select_best_models()

            print(f"[DEBUG_OPT] Mejores modelos seleccionados: {len(top_models)}")
            print(f"[DEBUG_OPT] Nivel de calidad: {quality_level}")

            if log_callback:
                log_callback("\n" + "=" * 80)
                log_callback(f"CALIDAD DE MODELOS ENCONTRADOS: {quality_level}")
                log_callback("=" * 80)
                log_callback(f"Excelentes (>=60%): {quality_counts['excellent']}")
                log_callback(f"Buenos (40-59%): {quality_counts['good']}")
                log_callback(f"Aceptables (20-39%): {quality_counts['acceptable']}")
                log_callback(f"Limitados (<20%): {quality_counts['poor']}")

                if quality_counts["excellent"] == 0 and quality_counts["good"] == 0:
                    log_callback("\nADVERTENCIA: Los datos son muy dificiles de predecir")
                    log_callback("Recomendacion: Revisar calidad de datos historicos")

            if progress_callback:
                progress_callback(95, "Guardando configuración óptima...")

            best_model = top_models[0] if top_models else None

            if best_model and regional_code:
                save_success = self.save_best_model_config(regional_code, best_model)

                if save_success and log_callback:
                    log_callback("\n" + "=" * 80)
                    log_callback("✓ CONFIGURACIÓN ÓPTIMA GUARDADA")
                    log_callback("=" * 80)
                    log_callback(f"Los parámetros óptimos de {regional_code} se usarán automáticamente")
                    log_callback("en futuras predicciones para esta regional.")
                    log_callback(f"Transformación: {best_model['transformation'].upper()}")
                    log_callback(f"Order: {best_model['order']}")
                    log_callback(f"Seasonal: {best_model['seasonal_order']}")
                    log_callback(f"Precisión: {best_model['precision_final']:.1f}%")

            if progress_callback:
                progress_callback(100, "Optimizacion completada")

            # Paso 6: Generar resumen final
            if log_callback:
                self._log_final_summary(log_callback, top_models, exog_info, regional_code)

        except (ValueError, KeyError, FileNotFoundError, pd.errors.EmptyDataError) as e:
            error_msg = f"Error durante optimizacion: {e!s}"
            print(f"[DEBUG_OPT_ERROR] {error_msg}")
            if log_callback:
                log_callback(f"ERROR: {error_msg}")
            raise OptimizationError(error_msg) from e
        else:
            # Preparar resultado solo si no hubo excepciones
            result = {
                "success": True,
                "top_models": top_models[:20],  # Limitar a top 20
                "total_evaluated": self.current_iteration,
                "best_model": best_model,
                "total_combinations": self.total_iterations,
                "transformation": best_model["transformation"] if best_model else None,
                "regional_code": regional_code,
                "transformation_stats": self.transformation_stats,
                "all_transformations_tested": self.AVAILABLE_TRANSFORMATIONS,
                "with_exogenous": exog_df is not None,
                "exogenous_vars": exog_info,
                "validation_method": "aligned_with_prediction",
                "optimization_method": "exhaustive_search",
                "quality_level": quality_level,
                "quality_counts": quality_counts,
                "config_saved": best_model is not None and regional_code is not None,
            }

            print("[DEBUG_OPT] Optimizacion finalizada exitosamente")

            return result

    def _reset_state(self):
        """Reiniciar estado interno del servicio."""
        print("[DEBUG_OPT] Reiniciando estado del servicio")

        self.all_models = []
        self.best_by_transformation = {}
        self.current_iteration = 0
        self.best_precision = 0.0
        self.best_rmse = float("inf")
        self.transformation_params = {}
        self.scaler = None
        self.exog_scaler = None

        # Reiniciar estadísticas de transformaciones
        self.transformation_stats = {
            t: {
                "count": 0,
                "best_precision": 0.0,
                "best_stability": 0.0,
                "best_model": None,
            }
            for t in self.AVAILABLE_TRANSFORMATIONS
        }

        # DEBUG
        self._debug_models_evaluated = 0
        self._debug_models_added = 0

        # Forzar recolección de basura
        gc.collect()


    def _load_and_validate_data(self,
                            file_path: str | None,
                            df_prepared: pd.DataFrame | None,
                            log_callback) -> tuple[pd.DataFrame, str, pd.DataFrame]:
        """
        Cargar y validar datos SAIDI.

        Returns:
            Tuple con (df_completo, nombre_columna_saidi, df_historico)

        """
        print("[DEBUG_OPT] Cargando datos SAIDI")

        # Cargar DataFrame
        df = self._load_dataframe(file_path, df_prepared, log_callback)

        # Asegurar índice datetime
        df = self._ensure_datetime_index(df)

        # Buscar y validar columna SAIDI
        col_saidi = self._find_saidi_column(df)

        # Filtrar y validar datos históricos
        historico = self._validate_historical_data(df, col_saidi)

        print(f"[DEBUG_OPT] Datos validados: {len(historico)} obs, columna: {col_saidi}")

        return df, col_saidi, historico


    def _load_dataframe(self,
                    file_path: str | None,
                    df_prepared: pd.DataFrame | None,
                    log_callback) -> pd.DataFrame:
        """Cargar DataFrame desde archivo o datos preparados."""
        if df_prepared is not None:
            if log_callback:
                log_callback("Usando datos SAIDI preparados del modelo")
            return df_prepared.copy()

        if file_path is not None:
            if log_callback:
                log_callback("Leyendo Excel SAIDI")
            return pd.read_excel(file_path, sheet_name="Hoja1")

        msg = "Debe proporcionar file_path o df_prepared"
        raise ValueError(msg)


    def _ensure_datetime_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """Asegurar que el DataFrame tenga un índice datetime."""
        if isinstance(df.index, pd.DatetimeIndex):
            return df

        if "Fecha" in df.columns:
            df["Fecha"] = pd.to_datetime(df["Fecha"])
            return df.set_index("Fecha")

        df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
        return df.set_index(df.columns[0])


    def _find_saidi_column(self, df: pd.DataFrame) -> str:
        """Buscar y validar la columna SAIDI en el DataFrame."""
        if "SAIDI" in df.columns:
            return "SAIDI"

        if "SAIDI Historico" in df.columns:
            return "SAIDI Historico"

        msg = "No se encontro la columna SAIDI en los datos"
        raise ValueError(msg)


    def _validate_historical_data(self, df: pd.DataFrame, col_saidi: str) -> pd.DataFrame:
        """Filtrar y validar datos históricos."""
        historico = df[df[col_saidi].notna()].copy()

        meses = 12
        if len(historico) < meses:
            msg = f"Datos insuficientes: solo {len(historico)} observaciones (minimo 12)"
            raise ValueError(msg)

        return historico

    def _configure_parameter_space(self, log_callback) -> list[tuple]:
        """
        Configurar espacio de búsqueda de parámetros.

        Returns:
            Lista de tuplas (p, d, q, P, D, Q, s)

        """
        print("[DEBUG_OPT] Configurando espacio de busqueda")

        # Rangos de parámetros
        #p_range = range(7)  # AR
        #d_range = range(3)  # Diferenciación
        #q_range = range(7)  # MA
        #P_range = range(6)  # AR estacional
        #D_range = range(3)  # Diferenciación estacional
        #Q_range = range(6)  # MA estacional
        s_range = [12]

        # Rangos de parámetros
        p_range = range(2)  # AR
        d_range = range(2)  # Diferenciación
        q_range = range(2)  # MA
        P_range = range(2)  # AR estacional # noqa: N806
        D_range = range(2)  # Diferenciación estacional # noqa: N806
        Q_range = range(2) # noqa: N806

        # Generar todas las combinaciones
        all_combinations = list(product(
            p_range, d_range, q_range,
            P_range, D_range, Q_range,
            s_range,
        ))

        print(f"[DEBUG_OPT] Combinaciones iniciales: {len(all_combinations)}")

        # Filtrar combinaciones inválidas
        valid_combinations = self._filter_invalid_combinations(all_combinations)

        print(f"[DEBUG_OPT] Combinaciones validas: {len(valid_combinations)}")

        if log_callback:
            log_callback(f"Combinaciones iniciales: {len(all_combinations)}")
            log_callback(f"Combinaciones validas: {len(valid_combinations)}")

        return valid_combinations

    def _filter_invalid_combinations(self, combinations: list[tuple]) -> list[tuple]:
        """
        Filtrar combinaciones de parámetros inválidas.

        Criterios:
        - Evitar modelos triviales (todos los parámetros en 0)
        - Evitar modelos extremadamente complejos
        - Debe tener al menos un componente AR o MA
        """
        valid = []

        parametros_rechazados = 14
        for p, d, q, P, D, Q, s in combinations: # noqa: N806
            # Calcular complejidad total
            total_params = p + d + q + P + D + Q

            # Rechazar si es trivial o demasiado complejo
            if total_params == 0 or total_params > parametros_rechazados:
                continue

            # Debe tener al menos un componente AR o MA
            if p == 0 and q == 0 and P == 0 and Q == 0:
                continue

            valid.append((p, d, q, P, D, Q, s))

        return valid

    def _run_parallel_optimization(self,
                               serie_original: pd.Series,
                               param_combinations: list[tuple],
                               exog_df: pd.DataFrame | None,
                               callbacks: OptimizationCallbacks,
                               max_workers: int | None):
        """Ejecutar optimización en paralelo usando ProcessPoolExecutor."""
        if max_workers is None:
            max_workers = max(1, mp.cpu_count() - 1)

        print(f"[DEBUG_OPT] Iniciando procesamiento paralelo con {max_workers} workers")

        if callbacks.log:
            callbacks.log(f"Procesamiento paralelo con {max_workers} workers")

        # Preparar tareas
        tasks = self._prepare_optimization_tasks(param_combinations, serie_original, exog_df)
        total_tasks = len(tasks)
        batch_size = 100

        print(f"[DEBUG_OPT] Total tareas: {total_tasks}, batch_size: {batch_size}")

        # Procesar en lotes
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for batch_start in range(0, total_tasks, batch_size):
                batch_end = min(batch_start + batch_size, total_tasks)
                batch_tasks = tasks[batch_start:batch_end]

                # Enviar y procesar lote
                self._process_batch(executor, batch_tasks, total_tasks, exog_df, callbacks)

                # Limpiar memoria después de cada lote
                gc.collect()

        print("[DEBUG_OPT] Procesamiento paralelo completado")


    def _prepare_optimization_tasks(self,
                                    param_combinations: list[tuple],
                                    serie_original: pd.Series,
                                    exog_df: pd.DataFrame | None) -> list[tuple]:
        """Preparar lista de tareas para optimización paralela."""
        tasks = []
        for params in param_combinations:
            p, d, q, P, D, Q, s = params  # noqa: N806
            order = (p, d, q)
            seasonal_order = (P, D, Q, s)

            for transformation in self.AVAILABLE_TRANSFORMATIONS:
                tasks.append((serie_original, order, seasonal_order, transformation, exog_df))

        return tasks


    def _process_batch(self,
                    executor: ProcessPoolExecutor,
                    batch_tasks: list[tuple],
                    total_tasks: int,
                    exog_df: pd.DataFrame | None,
                    callbacks: OptimizationCallbacks):
        """Procesar un lote de tareas en paralelo."""
        # Enviar lote al executor
        futures = {
            executor.submit(_evaluate_model_worker, task): task
            for task in batch_tasks
        }

        # Procesar resultados a medida que se completan
        for future in as_completed(futures):
            self.current_iteration += 1
            self._debug_models_evaluated += 1

            try:
                metrics = future.result(timeout=45)

                if metrics and self._is_valid_model(metrics):
                    task = futures[future]
                    self._store_model_metrics(task, metrics, exog_df)

            except TimeoutError:
                # Silenciar timeouts específicamente
                pass
            except (ValueError, KeyError, TypeError, RuntimeError) as e:
                # Capturar errores esperados durante la evaluación del modelo
                print(f"[DEBUG_OPT_WARN] Error en evaluacion: {type(e).__name__}")

            # Actualizar progreso
            self._update_progress(callbacks, total_tasks)


    def _store_model_metrics(self,
                            task: tuple,
                            metrics: dict,
                            exog_df: pd.DataFrame | None):
        """Almacenar métricas de un modelo válido."""
        transformation = task[3]
        order = task[1]
        seasonal_order = task[2]

        # Agregar métricas adicionales
        metrics["transformation"] = transformation
        metrics["order"] = order
        metrics["seasonal_order"] = seasonal_order
        metrics["with_exogenous"] = exog_df is not None

        # Almacenar modelo
        self._add_model(metrics)

        # Actualizar estadísticas
        self._update_transformation_stats(transformation, metrics)

        # Log de progreso si es relevante
        precision_mayor_60 = 60
        if metrics["precision_final"] > precision_mayor_60:
            print(f"[DEBUG_OPT] Modelo relevante encontrado: "
                f"{transformation} - Precision: {metrics['precision_final']:.1f}%")


    def _update_progress(self, callbacks: OptimizationCallbacks, total_tasks: int):
        """Actualizar callbacks de progreso e iteración."""
        if callbacks.progress:
            progress_pct = int((self.current_iteration / total_tasks) * 70 + 20)
            callbacks.progress(progress_pct,
                            f"Evaluando {self.current_iteration}/{total_tasks}")

        if callbacks.iteration and self.current_iteration % 100 == 0:
            self._update_iteration_status(callbacks.iteration)

    def _run_sequential_optimization(self,
                            serie_original: pd.Series,
                            param_combinations: list[tuple],
                            exog_df: pd.DataFrame | None,
                            progress_callback,
                            iteration_callback,
                            _log_callback):
        """Ejecutar optimización secuencialmente (un modelo a la vez)."""
        print("[DEBUG_OPT] Iniciando procesamiento secuencial")

        total_tasks = len(param_combinations) * len(self.AVAILABLE_TRANSFORMATIONS)

        precision_final_60 = 60
        for params in param_combinations:
            p, d, q, P, D, Q, s = params  # noqa: N806
            order = (p, d, q)
            seasonal_order = (P, D, Q, s)

            for transformation in self.AVAILABLE_TRANSFORMATIONS:
                self.current_iteration += 1
                self._debug_models_evaluated += 1

                # Evaluar modelo
                metrics = self._evaluate_single_model(
                    serie_original, order, seasonal_order, transformation, exog_df,
                )

                # Validar y almacenar
                if metrics and self._is_valid_model(metrics):
                    metrics["transformation"] = transformation
                    metrics["order"] = order
                    metrics["seasonal_order"] = seasonal_order
                    metrics["with_exogenous"] = exog_df is not None

                    self._add_model(metrics)
                    self._update_transformation_stats(transformation, metrics)

                    # Log relevante
                    if metrics["precision_final"] > precision_final_60:
                        print(f"[DEBUG_OPT] Modelo relevante: {transformation} - "
                            f"Precision: {metrics['precision_final']:.1f}%")

                # Actualizar progreso
                if progress_callback:
                    progress_pct = int((self.current_iteration / total_tasks) * 70 + 20)
                    progress_callback(progress_pct,
                                    f"Evaluando {self.current_iteration}/{total_tasks}")

                if iteration_callback and self.current_iteration % 100 == 0:
                    self._update_iteration_status(iteration_callback)

                # Limpieza periódica
                if self.current_iteration % 200 == 0:
                    gc.collect()

        print("[DEBUG_OPT] Procesamiento secuencial completado")

    def diagnose_exog_coverage(self,
                          serie_saidi: pd.Series,
                          exog_df: pd.DataFrame,
                          log_callback) -> bool:
        """Diagnosticar cobertura temporal de variables exógenas."""
        try:
            self._print_coverage_info(serie_saidi, exog_df)

            # Realizar todas las validaciones
            if not self._validate_index_coverage(serie_saidi, exog_df, log_callback):
                return False

            if not self._validate_nan_values(exog_df, log_callback):
                return False

            if not self._validate_infinite_values(exog_df):
                return False

            self._check_variance(exog_df)

        except (KeyError, IndexError, ValueError, TypeError) as e:
            print(f"[DIAGNOSTICO] ERROR durante diagnóstico: {e}")
            return False
        else:
            print("[DIAGNOSTICO] Cobertura temporal y calidad de datos OK")
            return True

    def _print_coverage_info(self, serie_saidi: pd.Series, exog_df: pd.DataFrame) -> None:
        """Imprimir información de cobertura temporal."""
        saidi_start = serie_saidi.index[0]
        saidi_end = serie_saidi.index[-1]
        exog_start = exog_df.index[0]
        exog_end = exog_df.index[-1]

        print(f"[DIAGNOSTICO] SAIDI: {saidi_start} a {saidi_end} ({len(serie_saidi)} obs)")
        print(f"[DIAGNOSTICO] EXOG:  {exog_start} a {exog_end} ({len(exog_df)} obs)")

    def _validate_index_coverage(self,
                                serie_saidi: pd.Series,
                                exog_df: pd.DataFrame,
                                log_callback) -> bool:
        """Verificar que los índices coinciden y no hay fechas faltantes críticas."""
        if exog_df.index.equals(serie_saidi.index):
            return True

        print("[DIAGNOSTICO] ADVERTENCIA: Índices no coinciden exactamente")
        missing_in_exog = [d for d in serie_saidi.index if d not in exog_df.index]

        if not missing_in_exog:
            return True

        pct_missing = len(missing_in_exog) / len(serie_saidi) * 100
        print(f"[DIAGNOSTICO] Fechas SAIDI faltantes en EXOG: {len(missing_in_exog)} ({pct_missing:.1f}%)")

        self._print_missing_dates(missing_in_exog)

        # CRÍTICO: Si falta >20% de fechas, rechazar
        cobertura = 20
        if pct_missing > cobertura:
            print("[DIAGNOSTICO] ERROR CRÍTICO: >20% de fechas faltantes")
            if log_callback:
                log_callback(f"ERROR: {pct_missing:.1f}% de fechas SAIDI no tienen datos climáticos")
            return False

        return True

    def _print_missing_dates(self, missing_dates: list) -> None:
        """Imprimir fechas faltantes de forma resumida."""
        meses_faltantes = 5
        if len(missing_dates) <= meses_faltantes:
            print(f"[DIAGNOSTICO]   Fechas faltantes: {missing_dates}")
        else:
            print(f"[DIAGNOSTICO]   Primeras faltantes: {missing_dates[:3]}")
            print(f"[DIAGNOSTICO]   Últimas faltantes: {missing_dates[-3:]}")

    def _validate_nan_values(self, exog_df: pd.DataFrame, log_callback) -> bool:
        """Verificar que no hay valores NaN en las columnas."""
        if not exog_df.isna().any().any():
            return True

        nan_cols = exog_df.columns[exog_df.isna().any()].tolist()
        nan_details = [
            f"{col}: {exog_df[col].isna().sum()} ({exog_df[col].isna().sum() / len(exog_df) * 100:.1f}%)"
            for col in nan_cols
        ]

        print("[DIAGNOSTICO] ERROR: Columnas con NaN encontradas:")
        for detail in nan_details:
            print(f"[DIAGNOSTICO]   - {detail}")

        if log_callback:
            log_callback("ERROR: Variables exógenas contienen valores NaN")

        return False

    def _validate_infinite_values(self, exog_df: pd.DataFrame) -> bool:
        """Verificar que no hay valores infinitos."""
        if np.isinf(exog_df.to_numpy()).any():
            print("[DIAGNOSTICO] ERROR: Variables exógenas contienen valores infinitos")
            return False
        return True

    def _check_variance(self, exog_df: pd.DataFrame) -> None:
        """Advertir sobre columnas con varianza cero."""
        for col in exog_df.columns:
            if exog_df[col].std() == 0:
                print(f"[DIAGNOSTICO] ADVERTENCIA: {col} tiene varianza cero")

    def _project_climate_intelligent(self,
                                climate_data: pd.DataFrame,
                                forecast_dates: pd.DatetimeIndex,
                                log_callback=None) -> pd.DataFrame:
        """
        Proyectar variables climáticas de forma inteligente usando promedios estacionales ponderados.

        Estrategia de proyección:
        1. Calcular promedios mensuales históricos (estacionalidad)
        2. Aplicar ponderación exponencial (más peso a años recientes)
        3. Detectar tendencias lineales con scipy.stats.linregress
        4. Aplicar ajuste de tendencia si correlación > 0.3
        5. Logging detallado de proyecciones vs histórico

        Esta estrategia es SUPERIOR al forward-fill naive porque:
        - Captura estacionalidad real (ciclos anuales)
        - Pondera más los años recientes (cambio climático)
        - Detecta y aplica tendencias (calentamiento, cambios de precipitación)
        - Proporciona valores más realistas para el modelo SARIMAX

        Args:
            climate_data: DataFrame con datos climáticos históricos
            forecast_dates: Índice de fechas futuras a proyectar
            log_callback: Función para logging detallado

        Returns:
            DataFrame con variables climáticas proyectadas para forecast_dates

        """
        print(f"[CLIMATE_PROJECTION] Iniciando proyección inteligente para {len(forecast_dates)} fechas")

        self._validate_climate_data(climate_data)
        self._log_projection_header(climate_data, forecast_dates, log_callback)

        projected_df = pd.DataFrame(index=forecast_dates)
        hist_start = climate_data.index[0]
        hist_end = climate_data.index[-1]

        # Crear contexto de proyección
        context = ClimateProjectionContext(
            forecast_dates=forecast_dates,
            hist_start=hist_start,
            hist_end=hist_end,
            log_callback=log_callback,
        )

        # Procesar cada variable climática
        for col in climate_data.columns:
            print(f"[CLIMATE_PROJECTION] Proyectando: {col}")
            projected_df[col] = self._project_single_variable(
                climate_data[col], col, context,
            )

        # Verificación final
        projected_df = self._handle_missing_values(projected_df)
        self._log_projection_footer(projected_df, log_callback)

        print("[CLIMATE_PROJECTION] Proyección completada exitosamente")
        return projected_df

    def _validate_climate_data(self, climate_data: pd.DataFrame) -> None:
        """Validar que hay datos climáticos históricos disponibles."""
        if climate_data is None or climate_data.empty:
            raise ValueError("No hay datos climáticos históricos para proyectar")

    def _log_projection_header(self,
                            climate_data: pd.DataFrame,
                            forecast_dates: pd.DatetimeIndex,
                            log_callback) -> None:
        """Registrar información inicial de la proyección."""
        hist_start = climate_data.index[0]
        hist_end = climate_data.index[-1]
        n_years_hist = (hist_end - hist_start).days / 365.25

        print(f"[CLIMATE_PROJECTION] Histórico: {hist_start} a {hist_end} ({n_years_hist:.1f} años)")

        if log_callback:
            log_callback("=" * 80)
            log_callback("PROYECCIÓN INTELIGENTE DE VARIABLES CLIMÁTICAS")
            log_callback("=" * 80)
            log_callback(f"Histórico disponible: {hist_start.strftime('%Y-%m')} a {hist_end.strftime('%Y-%m')}")
            log_callback(f"Periodo a proyectar: {forecast_dates[0].strftime('%Y-%m')} a {forecast_dates[-1].strftime('%Y-%m')}")
            log_callback("")

    def _project_single_variable(self,
                            var_series: pd.Series,
                            col_name: str,
                            context: ClimateProjectionContext) -> np.ndarray:
        """Proyectar una única variable climática."""
        try:
            var_series = var_series.dropna()

            meses = 12
            if len(var_series) < meses:
                print("[CLIMATE_PROJECTION]   ADVERTENCIA: Datos insuficientes, usando media")
                return np.full(len(context.forecast_dates), var_series.mean())

            # Calcular promedios mensuales ponderados
            monthly_avg = self._calculate_weighted_monthly_averages(var_series, context.hist_end)

            # Detectar tendencia
            trend_info = self._detect_linear_trend(var_series, context.hist_start)
            self._log_trend_info(col_name, trend_info, context.log_callback)

            # Proyectar valores futuros
            projected_values = self._project_future_values(
                context.forecast_dates, monthly_avg, trend_info, context.hist_end,
            )

            # Logging comparativo
            self._log_projection_comparison(
                var_series, projected_values, trend_info, context.log_callback,
            )

        except (ValueError, TypeError, KeyError, IndexError) as e:
            return self._handle_projection_error(
                col_name, var_series, context.forecast_dates, e, context.log_callback,
            )
        else:
            return projected_values

    def _calculate_weighted_monthly_averages(self,
                                            var_series: pd.Series,
                                            hist_end: pd.Timestamp) -> dict:
        """Calcular promedios mensuales con ponderación exponencial."""
        years_ago = (hist_end - var_series.index).days / 365.25
        weights = np.exp(-years_ago / 3.0)
        weights = weights / weights.sum()

        monthly_avg = {}
        for month in range(1, 13):
            mask = var_series.index.month == month
            month_values = var_series[mask]
            month_weights = weights[mask]

            if len(month_values) > 0:
                monthly_avg[month] = np.average(month_values, weights=month_weights)
            else:
                monthly_avg[month] = var_series.mean()

        return monthly_avg

    def _detect_linear_trend(self, var_series: pd.Series, hist_start: pd.Timestamp) -> dict:
        """Detectar tendencia lineal en la serie temporal."""
        time_numeric = np.array([(d - hist_start).days / 30.44 for d in var_series.index])
        values_numeric = var_series.to_numpy()

        slope, intercept, r_value, p_value, _ = linregress(time_numeric, values_numeric)

        abs_mayor = 0.3
        abs_menor = 0.05
        has_trend = (abs(r_value) > abs_mayor) and (p_value < abs_menor)

        return {
            "has_trend": has_trend,
            "slope": slope,
            "intercept": intercept,
            "r_value": r_value,
            "p_value": p_value,
            "time_numeric": time_numeric,
        }

    def _log_trend_info(self, col_name: str, trend_info: dict, log_callback) -> None:
        """Registrar información sobre la tendencia detectada."""
        if trend_info["has_trend"]:
            trend_direction = "ascendente" if trend_info["slope"] > 0 else "descendente"
            print(f"[CLIMATE_PROJECTION]   Tendencia {trend_direction} detectada: "
                f"r={trend_info['r_value']:.3f}, slope={trend_info['slope']:.4f}/mes")

            if log_callback:
                log_callback(f"{col_name}:")
                log_callback(f"  - Tendencia {trend_direction}: r={trend_info['r_value']:.3f}, p={trend_info['p_value']:.4f}")
                log_callback(f"  - Cambio proyectado: {trend_info['slope']*12:.4f} por año")
        else:
            print(f"[CLIMATE_PROJECTION]   Sin tendencia significativa (r={trend_info['r_value']:.3f})")
            if log_callback:
                log_callback(f"{col_name}: Sin tendencia (r={trend_info['r_value']:.3f})")

    def _project_future_values(self,
                            forecast_dates: pd.DatetimeIndex,
                            monthly_avg: dict,
                            trend_info: dict,
                            hist_end: pd.Timestamp) -> np.ndarray:
        """Proyectar valores futuros usando estacionalidad y tendencia."""
        projected_values = []

        for date in forecast_dates:
            base_value = monthly_avg[date.month]

            if trend_info["has_trend"]:
                months_ahead = (date - hist_end).days / 30.44
                time_point = trend_info["time_numeric"][-1] + months_ahead
                trend_adjustment = (
                    trend_info["slope"] * time_point + trend_info["intercept"] -
                    (trend_info["slope"] * trend_info["time_numeric"][-1] + trend_info["intercept"])
                )
                final_value = base_value + trend_adjustment
            else:
                final_value = base_value

            projected_values.append(final_value)

        return np.array(projected_values)

    def _log_projection_comparison(self,
                               var_series: pd.Series,
                               projected_values: np.ndarray,
                               trend_info: dict,
                               log_callback) -> None:
        """Comparar proyección con último valor histórico."""
        last_hist_value = var_series.iloc[-1]
        first_proj_value = projected_values[0]
        change_pct = ((first_proj_value - last_hist_value) / last_hist_value) * 100

        print(f"[CLIMATE_PROJECTION]   Último histórico: {last_hist_value:.2f}")
        print(f"[CLIMATE_PROJECTION]   Primera proyección: {first_proj_value:.2f} "
            f"({change_pct:+.1f}% cambio)")

        if log_callback and trend_info["has_trend"]:
            log_callback(f"  - Último valor histórico: {last_hist_value:.2f}")
            log_callback(f"  - Primera proyección: {first_proj_value:.2f} ({change_pct:+.1f}%)")

    def _handle_projection_error(self,
                                col_name: str,
                                var_series: pd.Series,
                                forecast_dates: pd.DatetimeIndex,
                                error: Exception,
                                log_callback) -> np.ndarray:
        """Manejar errores en la proyección usando fallback."""
        print(f"[CLIMATE_PROJECTION]   ERROR proyectando {col_name}: {error}")
        print("[CLIMATE_PROJECTION]   Fallback: usando último valor conocido")

        last_value = var_series.iloc[-1]

        if log_callback:
            log_callback(f"{col_name}: ERROR - usando último valor ({last_value:.2f})")

        return np.full(len(forecast_dates), last_value)

    def _handle_missing_values(self, projected_df: pd.DataFrame) -> pd.DataFrame:
        """Manejar valores faltantes en la proyección."""
        if projected_df.isna().any().any():
            print("[CLIMATE_PROJECTION] ADVERTENCIA: NaN detectados en proyección")
            projected_df = projected_df.fillna(projected_df.mean())
        return projected_df

    def _log_projection_footer(self, projected_df: pd.DataFrame, log_callback) -> None:
        """Registrar resumen final de la proyección."""
        if log_callback:
            log_callback("=" * 80)
            log_callback(f"Proyección completada: {len(projected_df.columns)} variables")
            log_callback("Método: Promedios estacionales ponderados + tendencias")
            log_callback("=" * 80)

    def _evaluate_single_model(self,
                serie_original: pd.Series,
                order: tuple[int, int, int],
                seasonal_order: tuple[int, int, int, int],
                transformation: str,
                exog_df: pd.DataFrame | None) -> dict[str, Any] | None:
        """
        Evaluar un modelo SARIMAX individual  .

        Estrategia de validación:
        - Train/test split adaptativo (20-30% test según cantidad de datos)
        - Validación estricta de cobertura exógena (100% sin NaN)
        - Alineación perfecta entre índices SAIDI y exógenas

        Args:
            serie_original: Serie temporal SAIDI
            order: Parámetros (p,d,q) del modelo
            seasonal_order: Parámetros estacionales (P,D,Q,s)
            transformation: Tipo de transformación a aplicar
            exog_df: DataFrame con variables exógenas EN ESCALA ORIGINAL (puede ser None)

        Returns:
            Dict con métricas del modelo o None si falla

        """
        observaciones_mayor_60 = 60
        observaciones_mayor_36 = 36
        meses = 12
        try:
            # Calcular porcentaje de validación adaptativo
            n_obs = len(serie_original)
            if n_obs >= observaciones_mayor_60:
                pct_validacion = 0.30
            elif n_obs >= observaciones_mayor_36:
                pct_validacion = 0.25
            else:
                pct_validacion = 0.20

            n_test = max(6, int(n_obs * pct_validacion))

            # Dividir en train/test
            train_original = serie_original[:-n_test]
            test_original = serie_original[-n_test:]

            if len(train_original) < meses:
                return None

            # Aplicar transformación a la serie SAIDI
            self.scaler = None
            self.transformation_params = {}

            train_transformed, _ = self._apply_transformation(
                train_original.to_numpy(), transformation,
            )
            train_transformed_series = pd.Series(train_transformed, index=train_original.index)

            # ========== PREPARAR EXÓGENAS SIN ESCALAR (VALIDACIÓN ESTRICTA) ==========
            exog_train = None
            exog_test = None

            if exog_df is not None:
                try:
                    train_index = train_original.index
                    test_index = test_original.index

                    # VALIDACIÓN 1: Verificar que exog_df contiene TODAS las fechas necesarias
                    missing_train = [idx for idx in train_index if idx not in exog_df.index]
                    missing_test = [idx for idx in test_index if idx not in exog_df.index]

                    if missing_train or missing_test:
                        # Rechazar: faltan fechas
                        return None

                    # VALIDACIÓN 2: Extraer subconjuntos con .loc (garantiza alineación)
                    exog_train = exog_df.loc[train_index].copy()
                    exog_test = exog_df.loc[test_index].copy()

                    # VALIDACIÓN 3: Verificar que NO hay NaN
                    if exog_train is not None and exog_train.isna().any().any():
                        return None
                    if exog_test is not None and exog_test.isna().any().any():
                        return None

                    # VALIDACIÓN 4: Verificar dimensiones correctas
                    if len(exog_train) != len(train_original) or len(exog_test) != n_test:
                        return None

                    # VALIDACIÓN 5: Verificar que no hay infinitos
                    if np.isinf(exog_train.to_numpy()).any() or np.isinf(exog_test.to_numpy()).any():
                        return None

                except (KeyError, ValueError, TypeError) as e:
                    # Cualquier error en preparación: rechazar modelo
                    self.logger.debug("Error preparing exogenous variables: %s", e)
                    return None

            try:
                model = SARIMAX(
                    train_transformed_series,
                    exog=exog_train,  # EN ESCALA ORIGINAL
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=True,
                    enforce_invertibility=True,
                )

                # Fit con maxiter limitado (timeout implícito)
                results = model.fit(disp=False, maxiter=50)

            except (ValueError, np.linalg.LinAlgError, RuntimeError) as e:
                # Convergencia fallida, parámetros inválidos, etc.
                self.logger.debug("Model fitting failed: %s", e)
                return None

            # ==================== PREDECIR EN TEST ====================
            try:
                pred = results.get_forecast(steps=n_test, exog=exog_test)  # EN ESCALA ORIGINAL
                pred_mean_transformed = pred.predicted_mean

            except (ValueError, KeyError) as e:
                # Error en forecast
                self.logger.debug("Forecast failed: %s", e)
                return None

            # ==================== REVERTIR TRANSFORMACIÓN ====================
            try:
                pred_mean_original = self._inverse_transformation(
                    pred_mean_transformed.to_numpy(), transformation,
                )
            except (ValueError, TypeError) as e:
                self.logger.debug("Inverse transformation failed: %s", e)
                return None

            # ==================== CALCULAR MÉTRICAS ====================
            test_values = test_original.to_numpy()
            pred_values = pred_mean_original

            # RMSE (Root Mean Squared Error)
            rmse = np.sqrt(mean_squared_error(test_values, pred_values))

            # MAE (Mean Absolute Error)
            mae = np.mean(np.abs(test_values - pred_values))

            # MAPE (Mean Absolute Percentage Error)
            epsilon = 1e-8
            mape = np.mean(np.abs((test_values - pred_values) /
                                (test_values + epsilon))) * 100

            # R² Score
            ss_res = np.sum((test_values - pred_values) ** 2)
            ss_tot = np.sum((test_values - np.mean(test_values)) ** 2)
            r2_score = 1 - (ss_res / (ss_tot + epsilon))

            # Precisión final (inversa del MAPE)
            precision_final = max(0.0, min(100.0, (1 - mape/100) * 100))

            # VALIDACIÓN: Verificar que métricas son válidas
            if np.isnan(precision_final) or np.isinf(precision_final):
                return None

            if np.isnan(rmse) or np.isinf(rmse):
                return None

            # Penalización por complejidad del modelo
            complexity_penalty = sum(order) + sum(seasonal_order[:3])
            composite_score = rmse + (complexity_penalty * 0.05)

            # Score de estabilidad
            stability_score = self._calculate_stability_numpy(
                test_values, pred_values, precision_final, mape,
            )

            # Retornar métricas completas
            return {
                "rmse": rmse,
                "mae": mae,
                "mape": mape,
                "r2_score": r2_score,
                "precision_final": precision_final,
                "aic": results.aic,
                "bic": results.bic,
                "composite_score": composite_score,
                "n_params": complexity_penalty,
                "n_test": n_test,
                "stability_score": stability_score,
                "validation_pct": pct_validacion * 100,
                "exog_scaled": False,
            }

        except (ValueError, TypeError, KeyError) as e:
            # Cualquier error no controlado: rechazar modelo
            self.logger.debug("Unexpected error in model evaluation: %s", e)
            return None


    def _calculate_stability_numpy(self,
                        actual_values: np.ndarray,
                        predicted_values: np.ndarray,
                        precision: float,
                        mape: float) -> float:
        """Calcular score de estabilidad del modelo."""
        coeficite_error = 1e-8
        mape_mayor = 50
        mape_menor = 30
        try:
            errors = actual_values - predicted_values

            mean_abs_error = np.mean(np.abs(errors))
            std_error = np.std(errors)

            # Coeficiente de variación de errores
            if mean_abs_error > coeficite_error:
                cv_error = std_error / mean_abs_error
                # Convertir a score (menor CV = mayor estabilidad)
                stability_cv = max(0, 100 * (1 - min(cv_error, 1)))
            else:
                # Si errores son muy pequeños, estabilidad neutral
                stability_cv = 50.0

            # Penalización adaptativa por MAPE
            if mape > mape_mayor:
                mape_penalty = 0.5  # Penalización fuerte
            elif mape > mape_menor:
                mape_penalty = 0.7  # Penalización moderada
            else:
                mape_penalty = 1.0  # Sin penalización

            stability_cv = stability_cv * mape_penalty

            # Combinar estabilidad con precisión (60% estabilidad, 40% precisión)
            stability = (stability_cv * 0.6) + (precision * 0.4)

            return min(100.0, max(0.0, stability))

        except (ValueError, TypeError, ZeroDivisionError) as e:
            self.logger.warning("Error calculating stability: %s", e)
            return 0.0


    def _calculate_stability(self,
                        actual: pd.Series,
                        predicted: np.ndarray,
                        precision: float,
                        mape: float) -> float:
        """
        Calcular score de estabilidad del modelo.

        Basado en:
        - Variabilidad de errores
        - Consistencia de predicciones
        - Penalización por MAPE alto
        """
        coeficite_error = 1e-8
        mape_mayor = 50
        mape_menor = 30
        try:
            errors = actual.to_numpy() - predicted

            # Coeficiente de variación de errores
            mean_abs_error = np.mean(np.abs(errors))
            std_error = np.std(errors)

            if mean_abs_error > coeficite_error:
                cv_error = std_error / mean_abs_error
                stability_cv = max(0, 100 * (1 - min(cv_error, 1)))
            else:
                stability_cv = 50.0

            # Penalización por MAPE alto
            if mape > mape_mayor:
                mape_penalty = 0.5
            elif mape > mape_menor:
                mape_penalty = 0.7
            else:
                mape_penalty = 1.0

            stability_cv = stability_cv * mape_penalty

            # Combinar con precisión
            stability = (stability_cv * 0.6) + (precision * 0.4)

            return min(100.0, max(0.0, stability))

        except (ValueError, TypeError, KeyError) as e:
            self.logger.warning("Error calculating stability: %s", e)
            return 0.0

    def _is_valid_model(self, metrics: dict[str, Any]) -> bool:
        """Verificar si un modelo es válido para consideración."""
        if metrics is None:
            return False

        # Verificar que las métricas críticas sean válidas
        if metrics["rmse"] == float("inf"):
            return False

        return metrics["precision_final"] >= 0

    def _add_model(self, metrics: dict[str, Any]):
        """Agregar modelo a la colección de resultados."""
        self.all_models.append(metrics)
        self._debug_models_added += 1

        transformation = metrics["transformation"]

        # Actualizar mejor modelo de esta transformación
        if transformation not in self.best_by_transformation:
            self.best_by_transformation[transformation] = metrics
        else:
            current_best = self.best_by_transformation[transformation]
            if metrics["precision_final"] > current_best["precision_final"]:
                self.best_by_transformation[transformation] = metrics

        # Actualizar mejores globales
        self.best_precision = max(self.best_precision, metrics["precision_final"])

        self.best_rmse = min(self.best_rmse, metrics["rmse"])

    def _update_transformation_stats(self, transformation: str, metrics: dict[str, Any]):
        """Actualizar estadísticas de una transformación específica."""
        stats = self.transformation_stats[transformation]

        stats["count"] += 1

        precision = metrics["precision_final"]
        stability = metrics.get("stability_score", 0.0)

        # Actualizar mejor precisión
        if precision > stats["best_precision"]:
            stats["best_precision"] = precision
            stats["best_model"] = metrics

        # Actualizar mejor estabilidad
        stats["best_stability"] = max(stats["best_stability"], stability)

    def _update_iteration_status(self, iteration_callback):
        """Actualizar callback de iteración con información del mejor modelo actual."""
        if not self.all_models:
            return

        # Obtener mejor modelo actual
        best_current = max(self.all_models, key=lambda m: m["precision_final"])

        transformation = best_current["transformation"]
        precision = best_current["precision_final"]
        exog_mark = " [+EXOG]" if best_current.get("with_exogenous") else ""
        stability = best_current.get("stability_score", 0)

        status_msg = (f"Evaluadas {self.current_iteration}/{self.total_iterations} | "
                    f"Mejor: {transformation} - {precision:.1f}%{exog_mark}")

        if stability > 0:
            status_msg += f" (Stability: {stability:.0f})"

        iteration_callback(status_msg)


    def _select_best_models(self) -> tuple[list[dict], str, dict[str, int]]:
        """
        Seleccionar y clasificar los mejores modelos encontrados.

        Sistema de fallback adaptativo:
        1. Prioridad: modelos 'excellent' y 'good'
        2. Fallback: modelos 'acceptable'
        3. Último recurso: mejores de 'poor'

        Returns:
            Tuple con (lista_modelos, nivel_calidad, conteos_calidad)

        """
        print(f"[DEBUG_OPT] Seleccionando mejores modelos de {len(self.all_models)} evaluados")

        if not self.all_models:
            print("[DEBUG_OPT_WARN] No hay modelos para seleccionar")
            return [], "SIN_MODELOS", {
                "excellent": 0, "good": 0, "acceptable": 0, "poor": 0,
            }

        # Ordenar todos los modelos por precisión (descendente)
        sorted_models = sorted(
            self.all_models,
            key=lambda m: m["precision_final"],
            reverse=True,
        )

        # Clasificar por calidad
        excellent = []
        good = []
        acceptable = []
        poor = []

        precision_excelente = 60
        precision_buena = 40
        precision_aceptable = 20

        for model in sorted_models:
            precision = model["precision_final"]

            if precision >= precision_excelente:
                model["quality"] = "excellent"
                excellent.append(model)
            elif precision >= precision_buena:
                model["quality"] = "good"
                good.append(model)
            elif precision >= precision_aceptable:
                model["quality"] = "acceptable"
                acceptable.append(model)
            else:
                model["quality"] = "poor"
                poor.append(model)

        quality_counts = {
            "excellent": len(excellent),
            "good": len(good),
            "acceptable": len(acceptable),
            "poor": len(poor),
        }

        print(f"[DEBUG_OPT] Clasificacion: Excellent={len(excellent)}, Good={len(good)}, "
            f"Acceptable={len(acceptable)}, Poor={len(poor)}")

        # Estrategia de selección adaptativa
        seleccion_5 = 5
        if len(excellent) >= seleccion_5:
            selected = excellent[:self.MAX_TOP_MODELS]
            quality_level = "EXCELENTE"
        elif len(excellent) + len(good) >= seleccion_5:
            selected = (excellent + good)[:self.MAX_TOP_MODELS]
            quality_level = "BUENO"
        elif len(excellent) + len(good) + len(acceptable) >= seleccion_5:
            selected = (excellent + good + acceptable)[:self.MAX_TOP_MODELS]
            quality_level = "ACEPTABLE"
        else:
            selected = sorted_models[:self.MAX_TOP_MODELS]
            quality_level = "LIMITADO"

        print(f"[DEBUG_OPT] Modelos seleccionados: {len(selected)}, Nivel: {quality_level}")

        return selected, quality_level, quality_counts

    def _prepare_exogenous_adaptive(self,
                        climate_data: pd.DataFrame,
                        regional_code: str,
                        historico: pd.DataFrame,
                        log_callback) -> tuple[pd.DataFrame | None, dict | None, dict]:
        """
        Preparar variables exógenas con estrategia de overlap inteligente.

        Estrategia:
        1. Identificar periodo de overlap entre SAIDI y datos climáticos
        2. Validar cobertura mínima (80% en overlap)
        3. Verificar varianza no-cero en overlap
        4. Forward-fill para fechas futuras (sin límite)
        5. Backward-fill para fechas pasadas (máx 3 meses)
        6. RETORNAR EN ESCALA ORIGINAL (sin StandardScaler)

        Args:
            climate_data: DataFrame con datos climáticos mensuales
            regional_code: Código de la regional
            historico: Serie temporal SAIDI histórica
            log_callback: Función para logging

        Returns:
            Tuple de (exog_df, exog_info, coverage_report)
            - exog_df: DataFrame con variables EN ESCALA ORIGINAL
            - exog_info: Dict con metadata de cada variable
            - coverage_report: Dict con estadísticas de cobertura

        """
        # Inicializar reporte de cobertura
        coverage_report = {
            "overlap_start": None,
            "overlap_end": None,
            "overlapping_months": 0,
            "extrapolation_months": 0,
            "backward_fill_months": 0,
            "variables_accepted": [],
            "variables_rejected": [],
            "rejection_reasons": {},
        }

        try:
            if climate_data is None or climate_data.empty:
                print("[EXOG_ADAPTIVE] Sin datos climáticos")
                if log_callback:
                    log_callback("Sin datos climáticos disponibles")
                return None, None, coverage_report

            if not regional_code or regional_code not in self.REGIONAL_EXOG_VARS:
                print(f"[EXOG_ADAPTIVE] Regional {regional_code} sin variables definidas")
                return None, None, coverage_report

            print(f"[EXOG_ADAPTIVE] Iniciando preparación para {regional_code}")
            print("[EXOG_ADAPTIVE] MODO: SIN ESCALADO (valores originales)")

            # ==================== VALIDAR ÍNDICE DATETIME ====================
            print(f"[EXOG_ADAPTIVE] Tipo de índice climate_data: {type(climate_data.index)}")

            if not isinstance(climate_data.index, pd.DatetimeIndex):
                print("[EXOG_ADAPTIVE] ADVERTENCIA: Índice no es DatetimeIndex")

                # Buscar columna de fecha
                fecha_col = None
                for col in ["fecha", "Fecha", "date", "Date", "month_date"]:
                    if col in climate_data.columns:
                        fecha_col = col
                        break

                if fecha_col is None:
                    print("[EXOG_ADAPTIVE] ERROR: No se encontró columna de fecha")
                    if log_callback:
                        log_callback("ERROR: Datos climáticos sin columna de fecha válida")
                    return None, None, coverage_report

                print(f"[EXOG_ADAPTIVE] Usando columna '{fecha_col}' como índice")

                climate_data = climate_data.copy()
                climate_data[fecha_col] = pd.to_datetime(climate_data[fecha_col])
                climate_data = climate_data.set_index(fecha_col)

                print(f"[EXOG_ADAPTIVE] Índice convertido: {climate_data.index[0]} a {climate_data.index[-1]}")

            # Verificar que ahora sí es DatetimeIndex
            if not isinstance(climate_data.index, pd.DatetimeIndex):
                print(f"[EXOG_ADAPTIVE] ERROR: Índice inválido: {type(climate_data.index)}")
                if log_callback:
                    log_callback("ERROR: Datos climáticos con formato de fecha inválido")
                return None, None, coverage_report

            # ==================== ANÁLISIS DE COBERTURA TEMPORAL ====================
            saidi_start = historico.index[0]
            saidi_end = historico.index[-1]
            clima_start = climate_data.index[0]
            clima_end = climate_data.index[-1]

            print(f"[EXOG_ADAPTIVE] SAIDI: {saidi_start} a {saidi_end} ({len(historico)} meses)")
            print(f"[EXOG_ADAPTIVE] CLIMA: {clima_start} a {clima_end} ({len(climate_data)} meses)")

            # Calcular periodo de overlap
            overlap_start = max(saidi_start, clima_start)
            overlap_end = min(saidi_end, clima_end)

            if overlap_start > overlap_end:
                print("[EXOG_ADAPTIVE] ERROR: Sin overlap entre SAIDI y CLIMA")
                if log_callback:
                    log_callback("ERROR: Periodos SAIDI y CLIMA no se traslapan")
                return None, None, coverage_report

            # Crear máscara de overlap
            overlap_mask = (historico.index >= overlap_start) & (historico.index <= overlap_end)
            overlap_months = overlap_mask.sum()

            # Validar overlap mínimo (12 meses)
            meses = 12
            if overlap_months < meses:
                print(f"[EXOG_ADAPTIVE] ERROR: Overlap insuficiente ({overlap_months} < 12 meses)")
                if log_callback:
                    log_callback(f"ERROR: Overlap insuficiente: {overlap_months} meses (mínimo: 12)")
                return None, None, coverage_report

            # Actualizar reporte
            coverage_report["overlap_start"] = overlap_start
            coverage_report["overlap_end"] = overlap_end
            coverage_report["overlapping_months"] = int(overlap_months)

            # Calcular meses de extrapolación y backfill
            future_mask = historico.index > overlap_end
            extrapolation_months = future_mask.sum()
            coverage_report["extrapolation_months"] = int(extrapolation_months)

            past_mask = historico.index < overlap_start
            backward_months = past_mask.sum()
            coverage_report["backward_fill_months"] = int(backward_months)

            # LOG INICIAL DE COBERTURA
            if log_callback:
                log_callback("=" * 80)
                log_callback("ANÁLISIS DE COBERTURA TEMPORAL")
                log_callback("=" * 80)
                log_callback(f"Periodo SAIDI: {saidi_start.strftime('%Y-%m')} a {saidi_end.strftime('%Y-%m')} ({len(historico)} meses)")
                log_callback(f"Periodo CLIMA: {clima_start.strftime('%Y-%m')} a {clima_end.strftime('%Y-%m')} ({len(climate_data)} meses)")
                log_callback(f"Periodo OVERLAP: {overlap_start.strftime('%Y-%m')} a {overlap_end.strftime('%Y-%m')} ({overlap_months} meses)")
                log_callback("=" * 80)
                log_callback(f"ENTRENAMIENTO usará: {overlap_months} meses con datos reales")
                if extrapolation_months > 0:
                    log_callback(f"PREDICCIÓN proyectará: {extrapolation_months} meses (proyección inteligente)")
                if backward_months > 0:
                    log_callback(f"HISTÓRICO rellenará: {backward_months} meses hacia pasado (máx 3 backfill)")
                log_callback("")

            # ==================== MAPEO AUTOMÁTICO DE COLUMNAS ====================
            exog_vars_config = self.REGIONAL_EXOG_VARS[regional_code]

            available_cols_normalized = {}
            for col in climate_data.columns:
                normalized = col.lower().strip().replace(" ", "_").replace("-", "_")
                available_cols_normalized[normalized] = col

            climate_column_mapping = {}

            mejor_score = 2
            for var_code in exog_vars_config:
                var_normalized = var_code.lower().strip()

                # Intento 1: Coincidencia exacta
                if var_normalized in available_cols_normalized:
                    climate_column_mapping[var_code] = available_cols_normalized[var_normalized]
                    continue

                # Intento 2: Coincidencia parcial
                var_parts = var_normalized.split("_")
                best_match = None
                best_match_score = 0

                for norm_col, orig_col in available_cols_normalized.items():
                    matches = sum(1 for part in var_parts if part in norm_col)
                    if matches > best_match_score:
                        best_match_score = matches
                        best_match = orig_col

                if best_match_score >= mejor_score:
                    climate_column_mapping[var_code] = best_match

            if not climate_column_mapping:
                print("[EXOG_ADAPTIVE] ERROR: No se pudo mapear ninguna variable")
                if log_callback:
                    log_callback("ERROR: Nombres de columnas climáticas no coinciden")
                return None, None, coverage_report

            print(f"[EXOG_ADAPTIVE] Variables mapeadas: {len(climate_column_mapping)}/{len(exog_vars_config)}")

            # ==================== PREPARACIÓN DE VARIABLES (SIN ESCALADO) ====================
            exog_df = pd.DataFrame(index=historico.index)
            exog_info = {}

            for var_code, var_nombre in exog_vars_config.items():
                climate_col = climate_column_mapping.get(var_code)

                if not climate_col or climate_col not in climate_data.columns:
                    coverage_report["variables_rejected"].append(var_nombre)
                    coverage_report["rejection_reasons"][var_code] = "Columna no encontrada"
                    print(f"[EXOG_ADAPTIVE]   X RECHAZADA {var_code}: columna no encontrada")
                    continue

                print(f"[EXOG_ADAPTIVE]   Procesando {var_code}...")

                overlap_menor_80 = 80

                # Extraer serie del clima
                var_series = climate_data[climate_col].copy()

                # Crear serie alineada (inicialmente vacía)
                aligned_series = pd.Series(index=historico.index, dtype=float)

                # Llenar datos donde hay overlap REAL
                for date in historico.index:
                    if date in var_series.index:
                        aligned_series[date] = var_series.loc[date]

                # VALIDACIÓN: Verificar cobertura en overlap
                overlap_data = aligned_series[overlap_mask]
                datos_reales_overlap = overlap_data.notna().sum()
                overlap_pct = (datos_reales_overlap / overlap_months) * 100

                print(f"[EXOG_ADAPTIVE]     Cobertura en overlap: {datos_reales_overlap}/{overlap_months} ({overlap_pct:.1f}%)")

                # RECHAZAR si cobertura < 80%
                if overlap_pct < overlap_menor_80:
                    coverage_report["variables_rejected"].append(var_nombre)
                    coverage_report["rejection_reasons"][var_code] = f"Cobertura insuficiente: {overlap_pct:.1f}%"
                    print(f"[EXOG_ADAPTIVE]   X RECHAZADA {var_code}: cobertura {overlap_pct:.1f}% < 80%")
                    continue

                # VALIDACIÓN: Verificar VARIANZA en overlap
                var_std = overlap_data.std()

                if pd.isna(var_std) or var_std == 0:
                    coverage_report["variables_rejected"].append(var_nombre)
                    coverage_report["rejection_reasons"][var_code] = "Varianza cero en overlap"
                    print(f"[EXOG_ADAPTIVE]   X RECHAZADA {var_code}: varianza = 0")
                    continue

                print(f"[EXOG_ADAPTIVE]     Varianza en overlap: {var_std:.4f}")

                # Forward-fill (sin límite) para meses futuros
                if extrapolation_months > 0:
                    aligned_series = aligned_series.fillna(method="ffill")
                    print(f"[EXOG_ADAPTIVE]     Forward-fill: {extrapolation_months} meses")

                # Backward-fill (limitado a 3 meses) para meses pasados
                if backward_months > 0:
                    aligned_series = aligned_series.fillna(method="bfill", limit=3)
                    print("[EXOG_ADAPTIVE]     Backward-fill: máx 3 meses")

                # Si TODAVÍA hay NaN, rellenar con media del overlap
                if aligned_series.isna().any():
                    mean_overlap = overlap_data.mean()
                    filled_count = aligned_series.isna().sum()
                    aligned_series = aligned_series.fillna(mean_overlap)
                    print(f"[EXOG_ADAPTIVE]     Rellenados {filled_count} NaN con media={mean_overlap:.2f}")

                # VERIFICACIÓN FINAL
                final_nan = aligned_series.isna().sum()
                if final_nan > 0:
                    coverage_report["variables_rejected"].append(var_nombre)
                    coverage_report["rejection_reasons"][var_code] = f"{final_nan} NaN después de relleno"
                    print(f"[EXOG_ADAPTIVE]   X RECHAZADA {var_code}: {final_nan} NaN finales")
                    continue

                # ===== CAMBIO CRÍTICO: GUARDAR EN ESCALA ORIGINAL =====
                # Ya NO se aplica StandardScaler
                exog_df[var_code] = aligned_series

                exog_info[var_code] = {
                    "nombre": var_nombre,
                    "columna_clima": climate_col,
                    "disponible": True,
                    "datos_reales_overlap": int(datos_reales_overlap),
                    "overlap_coverage_pct": float(overlap_pct),
                    "varianza_overlap": float(var_std),
                    "extrapolados": int(extrapolation_months),
                    "backfilled": min(int(backward_months), 3),
                    "scaled": False,
                }
                coverage_report["variables_accepted"].append(var_nombre)
                print(f"[EXOG_ADAPTIVE]   ✓ {var_code} -> ACEPTADA (escala original)")

            # VALIDACIÓN FINAL
            if exog_df.empty or exog_df.shape[1] == 0:
                print("[EXOG_ADAPTIVE] ERROR: Ninguna variable aceptada")
                if log_callback:
                    log_callback("ERROR: Ninguna variable exógena cumplió los criterios de calidad")
                    log_callback("Razones de rechazo:")
                    for var_code, reason in coverage_report["rejection_reasons"].items():
                        log_callback(f"  - {var_code}: {reason}")
                return None, None, coverage_report

            print(f"[EXOG_ADAPTIVE] Variables aceptadas: {len(exog_df.columns)}/{len(exog_vars_config)}")

            self.exog_scaler = None

        except (KeyError, ValueError, TypeError, AttributeError, IndexError) as e:
            print(f"[EXOG_ADAPTIVE] ERROR CRÍTICO: {e}")
            traceback.print_exc()
            if log_callback:
                log_callback(f"ERROR preparando variables exógenas: {e!s}")
            return None, None, coverage_report
        else:
            # LOG FINAL
            if log_callback:
                log_callback(f"Variables exógenas preparadas: {len(exog_df.columns)} (EN ESCALA ORIGINAL)")
                for var_data in exog_info.values():
                    log_callback(f"  ✓ {var_data['nombre']} "
                            f"(datos_reales={var_data['datos_reales_overlap']}, "
                            f"varianza={var_data['varianza_overlap']:.2f})")

                if coverage_report["variables_rejected"]:
                    log_callback(f"Variables rechazadas: {len(coverage_report['variables_rejected'])}")
                    for var_nombre in coverage_report["variables_rejected"]:
                        log_callback(f"  X {var_nombre}")

                log_callback("NOTA: SARIMAX maneja escalado internamente, valores en escala original")

            print(f"[EXOG_ADAPTIVE] Preparación completada: {len(exog_df.columns)} variables (SIN ESCALAR)")

            return exog_df, exog_info, coverage_report

    def _align_exog_to_saidi(self,
                            exog_series: pd.DataFrame,
                            df_saidi: pd.DataFrame,
                            var_code: str) -> pd.Series | None:
        """Alinear datos exógenos al índice temporal de SAIDI."""
        try:
            climate_dates = exog_series.index
            saidi_dates = df_saidi.index

            # Asegurar DatetimeIndex
            if not isinstance(climate_dates, pd.DatetimeIndex):
                climate_dates = pd.to_datetime(climate_dates)
            if not isinstance(saidi_dates, pd.DatetimeIndex):
                saidi_dates = pd.to_datetime(saidi_dates)

            # Crear serie resultado
            result = pd.Series(index=saidi_dates, dtype=float)

            # Llenar con valores disponibles
            for date in saidi_dates:
                if date in climate_dates:
                    result[date] = exog_series.loc[date].iloc[0]

            # Forward fill para fechas futuras
            max_climate_date = climate_dates.max()
            future_indices = saidi_dates > max_climate_date

            if future_indices.any():
                last_known_value = exog_series.iloc[-1].iloc[0]
                result.loc[future_indices] = last_known_value

            # Backward fill para fechas pasadas
            min_climate_date = climate_dates.min()
            past_indices = saidi_dates < min_climate_date

            if past_indices.any():
                first_known_value = exog_series.iloc[0].iloc[0]
                result.loc[past_indices] = first_known_value

        except (KeyError, IndexError, AttributeError, TypeError) as e:
            print(f"[DEBUG_OPT_ERROR] Error alineando {var_code}: {e}")
            return None
        else:
            return result

    def _apply_transformation(self,
                         data: np.ndarray,
                         transformation_type: str) -> tuple[np.ndarray, str]:
        """Aplicar transformación a los datos."""
        if transformation_type == "original":
            return data, "Sin transformacion"

        if transformation_type == "standard":
            self.scaler = StandardScaler()
            transformed = self.scaler.fit_transform(data.reshape(-1, 1)).flatten()
            return transformed, "StandardScaler"

        if transformation_type == "log":
            data_positive = np.maximum(data, 1e-10)
            transformed = np.log(data_positive)
            self.transformation_params["log_applied"] = True
            return transformed, "Log"

        if transformation_type == "sqrt":
            data_positive = np.maximum(data, 0)
            transformed = np.sqrt(data_positive)
            self.transformation_params["sqrt_applied"] = True
            return transformed, "Sqrt"

        if transformation_type == "boxcox":
            data_positive = np.maximum(data, 1e-10)
            transformed, lambda_param = stats.boxcox(data_positive)
            self.transformation_params["boxcox_lambda"] = lambda_param
            return transformed, f"Box-Cox (lambda={lambda_param:.4f})"

        return data, "Sin transformacion"

    def _inverse_transformation(self,
                           data: np.ndarray,
                           transformation_type: str) -> np.ndarray:
        """Revertir transformación a escala original."""
        if transformation_type == "original":
            return data

        # Diccionario de funciones de inversión
        inverse_transforms = {
            "standard": lambda d: (
                self.scaler.inverse_transform(d.reshape(-1, 1)).flatten()
                if self.scaler is not None
                else d
            ),
            "log": lambda d: np.exp(d),
            "sqrt": lambda d: np.power(d, 2),
            "boxcox": lambda d: self._inverse_boxcox(d),
        }

        # Aplicar la transformación correspondiente o retornar data sin cambios
        return inverse_transforms.get(transformation_type, lambda d: d)(data)

    def _inverse_boxcox(self, data: np.ndarray) -> np.ndarray:
        """Inversión específica para transformación Box-Cox."""
        lambda_param = self.transformation_params.get("boxcox_lambda", 0)
        if lambda_param == 0:
            return np.exp(data)
        return np.power(data * lambda_param + 1, 1 / lambda_param)

    def _log_final_summary(self,
                      log_callback,
                      top_models: list[dict],
                      exog_info: dict | None,
                      regional_code: str | None = None):  # Agregar regional_code como parámetro
        """Generar resumen final de optimizacion CON INFO DE CORRELACIONES."""
        # Definir mapa de correlaciones por regional
        correlations_map = {
            "SAIDI_O": {  # Ocaña
                "realfeel_min": 0.689,
                "windchill_avg": 0.520,
                "dewpoint_avg": 0.470,
                "windchill_max": 0.464,
                "dewpoint_min": 0.456,
                "precipitation_max_daily": 0.452,
                "precipitation_avg_daily": 0.438,
            },
            "SAIDI_C": {  # Cúcuta
                "realfeel_avg": 0.573,
                "wind_speed_max": 0.356,
                "pressure_abs_avg": -0.356,
            },
            "SAIDI_T": {  # Tibú
                "wind_dir_avg": -0.400,
                "uv_index_avg": 0.385,
                "heat_index_avg": 0.363,
                "temperature_min": 0.352,
                "temperature_avg": 0.338,
                "wind_gust_max": 0.318,
                "uv_index_max": 0.248,
            },
            "SAIDI_A": {  # Aguachica
                "uv_index_max": 0.664,
                "days_with_rain": 0.535,
            },
            "SAIDI_P": {  # Pamplona
                "precipitation_total": 0.577,
            },
        }

        log_callback("=" * 80)
        log_callback("RESUMEN FINAL - OPTIMIZACION COMPLETADA")
        if exog_info:
            log_callback("Con variables exogenas climaticas (basadas en correlacion)")
        log_callback("Validacion alineada con PredictionService (20-30% test)")
        log_callback("=" * 80)

        # Estadisticas por transformacion
        log_callback("\nESTADISTICAS POR TRANSFORMACION:")
        log_callback("-" * 80)
        for transform in self.AVAILABLE_TRANSFORMATIONS:
            stats = self.transformation_stats[transform]
            log_callback(f"{transform.upper():12s} | Modelos: {stats['count']:4d} | "
                        f"Mejor precision: {stats['best_precision']:.1f}% | "
                        f"Mejor stability: {stats['best_stability']:.0f}")

        # Top 10 mejores modelos
        log_callback("\nTOP 10 MEJORES MODELOS:")
        log_callback("-" * 80)

        top_1 = 1
        top_2 = 2
        top_3 = 3
        for i, modelo in enumerate(top_models[:10], 1):
            if i == top_1:
                medal = " Puesto 1"

            elif i == top_2:
                medal = " Puesto 2"
            elif i == top_3:
                medal = " Puesto 3"
            else:
                medal = f"   Puesto {i}"

            exog_mark = " [+EXOG]" if modelo.get("with_exogenous") else ""

            quality = modelo.get("quality", "poor")
            quality_map = {
                "excellent": "[EXCELENTE]",
                "good": "[BUENO]",
                "acceptable": "[ACEPTABLE]",
                "poor": "[LIMITADO]",
            }
            quality_symbol = quality_map.get(quality, "[?]")

            stability_str = ""
            if modelo.get("stability_score"):
                stability_str = f" | Stability: {modelo.get('stability_score', 0):.0f}"

            val_pct = modelo.get("validation_pct", 0)

            log_callback(f"\n{medal} {quality_symbol}:")
            log_callback(f"   Transformacion: {modelo['transformation'].upper()}{exog_mark}")
            log_callback(f"   Parametros: order={modelo['order']}, seasonal={modelo['seasonal_order']}")
            log_callback(f"   Precision: {modelo['precision_final']:.1f}%{stability_str}")
            log_callback(f"   RMSE: {modelo['rmse']:.4f} | R2: {modelo['r2_score']:.3f}")
            log_callback(f"   Validacion: {val_pct:.0f}% de datos como test ({modelo.get('n_test', 0)} meses)")

        # Variables exogenas utilizadas CON CORRELACIONES
        abs_fuerte = 0.6
        abs_moderada_fuerte = 0.4
        abs_moderada= 0.3
        if exog_info:
            log_callback("\nVARIABLES EXOGENAS UTILIZADAS:")
            log_callback("-" * 80)

            # Obtener correlaciones específicas de la regional
            regional_correlations = correlations_map.get(regional_code, {})

            for var_code, var_data in exog_info.items():
                # Obtener correlación específica de la regional
                corr = regional_correlations.get(var_code, 0.0)

                # Usar operador ternario
                corr_str = f"(r={corr:+.3f})" if corr != 0 else ""

                # Clasificar fuerza de correlacion
                abs_corr = abs(corr)
                if abs_corr >= abs_fuerte:
                    strength = "*** FUERTE ***"
                elif abs_corr >= abs_moderada_fuerte:
                    strength = "** MODERADA-FUERTE **"
                elif abs_corr >= abs_moderada:
                    strength = "* MODERADA *"
                else:
                    strength = ""

                log_callback(f"   {strength} {var_data['nombre']} {corr_str}")
                log_callback(f"      Columna: {var_data['columna_clima']}")

        # Modelo optimo seleccionado
        if top_models:
            best_model = top_models[0]
            precision = best_model["precision_final"]
            quality = best_model.get("quality", "poor")

            log_callback("\n" + "=" * 80)
            log_callback("MODELO OPTIMO SELECCIONADO:")
            log_callback("=" * 80)
            log_callback(f"Transformacion: {best_model['transformation'].upper()}")
            log_callback(f"Parametros: order={best_model['order']}, seasonal={best_model['seasonal_order']}")
            log_callback(f"Precision: {precision:.1f}%")
            log_callback(f"Calidad del modelo: {quality.upper()}")
            log_callback(f"Metodo de validacion: {best_model.get('validation_pct', 0):.0f}% test")

            if best_model.get("with_exogenous"):
                log_callback("Modelo incluye variables exogenas climaticas correlacionadas")

            stability = best_model.get("stability_score", 0)
            if stability:
                log_callback(f"Stability Score: {stability:.1f}/100")

            # Interpretacion
            if quality == "excellent":
                interpretacion = "EXCELENTE - Predicciones muy confiables"
            elif quality == "good":
                interpretacion = "BUENO - Predicciones confiables"
            elif quality == "acceptable":
                interpretacion = "ACEPTABLE - Usar con precaucion"
            else:
                interpretacion = "LIMITADO - Datos dificiles de predecir"
                log_callback("\nNOTA: Precision baja sugiere que los datos historicos")
                log_callback("      tienen alta variabilidad o patrones irregulares.")

            log_callback(f"Interpretacion: {interpretacion}")
            log_callback("Optimizacion: Sistema adaptativo con variables correlacionadas")

        log_callback("=" * 80)

    def save_best_model_config(self, regional_code: str, best_model: dict[str, Any]) -> bool:
        """
        Guardar la configuración del mejor modelo para una regional.

        Crea/actualiza un archivo JSON con los mejores parámetros encontrados
        para cada regional. Este archivo será leído por PredictionService.

        Args:
            regional_code: Código de la regional (ej: 'SAIDI_O')
            best_model: Dict con información del mejor modelo

        Returns:
            bool: True si se guardó correctamente

        """
        try:
            # Crear directorio de configuración si no existe
            config_dir = Path(__file__).parent.parent / "config"
            config_dir.mkdir(exist_ok=True)

            config_file = config_dir / "optimized_models.json"

            # Cargar configuraciones existentes
            if config_file.exists():
                with config_file.open(encoding="utf-8") as f:
                    configs = json.load(f)
            else:
                configs = {}

            # Preparar nueva configuración
            new_config = {
                "transformation": best_model.get("transformation", "original"),
                "order": list(best_model.get("order", (3, 1, 3))),
                "seasonal_order": list(best_model.get("seasonal_order", (1, 1, 1, 12))),
                "precision_final": float(best_model.get("precision_final", 0)),
                "rmse": float(best_model.get("rmse", 0)),
                "mape": float(best_model.get("mape", 0)),
                "r2_score": float(best_model.get("r2_score", 0)),
                "with_exogenous": bool(best_model.get("with_exogenous", False)),
                "stability_score": float(best_model.get("stability_score", 0)),
                "optimization_date": datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),  # noqa: UP017
                "quality": best_model.get("quality", "unknown"),
            }

            # Actualizar configuración de la regional
            configs[regional_code] = new_config

            # Guardar archivo
            with config_file.open("w", encoding="utf-8") as f:
                json.dump(configs, f, indent=2, ensure_ascii=False)

        except (OSError, json.JSONDecodeError) as e:
            print(f"[SAVE_CONFIG] ERROR guardando configuración: {e}")
            return False
        else:
            print(f"[SAVE_CONFIG] ✓ Configuración guardada para {regional_code}")
            print(f"[SAVE_CONFIG]   Archivo: {config_file}")
            print(f"[SAVE_CONFIG]   Transformación: {new_config['transformation']}")
            print(f"[SAVE_CONFIG]   Order: {new_config['order']}")
            print(f"[SAVE_CONFIG]   Seasonal: {new_config['seasonal_order']}")
            print(f"[SAVE_CONFIG]   Precisión: {new_config['precision_final']:.1f}%")
            return True

# FUNCIÓN WORKER PARA PROCESAMIENTO PARALELO
def _evaluate_model_worker(task: tuple) -> dict[str, Any] | None:
    """
    Worker function para evaluación paralela de modelos.

    Esta función se ejecuta en un proceso separado
    """
    logger = logging.getLogger(__name__)

    serie_original, order, seasonal_order, transformation, exog_df = task

    try:
        # Crear instancia temporal del servicio
        temp_service = OptimizationService()

        # Evaluar modelo
        # Se aplica noqa: SLF001 para permitir acceso explícito al método protegido
        # desde esta función worker auxiliar.
        metrics = temp_service._evaluate_single_model(  # noqa: SLF001
            serie_original, order, seasonal_order, transformation, exog_df,
        )

    except Exception:
        logger.exception(
            "[DEBUG_OPT_WARN] Error en worker paralelo %s",
            order,  # Variable 'order' pasada como argumento posicional
        )
        return None
    else:
        return metrics
