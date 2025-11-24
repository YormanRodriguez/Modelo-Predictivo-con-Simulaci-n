# services/optimization_service.py
import gc
import json
import multiprocessing as mp
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import linregress
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")


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
    AVAILABLE_TRANSFORMATIONS = ["original", "log", "boxcox", "standard", "sqrt"]

    # Máximo de modelos a retornar
    MAX_TOP_MODELS = 50

    # Variables exógenas por regional
    REGIONAL_EXOG_VARS = {
        "SAIDI_O": {  # Ocaña
            "realfeel_min": "Temperatura aparente mínima",           # r=0.689
            "windchill_avg": "Sensación térmica promedio",          # r=0.520
            "dewpoint_avg": "Punto de rocío promedio",              # r=0.470
            "windchill_max": "Sensación térmica máxima",            # r=0.464
            "dewpoint_min": "Punto de rocío mínimo",                # r=0.456
            # Adicionales relevantes
            "precipitation_max_daily": "Precipitación máxima diaria", # r=0.452
            "precipitation_avg_daily": "Precipitación promedio diaria", # r=0.438
        },

        "SAIDI_C": {  # Cúcuta
            "realfeel_avg": "Temperatura aparente promedio",        # r=0.573
            "pressure_rel_avg": "Presión relativa promedio",        # r=-0.358 (negativa)
            "wind_speed_max": "Velocidad máxima del viento",        # r=0.356
            "pressure_abs_avg": "Presión absoluta promedio",        # r=-0.356 (negativa)
        },

        "SAIDI_T": {  # Tibú

            "realfeel_avg": "Temperatura aparente promedio",        # r=0.906
            "wind_dir_avg": "Dirección promedio del viento",        # r=-0.400 (negativa)
            "uv_index_avg": "Índice UV promedio",                   # r=0.385
            "heat_index_avg": "Índice de calor promedio",           # r=0.363
            "temperature_min": "Temperatura mínima",                # r=0.352
            "windchill_min": "Sensación térmica mínima",            # r=0.340
            "temperature_avg": "Temperatura promedio",              # r=0.338
            "pressure_rel_avg": "Presión relativa promedio",        # r=-0.330 (negativa)
        },

        "SAIDI_A": {  # Aguachica
            "uv_index_max": "Índice UV máximo",                     # r=0.664
            "days_with_rain": "Días con lluvia",                    # r=0.535
        },

        "SAIDI_P": {  # Pamplona
            "precipitation_total": "Precipitación total",           # r=0.577
            "precipitation_avg_daily": "Precipitación promedio diaria", # r=0.552
            "realfeel_min": "Temperatura aparente mínima",          # r=0.344 (moderada-fuerte)
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
            use_parallel: Usar procesamiento paralelo
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
            df, col_saidi, historico = self._load_and_validate_data(
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
            exog_df, exog_info, coverage_report = self._prepare_exogenous_adaptive(
                climate_data, df, regional_code, historico, log_callback,
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
            comprobacion_paralela = 50
            if use_parallel and len(param_combinations) > comprobacion_paralela:
                print("[DEBUG_OPT] Usando procesamiento PARALELO")
                self._run_parallel_optimization(
                    historico[col_saidi], param_combinations, exog_df,
                    progress_callback, iteration_callback, log_callback, max_workers,
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

            # Preparar resultado
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
                "config_saved": best_model is not None and regional_code is not None,  # NUEVO
            }

            print("[DEBUG_OPT] Optimizacion finalizada exitosamente")

            return result

        except Exception as e:
            error_msg = f"Error durante optimizacion: {e!s}"
            print(f"[DEBUG_OPT_ERROR] {error_msg}")
            if log_callback:
                log_callback(f"ERROR: {error_msg}")
            raise Exception(error_msg)

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
        if df_prepared is not None:
            df = df_prepared.copy()
            if log_callback:
                log_callback("Usando datos SAIDI preparados del modelo")
        elif file_path is not None:
            df = pd.read_excel(file_path, sheet_name="Hoja1")
            if log_callback:
                log_callback("Leyendo Excel SAIDI")
        else:
            raise ValueError("Debe proporcionar file_path o df_prepared")

        # Asegurar índice datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            if "Fecha" in df.columns:
                df["Fecha"] = pd.to_datetime(df["Fecha"])
                df.set_index("Fecha", inplace=True)
            else:
                df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
                df.set_index(df.columns[0], inplace=True)

        # Buscar columna SAIDI
        col_saidi = None
        if "SAIDI" in df.columns:
            col_saidi = "SAIDI"
        elif "SAIDI Historico" in df.columns:
            col_saidi = "SAIDI Historico"

        if col_saidi is None:
            raise ValueError("No se encontro la columna SAIDI en los datos")

        # Filtrar datos históricos (no nulos)
        historico = df[df[col_saidi].notna()].copy()

        meses=12
        if len(historico) < meses:
            raise ValueError(f"Datos insuficientes: solo {len(historico)} observaciones (minimo {meses})")

        print(f"[DEBUG_OPT] Datos validados: {len(historico)} obs, columna: {col_saidi}")

        return df, col_saidi, historico

    def _configure_parameter_space(self, log_callback) -> list[tuple]:
        """
        Configurar espacio de búsqueda de parámetros.

        Returns:
            Lista de tuplas (p, d, q, P, D, Q, s)

        """
        print("[DEBUG_OPT] Configurando espacio de busqueda")

        # Rangos de parámetros
        p_range = range(7)  # AR
        d_range = range(3)  # Diferenciación
        q_range = range(7)  # MA
        P_range = range(6)  # AR estacional
        D_range = range(3)  # Diferenciación estacional
        Q_range = range(6)  # MA estacional
        s_range = [12]

        #p_range = range(2)  # AR
        #d_range = range(2)  # Diferenciación
        #q_range = range(2)  # MA
        #P_range = range(2)  # AR estacional
        #D_range = range(2)  # Diferenciación estacional
        #Q_range = range(2)  # MA estacional

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

        for p, d, q, P, D, Q, s in combinations:
            # Calcular complejidad total
            total_params = p + d + q + P + D + Q

            # Rechazar si es trivial o demasiado complejo
            combinacion_rechazada = 14
            if total_params == 0 or total_params > combinacion_rechazada:
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
                               progress_callback,
                               iteration_callback,
                               log_callback,
                               max_workers: int | None):
        """Ejecutar optimización en paralelo usando ProcessPoolExecutor."""
        if max_workers is None:
            max_workers = max(1, mp.cpu_count() - 1)

        print(f"[DEBUG_OPT] Iniciando procesamiento paralelo con {max_workers} workers")

        if log_callback:
            log_callback(f"Procesamiento paralelo con {max_workers} workers")

        # Preparar tareas
        tasks = []
        for params in param_combinations:
            p, d, q, P, D, Q, s = params
            order = (p, d, q)
            seasonal_order = (P, D, Q, s)

            for transformation in self.AVAILABLE_TRANSFORMATIONS:
                tasks.append((serie_original, order, seasonal_order, transformation, exog_df))

        total_tasks = len(tasks)
        batch_size = 100  # Procesar en lotes

        print(f"[DEBUG_OPT] Total tareas: {total_tasks}, batch_size: {batch_size}")

        # Procesar en lotes
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for batch_start in range(0, total_tasks, batch_size):
                batch_end = min(batch_start + batch_size, total_tasks)
                batch_tasks = tasks[batch_start:batch_end]

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
                            if metrics["precision_final"] > 60:
                                print(f"[DEBUG_OPT] Modelo relevante encontrado: "
                                    f"{transformation} - Precision: {metrics['precision_final']:.1f}%")

                    except Exception as e:
                        # Silenciar timeouts y errores menores
                        if "timeout" not in str(e).lower():
                            print(f"[DEBUG_OPT_WARN] Error en evaluacion: {type(e).__name__}")

                    # Actualizar progreso
                    if progress_callback:
                        progress_pct = int((self.current_iteration / total_tasks) * 70 + 20)
                        progress_callback(progress_pct,
                                        f"Evaluando {self.current_iteration}/{total_tasks}")

                    # Actualizar iteración actual
                    if iteration_callback and self.current_iteration % 100 == 0:
                        self._update_iteration_status(iteration_callback)

                # Limpiar memoria después de cada lote
                gc.collect()

        print("[DEBUG_OPT] Procesamiento paralelo completado")

    def _run_sequential_optimization(self,
                                serie_original: pd.Series,
                                param_combinations: list[tuple],
                                exog_df: pd.DataFrame | None,
                                progress_callback,
                                iteration_callback,
                                log_callback):
        """Ejecutar optimización secuencialmente (un modelo a la vez)."""
        print("[DEBUG_OPT] Iniciando procesamiento secuencial")

        total_tasks = len(param_combinations) * len(self.AVAILABLE_TRANSFORMATIONS)

        for params in param_combinations:
            p, d, q, P, D, Q, s = params
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

                    precision_aceptable = 60
                    # Log relevante
                    if metrics["precision_final"] > precision_aceptable:
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
            saidi_start = serie_saidi.index[0]
            saidi_end = serie_saidi.index[-1]
            exog_start = exog_df.index[0]
            exog_end = exog_df.index[-1]

            print(f"[DIAGNOSTICO] SAIDI: {saidi_start} a {saidi_end} ({len(serie_saidi)} obs)")
            print(f"[DIAGNOSTICO] EXOG:  {exog_start} a {exog_end} ({len(exog_df)} obs)")

            # 1. Verificar que los índices coinciden EXACTAMENTE
            if not exog_df.index.equals(serie_saidi.index):
                print("[DIAGNOSTICO] ADVERTENCIA: Índices no coinciden exactamente")

                # Verificar fechas faltantes
                missing_in_exog = [d for d in serie_saidi.index if d not in exog_df.index]

                if missing_in_exog:
                    pct_missing = len(missing_in_exog) / len(serie_saidi) * 100
                    print(f"[DIAGNOSTICO] Fechas SAIDI faltantes en EXOG: {len(missing_in_exog)} ({pct_missing:.1f}%)")

                    # Mostrar primeras y últimas fechas faltantes
                    fechas_faltantes_mostradas = 5
                    if len(missing_in_exog) <= fechas_faltantes_mostradas:
                        print(f"[DIAGNOSTICO]   Fechas faltantes: {missing_in_exog}")
                    else:
                        print(f"[DIAGNOSTICO]   Primeras faltantes: {missing_in_exog[:3]}")
                        print(f"[DIAGNOSTICO]   Últimas faltantes: {missing_in_exog[-3:]}")

                    fechas_faltantes_mostradas_critico = 20
                    # CRÍTICO: Si falta >20% de fechas, rechazar
                    if pct_missing > fechas_faltantes_mostradas_critico:
                        print("[DIAGNOSTICO] ERROR CRÍTICO: >20% de fechas faltantes")
                        if log_callback:
                            log_callback(f"ERROR: {pct_missing:.1f}% de fechas SAIDI no tienen datos climáticos")
                        return False

            # 2. Verificar que NO hay NaN en ninguna columna
            if exog_df.isnull().any().any():
                nan_cols = exog_df.columns[exog_df.isnull().any()].tolist()
                nan_details = []

                for col in nan_cols:
                    nan_count = exog_df[col].isnull().sum()
                    pct_nan = (nan_count / len(exog_df)) * 100
                    nan_details.append(f"{col}: {nan_count} ({pct_nan:.1f}%)")

                print("[DIAGNOSTICO] ERROR: Columnas con NaN encontradas:")
                for detail in nan_details:
                    print(f"[DIAGNOSTICO]   - {detail}")

                if log_callback:
                    log_callback("ERROR: Variables exógenas contienen valores NaN")

                return False

            # 3. Verificar valores infinitos
            if np.isinf(exog_df.values).any():
                print("[DIAGNOSTICO] ERROR: Variables exógenas contienen valores infinitos")
                return False

            # 4. Verificar que hay varianza en las variables
            for col in exog_df.columns:
                if exog_df[col].std() == 0:
                    print(f"[DIAGNOSTICO] ADVERTENCIA: {col} tiene varianza cero")

            print("[DIAGNOSTICO] ✓ Cobertura temporal y calidad de datos OK")
            return True

        except Exception as e:
            print(f"[DIAGNOSTICO] ERROR durante diagnóstico: {e}")
            return False

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

        if log_callback:
            log_callback("=" * 80)
            log_callback("PROYECCIÓN INTELIGENTE DE VARIABLES CLIMÁTICAS")
            log_callback("=" * 80)

        # Verificar que hay datos históricos
        if climate_data is None or climate_data.empty:
            raise ValueError("No hay datos climáticos históricos para proyectar")

        # Calcular periodo histórico disponible
        hist_start = climate_data.index[0]
        hist_end = climate_data.index[-1]
        n_years_hist = (hist_end - hist_start).days / 365.25

        print(f"[CLIMATE_PROJECTION] Histórico: {hist_start} a {hist_end} ({n_years_hist:.1f} años)")

        if log_callback:
            log_callback(f"Histórico disponible: {hist_start.strftime('%Y-%m')} a {hist_end.strftime('%Y-%m')}")
            log_callback(f"Periodo a proyectar: {forecast_dates[0].strftime('%Y-%m')} a {forecast_dates[-1].strftime('%Y-%m')}")
            log_callback("")

        # DataFrame para almacenar proyecciones
        projected_df = pd.DataFrame(index=forecast_dates)

        # Procesar cada variable climática
        for col in climate_data.columns:
            print(f"[CLIMATE_PROJECTION] Proyectando: {col}")

            try:
                var_series = climate_data[col].dropna()
                meses = 12
                if len(var_series) < meses:
                    # Datos insuficientes: usar media simple
                    projected_df[col] = var_series.mean()
                    print("[CLIMATE_PROJECTION]   ADVERTENCIA: Datos insuficientes, usando media")
                    continue

                # ========== PASO 1: CALCULAR PROMEDIOS MENSUALES PONDERADOS ==========
                # Crear ponderación exponencial (más peso a años recientes)
                years_ago = (hist_end - var_series.index).days / 365.25
                weights = np.exp(-years_ago / 3.0)  # Decaimiento exponencial con tau=3 años
                weights = weights / weights.sum()  # Normalizar

                # Calcular promedios mensuales ponderados
                monthly_avg = {}
                for month in range(1, 13):
                    mask = var_series.index.month == month
                    month_values = var_series[mask]
                    month_weights = weights[mask]

                    if len(month_values) > 0:
                        # Promedio ponderado
                        monthly_avg[month] = np.average(month_values, weights=month_weights)
                    else:
                        # Si no hay datos para este mes, usar promedio global
                        monthly_avg[month] = var_series.mean()

                # ========== PASO 2: DETECTAR TENDENCIA LINEAL ==========
                # Convertir fechas a números (meses desde inicio)
                time_numeric = np.array([(d - hist_start).days / 30.44 for d in var_series.index])
                values_numeric = var_series.values

                # Regresión lineal
                slope, intercept, r_value, p_value, std_err = linregress(time_numeric, values_numeric)

                tendencia_mayor = 0.3
                tendencia_menor= 0.05
                # Determinar si hay tendencia significativa
                has_trend = (abs(r_value) > tendencia_mayor) and (p_value < tendencia_menor)

                if has_trend:
                    trend_direction = "ascendente" if slope > 0 else "descendente"
                    print(f"[CLIMATE_PROJECTION]   Tendencia {trend_direction} detectada: "
                        f"r={r_value:.3f}, slope={slope:.4f}/mes")

                    if log_callback:
                        log_callback(f"{col}:")
                        log_callback(f"  - Tendencia {trend_direction}: r={r_value:.3f}, p={p_value:.4f}")
                        log_callback(f"  - Cambio proyectado: {slope*12:.4f} por año")
                else:
                    print(f"[CLIMATE_PROJECTION]   Sin tendencia significativa (r={r_value:.3f})")
                    if log_callback:
                        log_callback(f"{col}: Sin tendencia (r={r_value:.3f})")

                # ========== PASO 3: PROYECTAR VALORES FUTUROS ==========
                projected_values = []

                for date in forecast_dates:
                    month = date.month

                    # Valor base: promedio estacional ponderado
                    base_value = monthly_avg[month]

                    # Aplicar ajuste de tendencia si existe
                    if has_trend:
                        months_ahead = (date - hist_end).days / 30.44
                        time_point = time_numeric[-1] + months_ahead
                        trend_adjustment = slope * time_point + intercept - (slope * time_numeric[-1] + intercept)
                        final_value = base_value + trend_adjustment
                    else:
                        final_value = base_value

                    projected_values.append(final_value)

                projected_df[col] = projected_values

                # ========== PASO 4: LOGGING COMPARATIVO ==========
                # Comparar proyección vs último valor histórico
                last_hist_value = var_series.iloc[-1]
                first_proj_value = projected_values[0]
                change_pct = ((first_proj_value - last_hist_value) / last_hist_value) * 100

                print(f"[CLIMATE_PROJECTION]   Último histórico: {last_hist_value:.2f}")
                print(f"[CLIMATE_PROJECTION]   Primera proyección: {first_proj_value:.2f} "
                    f"({change_pct:+.1f}% cambio)")

                if log_callback and has_trend:
                    log_callback(f"  - Último valor histórico: {last_hist_value:.2f}")
                    log_callback(f"  - Primera proyección: {first_proj_value:.2f} ({change_pct:+.1f}%)")

            except Exception as e:
                # Si falla la proyección de esta variable, usar forward-fill simple
                print(f"[CLIMATE_PROJECTION]   ERROR proyectando {col}: {e}")
                print("[CLIMATE_PROJECTION]   Fallback: usando último valor conocido")

                last_value = climate_data[col].iloc[-1]
                projected_df[col] = last_value

                if log_callback:
                    log_callback(f"{col}: ERROR - usando último valor ({last_value:.2f})")

        # Verificación final
        if projected_df.isnull().any().any():
            print("[CLIMATE_PROJECTION] ADVERTENCIA: NaN detectados en proyección")
            # Rellenar con medias
            projected_df = projected_df.fillna(projected_df.mean())

        if log_callback:
            log_callback("=" * 80)
            log_callback(f"Proyección completada: {len(projected_df.columns)} variables")
            log_callback("Método: Promedios estacionales ponderados + tendencias")
            log_callback("=" * 80)

        print("[CLIMATE_PROJECTION] Proyección completada exitosamente")

        return projected_df

    def _evaluate_single_model(self,
                    serie_original: pd.Series,
                    order: tuple[int, int, int],
                    seasonal_order: tuple[int, int, int, int],
                    transformation: str,
                    exog_df: pd.DataFrame | None) -> dict[str, Any] | None:
        """
        Evaluar un modelo SARIMAX individual.

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
        try:
            # Calcular porcentaje de validación adaptativo
            observaciones_mayor_60 = 60
            observaciones_mayor_36 = 36
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

            meses = 12
            if len(train_original) < meses:
                return None

            # Aplicar transformación a la serie SAIDI
            self.scaler = None
            self.transformation_params = {}

            train_transformed, _ = self._apply_transformation(
                train_original.values, transformation,
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
                    if exog_train.isnull().any().any() or exog_test.isnull().any().any():
                        return None

                    # VALIDACIÓN 4: Verificar dimensiones correctas
                    if len(exog_train) != len(train_original) or len(exog_test) != n_test:
                        return None

                    # VALIDACIÓN 5: Verificar que no hay infinitos
                    if np.isinf(exog_train.values).any() or np.isinf(exog_test.values).any():
                        return None

                except Exception:
                    # Cualquier error en preparación: rechazar modelo
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

            except Exception:
                # Convergencia fallida, parámetros inválidos, etc.
                return None

            # ==================== PREDECIR EN TEST ====================
            try:
                pred = results.get_forecast(steps=n_test, exog=exog_test)  # EN ESCALA ORIGINAL
                pred_mean_transformed = pred.predicted_mean

            except Exception:
                # Error en forecast
                return None

            # ==================== REVERTIR TRANSFORMACIÓN ====================
            try:
                pred_mean_original = self._inverse_transformation(
                    pred_mean_transformed.values, transformation,
                )
            except Exception:
                return None

            # ==================== CALCULAR MÉTRICAS ====================
            test_values = test_original.values
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

        except Exception:
            # Cualquier error no controlado: rechazar modelo
            return None


    def _calculate_stability_numpy(self,
                            actual_values: np.ndarray,
                            predicted_values: np.ndarray,
                            precision: float,
                            mape: float) -> float:
        """Calcular score de estabilidad del modelo."""
        try:
            errors = actual_values - predicted_values

            mean_abs_error = np.mean(np.abs(errors))
            std_error = np.std(errors)

            # Coeficiente de variación de errores
            coeficiente_variacion = 1e-8
            if mean_abs_error > coeficiente_variacion:
                cv_error = std_error / mean_abs_error
                # Convertir a score (menor CV = mayor estabilidad)
                stability_cv = max(0, 100 * (1 - min(cv_error, 1)))
            else:
                # Si errores son muy pequeños, estabilidad neutral
                stability_cv = 50.0

            # Penalización adaptativa por MAPE
            mape_mayor=50
            mape_mayor_medio=30
            if mape > mape_mayor:
                mape_penalty = 0.5  # Penalización fuerte
            elif mape > mape_mayor_medio:
                mape_penalty = 0.7  # Penalización moderada
            else:
                mape_penalty = 1.0  # Sin penalización

            stability_cv = stability_cv * mape_penalty

            # Combinar estabilidad con precisión (60% estabilidad, 40% precisión)
            stability = (stability_cv * 0.6) + (precision * 0.4)

            return min(100.0, max(0.0, stability))

        except Exception:
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
        try:
            errors = actual.values - predicted

            # Coeficiente de variación de errores
            mean_abs_error = np.mean(np.abs(errors))
            std_error = np.std(errors)

            coeficiente_variacion = 1e-8
            if mean_abs_error > coeficiente_variacion:
                cv_error = std_error / mean_abs_error
                stability_cv = max(0, 100 * (1 - min(cv_error, 1)))
            else:
                stability_cv = 50.0

            # Penalización por MAPE alto
            mape_mayor=50
            mape_mayor_medio=30
            if mape > mape_mayor:
                mape_penalty = 0.5
            elif mape > mape_mayor_medio:
                mape_penalty = 0.7
            else:
                mape_penalty = 1.0

            stability_cv = stability_cv * mape_penalty

            # Combinar con precisión
            stability = (stability_cv * 0.6) + (precision * 0.4)

            return min(100.0, max(0.0, stability))

        except Exception:
            return 0.0

    def _is_valid_model(self, metrics: dict[str, Any]) -> bool:
        """
        Verificar si un modelo es válido para consideración.

        Criterio: Debe tener métricas computables (no infinitas)
        """
        if metrics is None:
            return False

        # Verificar que las métricas críticas sean válidas
        if metrics["rmse"] == float("inf"):
            return False

        precision_negativa=0
        if metrics["precision_final"] < precision_negativa:
            return False

        return True

    def _add_model(self, metrics: dict[str, Any]):
        """
        Agregar modelo a la colección de resultados.

        También actualiza el mejor modelo por transformación
        """
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

        for model in sorted_models:
            precision = model["precision_final"]
            precision_mayor_60=60
            precision_mayor_40=40
            precision_mayor_20=20
            if precision >= precision_mayor_60:
                model["quality"] = "excellent"
                excellent.append(model)
            elif precision >= precision_mayor_40:
                model["quality"] = "good"
                good.append(model)
            elif precision >= precision_mayor_20:
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
        len_menor5= 5
        if len(excellent) >= len_menor5:
            selected = excellent[:self.MAX_TOP_MODELS]
            quality_level = "EXCELENTE"
        elif len(excellent) + len(good) >= len_menor5:
            selected = (excellent + good)[:self.MAX_TOP_MODELS]
            quality_level = "BUENO"
        elif len(excellent) + len(good) + len(acceptable) >=len_menor5:
            selected = (excellent + good + acceptable)[:self.MAX_TOP_MODELS]
            quality_level = "ACEPTABLE"
        else:
            selected = sorted_models[:self.MAX_TOP_MODELS]
            quality_level = "LIMITADO"

        print(f"[DEBUG_OPT] Modelos seleccionados: {len(selected)}, Nivel: {quality_level}")

        return selected, quality_level, quality_counts

    def _prepare_exogenous_adaptive(self,
                        climate_data: pd.DataFrame,
                        df_saidi: pd.DataFrame,
                        regional_code: str,
                        historico: pd.DataFrame,
                        log_callback) -> tuple[pd.DataFrame | None, dict | None, dict]:
        """
        Preparar variables exógenas con estrategia de overlap inteligente.

        CAMBIO CLAVE: Ya NO se aplica escalado. SARIMAX maneja internamente
        la normalización de variables exógenas.

        Estrategia:
        1. Identificar periodo de overlap entre SAIDI y datos climáticos
        2. Validar cobertura mínima (80% en overlap)
        3. Verificar varianza no-cero en overlap
        4. Forward-fill para fechas futuras (sin límite)
        5. Backward-fill para fechas pasadas (máx 3 meses)
        6. RETORNAR EN ESCALA ORIGINAL (sin StandardScaler)

        Args:
            climate_data: DataFrame con datos climáticos mensuales
            df_saidi: DataFrame SAIDI completo
            regional_code: Código de la regional
            historico: Serie temporal SAIDI histórica
            log_callback: Función para logging

        Returns:
            Tuple de (exog_df, exog_info, coverage_report)
            - exog_df: DataFrame con variables EN ESCALA ORIGINAL
            - exog_info: Dict con metadata de cada variable
            - coverage_report: Dict con estadísticas de cobertura

        """
        try:
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

                try:
                    climate_data = climate_data.copy()
                    climate_data[fecha_col] = pd.to_datetime(climate_data[fecha_col])
                    climate_data = climate_data.set_index(fecha_col)

                    print(f"[EXOG_ADAPTIVE] Índice convertido: {climate_data.index[0]} a {climate_data.index[-1]}")

                except Exception as e:
                    print(f"[EXOG_ADAPTIVE] ERROR convirtiendo índice: {e}")
                    if log_callback:
                        log_callback(f"ERROR: No se pudo convertir columna de fecha: {e!s}")
                    return None, None, coverage_report

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
            meses=12
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

            for var_code in exog_vars_config.keys():
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

                mejor_de_2=2
                if best_match_score >= mejor_de_2:
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

                try:
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
                    cobertura_minima=80
                    if overlap_pct < cobertura_minima:
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
                    if aligned_series.isnull().any():
                        mean_overlap = overlap_data.mean()
                        filled_count = aligned_series.isnull().sum()
                        aligned_series = aligned_series.fillna(mean_overlap)
                        print(f"[EXOG_ADAPTIVE]     Rellenados {filled_count} NaN con media={mean_overlap:.2f}")

                    # VERIFICACIÓN FINAL
                    final_nan = aligned_series.isnull().sum()
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

                except Exception as e:
                    coverage_report["variables_rejected"].append(var_nombre)
                    coverage_report["rejection_reasons"][var_code] = f"Error: {e!s}"
                    print(f"[EXOG_ADAPTIVE]   X ERROR {var_code}: {e}")
                    continue

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

            # LOG FINAL
            if log_callback:
                log_callback(f"Variables exógenas preparadas: {len(exog_df.columns)} (EN ESCALA ORIGINAL)")
                for var_code, var_data in exog_info.items():
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

        except Exception as e:
            print(f"[EXOG_ADAPTIVE] ERROR CRÍTICO: {e}")
            traceback.print_exc()
            if log_callback:
                log_callback(f"ERROR preparando variables exógenas: {e!s}")
            return None, None, coverage_report

    def _align_exog_to_saidi(self,
                            exog_series: pd.DataFrame,
                            df_saidi: pd.DataFrame,
                            var_code: str,
                            log_callback) -> pd.Series | None:
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

            return result

        except Exception as e:
            print(f"[DEBUG_OPT_ERROR] Error alineando {var_code}: {e}")
            return None

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

        if transformation_type == "standard":
            if self.scaler is not None:
                return self.scaler.inverse_transform(data.reshape(-1, 1)).flatten()
            return data

        if transformation_type == "log":
            return np.exp(data)

        if transformation_type == "sqrt":
            return np.power(data, 2)

        if transformation_type == "boxcox":
            lambda_param = self.transformation_params.get("boxcox_lambda", 0)
            if lambda_param == 0:
                return np.exp(data)
            return np.power(data * lambda_param + 1, 1 / lambda_param)

        return data

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

        for i, modelo in enumerate(top_models[:10], 1):
            top_1=1
            top_2=2
            top_3=3
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
        if exog_info:
            log_callback("\nVARIABLES EXOGENAS UTILIZADAS:")
            log_callback("-" * 80)

            # Obtener correlaciones específicas de la regional
            regional_correlations = correlations_map.get(regional_code, {})

            for var_code, var_data in exog_info.items():
                # Obtener correlación específica de la regional
                corr = regional_correlations.get(var_code, 0.0)

                if corr != 0:
                    corr_str = f"(r={corr:+.3f})"
                else:
                    corr_str = ""

                # Clasificar fuerza de correlacion
                abs_corr = abs(corr)
                correlacion_fuerte=0.6
                correlacion_moderada_fuerte=0.4
                correlacion_moderada=0.3
                if abs_corr >= correlacion_fuerte:
                    strength = "*** FUERTE ***"
                elif abs_corr >= correlacion_moderada_fuerte:
                    strength = "** MODERADA-FUERTE **"
                elif abs_corr >= correlacion_moderada:
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
                with open(config_file, encoding="utf-8") as f:
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
                "optimization_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "quality": best_model.get("quality", "unknown"),
            }

            # Actualizar configuración de la regional
            configs[regional_code] = new_config

            # Guardar archivo
            with open(config_file, "w", encoding="utf-8") as f:
                json.dump(configs, f, indent=2, ensure_ascii=False)

            print(f"[SAVE_CONFIG] ✓ Configuración guardada para {regional_code}")
            print(f"[SAVE_CONFIG]   Archivo: {config_file}")
            print(f"[SAVE_CONFIG]   Transformación: {new_config['transformation']}")
            print(f"[SAVE_CONFIG]   Order: {new_config['order']}")
            print(f"[SAVE_CONFIG]   Seasonal: {new_config['seasonal_order']}")
            print(f"[SAVE_CONFIG]   Precisión: {new_config['precision_final']:.1f}%")

            return True

        except Exception as e:
            print(f"[SAVE_CONFIG] ERROR guardando configuración: {e}")
            return False

# FUNCIÓN WORKER PARA PROCESAMIENTO PARALELO
def _evaluate_model_worker(task: tuple) -> dict[str, Any] | None:
    """
    Worker function para evaluación paralela de modelos.

    Esta función se ejecuta en un proceso separado
    """
    serie_original, order, seasonal_order, transformation, exog_df = task

    try:
        # Crear instancia temporal del servicio
        temp_service = OptimizationService()

        # Evaluar modelo
        metrics = temp_service._evaluate_single_model(
            serie_original, order, seasonal_order, transformation, exog_df,
        )

        return metrics

    except Exception:
        # No propagar excepciones en workers paralelos
        return None
