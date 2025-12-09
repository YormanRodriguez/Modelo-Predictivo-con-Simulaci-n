# services/validation_service.py
import json
import tempfile
import traceback
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ClassVar

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX

# En la sección de imports (línea ~18)
from services.climate_simulation_service import (
    ClimateSimulationService,
    SimulationConfig,  # ← AGREGAR ESTO
)

warnings.filterwarnings("ignore")

@dataclass
class VariableProcessingContext:
    """Contexto para procesar una variable exógena individual."""

    var_code: str
    var_nombre: str
    climate_data: pd.DataFrame
    climate_column_mapping: dict
    historico: pd.DataFrame
    overlap_mask: pd.Series
    overlap_months: int
    regional_code: str

@dataclass
class ValidationMetricsParams:
    """Parámetros para el cálculo de métricas de validación."""

    order: tuple
    seasonal_order: tuple
    transformation: str
    with_exogenous: bool
    pct_validacion: float
    n_test: int

# Agregar al inicio del archivo con los otros dataclasses
@dataclass
class ValidationPlotConfig:
    """Configuración para generar gráfica de validación."""

    datos_entrenamiento: pd.Series
    datos_validacion: pd.Series
    predicciones_validacion: pd.Series
    order: tuple
    seasonal_order: tuple
    metricas: dict
    pct_validacion: float
    transformation: str
    exog_info: dict | None = None
    simulation_config: dict | None = None

class ValidationError(Exception):
    """Error general en el proceso de validación."""

class ValidationDataError(ValidationError):
    """Error relacionado con los datos de entrada para validación."""

class ModelFittingError(ValidationError):
    """Error al ajustar el modelo SARIMAX."""

class PredictionError(ValidationError):
    """Error al generar predicciones del modelo."""

class ValidationService:
    """Servicio para validar modelos SARIMAX con transformaciones por regional."""

    # Mapeo de regionales a sus transformaciones optimas
    REGIONAL_TRANSFORMATIONS: ClassVar[dict[str, str]] = {
        "SAIDI_O": "original",
        "SAIDI_C": "original",
        "SAIDI_A": "original",
        "SAIDI_P": "boxcox",
        "SAIDI_T": "sqrt",
        "SAIDI_Cens": "original",
    }

    # Variables exogenas por regional
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

    REGIONAL_ORDERS: ClassVar[dict[str, dict[str, tuple]]] = {
        "SAIDI_O": {"order": (3, 1, 6), "seasonal_order": (3, 1, 0, 12)},
        "SAIDI_C": {"order": (3, 1, 2), "seasonal_order": (1, 1, 2, 12)},
        "SAIDI_A": {"order": (2, 1, 3), "seasonal_order": (2, 1, 1, 12)},
        "SAIDI_P": {"order": (4, 1, 3), "seasonal_order": (1, 1, 4, 12)},
        "SAIDI_T": {"order": (3, 1, 3), "seasonal_order": (2, 1, 2, 12)},
        "SAIDI_Cens": {"order": (4, 1, 3), "seasonal_order": (1, 1, 4, 12)},
    }

    def __init__(self):
        self.default_order = (4, 1, 3)
        self.default_seasonal_order = (1, 1, 4, 12)
        self.plot_file_path = None
        self.scaler = None
        self.exog_scaler = None
        self.transformation_params = {}
        self.simulation_service = ClimateSimulationService()

    def load_optimized_config(self, regional_code: str) -> dict[str, Any] | None:
        """
        Cargar configuración optimizada para una regional.

        Lee el archivo JSON generado por OptimizationService y retorna
        los mejores parámetros encontrados previamente.

        Args:
            regional_code: Código de la regional (ej: 'SAIDI_O')

        Returns:
            Dict con configuración óptima o None si no existe

        """
        try:
            # Ubicación del archivo de configuración
            config_file = Path(__file__).parent.parent / "config" / "optimized_models.json"

            if not config_file.exists():
                print("[LOAD_CONFIG] No existe archivo de configuraciones optimizadas")
                return None

            # Cargar configuraciones usando Path.open()
            with config_file.open(encoding="utf-8") as f:
                configs = json.load(f)

            # Buscar configuración de la regional
            if regional_code not in configs:
                print(f"[LOAD_CONFIG] No hay configuración optimizada para {regional_code}")
                return None

            config = configs[regional_code]

        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            print(f"[LOAD_CONFIG] ERROR cargando configuración: {e}")
            return None
        else:
            # Este bloque se ejecuta solo si no hubo excepciones
            print(f"[LOAD_CONFIG] ✓ Configuración cargada para {regional_code}")
            print(f"[LOAD_CONFIG]   Transformación: {config['transformation']}")
            print(f"[LOAD_CONFIG]   Order: {config['order']}")
            print(f"[LOAD_CONFIG]   Seasonal: {config['seasonal_order']}")
            print(f"[LOAD_CONFIG]   Precisión: {config['precision_final']:.1f}%")
            print(f"[LOAD_CONFIG]   Optimizado: {config['optimization_date']}")

            return config

    def _get_orders_for_regional(self, regional_code):
        """
        MÉTODO ACTUALIZADO: Obtener órdenes SARIMAX específicos para una regional.

        Prioriza configuración optimizada sobre defaults hardcodeados.

        Args:
            regional_code: Código de la regional (ej: 'SAIDI_O')

        Returns:
            tuple: (order, seasonal_order) - Órdenes ARIMA y estacionales

        """
        if not regional_code:
            return self.default_order, self.default_seasonal_order

        # PRIORIDAD 1: Intentar cargar configuración optimizada
        optimized_config = self.load_optimized_config(regional_code)

        if optimized_config:
            order = tuple(optimized_config["order"])
            seasonal_order = tuple(optimized_config["seasonal_order"])

            print(f"[ORDERS] Usando parámetros OPTIMIZADOS para {regional_code}")
            print(f"[ORDERS]   Order: {order}")
            print(f"[ORDERS]   Seasonal: {seasonal_order}")
            print(f"[ORDERS]   Precisión documentada: {optimized_config['precision_final']:.1f}%")

            return order, seasonal_order

        # PRIORIDAD 2: Usar configuración hardcodeada
        if regional_code in self.REGIONAL_ORDERS:
            config = self.REGIONAL_ORDERS[regional_code]
            order = config["order"]
            seasonal_order = config["seasonal_order"]

            print(f"[ORDERS] Usando parámetros DEFAULT para {regional_code}")
            print(f"[ORDERS]   Order: {order}")
            print(f"[ORDERS]   Seasonal: {seasonal_order}")

            return order, seasonal_order

        # FALLBACK: Usar valores por defecto genéricos
        print(f"[ORDERS] Usando parámetros FALLBACK para {regional_code}")
        print(f"[ORDERS]   Order: {self.default_order}")
        print(f"[ORDERS]   Seasonal: {self.default_seasonal_order}")

        return self.default_order, self.default_seasonal_order

    def run_validation(self,
                    file_path: str | None = None,
                    df_prepared: pd.DataFrame | None = None,
                    order: tuple | None = None,
                    seasonal_order: tuple | None = None,
                    regional_code: str | None = None,
                    climate_data: pd.DataFrame | None = None,
                    simulation_config: dict | None = None,
                    progress_callback = None,
                    log_callback = None) -> dict[str, Any]:
            """
            Ejecutar validacion del modelo SARIMAX con transformacion especifica por regional.

            Carga automáticamente parámetros optimizados si existen

            Args:
                file_path: Ruta del archivo SAIDI Excel
                df_prepared: DataFrame de SAIDI ya preparado
                order: Orden ARIMA (opcional - si None usa el optimizado/default de la regional)
                seasonal_order: Orden estacional ARIMA (opcional - si None usa el optimizado/default)
                regional_code: Código de la regional
                climate_data: DataFrame con datos climáticos mensuales
                simulation_config: Configuración de simulación climática (opcional)
                progress_callback: Función para actualizar progreso
                log_callback: Función para loguear mensajes

            Returns:
                Diccionario con resultados de validacion

            """
            try:
                optimized_config = None

                if regional_code:
                    optimized_config = self.load_optimized_config(regional_code)

                    if optimized_config and log_callback:
                        log_callback("=" * 80)
                        log_callback("USANDO CONFIGURACIÓN OPTIMIZADA")
                        log_callback("=" * 80)
                        log_callback(f"Regional: {regional_code}")
                        log_callback(f"Transformación: {optimized_config['transformation'].upper()}")
                        log_callback(f"Order: {optimized_config['order']}")
                        log_callback(f"Seasonal: {optimized_config['seasonal_order']}")
                        log_callback(f"Precisión documentada: {optimized_config['precision_final']:.1f}%")
                        log_callback(f"Optimizado en: {optimized_config['optimization_date']}")
                        log_callback("=" * 80)

                # Obtener parámetros (prioriza optimizados > hardcoded > default)
                if order is None or seasonal_order is None:
                    order_regional, seasonal_regional = self._get_orders_for_regional(regional_code)

                    if order is None:
                        order = order_regional
                    if seasonal_order is None:
                        seasonal_order = seasonal_regional

                    if log_callback and regional_code and not optimized_config:
                        regional_nombre = {
                            "SAIDI_O": "Ocaña",
                            "SAIDI_C": "Cúcuta",
                            "SAIDI_A": "Aguachica",
                            "SAIDI_P": "Pamplona",
                            "SAIDI_T": "Tibú",
                            "SAIDI_Cens": "CENS",
                        }.get(regional_code, regional_code)

                        log_callback(f"Usando parametros default para regional {regional_nombre}")
                        log_callback(f"   Order: {order}")
                        log_callback(f"   Seasonal Order: {seasonal_order}")

                # Determinar transformacion (prioriza optimizada)
                transformation = self._get_transformation_for_regional(regional_code)

                # Detectar si hay simulacion
                simulation_applied = simulation_config and simulation_config.get("enabled", False)

                if log_callback:
                    log_callback(f"Iniciando validacion con parametros: order={order}, seasonal_order={seasonal_order}")
                    log_callback(f"Regional: {regional_code} - Transformacion: {transformation.upper()}")

                    if simulation_applied:
                        log_callback("=" * 60)
                        log_callback("VALIDACIÓN CON SIMULACIÓN CLIMÁTICA")
                        log_callback("=" * 60)

                        summary = simulation_config.get("summary", {})
                        log_callback(f"Escenario: {summary.get('escenario', 'N/A')}")
                        log_callback(f"Alcance: {simulation_config.get('alcance_meses', 'N/A')} meses")
                        log_callback(f"Días base: {simulation_config.get('dias_base', 'N/A')}")
                        log_callback(f"Ajuste: {simulation_config.get('slider_adjustment', 0):+d} días")
                        log_callback(f"Total días simulados: {summary.get('dias_simulados', 'N/A')}")

                        # Mostrar variables que se modificarán
                        if "variables_afectadas" in summary:
                            log_callback("\nVariables climáticas a modificar:")
                            vars_afectadas = summary["variables_afectadas"]
                            for var_info in vars_afectadas.values():
                                change = var_info.get("cambio_porcentual", 0)
                                arrow = "↑" if change > 0 else "↓" if change < 0 else "→"
                                log_callback(f"   {arrow} {var_info['nombre']}: {change:+.1f}%")

                        log_callback("")
                        log_callback("NOTA: Validación bajo condiciones climáticas HIPOTÉTICAS")
                        log_callback("Las métricas reflejan el comportamiento del modelo")
                        log_callback("bajo el escenario simulado, NO el clima real histórico")
                        log_callback("=" * 60)
                    else:
                        log_callback("Modo: Validacion estandar (sin simulacion)")

                if progress_callback:
                    progress_callback(10, "Cargando datos...")

                # Cargar datos SAIDI
                df = self._load_saidi_data(file_path, df_prepared, log_callback)

                if log_callback:
                    log_callback(f"Columnas encontradas: {df.columns.tolist()}")

                # Asegurar indice datetime
                df = self._ensure_datetime_index(df)

                # Buscar columna SAIDI
                col_saidi = self._find_saidi_column(df)

                historico = df[df[col_saidi].notna()]

                meses = 12
                self._validate_minimum_observations(historico, meses)

                if log_callback:
                    log_callback(f"Dataset: {len(historico)} observaciones")
                    log_callback(f"Periodo: {historico.index[0].strftime('%Y-%m')} a {historico.index[-1].strftime('%Y-%m')}")

                if progress_callback:
                    progress_callback(20, "Preparando variables exogenas...")

                # Preparar variables exogenas (si disponibles)
                exog_df = None
                exog_info = None

                if climate_data is not None and not climate_data.empty:
                    exog_df, exog_info = self._prepare_exogenous_variables(
                        climate_data, df, regional_code, log_callback,
                    )

                    if exog_df is not None:
                        if log_callback:
                            log_callback(f"Variables exogenas disponibles: {len(exog_df.columns)}")

                        # ========== NUEVO: Validar cobertura como OptimizationService ==========
                        if not self._diagnose_exog_coverage(historico[col_saidi], exog_df, log_callback):
                            if log_callback:
                                log_callback("=" * 60)
                                log_callback("ADVERTENCIA: Cobertura insuficiente")
                                log_callback("Las variables exogenas seran DESACTIVADAS")
                                log_callback("=" * 60)
                            exog_df = None
                            exog_info = None
                        else:
                            # Solo si pasa la validación, continuar con las exógenas
                            if log_callback:
                                for var_data in exog_info.values():
                                    log_callback(f"  - {var_data['nombre']}")

                            if simulation_applied:
                                # [Código de simulación - se omite por ahora]
                                pass
                            else:
                                if log_callback:
                                    log_callback("Variables exogenas en escala ORIGINAL")
                                    log_callback("SARIMAX las normalizara internamente")

                                # Guardar scaler solo para compatibilidad, pero NO transformar
                                self.exog_scaler = StandardScaler()
                                self.exog_scaler.fit(exog_df)  # Solo FIT, NO transform
                                # exog_df permanece completamente SIN ESCALAR
                    elif log_callback:
                        log_callback("No se pudieron preparar variables exogenas, continuando sin ellas")
                elif log_callback:
                    log_callback("No hay datos climaticos disponibles, validacion sin variables exogenas")

                if progress_callback:
                    progress_callback(30, "Dividiendo datos para validacion...")

                # Determinar porcentaje de validacion (IDENTICO a OptimizationService)
                n_obervaciones_mayor_60 = 60
                n_obervaciones_mayor_36 = 36
                n_obs = len(historico)
                if n_obs >= n_obervaciones_mayor_60:
                    pct_validacion = 0.30
                elif n_obs >= n_obervaciones_mayor_36:
                    pct_validacion = 0.25
                else:
                    pct_validacion = 0.20

                n_test = max(6, int(n_obs * pct_validacion))
                datos_entrenamiento_original = historico[col_saidi][:-n_test]
                datos_validacion_original = historico[col_saidi][-n_test:]

                if log_callback:
                    log_callback(f"Division: {len(datos_entrenamiento_original)} datos entrenamiento, {len(datos_validacion_original)} datos validacion")
                    log_callback(f"Porcentaje validacion: {pct_validacion*100:.0f}%")

                if progress_callback:
                    progress_callback(40, f"Aplicando transformacion {transformation.upper()}...")

                # Aplicar transformacion segun regional
                train_transformed, transform_info = self._apply_transformation(
                    datos_entrenamiento_original.values, transformation,
                )
                datos_entrenamiento_transformed = pd.Series(train_transformed, index=datos_entrenamiento_original.index)

                if log_callback:
                    log_callback(f"Transformacion aplicada: {transform_info}")

                if progress_callback:
                    progress_callback(50, "Entrenando modelo SARIMAX con datos transformados...")

                # Preparar variables exogenas para entrenamiento y validacion
                exog_train = None
                exog_test = None

                if exog_df is not None:
                    try:
                        # Extraer variables para entrenamiento (sin escalar)
                        exog_train = exog_df.loc[datos_entrenamiento_original.index]

                        if simulation_applied:
                            if log_callback:
                                log_callback("=" * 60)
                                log_callback("PREPARANDO VALIDACIÓN CON SIMULACIÓN CLIMÁTICA")
                                log_callback("=" * 60)

                            # Obtener variables SIN ESCALAR para el periodo de validación
                            exog_test_original = exog_df.loc[datos_validacion_original.index]

                            if log_callback:
                                log_callback("Variables de validación ANTES de simulación:")
                                log_callback(f"  Periodo: {exog_test_original.index[0].strftime('%Y-%m')} a {exog_test_original.index[-1].strftime('%Y-%m')}")
                                log_callback(f"  Variables: {len(exog_test_original.columns)}")
                                log_callback(f"  Shape: {exog_test_original.shape}")

                            # Aplicar simulación (retorna en escala original)
                            exog_test = self._apply_climate_simulation(
                                exog_test_original, simulation_config, log_callback,
                            )

                            if log_callback:
                                summary = simulation_config.get("summary", {})
                                log_callback("\n✓ Simulación aplicada a periodo de validación:")
                                log_callback(f"  - Escenario: {summary.get('escenario', 'N/A')}")
                                log_callback(f"  - Alcance: {simulation_config.get('alcance_meses', 'N/A')} meses")
                                log_callback(f"  - Días simulados: {summary.get('dias_simulados', 'N/A')}")
                                log_callback(f"  - Periodos afectados: {len(exog_test)}")
                                log_callback("=" * 60)
                        else:
                            # Sin simulación: usar directamente (sin escalar)
                            exog_test = exog_df.loc[datos_validacion_original.index]

                            if log_callback:
                                log_callback("Variables de validación SIN simulación (escala original)")

                        # Validación de dimensiones
                        if log_callback:
                            log_callback("\nVariables exógenas preparadas:")
                            log_callback(f"  - Entrenamiento: {len(exog_train)} periodos x {exog_train.shape[1]} variables")
                            log_callback(f"  - Validación: {len(exog_test)} periodos x {exog_test.shape[1]} variables")
                            if simulation_applied:
                                log_callback("  - Modo: CON SIMULACIÓN CLIMÁTICA")
                            else:
                                log_callback("  - Modo: SIN SIMULACIÓN")
                            log_callback("  - Escala: ORIGINAL (SARIMAX normaliza internamente)")

                    except (KeyError, IndexError, ValueError) as e:
                        if log_callback:
                            log_callback(f"ERROR preparando variables exógenas: {e!s}")
                            log_callback(traceback.format_exc())

                        # Fallback: desactivar exógenas
                        exog_train = None
                        exog_test = None

                        if log_callback:
                            log_callback("ADVERTENCIA: Variables exógenas desactivadas por error")

                # Entrenar modelo con datos TRANSFORMADOS
                try:
                    model = SARIMAX(
                        datos_entrenamiento_transformed,
                        exog=exog_train,
                        order=order,
                        seasonal_order=seasonal_order,
                        enforce_stationarity=True,
                        enforce_invertibility=True,
                    )
                    results = model.fit(disp=False)

                    if log_callback:
                        log_callback(f"Modelo SARIMAX ajustado con transformacion {transformation.upper()}")
                        if exog_train is not None:
                            log_callback(f"Modelo incluye {exog_train.shape[1]} variables exogenas")

                except (ValueError, np.linalg.LinAlgError) as e:
                    msg = f"Error ajustando modelo: {e!s}"
                    raise ModelFittingError(msg) from e

                if progress_callback:
                    progress_callback(70, "Generando predicciones de validacion...")

                # Predecir en escala TRANSFORMADA
                try:
                    pred = results.get_forecast(steps=n_test, exog=exog_test)
                    predicciones_transformed = pred.predicted_mean

                    # Revertir predicciones a escala ORIGINAL
                    predicciones_original = self._inverse_transformation(
                        predicciones_transformed.values, transformation,
                    )

                    predicciones_validacion = pd.Series(predicciones_original, index=predicciones_transformed.index)

                    if log_callback:
                        log_callback(f"Predicciones generadas y revertidas a escala original para {len(predicciones_validacion)} periodos")
                        if simulation_applied:
                            log_callback("  (basadas en condiciones climaticas simuladas)")

                except (ValueError, KeyError) as e:
                    msg = f"Error generando predicciones: {e!s}"
                    raise PredictionError(msg) from e

                if progress_callback:
                    progress_callback(85, "Calculando metricas de validacion...")

                # Calcular metricas IDENTICO a OptimizationService
                params = ValidationMetricsParams(
                    order=order,
                    seasonal_order=seasonal_order,
                    transformation=transformation,
                    with_exogenous=exog_df is not None,
                    pct_validacion=pct_validacion,
                    n_test=n_test,
                )

                metricas = self._calcular_metricas_validacion_optimized(
                    datos_validacion_original.values,
                    predicciones_original,
                    params,
                )

                # Calcular complejidad del modelo (IDENTICO a OptimizationService)
                complexity_penalty = sum(order) + sum(seasonal_order[:3])
                composite_score = metricas["rmse"] + (complexity_penalty * 0.05)

                # Calcular estabilidad (IDENTICO a OptimizationService)
                stability_score = self._calculate_stability_numpy(
                    datos_validacion_original.values,
                    predicciones_original,
                    metricas["precision_final"],
                    metricas["mape"],
                )

                # Agregar métricas adicionales
                metricas["composite_score"] = composite_score
                metricas["stability_score"] = stability_score
                metricas["complexity"] = complexity_penalty

                if log_callback:
                    log_callback("=" * 60)
                    if simulation_applied:
                        log_callback("METRICAS DE VALIDACION CON SIMULACION CLIMATICA")
                        summary = simulation_config.get("summary", {})
                        log_callback(f"Escenario simulado: {summary.get('escenario', 'N/A')}")
                        log_callback(f"Alcance: {simulation_config.get('alcance_meses', 'N/A')} meses")
                    else:
                        log_callback("METRICAS DEL MODELO (Calculadas como OptimizationService)")

                    log_callback("=" * 60)
                    log_callback(f"RMSE: {metricas['rmse']:.4f} minutos")
                    log_callback(f"MAE: {metricas['mae']:.4f} minutos")
                    log_callback(f"MAPE: {metricas['mape']:.1f}%")
                    log_callback(f"R2: {metricas['r2_score']:.3f}")
                    log_callback(f"PRECISION FINAL: {metricas['precision_final']:.1f}%")
                    log_callback(f"Stability Score: {stability_score:.1f}/100")
                    log_callback(f"Complejidad del modelo: {complexity_penalty} parametros")
                    log_callback(f"Composite Score: {composite_score:.4f}")

                    # Interpretación de precisión
                    precision_excelente = 60
                    precision_buena = 40
                    precision_aceptable = 20
                    precision = metricas["precision_final"]
                    if precision >= precision_excelente:
                        interpretacion = "EXCELENTE - Predicciones muy confiables"
                    elif precision >= precision_buena:
                        interpretacion = "BUENO - Predicciones confiables"
                    elif precision >= precision_aceptable:
                        interpretacion = "ACEPTABLE - Predicciones moderadamente confiables"
                    else:
                        interpretacion = "LIMITADO - Modelo poco confiable"

                    log_callback(f"INTERPRETACION: {interpretacion}")
                    log_callback(f"Validacion: {pct_validacion*100:.0f}% de datos como test ({n_test} meses)")

                    if simulation_applied:
                        log_callback("")
                        log_callback(" ADVERTENCIA: Métricas bajo condiciones climáticas SIMULADAS")
                        log_callback("   Los valores reflejan el desempeño del modelo bajo el escenario:")
                        log_callback(f"   '{summary.get('escenario', 'N/A')}' con {summary.get('dias_simulados', 'N/A')} días")
                        log_callback("   Los valores reales pueden DIFERIR significativamente")
                        log_callback("   si el clima no sigue este patrón hipotético")

                    if exog_info:
                        log_callback("\nVariables exogenas utilizadas en validacion:")
                        for var_data in exog_info.values():
                            correlacion_str = f" (r={var_data['correlacion']:.3f})" if var_data.get("correlacion", 0) != 0 else ""
                            log_callback(f"  - {var_data['nombre']}{correlacion_str}")

                        if simulation_applied:
                            log_callback("Estas variables fueron MODIFICADAS según el escenario simulado")

                    log_callback("=" * 60)

                if progress_callback:
                    progress_callback(95, "Generando grafica de validacion...")

                # Generar grafica con datos en escala ORIGINAL
                plot_config = ValidationPlotConfig(
                    datos_entrenamiento=datos_entrenamiento_original,
                    datos_validacion=datos_validacion_original,
                    predicciones_validacion=predicciones_validacion,
                    order=order,
                    seasonal_order=seasonal_order,
                    metricas=metricas,
                    pct_validacion=pct_validacion,
                    transformation=transformation,
                    exog_info=exog_info,
                    simulation_config=simulation_config if simulation_applied else None,
                )

                plot_path = self._generar_grafica_validacion(plot_config)

                if progress_callback:
                    progress_callback(100, "Validacion completada exitosamente")

                return {
                    "success": True,
                    "metrics": metricas,
                    "model_params": {
                        "order": order,
                        "seasonal_order": seasonal_order,
                        "transformation": transformation,
                        "regional_code": regional_code,
                        "with_exogenous": exog_df is not None,
                        "with_simulation": simulation_applied,
                        "complexity": complexity_penalty,
                    },
                    "predictions": {
                        "mean": predicciones_validacion.to_dict(),
                    },
                    "exogenous_vars": exog_info,
                    "simulation_config": simulation_config if simulation_applied else None,
                    "training_count": len(datos_entrenamiento_original),
                    "validation_count": len(datos_validacion_original),
                    "validation_percentage": pct_validacion * 100,
                    "training_period": {
                        "start": datos_entrenamiento_original.index[0].strftime("%Y-%m"),
                        "end": datos_entrenamiento_original.index[-1].strftime("%Y-%m"),
                    },
                    "validation_period": {
                        "start": datos_validacion_original.index[0].strftime("%Y-%m"),
                        "end": datos_validacion_original.index[-1].strftime("%Y-%m"),
                    },
                    "plot_file": plot_path,
                }

            except ValidationDataError:
                # Re-raise custom exceptions as-is
                raise
            except ModelFittingError:
                raise
            except PredictionError:
                raise
            except (KeyError, ValueError, pd.errors.EmptyDataError) as e:
                if log_callback:
                    log_callback(f"ERROR: {e!s}")
                msg = f"Error en validacion: {e!s}"
                raise ValidationError(msg) from e


    def _load_saidi_data(self, file_path: str | None, df_prepared: pd.DataFrame | None, log_callback) -> pd.DataFrame:
            """Carga los datos SAIDI desde archivo o DataFrame preparado."""
            if df_prepared is not None:
                df = df_prepared.copy()
                if log_callback:
                    log_callback("Usando datos preparados del modelo")
                return df

            if file_path is not None:
                df = pd.read_excel(file_path, sheet_name="Hoja1")
                if log_callback:
                    log_callback("Leyendo Excel en formato tradicional")
                return df

            msg = "Debe proporcionar file_path o df_prepared"
            raise ValidationDataError(msg)

    def _ensure_datetime_index(self, df: pd.DataFrame, log_callback=None) -> pd.DataFrame:
        """Asegurar que el DataFrame tenga un índice datetime."""
        if isinstance(df.index, pd.DatetimeIndex):
            return df

        # Para datos SAIDI
        if "Fecha" in df.columns:
            df = df.copy()
            df["Fecha"] = pd.to_datetime(df["Fecha"])
            return df.set_index("Fecha")

        # Para datos climáticos
        fecha_col = self._find_date_column(df)
        if fecha_col is not None:
            try:
                df = df.copy()
                df[fecha_col] = pd.to_datetime(df[fecha_col])
                return df.set_index(fecha_col)
            except (ValueError, TypeError, KeyError) as e:
                if log_callback:
                    log_callback(f"ERROR convirtiendo índice: {e!s}")
                return None

        # Fallback: usar primera columna
        df = df.copy()
        df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
        return df.set_index(df.columns[0])

    def _find_saidi_column(self, df: pd.DataFrame) -> str:
            """Busca y retorna el nombre de la columna SAIDI."""
            if "SAIDI" in df.columns:
                return "SAIDI"
            if "SAIDI Historico" in df.columns:
                return "SAIDI Historico"

            msg = "No se encontro la columna SAIDI"
            raise ValidationDataError(msg)

    def _validate_minimum_observations(self, historico: pd.DataFrame, meses: int) -> None:
            """Valida que haya suficientes observaciones históricas."""
            if len(historico) < meses:
                msg = f"Se necesitan al menos {meses} observaciones historicas para la validacion"
                raise ValidationDataError(msg)


    def _apply_climate_simulation(
    self,
    exog_forecast_original,
    simulation_config,
    log_callback=None,
    ):
        """
        Aplicar simulación climática a variables SIN ESCALAR.

        Args:
            exog_forecast_original: DataFrame SIN ESCALAR con variables exógenas
            simulation_config: Dict con configuración de simulación
            log_callback: Función para logging

        Returns:
            DataFrame con simulación aplicada (sin escalar para SARIMAX)

        """
        try:
            # Validar configuración
            if not self._validate_simulation_config(simulation_config, log_callback):
                return exog_forecast_original

            # Extraer parámetros
            sim_params = self._extract_simulation_params(simulation_config)

            # Validar percentiles
            if not sim_params["percentiles"]:
                self._log_missing_percentiles(log_callback)
                return exog_forecast_original

            # Logging de configuración
            self._log_simulation_config(sim_params, log_callback)

            # Aplicar simulación
            exog_simulated = self._execute_simulation(
                exog_forecast_original,
                sim_params,
                log_callback,
            )

        except (ValueError, TypeError, KeyError, AttributeError) as e:
            if log_callback:
                log_callback(f"ERROR CRÍTICO en _apply_climate_simulation: {e!s}")
                log_callback(traceback.format_exc())
            return exog_forecast_original
        else:
            return exog_simulated


    def _validate_simulation_config(
        self,
        simulation_config,
        log_callback,
    ) -> bool:
        """
        Validar que la simulación esté habilitada.

        Returns:
            bool: True si está habilitada

        """
        if simulation_config.get("enabled", False):
            return True

        if log_callback:
            log_callback("Simulación NO habilitada, usando valores originales")

        return False


    def _extract_simulation_params(self, simulation_config: dict) -> dict:
        """
        Extraer parámetros de configuración de simulación.

        Returns:
            dict: Parámetros extraídos y procesados

        """
        escenario = simulation_config.get(
            "scenario_name",
            simulation_config.get("escenario", "condiciones_normales"),
        )

        slider_adjustment = simulation_config.get("slider_adjustment", 0)
        dias_base = simulation_config.get("dias_base", 30)
        alcance_meses = simulation_config.get("alcance_meses", 3)
        percentiles = simulation_config.get("percentiles", {})
        regional_code = simulation_config.get("regional_code", "SAIDI_O")

        # Calcular factor de intensidad
        dias_simulados = dias_base + slider_adjustment
        intensity_adjustment = dias_simulados / dias_base if dias_base > 0 else 1.0

        return {
            "escenario": escenario,
            "slider_adjustment": slider_adjustment,
            "dias_base": dias_base,
            "alcance_meses": alcance_meses,
            "percentiles": percentiles,
            "regional_code": regional_code,
            "intensity_adjustment": intensity_adjustment,
        }


    def _log_missing_percentiles(self, log_callback) -> None:
        """Registrar error de percentiles faltantes."""
        if not log_callback:
            return

        log_callback("ERROR: No hay percentiles disponibles para simulación")
        log_callback("Usando valores originales sin simulación")


    def _log_simulation_config(self, sim_params: dict, log_callback) -> None:
        """Registrar configuración de simulación."""
        if not log_callback:
            return

        log_callback("=" * 60)
        log_callback("APLICANDO SIMULACIÓN CLIMÁTICA EN VALIDACIÓN")
        log_callback("=" * 60)
        log_callback("   Entrada: valores originales SIN ESCALAR")
        log_callback(f"   Escenario: {sim_params['escenario']}")
        log_callback(f"   Regional: {sim_params['regional_code']}")
        log_callback(
            f"   Slider: {sim_params['slider_adjustment']:+d} días "
            f"sobre base de {sim_params['dias_base']}",
        )
        log_callback(f"   Alcance: {sim_params['alcance_meses']} mes(es)")
        log_callback(f"   Intensidad calculada: {sim_params['intensity_adjustment']:.2f}x")


    def _execute_simulation(
    self,
    exog_forecast_original,
    sim_params: dict,
    log_callback,
    ):
        """
        Ejecutar la simulación climática.

        Returns:
            DataFrame con simulación aplicada

        """
        try:
            # Crear objeto SimulationConfig
            config = SimulationConfig(
                scenario_name=sim_params["escenario"],
                intensity_adjustment=sim_params["intensity_adjustment"],
                alcance_meses=sim_params["alcance_meses"],
                percentiles=sim_params["percentiles"],
                regional_code=sim_params["regional_code"],
            )

            # Llamar con el objeto config (como en prediction_service)
            exog_simulated = self.simulation_service.apply_simulation(
                exog_forecast=exog_forecast_original,
                config=config,  # ✓ Ahora usa el objeto SimulationConfig
            )

        except (ValueError, TypeError, KeyError, AttributeError, RuntimeError) as sim_error:
            if log_callback:
                log_callback(f"ERROR en apply_simulation: {sim_error!s}")
                log_callback(traceback.format_exc())
                log_callback("FALLBACK: Usando valores originales sin simulación")
            return exog_forecast_original

        else:
            # Logging de éxito
            self._log_simulation_success(
                exog_forecast_original,
                exog_simulated,
                sim_params,
                log_callback,
            )
            return exog_simulated


    def _log_simulation_success(
        self,
        exog_original,
        exog_simulated,
        sim_params: dict,
        log_callback,
    ) -> None:
        """Registrar éxito de simulación y cambios."""
        if not log_callback:
            return

        log_callback("Simulación aplicada correctamente en validación")

        # Mostrar cambios en el primer mes
        if sim_params["alcance_meses"] >= 1 and len(exog_simulated) > 0:
            self._log_first_month_changes(exog_original, exog_simulated, log_callback)

        log_callback("\n   Salida: valores SIMULADOS (escala original)")
        log_callback("=" * 60)


    def _log_first_month_changes(
        self,
        exog_original,
        exog_simulated,
        log_callback,
    ) -> None:
        """Registrar cambios en el primer mes de simulación."""
        log_callback("\n Cambios en primer mes (validación):")

        for col in exog_simulated.columns:
            try:
                original_val = exog_original.iloc[0][col]
                simulated_val = exog_simulated.iloc[0][col]

                change_pct = (
                    ((simulated_val - original_val) / original_val) * 100
                    if original_val != 0
                    else 0
                )

                log_callback(
                    f"     - {col}: {original_val:.2f} → {simulated_val:.2f} "
                    f"({change_pct:+.1f}%)",
                )

            except (ValueError, TypeError, KeyError, IndexError):
                log_callback(f"     - {col}: Error mostrando cambio")

    def _get_transformation_for_regional(self, regional_code: str | None) -> str:
        """
        Obtener transformación para la regional.

        Prioriza configuración optimizada sobre defaults hardcodeados.

        Args:
            regional_code: Código de la regional (ej: 'SAIDI_O')

        Returns:
            str: Tipo de transformación a aplicar

        """
        if not regional_code:
            return "original"

        # PRIORIDAD 1: Intentar cargar configuración optimizada
        optimized_config = self.load_optimized_config(regional_code)

        if optimized_config:
            transformation = optimized_config.get("transformation", "original")
            print(f"[TRANSFORMATION] Usando transformación OPTIMIZADA: {transformation}")
            return transformation

        # PRIORIDAD 2: Usar defaults hardcodeados
        if regional_code in self.REGIONAL_TRANSFORMATIONS:
            transformation = self.REGIONAL_TRANSFORMATIONS[regional_code]
            print(f"[TRANSFORMATION] Usando transformación DEFAULT: {transformation}")
            return transformation

        # FALLBACK: Original
        print("[TRANSFORMATION] Usando transformación FALLBACK: original")
        return "original"

    def _prepare_exogenous_variables(self,
                                 climate_data: pd.DataFrame,
                                 df_saidi: pd.DataFrame,
                                 regional_code: str | None,
                                 log_callback) -> tuple[pd.DataFrame | None, dict | None]:
        """
        Preparar variables exógenas climáticas SIN ESCALAR.

        Args:
            climate_data: DataFrame con datos climáticos mensuales
            df_saidi: DataFrame SAIDI completo
            regional_code: Código de la regional
            log_callback: Función para logging

        Returns:
            Tuple de (exog_df, exog_info) o (None, None) si falla
            - exog_df: DataFrame EN ESCALA ORIGINAL
            - exog_info: Dict con metadata de cada variable

        """
        try:
            # Ejecutar pipeline de preparación
            pipeline_result = self._execute_exog_preparation_pipeline(
                climate_data, df_saidi, regional_code, log_callback,
            )

            if pipeline_result is None:
                return None, None

            exog_df, exog_info = pipeline_result

            # Validar resultado final
            if not self._validate_final_exog_result(exog_df, log_callback):
                return None, None

            self._log_final_summary(exog_df, log_callback)

        except (ValueError, TypeError, KeyError, AttributeError, IndexError) as e:
            if log_callback:
                log_callback(f"ERROR CRÍTICO: {e!s}")
            return None, None
        else:
            return exog_df, exog_info if exog_info else None


    def _execute_exog_preparation_pipeline(
        self,
        climate_data: pd.DataFrame,
        df_saidi: pd.DataFrame,
        regional_code: str | None,
        log_callback,
    ) -> tuple[pd.DataFrame, dict] | None:
        """
        Ejecutar pipeline completo de preparación de variables exógenas.

        Returns:
            Tuple (exog_df, exog_info) o None si falla algún paso

        """
        # Paso 1: Validaciones iniciales
        if not self._validate_initial_inputs(climate_data, regional_code, log_callback):
            return None

        # Paso 2: Preparar índice datetime
        climate_data = self._ensure_datetime_index(climate_data, log_callback)
        if climate_data is None:
            if log_callback:
                log_callback("ERROR: No se pudo procesar índice de fechas")
            return None

        # Paso 3: Análisis de cobertura temporal
        overlap_info = self._analyze_temporal_coverage(
            climate_data, df_saidi, log_callback,
        )
        if overlap_info is None:
            return None

        # Paso 4: Mapeo de columnas
        climate_column_mapping = self._map_climate_columns(
            climate_data, regional_code, log_callback,
        )
        if not climate_column_mapping:
            return None

        # Paso 5: Construir DataFrame final
        exog_df, exog_info = self._build_exogenous_dataframe(
            climate_data,
            regional_code,
            climate_column_mapping,
            overlap_info,
            log_callback,
        )

        return (exog_df, exog_info) if exog_df is not None else None


    def _validate_final_exog_result(
        self,
        exog_df: pd.DataFrame | None,
        log_callback,
    ) -> bool:
        """
        Validar que el DataFrame de variables exógenas sea válido.

        Returns:
            bool: True si es válido, False si no

        """
        if exog_df is None or exog_df.empty or exog_df.shape[1] == 0:
            if log_callback:
                log_callback("ERROR: Ninguna variable aceptada")
            return False
        return True


    def _validate_initial_inputs(self,
                                climate_data: pd.DataFrame | None,
                                regional_code: str | None,
                                log_callback) -> bool:
        """Validar datos climáticos y código regional."""
        if climate_data is None or climate_data.empty:
            if log_callback:
                log_callback("Sin datos climáticos disponibles")
            return False

        if not regional_code or regional_code not in self.REGIONAL_EXOG_VARS:
            if log_callback:
                log_callback(f"Regional {regional_code} sin variables definidas")
            return False

        if log_callback:
            log_callback(f"Preparando variables para {regional_code}")
            log_callback("MODO: SIN ESCALADO (valores originales)")

        return True

    def _find_date_column(self, climate_data: pd.DataFrame) -> str | None:
        """Encontrar columna de fecha en el DataFrame."""
        date_column_names = ["fecha", "Fecha", "date", "Date", "month_date"]
        for col in date_column_names:
            if col in climate_data.columns:
                return col
        return None


    def _analyze_temporal_coverage(self,
                                climate_data: pd.DataFrame,
                                df_saidi: pd.DataFrame,
                                log_callback) -> dict | None:
        """Analizar cobertura temporal entre SAIDI y datos climáticos."""
        historico = self._get_historical_saidi(df_saidi)

        saidi_start = historico.index[0]
        saidi_end = historico.index[-1]
        clima_start = climate_data.index[0]
        clima_end = climate_data.index[-1]

        # Calcular periodo de overlap
        overlap_start = max(saidi_start, clima_start)
        overlap_end = min(saidi_end, clima_end)

        if overlap_start > overlap_end:
            if log_callback:
                log_callback("ERROR: Sin overlap entre SAIDI y CLIMA")
            return None

        overlap_mask = (historico.index >= overlap_start) & (historico.index <= overlap_end)
        overlap_months = overlap_mask.sum()

        # Validar overlap mínimo (12 meses)
        meses = 12
        if overlap_months < meses:
            if log_callback:
                log_callback(f"ERROR: Overlap insuficiente ({overlap_months} < 12 meses)")
            return None

        if log_callback:
            log_callback(f"SAIDI: {saidi_start.strftime('%Y-%m')} a {saidi_end.strftime('%Y-%m')} ({len(historico)} meses)")
            log_callback(f"CLIMA: {clima_start.strftime('%Y-%m')} a {clima_end.strftime('%Y-%m')} ({len(climate_data)} meses)")
            log_callback(f"OVERLAP: {overlap_start.strftime('%Y-%m')} a {overlap_end.strftime('%Y-%m')} ({overlap_months} meses)")

        return {
            "historico": historico,
            "overlap_mask": overlap_mask,
            "overlap_months": overlap_months,
            "overlap_start": overlap_start,
            "overlap_end": overlap_end,
        }


    def _get_historical_saidi(self, df_saidi: pd.DataFrame) -> pd.DataFrame:
        """Obtener datos históricos de SAIDI."""
        if "SAIDI" in df_saidi.columns:
            return df_saidi[df_saidi["SAIDI"].notna()]
        return df_saidi[df_saidi["SAIDI Historico"].notna()]


    def _map_climate_columns(self,
                            climate_data: pd.DataFrame,
                            regional_code: str,
                            log_callback) -> dict:
        """Mapear variables exógenas a columnas del clima."""
        exog_vars_config = self.REGIONAL_EXOG_VARS[regional_code]

        # Normalizar nombres disponibles
        available_cols_normalized = self._normalize_column_names(climate_data)

        # Mapear cada variable
        climate_column_mapping = {}
        for var_code in exog_vars_config:
            matched_col = self._find_matching_column(
                var_code, available_cols_normalized,
            )
            if matched_col:
                climate_column_mapping[var_code] = matched_col

        if not climate_column_mapping and log_callback:
            log_callback("ERROR: No se pudo mapear ninguna variable")

        return climate_column_mapping


    def _normalize_column_names(self, climate_data: pd.DataFrame) -> dict:
        """Normalizar nombres de columnas para búsqueda flexible."""
        normalized = {}
        for col in climate_data.columns:
            norm_name = col.lower().strip().replace(" ", "_").replace("-", "_")
            normalized[norm_name] = col
        return normalized


    def _find_matching_column(self,
                            var_code: str,
                            available_cols_normalized: dict) -> str | None:
        """Encontrar columna que coincida con el código de variable."""
        var_normalized = var_code.lower().strip()

        # Intento 1: Coincidencia exacta
        if var_normalized in available_cols_normalized:
            return available_cols_normalized[var_normalized]

        # Intento 2: Coincidencia parcial (al menos 2 partes)
        var_parts = var_normalized.split("_")
        best_match = None
        best_match_score = 0

        for norm_col, orig_col in available_cols_normalized.items():
            matches = sum(1 for part in var_parts if part in norm_col)
            if matches > best_match_score:
                best_match_score = matches
                best_match = orig_col

        mejor_score = 2
        return best_match if best_match_score >= mejor_score else None


    def _build_exogenous_dataframe(self,
                                climate_data: pd.DataFrame,
                                regional_code: str,
                                climate_column_mapping: dict,
                                overlap_info: dict,
                                log_callback) -> tuple[pd.DataFrame | None, dict | None]:
        """Construir DataFrame de variables exógenas."""
        exog_vars_config = self.REGIONAL_EXOG_VARS[regional_code]
        historico = overlap_info["historico"]
        overlap_mask = overlap_info["overlap_mask"]
        overlap_months = overlap_info["overlap_months"]

        exog_df = pd.DataFrame(index=historico.index)
        exog_info = {}

        for var_code, var_nombre in exog_vars_config.items():
            result = self._process_single_variable(
                var_code,
                var_nombre,
                climate_data,
                climate_column_mapping,
                historico,
                overlap_mask,
                overlap_months,
                regional_code,
                log_callback,
            )

            if result is not None:
                aligned_series, var_info = result
                exog_df[var_code] = aligned_series
                exog_info[var_code] = var_info

        return (exog_df, exog_info) if not exog_df.empty else (None, None)


    def _process_single_variable(self,  # noqa: PLR0913
                                var_code: str,
                                var_nombre: str,
                                climate_data: pd.DataFrame,
                                climate_column_mapping: dict,
                                historico: pd.DataFrame,
                                overlap_mask: pd.Series,
                                overlap_months: int,
                                regional_code: str,
                                log_callback) -> tuple[pd.Series, dict] | None:
        """Procesar una variable exógena individual."""
        climate_col = climate_column_mapping.get(var_code)

        if not climate_col or climate_col not in climate_data.columns:
            return None

        try:
            # Extraer y alinear serie
            var_series = climate_data[climate_col].copy()
            aligned_series = self._align_series_to_index(
                var_series, historico.index,
            )

            # Validar cobertura
            overlap_data = aligned_series[overlap_mask]
            coverage_valid, overlap_pct = self._validate_coverage(
                overlap_data, overlap_months, var_code, log_callback,
            )
            if not coverage_valid:
                return None

            # Validar varianza
            if not self._validate_variance(overlap_data, var_code, log_callback):
                return None

            # Rellenar valores faltantes
            aligned_series = self._fill_missing_values(
                aligned_series, overlap_data,
            )

            # Verificación final
            final_nan = aligned_series.isna().sum()
            if final_nan > 0:
                if log_callback:
                    log_callback(f"RECHAZADA {var_code}: {final_nan} NaN finales")
                return None

            var_info = {
                "nombre": var_nombre,
                "columna_clima": climate_col,
                "correlacion": self._get_correlation_for_var(var_code, regional_code),
                "scaled": False,
                "datos_reales_overlap": int(overlap_data.notna().sum()),
                "overlap_coverage_pct": float(overlap_pct),
                "varianza_overlap": float(overlap_data.std()),
            }

            if log_callback:
                log_callback(f" {var_code} -> ACEPTADA ({overlap_pct:.1f}% cobertura, escala original)")

            return aligned_series, var_info  # noqa: TRY300

        except (ValueError, TypeError, KeyError, IndexError) as e:
            if log_callback:
                log_callback(f"X ERROR {var_code}: {e}")
            return None


    def _align_series_to_index(self,
                            var_series: pd.Series,
                            target_index: pd.Index) -> pd.Series:
        """Alinear serie a un índice objetivo."""
        aligned_series = pd.Series(index=target_index, dtype=float)

        for date in target_index:
            if date in var_series.index:
                aligned_series[date] = var_series.loc[date]

        return aligned_series


    def _validate_coverage(self,
                        overlap_data: pd.Series,
                        overlap_months: int,
                        var_code: str,
                        log_callback) -> tuple[bool, float]:
        """Validar cobertura mínima de datos."""
        datos_reales_overlap = overlap_data.notna().sum()
        overlap_pct = (datos_reales_overlap / overlap_months) * 100

        overlap_min = 80
        if overlap_pct < overlap_min:
            if log_callback:
                log_callback(f"X RECHAZADA {var_code}: cobertura {overlap_pct:.1f}% < 80%")
            return False, overlap_pct

        return True, overlap_pct


    def _validate_variance(self,
                        overlap_data: pd.Series,
                        var_code: str,
                        log_callback) -> bool:
        """Validar que la variable tenga varianza."""
        var_std = overlap_data.std()

        if pd.isna(var_std) or var_std == 0:
            if log_callback:
                log_callback(f"RECHAZADA {var_code}: varianza = 0")
            return False

        return True


    def _fill_missing_values(self,
                            aligned_series: pd.Series,
                            overlap_data: pd.Series) -> pd.Series:
        """Rellenar valores faltantes en la serie."""
        # Forward-fill para fechas futuras
        aligned_series = aligned_series.fillna(method="ffill")

        # Backward-fill (máx 3 meses) para fechas pasadas
        aligned_series = aligned_series.fillna(method="bfill", limit=3)

        # Si AÚN hay NaN, rellenar con media del overlap
        if aligned_series.isna().any():
            mean_overlap = overlap_data.mean()
            aligned_series = aligned_series.fillna(mean_overlap)

        return aligned_series


    def _log_final_summary(self, exog_df: pd.DataFrame, log_callback) -> None:
        """Registrar resumen final de variables preparadas."""
        if not log_callback:
            return

        log_callback("=" * 60)
        log_callback(f"Variables preparadas: {len(exog_df.columns)}")
        log_callback("ESCALA: ORIGINAL (sin StandardScaler)")
        log_callback("Rangos:")
        for col in exog_df.columns:
            log_callback(f"    - {col}: [{exog_df[col].min():.2f}, {exog_df[col].max():.2f}]")
        log_callback("=" * 60)

    def _align_exog_to_saidi(self,
                        exog_series: pd.DataFrame,
                        df_saidi: pd.DataFrame,
                        var_code: str,
                        log_callback) -> pd.Series | None:
        """Alinear datos exogenos al indice de SAIDI."""
        try:
            climate_dates = exog_series.index
            saidi_dates = df_saidi.index

            # Asegurar que ambos indices sean DatetimeIndex
            if not isinstance(climate_dates, pd.DatetimeIndex):
                climate_dates = pd.to_datetime(climate_dates)
            if not isinstance(saidi_dates, pd.DatetimeIndex):
                saidi_dates = pd.to_datetime(saidi_dates)

            # Crear serie con valores para todas las fechas SAIDI
            result = pd.Series(index=saidi_dates, dtype=float)

            # Llenar con datos climaticos donde existan
            for date in saidi_dates:
                if date in climate_dates:
                    result[date] = exog_series.loc[date].iloc[0]

            # Forward fill: usar ultimo valor conocido para fechas futuras
            max_climate_date = climate_dates.max()
            future_indices = saidi_dates > max_climate_date

            if future_indices.any():
                last_known_value = exog_series.iloc[-1].iloc[0]
                result.loc[future_indices] = last_known_value

            # Backward fill: usar primer valor para fechas previas
            min_climate_date = climate_dates.min()
            past_indices = saidi_dates < min_climate_date

            if past_indices.any():
                first_known_value = exog_series.iloc[0].iloc[0]
                result.loc[past_indices] = first_known_value

        except (ValueError, TypeError, KeyError, IndexError, AttributeError) as e:
            if log_callback:
                log_callback(f"Error alineando variable {var_code}: {e!s}")
            return None
        else:
            return result

    def _apply_transformation(self, data: np.ndarray, transformation_type: str) -> tuple[np.ndarray, str]:
        """Aplicar transformacion a los datos."""
        if transformation_type == "original":
            return data, "Sin transformacion (datos originales)"

        if transformation_type == "standard":
            self.scaler = StandardScaler()
            transformed = self.scaler.fit_transform(data.reshape(-1, 1)).flatten()
            return transformed, f"StandardScaler (media={self.scaler.mean_[0]:.2f}, std={np.sqrt(self.scaler.var_[0]):.2f})"

        if transformation_type == "log":
            data_positive = np.maximum(data, 1e-10)
            transformed = np.log(data_positive)
            self.transformation_params["log_applied"] = True
            return transformed, "Transformacion logaritmica (log)"

        if transformation_type == "boxcox":
            data_positive = np.maximum(data, 1e-10)
            transformed, lambda_param = stats.boxcox(data_positive)
            self.transformation_params["boxcox_lambda"] = lambda_param
            return transformed, f"Box-Cox (lambda={lambda_param:.4f})"

        if transformation_type == "sqrt":
            data_positive = np.maximum(data, 0)
            transformed = np.sqrt(data_positive)
            self.transformation_params["sqrt_applied"] = True
            return transformed, "Sqrt"

        return data, "Sin transformacion (tipo desconocido)"

    def _inverse_transformation(self, data: np.ndarray, transformation_type: str) -> np.ndarray:
        """Revertir transformacion a escala original."""
        # Diccionario de transformaciones inversas
        transformations = {
            "original": lambda d: d,
            "standard": lambda d: (
                self.scaler.inverse_transform(d.reshape(-1, 1)).flatten()
                if self.scaler is not None
                else d
            ),
            "log": lambda d: np.exp(d),
            "sqrt": lambda d: np.power(d, 2),
            "boxcox": self._inverse_boxcox,
        }

        # Obtener la función de transformación o retornar datos sin cambios
        transform_func = transformations.get(transformation_type, lambda d: d)
        return transform_func(data)


    def _inverse_boxcox(self, data: np.ndarray) -> np.ndarray:
        """Aplicar transformación inversa de Box-Cox."""
        lambda_param = self.transformation_params.get("boxcox_lambda", 0)
        if lambda_param == 0:
            return np.exp(data)
        return np.power(data * lambda_param + 1, 1 / lambda_param)

    def _calcular_metricas_validacion_optimized(
    self,
    test_values: np.ndarray,
    pred_values: np.ndarray,
    params: ValidationMetricsParams,
    ) -> dict[str, float]:
        # Calcular RMSE
        rmse = np.sqrt(mean_squared_error(test_values, pred_values))

        # Calcular MAE
        mae = np.mean(np.abs(test_values - pred_values))

        # Calcular MAPE con epsilon
        epsilon = 1e-8
        mape = np.mean(np.abs((test_values - pred_values) /
                            (test_values + epsilon))) * 100

        # Calcular R2
        ss_res = np.sum((test_values - pred_values) ** 2)
        ss_tot = np.sum((test_values - np.mean(test_values)) ** 2)
        r2_score = 1 - (ss_res / (ss_tot + epsilon))

        # Calcular precision final (IDENTICO a OptimizationService)
        # Fórmula: max(0, min(100, (1 - mape/100) * 100))
        precision_final = max(0.0, min(100.0, (1 - mape/100) * 100))

        # Validación adicional
        if np.isnan(precision_final) or np.isinf(precision_final):
            precision_final = 0.0

        return {
            "rmse": rmse,
            "mae": mae,
            "mape": mape,
            "r2_score": r2_score,
            "precision_final": precision_final,
            "n_test": params.n_test,
            "validation_pct": params.pct_validacion * 100,
        }

    def _calculate_stability_numpy(self,
                               actual_values: np.ndarray,
                               predicted_values: np.ndarray,
                               precision: float,
                               mape: float) -> float:
        """Calcular score de estabilidad."""
        try:
            errors = actual_values - predicted_values

            mean_abs_error = np.mean(np.abs(errors))
            std_error = np.std(errors)

            comparacion = 1e-8
            penalizacion_maxima = 50
            penalizacion_minima = 30

            if mean_abs_error > comparacion:
                cv_error = std_error / mean_abs_error
                stability_cv = max(0, 100 * (1 - min(cv_error, 1)))
            else:
                stability_cv = 50.0

            # Penalización por MAPE alto
            if mape > penalizacion_maxima:
                mape_penalty = 0.5
            elif mape > penalizacion_minima:
                mape_penalty = 0.7
            else:
                mape_penalty = 1.0

            stability_cv = stability_cv * mape_penalty

            # Combinar con precisión
            stability = (stability_cv * 0.6) + (precision * 0.4)

            return min(100.0, max(0.0, stability))

        except (ValueError, TypeError, ZeroDivisionError, FloatingPointError):
            return 0.0

    def _generar_grafica_validacion(
        self,
        config: ValidationPlotConfig,
    ) -> str | None:
        """Generar grafica de validacion con metricas alineadas y soporte COMPLETO para simulacion."""
        try:
            # Validar datos
            if not self._validar_datos_grafica(config):
                return None

            # Crear archivo y figura
            plot_path = self._crear_archivo_temporal("saidi_validation")
            fig = plt.figure(figsize=(16, 10), dpi=100)
            plt.style.use("default")

            # Detectar simulación
            simulation_applied = config.simulation_config and config.simulation_config.get("enabled", False)

            # Graficar componentes
            self._plot_training_data(config)
            self._plot_validation_and_predictions(config, simulation_applied=simulation_applied)
            self._plot_info_boxes(config, simulation_applied=simulation_applied)

            # Configurar ejes y título
            self._configurar_ejes_validacion(config)
            self._configurar_titulo_validacion(simulation_applied=simulation_applied)

            # Guardar
            plt.savefig(plot_path, dpi=100, bbox_inches="tight", facecolor="white", edgecolor="none")
            plt.close(fig)

            self.plot_file_path = str(plot_path)
            return str(plot_path)

        except (ValueError, KeyError, IndexError, OSError) as e:
            print(f"Error generando grafica de validacion: {e}")
            return None

    def _validar_datos_grafica(self, config: ValidationPlotConfig) -> bool:
        """Validar que los datos de la gráfica sean válidos."""
        return (
            config.datos_entrenamiento is not None
            and len(config.datos_entrenamiento) > 0
            and config.datos_validacion is not None
            and len(config.datos_validacion) > 0
            and config.predicciones_validacion is not None
            and len(config.predicciones_validacion) > 0
        )


    def _crear_archivo_temporal(self, prefix: str) -> Path:
        """Crear archivo temporal para gráfica."""
        temp_dir = tempfile.gettempdir()
        timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")  # noqa: UP017
        filename = f"{prefix}_{timestamp}.png"
        return Path(temp_dir) / filename


    def _plot_training_data(self, config: ValidationPlotConfig) -> None:
        """Graficar datos de entrenamiento."""
        plt.plot(
            config.datos_entrenamiento.index,
            config.datos_entrenamiento.values,
            color="#2E86AB",
            linewidth=2,
            label="Datos Históricos (Entrenamiento)",
            marker="o",
            markersize=4,
            alpha=0.8,
        )


    def _plot_validation_and_predictions(
    self,
    config: ValidationPlotConfig,
    *,  # Forzar keyword-only arguments
    simulation_applied: bool,
    ) -> None:
        """Graficar datos de validación y predicciones."""
        # Datos reales de validación
        plt.plot(
            config.datos_validacion.index,
            config.datos_validacion.values,
            color="#A23B72",
            linewidth=2,
            label="Datos Reales (Validación)",
            marker="o",
            markersize=5,
            alpha=0.9,
        )

        # Predicciones
        pred_label = (
            "Predicciones (Con Simulación Climática)"
            if simulation_applied
            else "Predicciones del Modelo"
        )
        pred_color = "#FF6B35" if simulation_applied else "#F18F01"

        plt.plot(
            config.predicciones_validacion.index,
            config.predicciones_validacion.values,
            color=pred_color,
            linewidth=2.5,
            label=pred_label,
            marker="s",
            markersize=5,
            linestyle="--",
            alpha=0.9,
        )

    def _plot_info_boxes(
    self,
    config: ValidationPlotConfig,
    *,  # Forzar keyword-only arguments
    simulation_applied: bool,
    ) -> None:
        """Graficar cajas de información con métricas."""
        # Preparar texto de métricas
        metrics_text = self._format_metrics_text(config.metricas)

        # Preparar texto de parámetros
        params_text = self._format_params_text(config)

        # Caja de métricas
        plt.text(
            0.02, 0.98,
            metrics_text,
            transform=plt.gca().transAxes,
            fontsize=9,
            verticalalignment="top",
            bbox={"boxstyle": "round,pad=0.8", "facecolor": "lightyellow", "alpha": 0.9},
            family="monospace",
        )

        # Caja de parámetros
        plt.text(
            0.98, 0.98,
            params_text,
            transform=plt.gca().transAxes,
            fontsize=9,
            verticalalignment="top",
            horizontalalignment="right",
            bbox={"boxstyle": "round,pad=0.8", "facecolor": "lightblue", "alpha": 0.9},
            family="monospace",
        )

        # Advertencia de simulación si aplica
        if simulation_applied:
            self._plot_simulation_warning(config)


    def _format_metrics_text(self, metricas: dict) -> str:
        """Formatear texto de métricas."""
        return (
            f"MÉTRICAS DE VALIDACIÓN\n"
            f"{'='*25}\n"
            f"RMSE:      {metricas['rmse']:.2f} min\n"
            f"MAE:       {metricas['mae']:.2f} min\n"
            f"MAPE:      {metricas['mape']:.1f}%\n"
            f"R²:        {metricas['r2_score']:.3f}\n"
            f"Precisión: {metricas['precision_final']:.1f}%\n"
            f"Validación: {metricas['validation_pct']:.0f}% ({metricas['n_test']} meses)"
        )


    def _format_params_text(self, config: ValidationPlotConfig) -> str:
        """Formatear texto de parámetros."""
        return (
            f"PARÁMETROS DEL MODELO\n"
            f"{'='*25}\n"
            f"Order: {config.order}\n"
            f"Seasonal: {config.seasonal_order}\n"
            f"Transformación: {config.transformation.upper()}\n"
            f"Variables Exógenas: {'Sí' if config.exog_info else 'No'}"
        )


    def _plot_simulation_warning(self, config: ValidationPlotConfig) -> None:
        """Graficar advertencia de simulación."""
        if not config.simulation_config:
            return

        summary = config.simulation_config.get("summary", {})
        escenario = summary.get("escenario", "N/A")
        dias = summary.get("dias_simulados", "N/A")

        warning_text = (
            f"SIMULACIÓN CLIMÁTICA APLICADA\n"
            f"Escenario: {escenario}\n"
            f"Días simulados: {dias}"
        )

        plt.text(
            0.50, 0.02,
            warning_text,
            transform=plt.gca().transAxes,
            fontsize=8,
            verticalalignment="bottom",
            horizontalalignment="center",
            bbox={"boxstyle": "round,pad=0.6", "facecolor": "orange", "alpha": 0.8},
            family="monospace",
        )


    def _configurar_ejes_validacion(self, config: ValidationPlotConfig) -> None:  # noqa: ARG002
        """Configurar ejes de la gráfica."""
        plt.xlabel("Fecha", fontsize=12, fontweight="bold")
        plt.ylabel("SAIDI (minutos)", fontsize=12, fontweight="bold")
        plt.legend(loc="best", fontsize=10, framealpha=0.9)
        plt.grid(True, alpha=0.3, linestyle="--")

        # Formatear fechas en eje X
        ax = plt.gca()
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.xticks(rotation=45, ha="right")

        plt.tight_layout()


    def _configurar_titulo_validacion(
    self,
    *,  # Forzar keyword-only arguments
    simulation_applied: bool,
    ) -> None:
        """Configurar título de la gráfica."""
        if simulation_applied:
            titulo = "Validación del Modelo SARIMAX con Simulación Climática"
        else:
            titulo = "Validación del Modelo SARIMAX"

        plt.title(titulo, fontsize=14, fontweight="bold", pad=20)

    def cleanup_plot_file(self):
        """Limpiar archivo temporal de grafica."""
        if self.plot_file_path and Path(self.plot_file_path).exists():
            try:
                Path(self.plot_file_path).unlink()
            except (FileNotFoundError, PermissionError, OSError) as e:
                print(f"Error eliminando archivo temporal: {e}")
            finally:
                self.plot_file_path = None

    def _diagnose_exog_coverage(
    self,
    serie_saidi: pd.Series,
    exog_df: pd.DataFrame,
    log_callback,
    ) -> bool:
        """
        Diagnosticar cobertura temporal de variables exógenas.

        Valida:
        1. Índices coinciden exactamente
        2. No hay NaN en ninguna columna
        3. No hay valores infinitos
        4. Variables tienen varianza > 0

        Args:
            serie_saidi: Serie temporal SAIDI
            exog_df: DataFrame con variables exógenas
            log_callback: Función para logging

        Returns:
            bool: True si pasa todas las validaciones

        """
        try:
            # Mostrar diagnóstico inicial
            self._log_coverage_summary(serie_saidi, exog_df, log_callback)

            # Ejecutar validaciones en secuencia
            validations = [
                self._validate_index_alignment(serie_saidi, exog_df, log_callback),
                self._validate_no_nan_values(exog_df, log_callback),
                self._validate_no_infinite_values(exog_df, log_callback),
            ]

            # Si alguna validación crítica falla, retornar False
            if not all(validations):
                return False

            # Validación no crítica: verificar varianza
            self._check_zero_variance_vars(exog_df, log_callback)

        except (IndexError, KeyError, ValueError, AttributeError) as e:
            if log_callback:
                log_callback(f"ERROR durante diagnostico: {e}")
            return False
        else:
            # CORREGIDO: Return movido al bloque else
            if log_callback:
                log_callback("✓ Cobertura temporal y calidad de datos OK")
                log_callback("=" * 60)

            return True


    def _log_coverage_summary(
        self,
        serie_saidi: pd.Series,
        exog_df: pd.DataFrame,
        log_callback,
    ) -> None:
        """Mostrar resumen de cobertura temporal."""
        if not log_callback:
            return

        saidi_start = serie_saidi.index[0]
        saidi_end = serie_saidi.index[-1]
        exog_start = exog_df.index[0]
        exog_end = exog_df.index[-1]

        log_callback("=" * 60)
        log_callback("DIAGNOSTICO DE COBERTURA EXOGENA")
        log_callback("=" * 60)
        log_callback(
            f"SAIDI: {saidi_start.strftime('%Y-%m')} a {saidi_end.strftime('%Y-%m')} "
            f"({len(serie_saidi)} obs)",
        )
        log_callback(
            f"EXOG:  {exog_start.strftime('%Y-%m')} a {exog_end.strftime('%Y-%m')} "
            f"({len(exog_df)} obs)",
        )


    def _validate_index_alignment(
        self,
        serie_saidi: pd.Series,
        exog_df: pd.DataFrame,
        log_callback,
    ) -> bool:
        """
        Validar que los índices coincidan exactamente.

        Returns:
            bool: True si la alineación es válida

        """
        # Si los índices son idénticos, validación exitosa
        if exog_df.index.equals(serie_saidi.index):
            return True

        if log_callback:
            log_callback("ADVERTENCIA: Indices no coinciden exactamente")

        # Calcular fechas faltantes
        missing_in_exog = [d for d in serie_saidi.index if d not in exog_df.index]

        if not missing_in_exog:
            return True

        # Calcular porcentaje de datos faltantes
        pct_missing = len(missing_in_exog) / len(serie_saidi) * 100

        if log_callback:
            log_callback(
                f"Fechas SAIDI faltantes en EXOG: {len(missing_in_exog)} "
                f"({pct_missing:.1f}%)",
            )

        # CRÍTICO: Rechazar si falta >20% de fechas
        max_missing_pct = 20
        if pct_missing > max_missing_pct:
            if log_callback:
                log_callback("ERROR CRITICO: >20% de fechas faltantes")
                log_callback(
                    "Las variables exogenas NO cubren suficiente periodo historico",
                )
            return False

        return True


    def _validate_no_nan_values(
        self,
        exog_df: pd.DataFrame,
        log_callback,
    ) -> bool:
        """
        Validar que no haya valores NaN en las columnas.

        Returns:
            bool: True si no hay NaN

        """
        if not exog_df.isna().any().any():
            return True

        # Identificar columnas con NaN
        nan_cols = exog_df.columns[exog_df.isna().any()].tolist()

        if log_callback:
            log_callback("ERROR: Columnas con NaN encontradas:")
            for col in nan_cols:
                nan_count = exog_df[col].isna().sum()
                pct_nan = (nan_count / len(exog_df)) * 100
                log_callback(f"  - {col}: {nan_count} NaN ({pct_nan:.1f}%)")
            log_callback("Variables exogenas deben estar completamente rellenas")

        return False


    def _validate_no_infinite_values(
        self,
        exog_df: pd.DataFrame,
        log_callback,
    ) -> bool:
        """
        Validar que no haya valores infinitos.

        Returns:
            bool: True si no hay infinitos

        """
        if not np.isinf(exog_df.values).any():
            return True

        if log_callback:
            log_callback("ERROR: Variables exogenas contienen valores infinitos")

        return False


    def _check_zero_variance_vars(
        self,
        exog_df: pd.DataFrame,
        log_callback,
    ) -> None:
        """
        Verificar variables con varianza cero (validación no crítica).

        Solo advierte, no falla la validación.
        """
        if not log_callback:
            return

        zero_variance_vars = [
            col for col in exog_df.columns if exog_df[col].std() == 0
        ]

        if not zero_variance_vars:
            return

        log_callback("ADVERTENCIA: Variables con varianza cero:")
        for var in zero_variance_vars:
            log_callback(f"  - {var}")
        log_callback("Estas variables no aportan informacion al modelo")

    def _get_correlation_for_var(self, var_code: str, regional_code: str) -> float:
        """
        Obtener correlación documentada de una variable específica.

        Args:
            var_code: Código de la variable (ej: 'realfeel_min')
            regional_code: Código de la regional (ej: 'SAIDI_O')

        Returns:
            float: Correlación documentada o 0.0 si no existe

        """
        # Correlaciones REALES documentadas por regional
        correlations = {
            "SAIDI_O": {  # Ocaña
                "realfeel_min": 0.689,              # *** FUERTE
                "windchill_avg": 0.520,             # ** MODERADA-FUERTE
                "dewpoint_avg": 0.470,              # ** MODERADA-FUERTE
                "windchill_max": 0.464,             # ** MODERADA-FUERTE
                "dewpoint_min": 0.456,              # ** MODERADA-FUERTE
                "precipitation_max_daily": 0.452,
                "precipitation_avg_daily": 0.438,
            },

            "SAIDI_C": {  # Cúcuta
                "realfeel_avg": 0.573,              # ** MODERADA-FUERTE
                "pressure_rel_avg": -0.358,         # Negativa
                "wind_speed_max": 0.356,
                "pressure_abs_avg": -0.356,         # Negativa
            },

            "SAIDI_T": {  # Tibú
                "realfeel_avg": 0.906,              # *** MUY FUERTE
                "wind_dir_avg": -0.400,             # Negativa
                "uv_index_avg": 0.385,
                "heat_index_avg": 0.363,
                "temperature_min": 0.352,
                "windchill_min": 0.340,
                "temperature_avg": 0.338,
                "pressure_rel_avg": -0.330,         # Negativa
            },

            "SAIDI_A": {  # Aguachica
                "uv_index_max": 0.664,              # *** FUERTE
                "days_with_rain": 0.535,            # ** MODERADA-FUERTE
            },

            "SAIDI_P": {  # Pamplona
                "precipitation_total": 0.577,       # ** MODERADA-FUERTE
                "precipitation_avg_daily": 0.552,
                "realfeel_min": 0.344,
            },
        }

        # Buscar correlación específica
        if regional_code in correlations and var_code in correlations[regional_code]:
            return correlations[regional_code][var_code]

        return 0.0
