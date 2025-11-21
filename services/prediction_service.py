# services/prediction_service.py
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

import json
import tempfile
import traceback
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ClassVar

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX

from services.climate_simulation_service import ClimateSimulationService
from services.export_service import ExportService
from services.uncertainty_service import UncertaintyService

warnings.filterwarnings("ignore")

@dataclass
class ExportConfig:
    """Configuración para exportación de predicciones."""

    output_dir: str | None = None
    include_intervals: bool = True
    model_params: dict[str, Any] | None = None
    metrics: dict[str, Any] | None = None

@dataclass
class AlignmentContext:
    """Contexto para alineación de datos exógenos."""

    exog_series: pd.Series
    climate_dates: pd.DatetimeIndex
    saidi_dates: pd.DatetimeIndex
    var_code: str
    log_callback: Callable[[str], None] | None = None


class PredictionService:
    """Servicio para generar predicciones SAIDI con variables exogenas climaticas, simulacion e intervalos de confianza."""

    # Mapeo de regionales a sus transformaciones optimas
    REGIONAL_TRANSFORMATIONS: ClassVar[dict[str, str]] = {
        "SAIDI_O": "original",
        "SAIDI_C": "original",
        "SAIDI_A": "original",
        "SAIDI_P": "boxcox",
        "SAIDI_T": "sqrt",
        "SAIDI_Cens": "original",
    }

    REGIONAL_ORDERS: ClassVar[dict[str, dict[str, tuple]]] = {
        "SAIDI_O": {"order": (3, 1, 6), "seasonal_order": (3, 1, 0, 12)},
        "SAIDI_C": {"order": (3, 1, 2), "seasonal_order": (1, 1, 2, 12)},
        "SAIDI_A": {"order": (2, 1, 3), "seasonal_order": (2, 1, 1, 12)},
        "SAIDI_P": {"order": (4, 1, 3), "seasonal_order": (1, 1, 4, 12)},
        "SAIDI_T": {"order": (3, 1, 3), "seasonal_order": (2, 1, 2, 12)},
        "SAIDI_Cens": {"order": (4, 1, 3), "seasonal_order": (1, 1, 4, 12)},
    }

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
        self.default_order = (9, 9, 9)
        self.default_seasonal_order = (8, 8, 8, 12)
        self.plot_file_path = None
        self.scaler = None
        self.exog_scaler = None
        self.transformation_params = {}
        self.simulation_service = ClimateSimulationService()
        self.uncertainty_service = UncertaintyService()
        self.export_service = ExportService()

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
        # Ubicación del archivo de configuración
        config_file = Path(__file__).parent.parent / "config" / "optimized_models.json"

        # Validar existencia del archivo
        if not config_file.exists():
            print("[LOAD_CONFIG] No existe archivo de configuraciones optimizadas")
            return None

        try:
            # Cargar configuraciones
            with config_file.open(encoding="utf-8") as f:
                configs = json.load(f)

            # Buscar configuración de la regional
            if regional_code not in configs:
                print(f"[LOAD_CONFIG] No hay configuración optimizada para {regional_code}")
                return None

            config = configs[regional_code]

        except (FileNotFoundError, json.JSONDecodeError, KeyError, OSError) as e:
            error_messages = {
                FileNotFoundError: "Archivo de configuración no encontrado",
                json.JSONDecodeError: f"Archivo JSON inválido: {e}",
                KeyError: f"Clave faltante en configuración: {e}",
                OSError: f"ERROR de E/S al leer archivo: {e}",
            }
            error_msg = error_messages.get(type(e), f"Error inesperado: {e}")
            print(f"[LOAD_CONFIG] ERROR: {error_msg}")
            return None
        else:
            # Este código solo se ejecuta si NO hubo excepciones
            print(f"[LOAD_CONFIG] ✓ Configuración cargada para {regional_code}")
            print(f"[LOAD_CONFIG]   Transformación: {config['transformation']}")
            print(f"[LOAD_CONFIG]   Order: {config['order']}")
            print(f"[LOAD_CONFIG]   Seasonal: {config['seasonal_order']}")
            print(f"[LOAD_CONFIG]   Precisión: {config['precision_final']:.1f}%")
            print(f"[LOAD_CONFIG]   Optimizado: {config['optimization_date']}")

            return config

    def _get_transformation_for_regional(self, regional_code):
        """
        Obtener transformación para la regional.

        Primero intenta cargar de configuración optimizada,
        si no existe usa los defaults hardcodeados.
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

    def _get_orders_for_regional(self, regional_code):
        """
        Obtener órdenes SARIMAX específicos para una regional.

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

    def export_predictions(
        self,
        predictions_dict: dict,
        regional_code: str,
        regional_nombre: str,
        config: ExportConfig | None = None,
    ) -> str | None:
        """
        Exportar predicciones a Excel usando ExportService.

        Args:
            predictions_dict: Diccionario con predicciones {fecha: valor o dict}
            regional_code: Codigo de regional (ej: 'SAIDI_O')
            regional_nombre: Nombre de regional (ej: 'Ocana')
            config: Configuración de exportación (ExportConfig)

        Returns:
            str: Ruta del archivo exportado o None si hay error

        """
        # Usar configuración por defecto si no se proporciona
        if config is None:
            config = ExportConfig()

        try:
            # Preparar informacion del modelo para exportacion
            model_info = {}

            if config.model_params:
                model_info.update(config.model_params)

            if config.metrics:
                model_info["metrics"] = config.metrics

            # Llamar al servicio de exportacion
            filepath = self.export_service.export_predictions_to_excel(
                predictions_dict=predictions_dict,
                regional_code=regional_code,
                regional_nombre=regional_nombre,
                output_dir=config.output_dir,
                include_confidence_intervals=config.include_intervals,
                model_info=model_info,
            )

        except (OSError, PermissionError, ValueError, TypeError) as e:
            print(f"Error exportando predicciones: {e!s}")
            return None
        else:
            return filepath

    def run_prediction(
    self,
    file_path=None,
    df_prepared=None,
    order=None,
    seasonal_order=None,
    regional_code=None,
    climate_data=None,
    simulation_config=None,
    progress_callback=None,
    log_callback=None,
    ):
        """
        Ejecutar prediccion SAIDI con variables exogenas climaticas, simulacion e intervalos de confianza.

        Carga automáticamente parámetros optimizados si existen

        Args:
            file_path: Ruta del archivo SAIDI Excel
            df_prepared: DataFrame de SAIDI ya preparado
            order: Orden ARIMA (opcional - si None usa el optimizado/default de la regional)
            seasonal_order: Orden estacional ARIMA (opcional - si None usa el optimizado/default)
            regional_code: Codigo de la regional
            climate_data: DataFrame con datos climaticos mensuales
            simulation_config: Configuracion de simulacion climatica (opcional)
            progress_callback: Funcion para actualizar progreso
            log_callback: Funcion para loguear mensajes

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
                order_regional, seasonal_regional = self._get_orders_for_regional(
                    regional_code,
                )

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

                    log_callback(
                        f"✓ Usando parametros default para regional {regional_nombre}",
                    )
                    log_callback(f"   Order: {order}")
                    log_callback(f"   Seasonal Order: {seasonal_order}")

            # Determinar transformacion segun regional (prioriza optimizada)
            transformation = self._get_transformation_for_regional(regional_code)

            if log_callback:
                order_str = str(order)
                seasonal_str = str(seasonal_order)
                log_callback(
                    f"Iniciando prediccion con parametros: order={order_str}, seasonal_order={seasonal_str}",
                )
                log_callback(
                    f"Regional: {regional_code} - Transformacion: {transformation.upper()}",
                )
                log_callback("Modo: CON VARIABLES EXOGENAS E INTERVALOS DE CONFIANZA")

            if progress_callback:
                progress_callback(10, "Cargando datos...")

            # Cargar datos SAIDI
            if df_prepared is not None:
                df = df_prepared.copy()
                if log_callback:
                    log_callback("Usando datos SAIDI preparados del modelo")
            elif file_path is not None:
                df = pd.read_excel(file_path, sheet_name="Hoja1")
                if log_callback:
                    log_callback("Leyendo Excel SAIDI en formato tradicional")
            else:
                raise Exception("Debe proporcionar file_path o df_prepared")

            # Asegurar indice datetime
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
                raise Exception("No se encontro la columna SAIDI")

            if progress_callback:
                progress_callback(20, "Preparando variables exogenas...")

            # Preparar variables exogenas SIN ESCALAR (como OptimizationService)
            exog_df = None
            exog_info = None
            simulation_applied = False

            if climate_data is not None and not climate_data.empty:
                exog_df, exog_info = self._prepare_exogenous_variables(
                    climate_data, df, regional_code, log_callback,
                )

                if exog_df is not None:
                    if log_callback:
                        log_callback(
                            f"Variables exogenas disponibles: {len(exog_df.columns)}",
                        )

                    # ========== NUEVO: Validar cobertura como OptimizationService ==========
                    historico_temp = df[df[col_saidi].notna()]
                    if not self._diagnose_exog_coverage(historico_temp[col_saidi], exog_df, log_callback):
                        if log_callback:
                            log_callback("=" * 60)
                            log_callback("ADVERTENCIA: Cobertura insuficiente")
                            log_callback("Las variables exogenas seran DESACTIVADAS")
                            log_callback("=" * 60)
                        exog_df = None
                        exog_info = None
                    # ========== CORREGIDO: NO escalar aquí ==========
                    # SARIMAX normaliza internamente las variables exógenas
                    # Escalar manualmente causa DOBLE ESCALADO y pérdida de precisión

                    elif simulation_config and simulation_config.get("enabled", False):
                        if log_callback:
                            log_callback("=" * 60)
                            log_callback("SIMULACION CLIMATICA ACTIVADA")
                            log_callback("=" * 60)
                            log_callback(
                                "Variables exogenas SIN ESCALAR (para simulacion)",
                            )
                        simulation_applied = True
                        # Solo guardar scaler para referencia, pero NO aplicar transform
                        self.exog_scaler = StandardScaler()
                        self.exog_scaler.fit(exog_df)  # Solo FIT, no transform
                        # exog_df permanece SIN ESCALAR
                    else:
                        # ========== CAMBIO CRÍTICO: Sin simulación tampoco escalar ==========
                        if log_callback:
                            log_callback("Variables exogenas en escala ORIGINAL")
                            log_callback("SARIMAX las normalizara internamente")
                            log_callback("(Escalado manual eliminado para evitar doble normalización)")

                        # Guardar scaler solo para compatibilidad, pero NO transformar
                        self.exog_scaler = StandardScaler()
                        self.exog_scaler.fit(exog_df)  # Solo FIT, NO transform
                        # exog_df permanece completamente SIN ESCALAR

                        if log_callback:
                            log_callback("Rango de valores exogenas (sin escalar):")
                            for col in exog_df.columns[:3]:  # Mostrar primeras 3
                                log_callback(f"  - {col}: [{exog_df[col].min():.2f}, {exog_df[col].max():.2f}]")
                elif log_callback:
                    log_callback(
                        "No se pudieron preparar variables exogenas, continuando sin ellas",
                    )
            elif log_callback:
                log_callback(
                    "No hay datos climaticos disponibles, prediccion sin variables exogenas",
                )

            if progress_callback:
                progress_callback(25, "Procesando datos historicos...")

            # Separar datos historicos y faltantes
            faltantes = df[df[col_saidi].isna()]
            historico = df[df[col_saidi].notna()]

            if faltantes.empty:
                if log_callback:
                    log_callback("No hay meses faltantes, creando predicciones futuras")
                ultimo_mes = historico.index[-1]
                fechas_futuras = pd.date_range(
                    start=ultimo_mes + pd.DateOffset(months=1), periods=6, freq="MS",
                )
                for fecha in fechas_futuras:
                    df.loc[fecha, col_saidi] = np.nan
                faltantes = df[df[col_saidi].isna()]

            if log_callback:
                log_callback(f"Datos historicos SAIDI: {len(historico)} observaciones")
                log_callback(f"Meses a predecir: {len(faltantes)} observaciones")
                log_callback(
                    f"Periodo historico: {historico.index[0].strftime('%Y-%m')} a {historico.index[-1].strftime('%Y-%m')}",
                )

            if progress_callback:
                progress_callback(
                    30, f"Aplicando transformacion {transformation.upper()}...",
                )

            # Aplicar transformacion
            historico_values_original = historico[col_saidi].values
            historico_transformed, transform_info = self._apply_transformation(
                historico_values_original, transformation,
            )
            historico_transformed_series = pd.Series(
                historico_transformed, index=historico.index,
            )

            if log_callback:
                log_callback(f"Transformacion aplicada: {transform_info}")

            if progress_callback:
                progress_callback(40, "Calculando metricas del modelo...")

            # CORREGIDO: Calcular metricas con mismo metodo que OptimizationService
            historico_original_series = pd.Series(
                historico_values_original, index=historico.index,
            )
            metricas = self._calcular_metricas_modelo(
                historico_original_series,
                {
                    "order": order,
                    "seasonal_order": seasonal_order,
                    "transformation": transformation,
                },
                exog_df,
            )

            if metricas and log_callback:
                log_callback("=" * 60)
                log_callback("METRICAS DEL MODELO (en escala original):")
                log_callback("=" * 60)
                log_callback(f"  - RMSE: {metricas['rmse']:.4f} minutos")
                log_callback(f"  - MAE: {metricas['mae']:.4f} minutos")
                log_callback(f"  - MAPE: {metricas['mape']:.1f}%")
                log_callback(f"  - R2 Score: {metricas['r2_score']:.4f}")
                log_callback(f"  - Precision Final: {metricas['precision_final']:.1f}%")
                if "stability_score" in metricas:
                    log_callback(f"  - Stability Score: {metricas['stability_score']:.1f}")
                log_callback(f"  - Validacion: {metricas.get('validation_pct', 0):.0f}% test ({metricas.get('n_test', 0)} meses)")
                if exog_df is not None:
                    log_callback(f"  - Variables exogenas: {len(exog_df.columns)} (sin escalar)")
                log_callback("=" * 60)

            if progress_callback:
                progress_callback(60, "Ajustando modelo SARIMAX...")

            # Ajustar modelo con variables exogenas SIN ESCALAR
            try:
                exog_train = None
                if exog_df is not None:
                    exog_train = exog_df.loc[historico.index]
                    if log_callback:
                        log_callback(
                            f"Variables exogenas de entrenamiento: {len(exog_train)} periodos",
                        )
                        log_callback(
                            "NOTA: Variables en escala ORIGINAL (SARIMAX normaliza internamente)",
                        )
                        if simulation_applied:
                            log_callback("Modo: Entrenamiento con datos SIN ESCALAR para simulacion")

                model = SARIMAX(
                    historico_transformed_series,
                    exog=exog_train,  # SIN ESCALAR - CRÍTICO
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=True,
                    enforce_invertibility=True,
                )
                results = model.fit(disp=False)

                if log_callback:
                    log_callback("Modelo SARIMAX ajustado correctamente")
                    if exog_train is not None:
                        log_callback(
                            f"  Variables exogenas incluidas: {exog_train.shape[1]}",
                        )
                        log_callback("  Escalado: Interno de SARIMAX (no manual)")
            except Exception as e:
                raise Exception(f"Error ajustando modelo: {e!s}")

            if progress_callback:
                progress_callback(
                    80, "Generando predicciones con intervalos de confianza...",
                )

            # Generar predicciones con intervalos de confianza
            try:
                # Preparar variables exogenas para prediccion
                exog_forecast = None
                exog_forecast_original = None

                if exog_df is not None:
                    # Extender SIN ESCALAR primero
                    exog_forecast_original = self._extend_exogenous_for_forecast(
                        exog_df,
                        faltantes.index,
                        log_callback,
                        unscale=False,  # Ya están sin escalar
                    )

                    # Aplicar simulacion SI esta habilitada
                    if simulation_applied and simulation_config:
                        exog_forecast = self._apply_climate_simulation(
                            exog_forecast_original, simulation_config, log_callback,
                        )
                    else:
                        # Si no hay simulación, usar directamente (sin escalar)
                        exog_forecast = exog_forecast_original
                        if log_callback:
                            log_callback(f"Variables exogenas extendidas: {len(exog_forecast)} periodos (sin escalar)")

                if log_callback:
                    log_callback("Calculando intervalos de confianza ajustados...")

                # Obtener predicción con intervalos del modelo
                pred = results.get_forecast(steps=len(faltantes), exog=exog_forecast)
                pred_mean_transformed = pred.predicted_mean.values

                # Calcular intervalos en escala transformada (95%)
                conf_int_transformed = pred.conf_int(alpha=0.05)
                lower_transformed = conf_int_transformed.iloc[:, 0].values
                upper_transformed = conf_int_transformed.iloc[:, 1].values

                # Revertir transformacion para valores predichos
                pred_mean_original = self._inverse_transformation(
                    pred_mean_transformed, transformation,
                )
                lower_bound_original = self._inverse_transformation(
                    lower_transformed, transformation,
                )
                upper_bound_original = self._inverse_transformation(
                    upper_transformed, transformation,
                )

                # Ajustar intervalos basado en precisión del modelo
                adjustment_factor = 1.0
                mape_mayor_menor = 15
                mape_mayor_medio = 20
                mape_mayor_mayor = 30
                if metricas:
                    if metricas["mape"] > mape_mayor_mayor:
                        adjustment_factor = 1.15
                    elif metricas["mape"] > mape_mayor_medio:
                        adjustment_factor = 1.10
                    elif metricas["mape"] > mape_mayor_menor:
                        adjustment_factor = 1.05
                    else:
                        adjustment_factor = 1.02

                    if log_callback:
                        log_callback(
                            f"  Factor de ajuste de intervalos: {adjustment_factor:.2f}x (basado en MAPE={metricas['mape']:.1f}%)",
                        )

                # Aplicar ajuste conservador
                for i in range(len(pred_mean_original)):
                    center = pred_mean_original[i]
                    half_width = (upper_bound_original[i] - lower_bound_original[i]) / 2
                    adjusted_half_width = half_width * adjustment_factor

                    lower_bound_original[i] = max(0, center - adjusted_half_width)
                    upper_bound_original[i] = center + adjusted_half_width

                # Validación: Asegurar intervalos razonables (máx ±50%)
                for i in range(len(pred_mean_original)):
                    center = pred_mean_original[i]
                    if center > 0:
                        max_reasonable_width = center * 0.50

                        current_lower = lower_bound_original[i]
                        current_upper = upper_bound_original[i]

                        if (center - current_lower) > max_reasonable_width:
                            lower_bound_original[i] = max(
                                0, center - max_reasonable_width,
                            )

                        if (current_upper - center) > max_reasonable_width:
                            upper_bound_original[i] = center + max_reasonable_width

                if log_callback:
                    avg_width_pct = 0
                    for i in range(len(pred_mean_original)):
                        if pred_mean_original[i] > 0:
                            width = upper_bound_original[i] - lower_bound_original[i]
                            width_pct = (width / (2 * pred_mean_original[i])) * 100
                            avg_width_pct += width_pct
                    avg_width_pct /= len(pred_mean_original)
                    log_callback(
                        f"  Ancho promedio de intervalos: ±{avg_width_pct:.0f}% del valor predicho",
                    )

                # Calcular margen de error
                z_score = stats.norm.ppf(0.975)
                margin_error = (upper_bound_original - lower_bound_original) / (
                    2 * z_score
                )

                # Crear Series con indices correctos
                pred_mean = pd.Series(pred_mean_original, index=faltantes.index)
                lower_bound = pd.Series(lower_bound_original, index=faltantes.index)
                upper_bound = pd.Series(upper_bound_original, index=faltantes.index)
                margin_error_series = pd.Series(margin_error, index=faltantes.index)

                # Crear resultado compatible
                uncertainty_result = {
                    "predictions": pred_mean_original,
                    "lower_bound": lower_bound_original,
                    "upper_bound": upper_bound_original,
                    "margin_error": margin_error,
                    "method": "parametric_adjusted",
                    "confidence_level": 0.95,
                }

                if log_callback:
                    log_callback(
                        f"Predicciones con intervalos de confianza generadas para {len(pred_mean)} periodos",
                    )
                    log_callback("Metodo: Intervalos parametricos ajustados")
                    if exog_forecast is not None:
                        if simulation_applied:
                            log_callback("Usando variables exogenas SIMULADAS")
                        else:
                            log_callback("Usando variables exogenas proyectadas")
                    log_callback("Intervalos de confianza: 95%")
                    avg_margin_pct = np.mean(margin_error / pred_mean_original) * 100
                    log_callback(f"Margen de error promedio: ±{avg_margin_pct:.1f}%")

            except (ValueError, TypeError, KeyError, IndexError, AttributeError) as e:
                msg = f"Error generando predicciones: {e!s}"
                raise RuntimeError(msg) from e

            if progress_callback:
                progress_callback(90, "Generando grafica...")

            # Construir DataFrame con resultados
            df_pred = df.copy()
            df_pred.loc[faltantes.index, col_saidi] = pred_mean

            # LOGGING de resultados con intervalos
            if log_callback:
                log_callback("=" * 60)
                if simulation_applied:
                    log_callback(
                        "PREDICCION SAIDI CON SIMULACION CLIMATICA E INTERVALOS",
                    )
                else:
                    log_callback(
                        "RESUMEN DE PREDICCIONES CON INTERVALOS DE CONFIANZA",
                    )
                log_callback("=" * 60)

                if simulation_applied and simulation_config:
                    summary = simulation_config.get("summary", {})
                    log_callback(
                        f"Escenario simulado: {summary.get('escenario', 'N/A')}",
                    )
                    log_callback(
                        f"Dias simulados: {summary.get('dias_simulados', 'N/A')}",
                    )
                    log_callback(
                        f"Alcance: {summary.get('alcance_meses', 'N/A')} meses",
                    )

                log_callback(f"Predicciones generadas: {len(pred_mean)}")
                log_callback(
                    f"Periodo: {faltantes.index[0].strftime('%Y-%m')} a {faltantes.index[-1].strftime('%Y-%m')}",
                )
                log_callback("Valores predichos con intervalos de confianza (95%):")

                for fecha, valor, inferior, superior, margen in zip(
                    faltantes.index,
                    pred_mean,
                    lower_bound,
                    upper_bound,
                    margin_error_series,
                    strict=True,
                ):
                    margen_sup = superior - valor
                    margen_inf = valor - inferior
                    margen_pct = (margen / valor * 100) if valor > 0 else 0

                    log_callback(
                        f"  • {fecha.strftime('%Y-%m')}: {valor:.2f} min "
                        f"[IC: {inferior:.2f} - {superior:.2f}] "
                        f"(+{margen_sup:.2f}/-{margen_inf:.2f} | ±{margen_pct:.0f}%)",
                    )

                if exog_info:
                    log_callback("\nVariables exogenas utilizadas (escala original):")
                    for var_info in exog_info.values():
                        log_callback(
                            f"  - {var_info['nombre']}: correlacion {var_info['correlacion']}",
                        )

                log_callback("=" * 60)

            # Generar grafica con intervalos de confianza
            plot_path = self._generar_grafica(
                data_config={
                    "historico": historico,
                    "pred_mean": pred_mean,
                    "faltantes": faltantes,
                    "df": df,
                    "col_saidi": col_saidi,
                },
                model_config={
                    "order": order,
                    "seasonal_order": seasonal_order,
                    "metricas": metricas,
                    "transformation": transformation,
                },
                pred_config={
                    "exog_info": exog_info,
                    "simulation_config": simulation_config,
                    "lower_bound": lower_bound,
                    "upper_bound": upper_bound,
                },
            )

            if progress_callback:
                progress_callback(100, "Prediccion completada")

            # Preparar resultados con intervalos de confianza
            predicciones_dict = {}
            for fecha, valor, inferior, superior, margen in zip(
                faltantes.index,
                pred_mean,
                lower_bound,
                upper_bound,
                margin_error_series,
                strict=True,
            ):
                predicciones_dict[fecha.strftime("%Y-%m")] = {
                    "valor_predicho": float(valor),
                    "limite_inferior": float(inferior),
                    "limite_superior": float(superior),
                    "margen_error": float(margen),
                }

            return {
                "success": True,
                "predictions": predicciones_dict,
                "metrics": metricas,
                "model_params": {
                    "order": list(order),
                    "seasonal_order": list(seasonal_order),
                    "transformation": transformation,
                    "regional_code": regional_code,
                    "with_exogenous": exog_df is not None,
                    "with_simulation": simulation_applied,
                    "confidence_level": 0.95,
                    "uncertainty_method": uncertainty_result["method"],
                    "exog_scaled": False,  # CRÍTICO: Marcar que NO están escaladas
                },
                "exogenous_vars": exog_info,
                "simulation_config": simulation_config if simulation_applied else None,
                "historical_count": len(historico),
                "prediction_count": len(pred_mean),
                "plot_file": plot_path,
                "export_service": self.export_service,
            }

        except (ValueError, TypeError, KeyError, IndexError, AttributeError) as e:
            if log_callback:
                log_callback(f"ERROR: {e!s}")
            msg = f"Error en prediccion: {e!s}"
            raise ValueError(msg) from e


    def _apply_climate_simulation(
        self, exog_forecast_original, simulation_config, log_callback=None,
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
        # Validar que la simulación esté habilitada
        if not simulation_config.get("enabled", False):
            self._log(log_callback, "Simulación NO habilitada, usando valores originales")
            return exog_forecast_original

        self._log_simulation_header(log_callback)

        try:
            # Extraer y validar configuración
            config = self._extract_simulation_config(simulation_config)

            if not config["percentiles"]:
                self._log(log_callback, "ERROR: No hay percentiles disponibles para simulación")
                self._log(log_callback, "Usando valores originales sin simulación")
                return exog_forecast_original

            self._log_simulation_config(log_callback, config)

            # Calcular factor de intensidad
            intensity_adjustment = self._calculate_intensity(
                config["dias_base"],
                config["slider_adjustment"],
            )
            self._log(log_callback, f"   Intensidad calculada: {intensity_adjustment:.2f}x")

            # Aplicar simulación
            exog_simulated = self.simulation_service.apply_simulation(
                exog_forecast=exog_forecast_original,
                scenario_name=config["escenario"],
                intensity_adjustment=intensity_adjustment,
                alcance_meses=config["alcance_meses"],
                percentiles=config["percentiles"],
                regional_code=config["regional_code"],
            )

        except (KeyError, ValueError, TypeError, AttributeError) as e:
            return self._handle_simulation_error(
                log_callback, e, "configuración o datos", exog_forecast_original,
            )

        except (ZeroDivisionError, ArithmeticError) as e:
            return self._handle_simulation_error(
                log_callback, e, "cálculos de simulación", exog_forecast_original,
            )

        else:
            # Logging de resultados exitosos
            self._log_simulation_results(
                log_callback,
                exog_simulated,
                exog_forecast_original,
                config["alcance_meses"],
            )
            return exog_simulated

    def _log(self, log_callback, message):
        """Helper para logging condicional."""
        if log_callback:
            log_callback(message)

    def _log_simulation_header(self, log_callback):
        """Log del encabezado de simulación."""
        if log_callback:
            log_callback("=" * 60)
            log_callback("APLICANDO SIMULACIÓN CLIMÁTICA")
            log_callback("=" * 60)
            log_callback("   Entrada: valores originales SIN ESCALAR")

    def _extract_simulation_config(self, simulation_config):
        """Extraer y organizar configuración de simulación."""
        return {
            "escenario": simulation_config.get(
                "scenario_name",
                simulation_config.get("escenario", "condiciones_normales"),
            ),
            "slider_adjustment": simulation_config.get("slider_adjustment", 0),
            "dias_base": simulation_config.get("dias_base", 30),
            "alcance_meses": simulation_config.get("alcance_meses", 3),
            "percentiles": simulation_config.get("percentiles", {}),
            "regional_code": simulation_config.get("regional_code", "SAIDI_O"),
        }

    def _calculate_intensity(self, dias_base, slider_adjustment):
        """Calcular factor de intensidad de simulación."""
        if dias_base <= 0:
            return 1.0
        dias_simulados = dias_base + slider_adjustment
        return dias_simulados / dias_base

    def _log_simulation_config(self, log_callback, config):
        """Log de configuración de simulación."""
        if log_callback:
            log_callback(f"Escenario: {config['escenario']}")
            log_callback(f"Regional: {config['regional_code']}")
            log_callback(
                f"Slider: {config['slider_adjustment']:+d} días sobre base de {config['dias_base']}",
            )
            log_callback(f"Alcance: {config['alcance_meses']} mes(es)")

    def _handle_simulation_error(self, log_callback, error, error_type, fallback_data):
        """Manejar errores de simulación con logging y fallback."""
        if log_callback:
            log_callback(f"ERROR en {error_type}: {error!s}")
            log_callback(traceback.format_exc())
            log_callback("FALLBACK: Usando valores originales sin simulación")
        return fallback_data

    def _log_simulation_results(
            self, log_callback, exog_simulated, exog_forecast_original, alcance_meses,
        ):
        """Log de resultados de simulación exitosa."""
        if not log_callback:
            return

        log_callback("Simulación aplicada correctamente")

        # Mostrar cambios en el primer mes
        if alcance_meses >= 1 and len(exog_simulated) > 0:
            log_callback("\nCambios en primer mes:")
            self._log_column_changes(
                log_callback,
                exog_simulated,
                exog_forecast_original,
            )

        log_callback("\nSalida: valores SIMULADOS (escala original)")
        log_callback("=" * 60)

    def _log_column_changes(self, log_callback, exog_simulated, exog_forecast_original):
        """Log de cambios por columna."""
        for col in exog_simulated.columns:
            try:
                original_val = exog_forecast_original.iloc[0][col]
                simulated_val = exog_simulated.iloc[0][col]

                change_pct = self._calculate_change_percentage(original_val, simulated_val)

                log_callback(
                    f"     - {col}: {original_val:.2f} → {simulated_val:.2f} "
                    f"({change_pct:+.1f}%)",
                )
            except (KeyError, IndexError, ValueError, ZeroDivisionError):
                # Silenciosamente continuar si hay error mostrando un cambio específico
                pass

    def _calculate_change_percentage(self, original_val, simulated_val):
        """Calcular porcentaje de cambio entre valores."""
        if original_val == 0:
            return 0.0
        return ((simulated_val - original_val) / original_val) * 100

    def _prepare_exogenous_variables(
    self, climate_data, df_saidi, regional_code, log_callback=None,
    ):
        """
        Preparar variables exógenas climáticas SIN ESCALAR.

        ALINEADO CON OptimizationService para obtener métricas consistentes

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
            # Validaciones y preparación inicial
            preparacion_inicial = self._realizar_preparacion_inicial(
                climate_data, df_saidi, regional_code, log_callback,
            )
            if preparacion_inicial is None:
                return None, None

            climate_data = preparacion_inicial["climate_data"]
            overlap_info = preparacion_inicial["overlap_info"]
            historico = overlap_info["historico"]

            # Preparar variables sin escalado
            exog_df, exog_info = self._preparar_variables_sin_escalado(
                climate_data, historico, regional_code, overlap_info, log_callback,
            )

            # Validar resultado final
            if exog_df is None or exog_df.empty or exog_df.shape[1] == 0:
                if log_callback:
                    log_callback("ERROR: Ninguna variable aceptada")
                return None, None

            if log_callback:
                self._log_resumen_variables(exog_df, log_callback)

        except (ValueError, TypeError, KeyError, IndexError, AttributeError) as e:
            if log_callback:
                log_callback(f"ERROR CRÍTICO: {e!s}")
            return None, None
        else:
            return exog_df, exog_info if exog_info else None


    def _realizar_preparacion_inicial(
        self, climate_data, df_saidi, regional_code, log_callback,
    ):
        """Realizar validaciones iniciales y preparar datos."""
        # Validaciones iniciales
        if not self._validar_datos_iniciales(climate_data, regional_code, log_callback):
            return None

        if log_callback:
            log_callback(f"Preparando variables para {regional_code}")
            log_callback("MODO: SIN ESCALADO (valores originales)")

        # Validar y preparar índice datetime
        climate_data = self._validar_indice_datetime(climate_data, log_callback)
        if climate_data is None:
            return None

        # Análisis de cobertura temporal
        overlap_info = self._analizar_cobertura_temporal(
            climate_data, df_saidi, log_callback,
        )
        if overlap_info is None:
            return None

        # Mapeo automático de columnas
        climate_column_mapping = self._mapear_columnas_climaticas(
            climate_data, regional_code, log_callback,
        )
        if not climate_column_mapping:
            return None

        # Preparar info consolidada
        overlap_info["climate_column_mapping"] = climate_column_mapping

        return {
            "climate_data": climate_data,
            "overlap_info": overlap_info,
        }


    def _validar_datos_iniciales(self, climate_data, regional_code, log_callback):
        """Validar datos climáticos y código regional."""
        if climate_data is None or climate_data.empty:
            if log_callback:
                log_callback("Sin datos climáticos disponibles")
            return False

        if not regional_code or regional_code not in self.REGIONAL_EXOG_VARS:
            if log_callback:
                log_callback(f"Regional {regional_code} sin variables definidas")
            return False

        return True


    def _validar_indice_datetime(self, climate_data, log_callback):
        """Validar y convertir índice a DatetimeIndex."""
        if isinstance(climate_data.index, pd.DatetimeIndex):
            return climate_data

        # Buscar columna de fecha
        fecha_col = None
        for col in ["fecha", "Fecha", "date", "Date", "month_date"]:
            if col in climate_data.columns:
                fecha_col = col
                break

        if fecha_col is None:
            if log_callback:
                log_callback("ERROR: No se encontró columna de fecha válida")
            return None

        try:
            climate_data = climate_data.copy()
            climate_data[fecha_col] = pd.to_datetime(climate_data[fecha_col])
            climate_data = climate_data.set_index(fecha_col)
        except (ValueError, TypeError, KeyError) as e:
            if log_callback:
                log_callback(f"ERROR convirtiendo índice: {e!s}")
            return None

        if not isinstance(climate_data.index, pd.DatetimeIndex):
            if log_callback:
                log_callback("ERROR: Formato de fecha inválido")
            return None

        return climate_data


    def _analizar_cobertura_temporal(self, climate_data, df_saidi, log_callback):
        """Analizar cobertura temporal entre SAIDI y datos climáticos."""
        historico = df_saidi[
            df_saidi["SAIDI"].notna() if "SAIDI" in df_saidi.columns
            else df_saidi["SAIDI Historico"].notna()
        ]

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
        }


    def _mapear_columnas_climaticas(self, climate_data, regional_code, log_callback):
        """Mapear columnas climáticas con búsqueda flexible."""
        exog_vars_config = self.REGIONAL_EXOG_VARS[regional_code]

        # Normalizar nombres disponibles
        available_cols_normalized = {}
        for col in climate_data.columns:
            normalized = col.lower().strip().replace(" ", "_").replace("-", "_")
            available_cols_normalized[normalized] = col

        # Mapear cada variable con búsqueda flexible
        climate_column_mapping = {}

        for var_code in exog_vars_config:
            var_normalized = var_code.lower().strip()

            # Intento 1: Coincidencia exacta
            if var_normalized in available_cols_normalized:
                climate_column_mapping[var_code] = available_cols_normalized[var_normalized]
                continue

            # Intento 2: Coincidencia parcial
            best_match = self._buscar_mejor_coincidencia(
                var_normalized, available_cols_normalized,
            )
            if best_match:
                climate_column_mapping[var_code] = best_match

        if not climate_column_mapping and log_callback:
            log_callback("ERROR: No se pudo mapear ninguna variable")

        return climate_column_mapping


    def _buscar_mejor_coincidencia(self, var_normalized, available_cols_normalized):
        """Buscar la mejor coincidencia parcial para una variable."""
        var_parts = var_normalized.split("_")
        best_match = None
        best_match_score = 0

        for norm_col, orig_col in available_cols_normalized.items():
            matches = sum(1 for part in var_parts if part in norm_col)
            if matches > best_match_score:
                best_match_score = matches
                best_match = orig_col

        mayor_match_score = 2
        if best_match_score >= mayor_match_score:
            return best_match
        return None


    def _preparar_variables_sin_escalado(
    self, climate_data, historico, regional_code, overlap_info, log_callback,
    ):
        """Preparar variables exógenas sin escalado."""
        exog_vars_config = self.REGIONAL_EXOG_VARS[regional_code]
        exog_df = pd.DataFrame(index=historico.index)
        exog_info = {}

        overlap_mask = overlap_info["overlap_mask"]
        overlap_months = overlap_info["overlap_months"]
        climate_column_mapping = overlap_info["climate_column_mapping"]

        for var_code, var_nombre in exog_vars_config.items():
            climate_col = climate_column_mapping.get(var_code)

            if not climate_col or climate_col not in climate_data.columns:
                continue

            var_config = {
                "var_nombre": var_nombre,
                "climate_col": climate_col,
                "overlap_mask": overlap_mask,
                "overlap_months": overlap_months,
                "regional_code": regional_code,
            }

            result = self._procesar_variable_exogena(
                climate_data, historico, var_code, var_config, log_callback,
            )

            if result is not None:
                exog_df[var_code] = result["series"]
                exog_info[var_code] = result["info"]

        return exog_df, exog_info


    def _procesar_variable_exogena(
    self, climate_data, historico, var_code, var_config, log_callback,
    ):
        """Procesar una variable exógena individual."""
        try:
            # Extraer configuración
            var_nombre = var_config["var_nombre"]
            climate_col = var_config["climate_col"]
            overlap_mask = var_config["overlap_mask"]
            overlap_months = var_config["overlap_months"]
            regional_code = var_config["regional_code"]

            # Extraer y alinear serie
            var_series = climate_data[climate_col].copy()
            aligned_series = self._alinear_serie_climatica(var_series, historico)

            # Validar cobertura y varianza
            validacion = self._validar_variable_exogena(
                aligned_series, overlap_mask, overlap_months, var_code, log_callback,
            )
            if validacion is None:
                return None

            # Rellenar NaN
            aligned_series = self._rellenar_valores_faltantes(
                aligned_series, validacion["overlap_data"],
            )

            # Verificación final
            final_nan = aligned_series.isna().sum()
            if final_nan > 0:
                if log_callback:
                    log_callback(f"X RECHAZADA {var_code}: {final_nan} NaN finales")
                return None

            if log_callback:
                log_callback(f"✓ {var_code} -> ACEPTADA ({validacion['overlap_pct']:.1f}% cobertura, escala original)")

            return {
                "series": aligned_series,
                "info": {
                    "nombre": var_nombre,
                    "columna_clima": climate_col,
                    "correlacion": self._get_correlation_for_var(var_code, regional_code),
                    "scaled": False,
                    "datos_reales_overlap": validacion["datos_reales_overlap"],
                    "overlap_coverage_pct": validacion["overlap_pct"],
                    "varianza_overlap": validacion["var_std"],
                },
            }

        except (ValueError, TypeError, KeyError, IndexError) as e:
            if log_callback:
                log_callback(f"X ERROR {var_code}: {e}")
            return None


    def _alinear_serie_climatica(self, var_series, historico):
        """Alinear serie climática con índice histórico."""
        aligned_series = pd.Series(index=historico.index, dtype=float)

        for date in historico.index:
            if date in var_series.index:
                aligned_series[date] = var_series.loc[date]

        return aligned_series


    def _validar_variable_exogena(
        self, aligned_series, overlap_mask, overlap_months, var_code, log_callback,
    ):
        """Validar cobertura y varianza de variable exógena."""
        overlap_data = aligned_series[overlap_mask]
        datos_reales_overlap = overlap_data.notna().sum()
        overlap_pct = (datos_reales_overlap / overlap_months) * 100

        cobertura_minima = 80
        if overlap_pct < cobertura_minima:
            if log_callback:
                log_callback(f"X RECHAZADA {var_code}: cobertura {overlap_pct:.1f}% < 80%")
            return None

        var_std = overlap_data.std()
        if pd.isna(var_std) or var_std == 0:
            if log_callback:
                log_callback(f"X RECHAZADA {var_code}: varianza = 0")
            return None

        return {
            "overlap_data": overlap_data,
            "datos_reales_overlap": int(datos_reales_overlap),
            "overlap_pct": float(overlap_pct),
            "var_std": float(var_std),
        }


    def _rellenar_valores_faltantes(self, aligned_series, overlap_data):
        """Rellenar valores faltantes con estrategia forward/backward fill."""
        # Forward-fill para fechas futuras
        aligned_series = aligned_series.fillna(method="ffill")

        # Backward-fill (máx 3 meses) para fechas pasadas
        aligned_series = aligned_series.fillna(method="bfill", limit=3)

        # Si aún hay NaN, rellenar con media del overlap
        if aligned_series.isna().any():
            mean_overlap = overlap_data.mean()
            aligned_series = aligned_series.fillna(mean_overlap)

        return aligned_series


    def _log_resumen_variables(self, exog_df, log_callback):
        """Registrar resumen de variables preparadas."""
        log_callback("=" * 60)
        log_callback(f"✓ Variables preparadas: {len(exog_df.columns)}")
        log_callback("  ESCALA: ORIGINAL (sin StandardScaler)")
        log_callback("  Rangos:")
        for col in exog_df.columns:
            log_callback(f"    - {col}: [{exog_df[col].min():.2f}, {exog_df[col].max():.2f}]")
        log_callback("=" * 60)

    def _align_exog_to_saidi(self, exog_series, df_saidi, var_code, log_callback=None):
        """
        Alinear datos exogenos al indice de SAIDI.

        Estrategia:
        - Para fechas con datos climaticos: usar el valor directo
        - Para fechas sin datos climaticos: extrapolar usando ultimo valor
        """
        try:
            # Normalizar índices a DatetimeIndex
            climate_dates = self._ensure_datetime_index(exog_series.index)
            saidi_dates = self._ensure_datetime_index(df_saidi.index)

            # Crear contexto de alineación
            context = AlignmentContext(
                exog_series=exog_series,
                climate_dates=climate_dates,
                saidi_dates=saidi_dates,
                var_code=var_code,
                log_callback=log_callback,
            )

            # Crear serie resultado con datos climáticos existentes
            result = self._fill_existing_climate_data(context)

            # Proyectar valores para fechas futuras y pasadas
            self._fill_future_dates(result, context)
            self._fill_past_dates(result, context)

        except (KeyError, IndexError, AttributeError) as e:
            if log_callback:
                log_callback(f"Error accediendo a datos para {var_code}: {e!s}")
            return None

        except (ValueError, TypeError) as e:
            if log_callback:
                log_callback(f"Error de tipo/valor al alinear {var_code}: {e!s}")
            return None

        except pd.errors.ParserError as e:
            if log_callback:
                log_callback(f"Error parseando fechas para {var_code}: {e!s}")
            return None

        else:
            return result

    def _ensure_datetime_index(self, index):
        """Asegurar que un índice sea DatetimeIndex."""
        if isinstance(index, pd.DatetimeIndex):
            return index
        return pd.to_datetime(index)

    def _fill_existing_climate_data(self, context: AlignmentContext):
        """Crear serie con datos climáticos donde existan."""
        result = pd.Series(index=context.saidi_dates, dtype=float)

        for date in context.saidi_dates:
            if date in context.climate_dates:
                result[date] = context.exog_series.loc[date].iloc[0]

        return result

    def _fill_future_dates(self, result: pd.Series, context: AlignmentContext):
        """Forward fill: proyectar valores para fechas futuras."""
        max_climate_date = context.climate_dates.max()
        future_indices = context.saidi_dates > max_climate_date

        if not future_indices.any():
            return

        last_known_value = context.exog_series.iloc[-1].iloc[0]
        result.loc[future_indices] = last_known_value

        if context.log_callback:
            n_future = future_indices.sum()
            context.log_callback(
                f"    {context.var_code}: {n_future} valores futuros proyectados",
            )

    def _fill_past_dates(self, result: pd.Series, context: AlignmentContext):
        """Backward fill: proyectar valores para fechas pasadas."""
        min_climate_date = context.climate_dates.min()
        past_indices = context.saidi_dates < min_climate_date

        if not past_indices.any():
            return

        first_known_value = context.exog_series.iloc[0].iloc[0]
        result.loc[past_indices] = first_known_value

        if context.log_callback:
            n_past = past_indices.sum()
            context.log_callback(
                f"    {context.var_code}: {n_past} valores pasados proyectados",
            )

    def _get_correlation_for_var(self, var_code, regional_code):
        """
        Obtener correlación documentada de una variable específica.

        Args:
            var_code: Código de la variable (ej: 'realfeel_min')
            regional_code: Código de la regional (ej: 'SAIDI_O')

        Returns:
            float: Correlación documentada o 0.0 si no existe

        """
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

    def _extend_exogenous_for_forecast(
        self, exog_df, forecast_dates, log_callback=None, *, unscale=False,
    ):
        """
        Extender variables exogenas para prediccion.

        Args:
            exog_df: DataFrame con variables exogenas (pueden estar escaladas o no)
            forecast_dates: Fechas para las que se necesitan predicciones
            log_callback: Funcion para logging
            unscale: Si True, des-escalar antes de extender (keyword-only)

        Returns:
            DataFrame SIN ESCALAR con valores extendidos

        """
        try:
            # Si unscale=True, des-escalar primero
            if unscale and self.exog_scaler is not None:
                exog_df_original = pd.DataFrame(
                    self.exog_scaler.inverse_transform(exog_df),
                    index=exog_df.index,
                    columns=exog_df.columns,
                )
                if log_callback:
                    log_callback("  Des-escalando variables para simulacion...")
            else:
                # Ya estan sin escalar o no hay scaler
                exog_df_original = exog_df.copy()

            # Crear forecast con valores originales (sin escalar)
            exog_forecast = pd.DataFrame(
                index=forecast_dates, columns=exog_df_original.columns,
            )

            for col in exog_df_original.columns:
                last_value = exog_df_original[col].iloc[-1]
                exog_forecast[col] = last_value

            if log_callback:
                log_callback(
                    f"  Variables extendidas: {len(forecast_dates)} periodos (sin escalar)",
                )

        except (AttributeError, KeyError, IndexError, ValueError) as e:
            if log_callback:
                log_callback(f"ERROR extendiendo variables exogenas: {e!s}")
            return None
        else:
            return exog_forecast  # Retorna SIN ESCALAR

    def _get_correlation_for_var(self, var_code, regional_code):
        """Obtener la correlacion documentada de una variable."""
        correlations = {
            "SAIDI_O": {"temp_max": 0.450, "humedad_avg": 0.380, "precip_total": 0.420},
            "SAIDI_C": {"temp_max": 0.450, "humedad_avg": 0.380, "precip_total": 0.420},
            "SAIDI_A": {"temp_max": 0.450, "humedad_avg": 0.380, "precip_total": 0.420},
            "SAIDI_P": {"temp_max": 0.450, "humedad_avg": 0.380, "precip_total": 0.420},
            "SAIDI_T": {"temp_max": 0.450, "humedad_avg": 0.380, "precip_total": 0.420},
        }

        if regional_code in correlations and var_code in correlations[regional_code]:
            return correlations[regional_code][var_code]
        return 0.0

    def _diagnose_exog_coverage(self,
                      serie_saidi: pd.Series,
                      exog_df: pd.DataFrame,
                      log_callback) -> bool:
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
            self._log_coverage_header(serie_saidi, exog_df, log_callback)

            if not self._validate_index_alignment(serie_saidi, exog_df, log_callback):
                return False

            if not self._validate_no_nan_values(exog_df, log_callback):
                return False

            if not self._validate_no_infinite_values(exog_df, log_callback):
                return False

            self._check_variable_variance(exog_df, log_callback)

            if log_callback:
                log_callback("✓ Cobertura temporal y calidad de datos OK")
                log_callback("=" * 60)

        except (AttributeError, KeyError, IndexError, ValueError, TypeError) as e:
            if log_callback:
                log_callback(f"ERROR durante diagnostico: {e}")
            return False
        else:
            return True

    def _log_coverage_header(self,
                            serie_saidi: pd.Series,
                            exog_df: pd.DataFrame,
                            log_callback) -> None:
        """Log header information about coverage."""
        if not log_callback:
            return

        saidi_start = serie_saidi.index[0]
        saidi_end = serie_saidi.index[-1]
        exog_start = exog_df.index[0]
        exog_end = exog_df.index[-1]

        log_callback("=" * 60)
        log_callback("DIAGNOSTICO DE COBERTURA EXOGENA")
        log_callback("=" * 60)
        log_callback(f"SAIDI: {saidi_start.strftime('%Y-%m')} a {saidi_end.strftime('%Y-%m')} ({len(serie_saidi)} obs)")
        log_callback(f"EXOG:  {exog_start.strftime('%Y-%m')} a {exog_end.strftime('%Y-%m')} ({len(exog_df)} obs)")

    def _validate_index_alignment(self,
                                  serie_saidi: pd.Series,
                                  exog_df: pd.DataFrame,
                                  log_callback) -> bool:
        """Validate that indices align exactly."""
        if exog_df.index.equals(serie_saidi.index):
            return True

        if log_callback:
            log_callback("ADVERTENCIA: Indices no coinciden exactamente")

        missing_in_exog = [d for d in serie_saidi.index if d not in exog_df.index]

        if not missing_in_exog:
            return True

        pct_missing = len(missing_in_exog) / len(serie_saidi) * 100

        if log_callback:
            log_callback(f"Fechas SAIDI faltantes en EXOG: {len(missing_in_exog)} ({pct_missing:.1f}%)")

        porcentaje_falla = 20
        if pct_missing > porcentaje_falla:
            if log_callback:
                log_callback("ERROR CRITICO: >20% de fechas faltantes")
                log_callback("Las variables exogenas NO cubren suficiente periodo historico")
            return False

        return True

    def _validate_no_nan_values(self, exog_df: pd.DataFrame, log_callback) -> bool:
        """Validate that there are no NaN values in any column."""
        if not exog_df.isna().any().any():
            return True

        nan_cols = exog_df.columns[exog_df.isna().any()].tolist()

        if log_callback:
            log_callback("ERROR: Columnas con NaN encontradas:")
            for col in nan_cols:
                nan_count = exog_df[col].isna().sum()
                pct_nan = (nan_count / len(exog_df)) * 100
                log_callback(f"  - {col}: {nan_count} NaN ({pct_nan:.1f}%)")
            log_callback("Variables exogenas deben estar completamente rellenas")

        return False

    def _validate_no_infinite_values(self, exog_df: pd.DataFrame, log_callback) -> bool:
        """Validate that there are no infinite values."""
        if not np.isinf(exog_df.values).any():
            return True

        if log_callback:
            log_callback("ERROR: Variables exogenas contienen valores infinitos")

        return False

    def _check_variable_variance(self, exog_df: pd.DataFrame, log_callback) -> None:
        """Check for variables with zero variance (warning only)."""
        zero_variance_vars = [col for col in exog_df.columns if exog_df[col].std() == 0]

        if zero_variance_vars and log_callback:
            log_callback("ADVERTENCIA: Variables con varianza cero:")
            for var in zero_variance_vars:
                log_callback(f"  - {var}")
            log_callback("Estas variables no aportan informacion al modelo")

    def _apply_transformation(self, data, transformation_type):
        """Aplicar transformacion a los datos."""
        if transformation_type == "original":
            return data, "Sin transformacion (datos originales)"

        if transformation_type == "standard":
            self.scaler = StandardScaler()
            transformed = self.scaler.fit_transform(data.reshape(-1, 1)).flatten()
            return transformed, "StandardScaler"

        if transformation_type == "log":
            data_positive = np.maximum(data, 1e-10)
            transformed = np.log(data_positive)
            self.transformation_params["log_applied"] = True
            return transformed, "Transformacion logaritmica"

        if transformation_type == "boxcox":
            data_positive = np.maximum(data, 1e-10)
            transformed, lambda_param = stats.boxcox(data_positive)
            self.transformation_params["boxcox_lambda"] = lambda_param
            return transformed, f"Box-Cox (lambda={lambda_param:.4f})"

        if transformation_type == "sqrt":
            data_positive = np.maximum(data, 1e-10)
            transformed = np.sqrt(data_positive)
            return transformed, "Transformacion raiz cuadrada"

        return data, "Sin transformacion"

    def _inverse_transformation(self, data, transformation_type):
        """Revertir transformacion a escala original."""
        if transformation_type == "original":
            return data

        if transformation_type == "standard":
            if self.scaler is not None:
                return self.scaler.inverse_transform(data.reshape(-1, 1)).flatten()
            return data

        # Diccionario de transformaciones inversas
        transformations = {
            "log": lambda d: np.exp(d),
            "sqrt": lambda d: np.power(d, 2),
            "boxcox": self._inverse_boxcox,
        }

        transform_func = transformations.get(transformation_type)
        if transform_func is not None:
            return transform_func(data)

        return data

    def _inverse_boxcox(self, data):
        """Aplicar transformacion inversa de Box-Cox."""
        lambda_param = self.transformation_params.get("boxcox_lambda", 0)
        if lambda_param == 0:
            return np.exp(data)
        return np.power(data * lambda_param + 1, 1 / lambda_param)

    def _calcular_metricas_modelo(
    self, serie_original, model_params, exog_df=None,
    ):
        """
        Calcular métricas del modelo.

        CAMBIOS CRÍTICOS:
        1. Usa train/test split adaptativo (20-30% según cantidad de datos)
        2. Variables exógenas SIN ESCALAR (SARIMAX normaliza internamente)
        3. Validación estricta de alineación de índices
        4. Calcula stability_score como OptimizationService

        Args:
            serie_original: Serie temporal SAIDI en escala original
            model_params: Diccionario con order, seasonal_order, transformation
            exog_df: DataFrame con variables exógenas EN ESCALA ORIGINAL (opcional)

        Returns:
            Dict con métricas del modelo o None si falla

        """
        try:
            order = model_params["order"]
            seasonal_order = model_params["seasonal_order"]
            transformation = model_params["transformation"]

            # Calcular porcentaje de validación y dividir datos
            split_data = self._preparar_train_test_split(serie_original)
            if split_data is None:
                return None

            train_original = split_data["train_original"]
            test_original = split_data["test_original"]
            n_test = split_data["n_test"]
            pct_validacion = split_data["pct_validacion"]

            # Aplicar transformación
            train_transformed_series = self._aplicar_transformacion_train(
                train_original, transformation,
            )

            # Preparar variables exógenas
            exog_result = self._preparar_exogenas_metricas(
                exog_df, train_original, test_original, n_test,
            )
            if exog_result is None and exog_df is not None:
                return None

            exog_train = exog_result["exog_train"] if exog_result else None
            exog_test = exog_result["exog_test"] if exog_result else None

            # Entrenar modelo
            results = self._entrenar_modelo_metricas(
                train_transformed_series, exog_train, order, seasonal_order,
            )
            if results is None:
                return None

            # Predecir y evaluar
            pred_mean_original = self._predecir_y_revertir(
                results, n_test, exog_test, transformation,
            )
            if pred_mean_original is None:
                return None

            # Calcular todas las métricas
            metricas = self._calcular_todas_metricas(
                test_original.values,
                pred_mean_original,
                results,
                {"order": order, "seasonal_order": seasonal_order},
                {"n_test": n_test, "pct_validacion": pct_validacion},
            )

        except (ValueError, TypeError, KeyError) as e:
            print(f"Error calculando metricas: {e}")
            return None
        else:
            return metricas

    def _preparar_train_test_split(self, serie_original):
        """Preparar división train/test con porcentaje adaptativo."""
        n_obs = len(serie_original)
        comparcion_mayor = 60
        comparcion_menor = 36

        if n_obs >= comparcion_mayor:
            pct_validacion = 0.30
        elif n_obs >= comparcion_menor:
            pct_validacion = 0.25
        else:
            pct_validacion = 0.20

        n_test = max(6, int(n_obs * pct_validacion))
        train_original = serie_original[:-n_test]
        test_original = serie_original[-n_test:]

        train_maximo = 12
        if len(train_original) < train_maximo:
            return None

        return {
            "train_original": train_original,
            "test_original": test_original,
            "n_test": n_test,
            "pct_validacion": pct_validacion,
        }

    def _aplicar_transformacion_train(self, train_original, transformation):
        """Aplicar transformación a datos de entrenamiento."""
        self.scaler = None
        self.transformation_params = {}

        train_transformed, _ = self._apply_transformation(
            train_original.values, transformation,
        )
        return pd.Series(train_transformed, index=train_original.index)

    def _preparar_exogenas_metricas(self, exog_df, train_original, test_original, n_test):
        """Preparar y validar variables exógenas SIN ESCALAR."""
        if exog_df is None:
            return None

        try:
            train_index = train_original.index
            test_index = test_original.index

            # VALIDACIÓN 1: Verificar que exog_df contiene TODAS las fechas
            missing_train = [idx for idx in train_index if idx not in exog_df.index]
            missing_test = [idx for idx in test_index if idx not in exog_df.index]

            if missing_train or missing_test:
                print(f"[METRICAS] Rechazado: faltan {len(missing_train)} fechas train, {len(missing_test)} test")
                resultado = None
            else:
                # VALIDACIÓN 2: Extraer subconjuntos con .loc
                exog_train = exog_df.loc[train_index].copy()
                exog_test = exog_df.loc[test_index].copy()

                # VALIDACIONES 3-5: Verificar calidad de datos
                if (exog_train.isna().any().any() or exog_test.isna().any().any()):
                    print("[METRICAS] Rechazado: NaN en exógenas")
                    resultado = None
                elif (len(exog_train) != len(train_original) or len(exog_test) != n_test):
                    print("[METRICAS] Rechazado: dimensiones incorrectas")
                    resultado = None
                elif (np.isinf(exog_train.values).any() or np.isinf(exog_test.values).any()):
                    print("[METRICAS] Rechazado: infinitos en exógenas")
                    resultado = None
                else:
                    resultado = {"exog_train": exog_train, "exog_test": exog_test}

        except (ValueError, KeyError, IndexError) as e:
            print(f"[METRICAS] Error preparando exógenas: {e}")
            return None
        else:
            return resultado

    def _entrenar_modelo_metricas(self, train_data, exog_train, order, seasonal_order):
        """Entrenar modelo SARIMAX."""
        try:
            model = SARIMAX(
                train_data,
                exog=exog_train,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=True,
                enforce_invertibility=True,
            )
            return model.fit(disp=False, maxiter=50)

        except (ValueError, np.linalg.LinAlgError) as e:
            print(f"[METRICAS] Error en fit: {e}")
            return None

    def _predecir_y_revertir(self, results, n_test, exog_test, transformation):
        """Predecir y revertir transformación."""
        try:
            pred = results.get_forecast(steps=n_test, exog=exog_test)
            pred_mean_transformed = pred.predicted_mean

            return self._inverse_transformation(
                pred_mean_transformed.values, transformation,
            )

        except (ValueError, KeyError) as e:
            print(f"[METRICAS] Error en predicción: {e}")
            return None

    def _calcular_todas_metricas(
        self, test_values, pred_values, results, model_info, test_info,
    ):
        """
        Calcular todas las métricas del modelo.

        Args:
            test_values: Valores reales del conjunto de prueba
            pred_values: Valores predichos
            results: Resultados del modelo SARIMAX
            model_info: Diccionario con order y seasonal_order
            test_info: Diccionario con n_test y pct_validacion

        """
        order = model_info["order"]
        seasonal_order = model_info["seasonal_order"]
        n_test = test_info["n_test"]
        pct_validacion = test_info["pct_validacion"]

        # RMSE
        rmse = np.sqrt(mean_squared_error(test_values, pred_values))

        # MAE
        mae = np.mean(np.abs(test_values - pred_values))

        # MAPE
        epsilon = 1e-8
        mape = np.mean(np.abs((test_values - pred_values) /
                            (test_values + epsilon))) * 100

        # R² Score
        ss_res = np.sum((test_values - pred_values) ** 2)
        ss_tot = np.sum((test_values - np.mean(test_values)) ** 2)
        r2_score = 1 - (ss_res / (ss_tot + epsilon))

        # Precisión final
        precision_final = max(0.0, min(100.0, (1 - mape/100) * 100))

        # Validar métricas
        if np.isnan(precision_final) or np.isinf(precision_final):
            return None
        if np.isnan(rmse) or np.isinf(rmse):
            return None

        # Penalización por complejidad
        complexity_penalty = sum(order) + sum(seasonal_order[:3])
        composite_score = rmse + (complexity_penalty * 0.05)

        # Score de estabilidad
        stability_score = self._calculate_stability_score(
            test_values, pred_values, precision_final, mape,
        )

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

    def _calculate_stability_score(self,
                               actual_values: np.ndarray,
                               predicted_values: np.ndarray,
                               precision: float,
                               mape: float) -> float:
        """
        Calcular score de estabilidad del modelo.

        Basado en:
        - Coeficiente de variación de errores
        - Penalización por MAPE alto
        - Combinación con precisión (60% estabilidad, 40% precisión)

        Args:
            actual_values: Valores reales
            predicted_values: Valores predichos
            precision: Precisión del modelo (%)
            mape: MAPE del modelo (%)

        Returns:
            float: Score de estabilidad (0-100)

        """
        try:
            errors = actual_values - predicted_values

            mean_abs_error = np.mean(np.abs(errors))
            std_error = np.std(errors)

            # Coeficiente de variación de errores
            coeficientemayor = 1e-8
            if mean_abs_error > coeficientemayor:
                cv_error = std_error / mean_abs_error
                # Convertir a score (menor CV = mayor estabilidad)
                stability_cv = max(0, 100 * (1 - min(cv_error, 1)))
            else:
                # Si errores son muy pequeños, estabilidad neutral
                stability_cv = 50.0

            # Penalización adaptativa por MAPE
            mape_mayor = 50
            mape_medio = 30
            if mape > mape_mayor:
                mape_penalty = 0.5  # Penalización fuerte
            elif mape > mape_medio:
                mape_penalty = 0.7  # Penalización moderada
            else:
                mape_penalty = 1.0  # Sin penalización

            stability_cv = stability_cv * mape_penalty

            # Combinar estabilidad con precisión (60% estabilidad, 40% precisión)
            stability = (stability_cv * 0.6) + (precision * 0.4)

            return min(100.0, max(0.0, stability))

        except (ValueError, TypeError, ZeroDivisionError) as e:
            # Manejo específico de errores esperados en operaciones numéricas
            print(f"Error al calcular stability score: {e}")
            return 0.0

    def _generar_grafica(
        self,
        data_config,
        model_config,
        pred_config,
    ):
        """
        Generar grafica de prediccion con intervalos de confianza y indicador de simulacion.

        Args:
            data_config: Diccionario con historico, pred_mean, faltantes, df, col_saidi
            model_config: Diccionario con order, seasonal_order, metricas, transformation
            pred_config: Diccionario con exog_info, simulation_config, lower_bound, upper_bound

        """
        historico = data_config["historico"]
        pred_mean = data_config["pred_mean"]

        if historico.empty or pred_mean.empty:
            return None

        try:
            plot_path = self._crear_archivo_grafica()
            fig = plt.figure(figsize=(16, 10), dpi=100)

            self._graficar_historico(historico, data_config["col_saidi"])
            self._graficar_predicciones(
                historico, pred_mean, data_config["col_saidi"],
                {
                    "transformation": model_config["transformation"],
                    "exog_info": pred_config.get("exog_info"),
                    "simulation_config": pred_config.get("simulation_config"),
                    "lower_bound": pred_config.get("lower_bound"),
                    "upper_bound": pred_config.get("upper_bound"),
                },
            )
            self._configurar_ejes(historico, data_config["faltantes"], data_config["df"])
            self._configurar_titulo(
                model_config["order"],
                model_config["seasonal_order"],
                model_config["transformation"],
                model_config["metricas"],
                {
                    "exog_info": pred_config.get("exog_info"),
                    "simulation_config": pred_config.get("simulation_config"),
                },
            )
            self._agregar_notas_pie(
                pred_config.get("exog_info"),
                pred_config.get("simulation_config"),
            )

            plt.savefig(
                plot_path,
                dpi=100,
                bbox_inches="tight",
                facecolor="white",
                edgecolor="none",
            )
            plt.close(fig)

            self.plot_file_path = str(plot_path)
            return str(plot_path)

        except (OSError, ValueError, TypeError) as e:
            print(f"Error generando grafica: {e}")
            return None

    def _crear_archivo_grafica(self):
        """Crear ruta para archivo temporal de grafica."""
        temp_dir = tempfile.gettempdir()
        timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")  # noqa: UP017
        return Path(temp_dir) / f"saidi_prediction_{timestamp}.png"

    def _graficar_historico(self, historico, col_saidi):
        """Graficar datos historicos."""
        plt.plot(
            historico.index,
            historico[col_saidi],
            label="SAIDI Historico",
            color="blue",
            linewidth=3,
            marker="o",
            markersize=5,
        )

    def _graficar_predicciones(
        self, historico, pred_mean, col_saidi, pred_config,
    ):
        """
        Graficar predicciones con intervalos de confianza.

        Args:
            historico: DataFrame con datos históricos
            pred_mean: Serie con predicciones
            col_saidi: Nombre de la columna SAIDI
            pred_config: Diccionario con transformation, exog_info, simulation_config,
                        lower_bound, upper_bound

        """
        if historico.empty or len(pred_mean) == 0:
            return

        ultimo_real_x = historico.index[-1]
        ultimo_real_y = historico[col_saidi].iloc[-1]

        x_pred = [ultimo_real_x, *pred_mean.index]
        y_pred = [ultimo_real_y, *pred_mean.to_numpy()]

        pred_label = self._obtener_etiqueta_prediccion(
            pred_config["transformation"],
            pred_config.get("exog_info"),
            pred_config.get("simulation_config"),
        )

        plt.plot(
            x_pred,
            y_pred,
            label=pred_label,
            color="orange",
            linewidth=3,
            marker="^",
            markersize=7,
            zorder=5,
        )

        self._graficar_intervalo_confianza(
            pred_mean,
            pred_config.get("lower_bound"),
            pred_config.get("upper_bound"),
        )
        self._agregar_etiquetas_valores(pred_mean)
        self._agregar_linea_divisoria(historico)

    def _obtener_etiqueta_prediccion(self, transformation, exog_info, simulation_config):
        """Obtener etiqueta apropiada para la prediccion."""
        if simulation_config and simulation_config.get("enabled", False):
            summary = simulation_config.get("summary", {})
            return f"Prediccion SIMULADA: {summary.get('escenario', 'N/A')}"
        if exog_info:
            return "Prediccion CON variables exogenas"
        return f"Prediccion ({transformation.upper()})"

    def _graficar_intervalo_confianza(self, pred_mean, lower_bound, upper_bound):
        """Graficar banda de intervalo de confianza."""
        if lower_bound is not None and upper_bound is not None:
            plt.fill_between(
                pred_mean.index,
                lower_bound.to_numpy(),
                upper_bound.to_numpy(),
                color="orange",
                alpha=0.25,
                label="Intervalo de confianza 95%",
                zorder=3,
            )

    def _agregar_etiquetas_valores(self, pred_mean):
        """Agregar etiquetas con valores predichos."""
        for x, y in zip(pred_mean.index, pred_mean.to_numpy(), strict=False):
            plt.text(
                x,
                y + 0.4,
                f"{y:.1f}",
                color="orange",
                fontsize=9,
                ha="center",
                va="bottom",
                weight="bold",
            )

    def _agregar_linea_divisoria(self, historico):
        """Agregar linea divisoria entre historico y prediccion."""
        if not historico.empty:
            plt.axvline(
                x=historico.index[-1],
                color="gray",
                linestyle="--",
                alpha=0.8,
                linewidth=2,
            )

    def _configurar_ejes(self, historico, faltantes, df):
        """Configurar ejes de la grafica."""
        ax = plt.gca()
        x_min = historico.index[0] if not historico.empty else df.index[0]
        x_max = faltantes.index[-1] if not faltantes.empty else historico.index[-1]
        plt.xlim(x_min, x_max)

        self._configurar_etiquetas_fechas(ax, x_min, x_max)

        plt.xlabel("Fecha", fontsize=12, weight="bold")
        plt.ylabel("SAIDI (minutos)", fontsize=12, weight="bold")
        plt.legend(fontsize=10, loc="upper left", frameon=True, shadow=True)
        plt.grid(True, alpha=0.4, linestyle="-", linewidth=0.8)
        plt.tight_layout()

    def _configurar_etiquetas_fechas(self, ax, x_min, x_max):
        """Configurar etiquetas de fechas en el eje X."""
        meses_espanol = [
            "Ene", "Feb", "Mar", "Abr", "May", "Jun",
            "Jul", "Ago", "Sep", "Oct", "Nov", "Dic",
        ]

        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        fechas_mensuales = pd.date_range(start=x_min, end=x_max, freq="MS")
        labels_mensuales = [
            f"{meses_espanol[f.month - 1]}\n{f.year}" for f in fechas_mensuales
        ]

        if len(fechas_mensuales) > 0:
            ax.set_xticks(fechas_mensuales)
            ax.set_xticklabels(
                labels_mensuales, rotation=45, ha="right", fontsize=9,
            )

    def _configurar_titulo(
        self, order, seasonal_order, transformation, metricas, config_dict,
    ):
        """
        Configurar titulo de la grafica.

        Args:
            order: Orden del modelo ARIMA
            seasonal_order: Orden estacional del modelo
            transformation: Tipo de transformación aplicada
            metricas: Diccionario con métricas del modelo
            config_dict: Diccionario con exog_info y simulation_config

        """
        exog_info = config_dict.get("exog_info")
        simulation_config = config_dict.get("simulation_config")

        exog_info_text = self._obtener_texto_variables(exog_info, simulation_config)
        precision_text = (
            f" - Precision: {metricas['precision_final']:.1f}%" if metricas else ""
        )
        order_str = f"({order[0]},{order[1]},{order[2]})"
        seasonal_str = f"({seasonal_order[0]},{seasonal_order[1]},{seasonal_order[2]},{seasonal_order[3]})"

        plt.title(
            f"SAIDI: Prediccion SARIMAX{order_str}x{seasonal_str} + {transformation.upper()}{exog_info_text}{precision_text}",
            fontsize=16,
            fontweight="bold",
            pad=20,
        )

    def _obtener_texto_variables(self, exog_info, simulation_config):
        """Obtener texto descriptivo de variables exogenas o simulacion."""
        if simulation_config and simulation_config.get("enabled", False):
            summary = simulation_config.get("summary", {})
            escenario_name = summary.get("escenario", "Simulado")
            return f" [SIMULACION: {escenario_name}]"
        if exog_info:
            vars_names = " + ".join(v["nombre"] for v in exog_info.values())
            return f" [Con: {vars_names}]"
        return ""

    def _agregar_notas_pie(self, exog_info, simulation_config):
        """Agregar notas al pie de la grafica."""
        footer_y = 0.01

        if simulation_config and simulation_config.get("enabled", False):
            footer_y = self._agregar_nota_simulacion(simulation_config, footer_y)

        if exog_info and not simulation_config:
            self._agregar_nota_variables(exog_info, footer_y)

    def _agregar_nota_simulacion(self, simulation_config, footer_y):
        """Agregar nota de simulacion climatica."""
        summary = simulation_config.get("summary", {})
        footer_text = f"SIMULACION CLIMATICA APLICADA: {summary.get('escenario', 'N/A')} | "
        footer_text += f"Alcance: {summary.get('alcance_meses', 'N/A')} meses | "
        footer_text += "Intervalos reflejan incertidumbre estadistica, no del escenario simulado"

        plt.figtext(
            0.5,
            footer_y,
            footer_text,
            ha="center",
            fontsize=9,
            style="italic",
            color="darkred",
            weight="bold",
            bbox={
                "boxstyle": "round,pad=0.5",
                "facecolor": "#FFEBEE",
                "alpha": 0.9,
                "edgecolor": "#F44336",
                "linewidth": 2,
            },
        )
        return footer_y + 0.04

    def _agregar_nota_variables(self, exog_info, footer_y):
        """Agregar nota de variables exogenas."""
        footer_text = "Con variables exogenas: "
        for var_data in exog_info.values():
            footer_text += f"{var_data['nombre']} (r={var_data['correlacion']:.3f}) "
        footer_text += " | Intervalos de confianza: 95%"

        plt.figtext(
            0.5,
            footer_y,
            footer_text,
            ha="center",
            fontsize=9,
            style="italic",
            color="darkblue",
            weight="bold",
            bbox={
                "boxstyle": "round,pad=0.5",
                "facecolor": "lightyellow",
                "alpha": 0.8,
            },
        )

    def cleanup_plot_file(self):
        """Limpiar archivo temporal de grafica."""
        if self.plot_file_path:
            try:
                Path(self.plot_file_path).unlink(missing_ok=True)
            except (OSError, PermissionError) as e:
                print(f"Error eliminando archivo temporal: {e}")
            finally:
                self.plot_file_path = None
