# services/prediction_service.py
import matplotlib
import numpy as np
import pandas as pd
matplotlib.use("Agg")
import os
import tempfile
import warnings
from datetime import datetime
from typing import Optional, Dict, Any

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


class PredictionService:
    """Servicio para generar predicciones SAIDI con variables exogenas climaticas, simulacion e intervalos de confianza"""

    # Mapeo de regionales a sus transformaciones optimas
    REGIONAL_TRANSFORMATIONS = {
        "SAIDI_O": "original", 
        "SAIDI_C": "original",
        "SAIDI_A": "original",
        "SAIDI_P": "boxcox",
        "SAIDI_T": "sqrt",
        "SAIDI_Cens": "original",
    }

    REGIONAL_ORDERS = {
        "SAIDI_O": {"order": (3, 1, 6), "seasonal_order": (3, 1, 0, 12)},
        "SAIDI_C": {"order": (3, 1, 2), "seasonal_order": (1, 1, 2, 12)},
        "SAIDI_A": {"order": (2, 1, 3), "seasonal_order": (2, 1, 1, 12)},
        "SAIDI_P": {"order": (4, 1, 3), "seasonal_order": (1, 1, 4, 12)},
        "SAIDI_T": {"order": (3, 1, 3), "seasonal_order": (2, 1, 2, 12)},
        "SAIDI_Cens": {"order": (4, 1, 3), "seasonal_order": (1, 1, 4, 12)},
    }

    # Variables exogenas por regional
    REGIONAL_EXOG_VARS = {
    'SAIDI_O': {  # Ocaña - 7 variables correlacionadas
        'realfeel_min': 'Temperatura aparente mínima',           # r=0.689 *** FUERTE
        'windchill_avg': 'Sensación térmica promedio',          # r=0.520 ** MODERADA-FUERTE
        'dewpoint_avg': 'Punto de rocío promedio',              # r=0.470 ** MODERADA-FUERTE
        'windchill_max': 'Sensación térmica máxima',            # r=0.464 ** MODERADA-FUERTE
        'dewpoint_min': 'Punto de rocío mínimo',                # r=0.456 ** MODERADA-FUERTE
        'precipitation_max_daily': 'Precipitación máxima diaria', # r=0.452
        'precipitation_avg_daily': 'Precipitación promedio diaria', # r=0.438
    },
    
    'SAIDI_C': {  # Cúcuta - 4 variables correlacionadas
        'realfeel_avg': 'Temperatura aparente promedio',        # r=0.573 ** MODERADA-FUERTE
        'pressure_rel_avg': 'Presión relativa promedio',        # r=-0.358 (negativa)
        'wind_speed_max': 'Velocidad máxima del viento',        # r=0.356
        'pressure_abs_avg': 'Presión absoluta promedio',        # r=-0.356 (negativa)
    },
    
    'SAIDI_T': {  # Tibú - 8 variables correlacionadas
        'realfeel_avg': 'Temperatura aparente promedio',        # r=0.906 *** MUY FUERTE
        'wind_dir_avg': 'Dirección promedio del viento',        # r=-0.400 (negativa)
        'uv_index_avg': 'Índice UV promedio',                   # r=0.385
        'heat_index_avg': 'Índice de calor promedio',           # r=0.363
        'temperature_min': 'Temperatura mínima',                # r=0.352
        'windchill_min': 'Sensación térmica mínima',            # r=0.340
        'temperature_avg': 'Temperatura promedio',              # r=0.338
        'pressure_rel_avg': 'Presión relativa promedio',        # r=-0.330 (negativa)
    },
    
    'SAIDI_A': {  # Aguachica - 2 variables correlacionadas
        'uv_index_max': 'Índice UV máximo',                     # r=0.664 *** FUERTE
        'days_with_rain': 'Días con lluvia',                    # r=0.535 ** MODERADA-FUERTE
    },
    
    'SAIDI_P': {  # Pamplona - 3 variables correlacionadas
        'precipitation_total': 'Precipitación total',           # r=0.577 ** MODERADA-FUERTE
        'precipitation_avg_daily': 'Precipitación promedio diaria', # r=0.552
        'realfeel_min': 'Temperatura aparente mínima',          # r=0.344
    },
}

    def __init__(self):
        self.default_order = (4, 1, 3)
        self.default_seasonal_order = (1, 1, 4, 12)
        self.plot_file_path = None
        self.scaler = None
        self.exog_scaler = None
        self.transformation_params = {}
        self.simulation_service = ClimateSimulationService()
        self.uncertainty_service = UncertaintyService()
        self.export_service = ExportService()

    def load_optimized_config(self, regional_code: str) -> Optional[Dict[str, Any]]:
        """
        Cargar configuración optimizada para una regional
        
        Lee el archivo JSON generado por OptimizationService y retorna
        los mejores parámetros encontrados previamente.
        
        Args:
            regional_code: Código de la regional (ej: 'SAIDI_O')
        
        Returns:
            Dict con configuración óptima o None si no existe
        """
        try:
            import json
            from pathlib import Path
            
            # Ubicación del archivo de configuración
            config_file = Path(__file__).parent.parent / 'config' / 'optimized_models.json'
            
            if not config_file.exists():
                print("[LOAD_CONFIG] No existe archivo de configuraciones optimizadas")
                return None
            
            # Cargar configuraciones
            with open(config_file, 'r', encoding='utf-8') as f:
                configs = json.load(f)
            
            # Buscar configuración de la regional
            if regional_code not in configs:
                print(f"[LOAD_CONFIG] No hay configuración optimizada para {regional_code}")
                return None
            
            config = configs[regional_code]
            
            print(f"[LOAD_CONFIG] ✓ Configuración cargada para {regional_code}")
            print(f"[LOAD_CONFIG]   Transformación: {config['transformation']}")
            print(f"[LOAD_CONFIG]   Order: {config['order']}")
            print(f"[LOAD_CONFIG]   Seasonal: {config['seasonal_order']}")
            print(f"[LOAD_CONFIG]   Precisión: {config['precision_final']:.1f}%")
            print(f"[LOAD_CONFIG]   Optimizado: {config['optimization_date']}")
            
            return config
            
        except Exception as e:
            print(f"[LOAD_CONFIG] ERROR cargando configuración: {e}")
            return None

    def _get_transformation_for_regional(self, regional_code):
        """
        MÉTODO ACTUALIZADO: Obtener transformación para la regional
        
        Primero intenta cargar de configuración optimizada,
        si no existe usa los defaults hardcodeados.
        """
        if not regional_code:
            return "original"
        
        # PRIORIDAD 1: Intentar cargar configuración optimizada
        optimized_config = self.load_optimized_config(regional_code)
        
        if optimized_config:
            transformation = optimized_config.get('transformation', 'original')
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
        MÉTODO ACTUALIZADO: Obtener órdenes SARIMAX específicos para una regional
        
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
            order = tuple(optimized_config['order'])
            seasonal_order = tuple(optimized_config['seasonal_order'])
            
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
        predictions_dict,
        regional_code,
        regional_nombre,
        output_dir=None,
        include_intervals=True,
        model_params=None,
        metrics=None,
    ):
        """
        Exportar predicciones a Excel usando ExportService

        Args:
            predictions_dict: Diccionario con predicciones {fecha: valor o dict}
            regional_code: Codigo de regional (ej: 'SAIDI_O')
            regional_nombre: Nombre de regional (ej: 'Ocana')
            output_dir: Directorio de salida (None = Desktop)
            include_intervals: Incluir intervalos de confianza
            model_params: Parametros del modelo
            metrics: Metricas del modelo

        Returns:
            str: Ruta del archivo exportado o None si hay error
        """
        try:
            # Preparar informacion del modelo para exportacion
            model_info = {}

            if model_params:
                model_info.update(model_params)

            if metrics:
                model_info["metrics"] = metrics

            # Llamar al servicio de exportacion
            filepath = self.export_service.export_predictions_to_excel(
                predictions_dict=predictions_dict,
                regional_code=regional_code,
                regional_nombre=regional_nombre,
                output_dir=output_dir,
                include_confidence_intervals=include_intervals,
                model_info=model_info,
            )

            return filepath

        except Exception as e:
            print(f"Error exportando predicciones: {str(e)}")
            return None

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
        Ejecutar prediccion SAIDI con variables exogenas climaticas, simulacion e intervalos de confianza
        
        ACTUALIZADO: Carga automáticamente parámetros optimizados si existen
        
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
            # ========== NUEVO: CARGAR CONFIGURACIÓN OPTIMIZADA ==========
            optimized_config = None
            
            if regional_code:
                optimized_config = self.load_optimized_config(regional_code)
                
                if optimized_config and log_callback:
                    log_callback("=" * 80)
                    log_callback("⚙️  USANDO CONFIGURACIÓN OPTIMIZADA")
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
                    regional_code
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
                        f"✓ Usando parametros default para regional {regional_nombre}"
                    )
                    log_callback(f"   Order: {order}")
                    log_callback(f"   Seasonal Order: {seasonal_order}")

            # Determinar transformacion segun regional (prioriza optimizada)
            transformation = self._get_transformation_for_regional(regional_code)

            if log_callback:
                order_str = str(order)
                seasonal_str = str(seasonal_order)
                log_callback(
                    f"Iniciando prediccion con parametros: order={order_str}, seasonal_order={seasonal_str}"
                )
                log_callback(
                    f"Regional: {regional_code} - Transformacion: {transformation.upper()}"
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
                    climate_data, df, regional_code, log_callback
                )

                if exog_df is not None:
                    if log_callback:
                        log_callback(
                            f"Variables exogenas disponibles: {len(exog_df.columns)}"
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
                    else:
                        # ========== CORREGIDO: NO escalar aquí ==========
                        # SARIMAX normaliza internamente las variables exógenas
                        # Escalar manualmente causa DOBLE ESCALADO y pérdida de precisión
                        
                        if simulation_config and simulation_config.get("enabled", False):
                            if log_callback:
                                log_callback("=" * 60)
                                log_callback("SIMULACION CLIMATICA ACTIVADA")
                                log_callback("=" * 60)
                                log_callback(
                                    "Variables exogenas SIN ESCALAR (para simulacion)"
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
                else:
                    if log_callback:
                        log_callback(
                            "No se pudieron preparar variables exogenas, continuando sin ellas"
                        )
            else:
                if log_callback:
                    log_callback(
                        "No hay datos climaticos disponibles, prediccion sin variables exogenas"
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
                    start=ultimo_mes + pd.DateOffset(months=1), periods=6, freq="MS"
                )
                for fecha in fechas_futuras:
                    df.loc[fecha, col_saidi] = np.nan
                faltantes = df[df[col_saidi].isna()]

            if log_callback:
                log_callback(f"Datos historicos SAIDI: {len(historico)} observaciones")
                log_callback(f"Meses a predecir: {len(faltantes)} observaciones")
                log_callback(
                    f"Periodo historico: {historico.index[0].strftime('%Y-%m')} a {historico.index[-1].strftime('%Y-%m')}"
                )

            if progress_callback:
                progress_callback(
                    30, f"Aplicando transformacion {transformation.upper()}..."
                )

            # Aplicar transformacion
            historico_values_original = historico[col_saidi].values
            historico_transformed, transform_info = self._apply_transformation(
                historico_values_original, transformation
            )
            historico_transformed_series = pd.Series(
                historico_transformed, index=historico.index
            )

            if log_callback:
                log_callback(f"Transformacion aplicada: {transform_info}")

            if progress_callback:
                progress_callback(40, "Calculando metricas del modelo...")

            # CORREGIDO: Calcular metricas con mismo metodo que OptimizationService
            historico_original_series = pd.Series(
                historico_values_original, index=historico.index
            )
            metricas = self._calcular_metricas_modelo(
                historico_original_series,
                order,
                seasonal_order,
                transformation,
                exog_df,  # Sin escalar
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
                if 'stability_score' in metricas:
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
                            f"Variables exogenas de entrenamiento: {len(exog_train)} periodos"
                        )
                        log_callback(
                            "NOTA: Variables en escala ORIGINAL (SARIMAX normaliza internamente)"
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
                            f"  Variables exogenas incluidas: {exog_train.shape[1]}"
                        )
                        log_callback("  Escalado: Interno de SARIMAX (no manual)")
            except Exception as e:
                raise Exception(f"Error ajustando modelo: {str(e)}")

            if progress_callback:
                progress_callback(
                    80, "Generando predicciones con intervalos de confianza..."
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
                            exog_forecast_original, simulation_config, log_callback
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
                    pred_mean_transformed, transformation
                )
                lower_bound_original = self._inverse_transformation(
                    lower_transformed, transformation
                )
                upper_bound_original = self._inverse_transformation(
                    upper_transformed, transformation
                )

                # Ajustar intervalos basado en precisión del modelo
                adjustment_factor = 1.0

                if metricas:
                    if metricas["mape"] > 30:
                        adjustment_factor = 1.15
                    elif metricas["mape"] > 20:
                        adjustment_factor = 1.10
                    elif metricas["mape"] > 15:
                        adjustment_factor = 1.05
                    else:
                        adjustment_factor = 1.02

                    if log_callback:
                        log_callback(
                            f"  Factor de ajuste de intervalos: {adjustment_factor:.2f}x (basado en MAPE={metricas['mape']:.1f}%)"
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
                                0, center - max_reasonable_width
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
                        f"  Ancho promedio de intervalos: ±{avg_width_pct:.0f}% del valor predicho"
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
                        f"Predicciones con intervalos de confianza generadas para {len(pred_mean)} periodos"
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

            except Exception as e:
                raise Exception(f"Error generando predicciones: {str(e)}")

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
                        "PREDICCION SAIDI CON SIMULACION CLIMATICA E INTERVALOS"
                    )
                else:
                    log_callback(
                        "RESUMEN DE PREDICCIONES CON INTERVALOS DE CONFIANZA"
                    )
                log_callback("=" * 60)

                if simulation_applied and simulation_config:
                    summary = simulation_config.get("summary", {})
                    log_callback(
                        f"Escenario simulado: {summary.get('escenario', 'N/A')}"
                    )
                    log_callback(
                        f"Dias simulados: {summary.get('dias_simulados', 'N/A')}"
                    )
                    log_callback(
                        f"Alcance: {summary.get('alcance_meses', 'N/A')} meses"
                    )

                log_callback(f"Predicciones generadas: {len(pred_mean)}")
                log_callback(
                    f"Periodo: {faltantes.index[0].strftime('%Y-%m')} a {faltantes.index[-1].strftime('%Y-%m')}"
                )
                log_callback("Valores predichos con intervalos de confianza (95%):")

                for fecha, valor, inferior, superior, margen in zip(
                    faltantes.index,
                    pred_mean,
                    lower_bound,
                    upper_bound,
                    margin_error_series,
                ):
                    margen_sup = superior - valor
                    margen_inf = valor - inferior
                    margen_pct = (margen / valor * 100) if valor > 0 else 0

                    log_callback(
                        f"  • {fecha.strftime('%Y-%m')}: {valor:.2f} min "
                        f"[IC: {inferior:.2f} - {superior:.2f}] "
                        f"(+{margen_sup:.2f}/-{margen_inf:.2f} | ±{margen_pct:.0f}%)"
                    )

                if exog_info:
                    log_callback("\nVariables exogenas utilizadas (escala original):")
                    for var_code, var_info in exog_info.items():
                        log_callback(
                            f"  - {var_info['nombre']}: correlacion {var_info['correlacion']}"
                        )

                log_callback("=" * 60)

            # Generar grafica con intervalos de confianza
            plot_path = self._generar_grafica(
                historico,
                pred_mean,
                faltantes,
                df,
                col_saidi,
                order,
                seasonal_order,
                metricas,
                transformation,
                exog_info,
                simulation_config if simulation_applied else None,
                lower_bound,
                upper_bound,
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

        except Exception as e:
            if log_callback:
                log_callback(f"ERROR: {str(e)}")
            raise Exception(f"Error en prediccion: {str(e)}")

    
    def _apply_climate_simulation(
        self, exog_forecast_original, simulation_config, log_callback=None
    ):
        """
        Aplicar simulación climática a variables SIN ESCALAR
        CORREGIDO: Manejo correcto de la estructura de simulation_config
        
        Args:
            exog_forecast_original: DataFrame SIN ESCALAR con variables exógenas
            simulation_config: Dict con configuración de simulación
            log_callback: Función para logging
        
        Returns:
            DataFrame con simulación aplicada (sin escalar para SARIMAX)
        """
        try:
            # Validar que la simulación esté habilitada
            if not simulation_config.get("enabled", False):
                if log_callback:
                    log_callback("Simulación NO habilitada, usando valores originales")
                return exog_forecast_original
            
            if log_callback:
                log_callback("=" * 60)
                log_callback("APLICANDO SIMULACIÓN CLIMÁTICA")
                log_callback("=" * 60)
                log_callback("   Entrada: valores originales SIN ESCALAR")
            
            escenario = simulation_config.get("scenario_name", 
                        simulation_config.get("escenario", "condiciones_normales"))
            
            # Extraer configuración del slider
            slider_adjustment = simulation_config.get("slider_adjustment", 0)
            dias_base = simulation_config.get("dias_base", 30)
            alcance_meses = simulation_config.get("alcance_meses", 3)
            percentiles = simulation_config.get("percentiles", {})
            regional_code = simulation_config.get("regional_code", "SAIDI_O")
            
            # Validar que tengamos los datos necesarios
            if not percentiles:
                if log_callback:
                    log_callback("ERROR: No hay percentiles disponibles para simulación")
                    log_callback("Usando valores originales sin simulación")
                return exog_forecast_original
            
            if log_callback:
                log_callback(f"   Escenario: {escenario}")
                log_callback(f"   Regional: {regional_code}")
                log_callback(f"   Slider: {slider_adjustment:+d} días sobre base de {dias_base}")
                log_callback(f"   Alcance: {alcance_meses} mes(es)")
            
            # Calcular factor de intensidad
            dias_simulados = dias_base + slider_adjustment
            intensity_adjustment = dias_simulados / dias_base if dias_base > 0 else 1.0
            
            if log_callback:
                log_callback(f"   Intensidad calculada: {intensity_adjustment:.2f}x")
            
            # ========== APLICAR SIMULACIÓN ==========
            try:
                exog_simulated = self.simulation_service.apply_simulation(
                    exog_forecast=exog_forecast_original,
                    scenario_name=escenario,
                    intensity_adjustment=intensity_adjustment,
                    alcance_meses=alcance_meses,
                    percentiles=percentiles,
                    regional_code=regional_code
                )
                
                if log_callback:
                    log_callback("   ✓ Simulación aplicada correctamente")
                    
                    # Mostrar cambios en el primer mes
                    if alcance_meses >= 1 and len(exog_simulated) > 0:
                        log_callback("\n   📊 Cambios en primer mes:")
                        for col in exog_simulated.columns:
                            try:
                                original_val = exog_forecast_original.iloc[0][col]
                                simulated_val = exog_simulated.iloc[0][col]
                                
                                if original_val != 0:
                                    change_pct = ((simulated_val - original_val) / original_val) * 100
                                else:
                                    change_pct = 0
                                
                                log_callback(
                                    f"     - {col}: {original_val:.2f} → {simulated_val:.2f} "
                                    f"({change_pct:+.1f}%)"
                                )
                            except Exception as e:
                                log_callback(f"     - {col}: Error mostrando cambio: {e}")
                    
                    log_callback("\n   Salida: valores SIMULADOS (escala original)")
                    log_callback("=" * 60)
                
                return exog_simulated
                
            except Exception as sim_error:
                if log_callback:
                    log_callback(f"ERROR en apply_simulation: {str(sim_error)}")
                    import traceback
                    log_callback(traceback.format_exc())
                
                # Fallback: retornar valores originales
                if log_callback:
                    log_callback("FALLBACK: Usando valores originales sin simulación")
                return exog_forecast_original
            
        except Exception as e:
            if log_callback:
                log_callback(f"ERROR CRÍTICO en _apply_climate_simulation: {str(e)}")
                import traceback
                log_callback(traceback.format_exc())
            
            # En caso de error total, retornar valores originales
            return exog_forecast_original

    def _prepare_exogenous_variables(
    self, climate_data, df_saidi, regional_code, log_callback=None
    ):
        """
        Preparar variables exógenas climáticas SIN ESCALAR
        ALINEADO CON OptimizationService para obtener métricas consistentes
        
        CAMBIOS CRÍTICOS:
        1. Mapeo de columnas con coincidencia parcial (no solo exacta)
        2. Validación de cobertura temporal (80% mínimo en overlap)
        3. Validación de varianza no-cero en overlap
        4. Relleno inteligente: forward-fill + backward-fill (máx 3) + media
        5. RETORNA EN ESCALA ORIGINAL (sin StandardScaler)
        
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
            # Validaciones iniciales
            if climate_data is None or climate_data.empty:
                if log_callback:
                    log_callback("Sin datos climáticos disponibles")
                return None, None
            
            if not regional_code or regional_code not in self.REGIONAL_EXOG_VARS:
                if log_callback:
                    log_callback(f"Regional {regional_code} sin variables definidas")
                return None, None
            
            if log_callback:
                log_callback(f"Preparando variables para {regional_code}")
                log_callback("MODO: SIN ESCALADO (valores originales)")
            
            # ========== VALIDAR ÍNDICE DATETIME ==========
            if not isinstance(climate_data.index, pd.DatetimeIndex):
                # Buscar columna de fecha
                fecha_col = None
                for col in ['fecha', 'Fecha', 'date', 'Date', 'month_date']:
                    if col in climate_data.columns:
                        fecha_col = col
                        break
                
                if fecha_col is None:
                    if log_callback:
                        log_callback("ERROR: No se encontró columna de fecha válida")
                    return None, None
                
                try:
                    climate_data = climate_data.copy()
                    climate_data[fecha_col] = pd.to_datetime(climate_data[fecha_col])
                    climate_data = climate_data.set_index(fecha_col)
                except Exception as e:
                    if log_callback:
                        log_callback(f"ERROR convirtiendo índice: {str(e)}")
                    return None, None
            
            # Verificar que ahora es DatetimeIndex
            if not isinstance(climate_data.index, pd.DatetimeIndex):
                if log_callback:
                    log_callback("ERROR: Formato de fecha inválido")
                return None, None
            
            # ========== ANÁLISIS DE COBERTURA TEMPORAL ==========
            historico = df_saidi[df_saidi['SAIDI'].notna() if 'SAIDI' in df_saidi.columns else df_saidi['SAIDI Historico'].notna()]
            
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
                return None, None
            
            overlap_mask = (historico.index >= overlap_start) & (historico.index <= overlap_end)
            overlap_months = overlap_mask.sum()
            
            # Validar overlap mínimo (12 meses)
            if overlap_months < 12:
                if log_callback:
                    log_callback(f"ERROR: Overlap insuficiente ({overlap_months} < 12 meses)")
                return None, None
            
            if log_callback:
                log_callback(f"SAIDI: {saidi_start.strftime('%Y-%m')} a {saidi_end.strftime('%Y-%m')} ({len(historico)} meses)")
                log_callback(f"CLIMA: {clima_start.strftime('%Y-%m')} a {clima_end.strftime('%Y-%m')} ({len(climate_data)} meses)")
                log_callback(f"OVERLAP: {overlap_start.strftime('%Y-%m')} a {overlap_end.strftime('%Y-%m')} ({overlap_months} meses)")
            
            # ========== MAPEO AUTOMÁTICO DE COLUMNAS ==========
            exog_vars_config = self.REGIONAL_EXOG_VARS[regional_code]
            
            # Normalizar nombres disponibles
            available_cols_normalized = {}
            for col in climate_data.columns:
                normalized = col.lower().strip().replace(' ', '_').replace('-', '_')
                available_cols_normalized[normalized] = col
            
            # Mapear cada variable con búsqueda flexible
            climate_column_mapping = {}
            
            for var_code in exog_vars_config.keys():
                var_normalized = var_code.lower().strip()
                
                # Intento 1: Coincidencia exacta
                if var_normalized in available_cols_normalized:
                    climate_column_mapping[var_code] = available_cols_normalized[var_normalized]
                    continue
                
                # Intento 2: Coincidencia parcial (al menos 2 partes)
                var_parts = var_normalized.split('_')
                best_match = None
                best_match_score = 0
                
                for norm_col, orig_col in available_cols_normalized.items():
                    matches = sum(1 for part in var_parts if part in norm_col)
                    if matches > best_match_score:
                        best_match_score = matches
                        best_match = orig_col
                
                if best_match_score >= 2:
                    climate_column_mapping[var_code] = best_match
            
            if not climate_column_mapping:
                if log_callback:
                    log_callback("ERROR: No se pudo mapear ninguna variable")
                return None, None
            
            # ========== PREPARACIÓN DE VARIABLES SIN ESCALADO ==========
            exog_df = pd.DataFrame(index=historico.index)
            exog_info = {}
            
            for var_code, var_nombre in exog_vars_config.items():
                climate_col = climate_column_mapping.get(var_code)
                
                if not climate_col or climate_col not in climate_data.columns:
                    continue
                
                try:
                    # Extraer serie del clima
                    var_series = climate_data[climate_col].copy()
                    
                    # Crear serie alineada (inicialmente vacía)
                    aligned_series = pd.Series(index=historico.index, dtype=float)
                    
                    # Llenar datos donde hay overlap REAL
                    for date in historico.index:
                        if date in var_series.index:
                            aligned_series[date] = var_series.loc[date]
                    
                    # VALIDACIÓN: Cobertura en overlap
                    overlap_data = aligned_series[overlap_mask]
                    datos_reales_overlap = overlap_data.notna().sum()
                    overlap_pct = (datos_reales_overlap / overlap_months) * 100
                    
                    # RECHAZAR si cobertura < 80%
                    if overlap_pct < 80:
                        if log_callback:
                            log_callback(f"X RECHAZADA {var_code}: cobertura {overlap_pct:.1f}% < 80%")
                        continue
                    
                    # VALIDACIÓN: Varianza en overlap
                    var_std = overlap_data.std()
                    
                    if pd.isna(var_std) or var_std == 0:
                        if log_callback:
                            log_callback(f"X RECHAZADA {var_code}: varianza = 0")
                        continue
                    
                    # Forward-fill para fechas futuras
                    aligned_series = aligned_series.fillna(method='ffill')
                    
                    # Backward-fill (máx 3 meses) para fechas pasadas
                    aligned_series = aligned_series.fillna(method='bfill', limit=3)
                    
                    # Si AÚN hay NaN, rellenar con media del overlap
                    if aligned_series.isnull().any():
                        mean_overlap = overlap_data.mean()
                        aligned_series = aligned_series.fillna(mean_overlap)
                    
                    # VERIFICACIÓN FINAL
                    final_nan = aligned_series.isnull().sum()
                    if final_nan > 0:
                        if log_callback:
                            log_callback(f"X RECHAZADA {var_code}: {final_nan} NaN finales")
                        continue
                    
                    # ===== GUARDAR EN ESCALA ORIGINAL =====
                    exog_df[var_code] = aligned_series
                    
                    exog_info[var_code] = {
                        'nombre': var_nombre,
                        'columna_clima': climate_col,
                        'correlacion': self._get_correlation_for_var(var_code, regional_code),
                        'scaled': False,  # CRÍTICO
                        'datos_reales_overlap': int(datos_reales_overlap),
                        'overlap_coverage_pct': float(overlap_pct),
                        'varianza_overlap': float(var_std)
                    }
                    
                    if log_callback:
                        log_callback(f"✓ {var_code} -> ACEPTADA ({overlap_pct:.1f}% cobertura, escala original)")
                        
                except Exception as e:
                    if log_callback:
                        log_callback(f"X ERROR {var_code}: {e}")
                    continue
            
            # VALIDACIÓN FINAL
            if exog_df.empty or exog_df.shape[1] == 0:
                if log_callback:
                    log_callback("ERROR: Ninguna variable aceptada")
                return None, None
            
            if log_callback:
                log_callback("=" * 60)
                log_callback(f"✓ Variables preparadas: {len(exog_df.columns)}")
                log_callback("  ESCALA: ORIGINAL (sin StandardScaler)")
                log_callback("  Rangos:")
                for col in exog_df.columns:
                    log_callback(f"    - {col}: [{exog_df[col].min():.2f}, {exog_df[col].max():.2f}]")
                log_callback("=" * 60)
            
            return exog_df, exog_info if exog_info else None
            
        except Exception as e:
            if log_callback:
                log_callback(f"ERROR CRÍTICO: {str(e)}")
            return None, None


    def _align_exog_to_saidi(self, exog_series, df_saidi, var_code, log_callback=None):
        """
        Alinear datos exogenos al indice de SAIDI
        SIN MODIFICAR: Este método ya funciona correctamente
        
        Estrategia:
        - Para fechas con datos climaticos: usar el valor directo
        - Para fechas sin datos climaticos: extrapolar usando ultimo valor
        """
        try:
            # Obtener las fechas del clima
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

                if log_callback:
                    n_future = future_indices.sum()
                    log_callback(
                        f"    {var_code}: {n_future} valores futuros proyectados"
                    )

            # Backward fill: usar primer valor para fechas previas
            min_climate_date = climate_dates.min()
            past_indices = saidi_dates < min_climate_date

            if past_indices.any():
                first_known_value = exog_series.iloc[0].iloc[0]
                result.loc[past_indices] = first_known_value

                if log_callback:
                    n_past = past_indices.sum()
                    log_callback(
                        f"    {var_code}: {n_past} valores pasados proyectados"
                    )

            return result

        except Exception as e:
            if log_callback:
                log_callback(f"Error alineando variable {var_code}: {str(e)}")
            return None


    def _get_correlation_for_var(self, var_code, regional_code):
        """
        Obtener correlación documentada de una variable específica
        ACTUALIZADO con correlaciones REALES de OptimizationService
        
        Args:
            var_code: Código de la variable (ej: 'realfeel_min')
            regional_code: Código de la regional (ej: 'SAIDI_O')
        
        Returns:
            float: Correlación documentada o 0.0 si no existe
        """
        # ❌ ANTIGUO (correlaciones genéricas ficticias):
        """
        correlations = {
            "SAIDI_O": {"temp_max": 0.450, "humedad_avg": 0.380, "precip_total": 0.420},
            # ... todas iguales ...
        }
        """
        
        # ✅ NUEVO (correlaciones REALES documentadas):
        correlations = {
            'SAIDI_O': {  # Ocaña
                'realfeel_min': 0.689,              # *** FUERTE
                'windchill_avg': 0.520,             # ** MODERADA-FUERTE
                'dewpoint_avg': 0.470,              # ** MODERADA-FUERTE
                'windchill_max': 0.464,             # ** MODERADA-FUERTE
                'dewpoint_min': 0.456,              # ** MODERADA-FUERTE
                'precipitation_max_daily': 0.452,
                'precipitation_avg_daily': 0.438,
            },
            
            'SAIDI_C': {  # Cúcuta
                'realfeel_avg': 0.573,              # ** MODERADA-FUERTE
                'pressure_rel_avg': -0.358,         # Negativa
                'wind_speed_max': 0.356,
                'pressure_abs_avg': -0.356,         # Negativa
            },
            
            'SAIDI_T': {  # Tibú
                'realfeel_avg': 0.906,              # *** MUY FUERTE
                'wind_dir_avg': -0.400,             # Negativa
                'uv_index_avg': 0.385,
                'heat_index_avg': 0.363,
                'temperature_min': 0.352,
                'windchill_min': 0.340,
                'temperature_avg': 0.338,
                'pressure_rel_avg': -0.330,         # Negativa
            },
            
            'SAIDI_A': {  # Aguachica
                'uv_index_max': 0.664,              # *** FUERTE
                'days_with_rain': 0.535,            # ** MODERADA-FUERTE
            },
            
            'SAIDI_P': {  # Pamplona
                'precipitation_total': 0.577,       # ** MODERADA-FUERTE
                'precipitation_avg_daily': 0.552,
                'realfeel_min': 0.344,
            },
        }
        
        # Buscar correlación específica
        if regional_code in correlations and var_code in correlations[regional_code]:
            return correlations[regional_code][var_code]
        
        return 0.0

    def _align_exog_to_saidi(self, exog_series, df_saidi, var_code, log_callback=None):
        """
        Alinear datos exogenos al indice de SAIDI

        Estrategia:
        - Para fechas con datos climaticos: usar el valor directo
        - Para fechas sin datos climaticos: extrapolar usando ultimo valor
        """
        try:
            # Obtener las fechas del clima
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
            # (despues del periodo de datos climaticos)
            max_climate_date = climate_dates.max()

            # Usar indexacion booleana correctamente
            future_indices = saidi_dates > max_climate_date

            if future_indices.any():
                last_known_value = exog_series.iloc[-1].iloc[0]
                result.loc[future_indices] = last_known_value

                if log_callback:
                    n_future = future_indices.sum()
                    log_callback(
                        f"  - {var_code}: {n_future} valores futuros proyectados desde ultima observacion"
                    )

            # Backward fill: usar primer valor para fechas previas
            # (antes del periodo de datos climaticos)
            min_climate_date = climate_dates.min()

            # Usar indexacion booleana correctamente
            past_indices = saidi_dates < min_climate_date

            if past_indices.any():
                first_known_value = exog_series.iloc[0].iloc[0]
                result.loc[past_indices] = first_known_value

                if log_callback:
                    n_past = past_indices.sum()
                    log_callback(
                        f"  - {var_code}: {n_past} valores pasados proyectados desde primera observacion"
                    )

            # Verificar cobertura
            if log_callback:
                n_direct = (~past_indices & ~future_indices).sum()
                log_callback(f"  - {var_code}: {n_direct} valores directos del clima")

            return result

        except Exception as e:
            if log_callback:
                log_callback(f"Error alineando variable {var_code}: {str(e)}")
                import traceback

                log_callback(traceback.format_exc())
            return None

    def _extend_exogenous_for_forecast(
        self, exog_df, forecast_dates, log_callback=None, unscale=False
    ):
        """
        Extender variables exogenas para prediccion

        Args:
            exog_df: DataFrame con variables exogenas (pueden estar escaladas o no)
            forecast_dates: Fechas para las que se necesitan predicciones
            log_callback: Funcion para logging
            unscale: Si True, des-escalar antes de extender

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
                index=forecast_dates, columns=exog_df_original.columns
            )

            for col in exog_df_original.columns:
                last_value = exog_df_original[col].iloc[-1]
                exog_forecast[col] = last_value

            if log_callback:
                log_callback(
                    f"  Variables extendidas: {len(forecast_dates)} periodos (sin escalar)"
                )

            return exog_forecast  # Retorna SIN ESCALAR

        except Exception as e:
            if log_callback:
                log_callback(f"ERROR extendiendo variables exogenas: {str(e)}")
            return None

    def _get_correlation_for_var(self, var_code, regional_code):
        """Obtener la correlacion documentada de una variable"""
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
        Diagnosticar cobertura temporal de variables exógenas
        COPIADO DE OptimizationService para consistencia
        
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
            saidi_start = serie_saidi.index[0]
            saidi_end = serie_saidi.index[-1]
            exog_start = exog_df.index[0]
            exog_end = exog_df.index[-1]
            
            if log_callback:
                log_callback("=" * 60)
                log_callback("DIAGNOSTICO DE COBERTURA EXOGENA")
                log_callback("=" * 60)
                log_callback(f"SAIDI: {saidi_start.strftime('%Y-%m')} a {saidi_end.strftime('%Y-%m')} ({len(serie_saidi)} obs)")
                log_callback(f"EXOG:  {exog_start.strftime('%Y-%m')} a {exog_end.strftime('%Y-%m')} ({len(exog_df)} obs)")
            
            # 1. Verificar que los índices coinciden EXACTAMENTE
            if not exog_df.index.equals(serie_saidi.index):
                if log_callback:
                    log_callback("ADVERTENCIA: Indices no coinciden exactamente")
                
                # Verificar fechas faltantes
                missing_in_exog = [d for d in serie_saidi.index if d not in exog_df.index]
                
                if missing_in_exog:
                    pct_missing = len(missing_in_exog) / len(serie_saidi) * 100
                    
                    if log_callback:
                        log_callback(f"Fechas SAIDI faltantes en EXOG: {len(missing_in_exog)} ({pct_missing:.1f}%)")
                    
                    # CRÍTICO: Si falta >20% de fechas, rechazar
                    if pct_missing > 20:
                        if log_callback:
                            log_callback("ERROR CRITICO: >20% de fechas faltantes")
                            log_callback("Las variables exogenas NO cubren suficiente periodo historico")
                        return False
            
            # 2. Verificar que NO hay NaN en ninguna columna
            if exog_df.isnull().any().any():
                nan_cols = exog_df.columns[exog_df.isnull().any()].tolist()
                
                if log_callback:
                    log_callback("ERROR: Columnas con NaN encontradas:")
                    for col in nan_cols:
                        nan_count = exog_df[col].isnull().sum()
                        pct_nan = (nan_count / len(exog_df)) * 100
                        log_callback(f"  - {col}: {nan_count} NaN ({pct_nan:.1f}%)")
                    log_callback("Variables exogenas deben estar completamente rellenas")
                
                return False
            
            # 3. Verificar valores infinitos
            if np.isinf(exog_df.values).any():
                if log_callback:
                    log_callback("ERROR: Variables exogenas contienen valores infinitos")
                return False
            
            # 4. Verificar que hay varianza en las variables
            zero_variance_vars = []
            for col in exog_df.columns:
                if exog_df[col].std() == 0:
                    zero_variance_vars.append(col)
            
            if zero_variance_vars:
                if log_callback:
                    log_callback("ADVERTENCIA: Variables con varianza cero:")
                    for var in zero_variance_vars:
                        log_callback(f"  - {var}")
                    log_callback("Estas variables no aportan informacion al modelo")
                # No rechazar por esto, solo advertir
            
            if log_callback:
                log_callback("✓ Cobertura temporal y calidad de datos OK")
                log_callback("=" * 60)
            
            return True
            
        except Exception as e:
            if log_callback:
                log_callback(f"ERROR durante diagnostico: {e}")
            return False

    def _apply_transformation(self, data, transformation_type):
        """Aplicar transformacion a los datos"""
        if transformation_type == "original":
            return data, "Sin transformacion (datos originales)"

        elif transformation_type == "standard":
            self.scaler = StandardScaler()
            transformed = self.scaler.fit_transform(data.reshape(-1, 1)).flatten()
            return transformed, "StandardScaler"

        elif transformation_type == "log":
            data_positive = np.maximum(data, 1e-10)
            transformed = np.log(data_positive)
            self.transformation_params["log_applied"] = True
            return transformed, "Transformacion logaritmica"

        elif transformation_type == "boxcox":
            data_positive = np.maximum(data, 1e-10)
            transformed, lambda_param = stats.boxcox(data_positive)
            self.transformation_params["boxcox_lambda"] = lambda_param
            return transformed, f"Box-Cox (lambda={lambda_param:.4f})"

        elif transformation_type == "sqrt":
            data_positive = np.maximum(data, 1e-10)
            transformed = np.sqrt(data_positive)
            return transformed, "Transformacion raiz cuadrada"

        else:
            return data, "Sin transformacion"

    def _inverse_transformation(self, data, transformation_type):
        """Revertir transformacion a escala original"""
        if transformation_type == "original":
            return data
        elif transformation_type == "standard":
            if self.scaler is not None:
                return self.scaler.inverse_transform(data.reshape(-1, 1)).flatten()
            return data
        elif transformation_type == "log":
            return np.exp(data)
        elif transformation_type == "boxcox":
            lambda_param = self.transformation_params.get("boxcox_lambda", 0)
            if lambda_param == 0:
                return np.exp(data)
            else:
                return np.power(data * lambda_param + 1, 1 / lambda_param)
        elif transformation_type == "sqrt":
            return np.power(data, 2)
        else:
            return data

    def _calcular_metricas_modelo(
    self, serie_original, order, seasonal_order, transformation, exog_df=None
    ):
        """
        Calcular métricas del modelo con VALIDACIÓN IDÉNTICA a OptimizationService
        
        CAMBIOS CRÍTICOS:
        1. Usa train/test split adaptativo (20-30% según cantidad de datos)
        2. Variables exógenas SIN ESCALAR (SARIMAX normaliza internamente)
        3. Validación estricta de alineación de índices
        4. Calcula stability_score como OptimizationService
        
        Args:
            serie_original: Serie temporal SAIDI en escala original
            order: Parámetros (p,d,q) del modelo
            seasonal_order: Parámetros estacionales (P,D,Q,s)
            transformation: Tipo de transformación aplicada
            exog_df: DataFrame con variables exógenas EN ESCALA ORIGINAL (opcional)
        
        Returns:
            Dict con métricas del modelo o None si falla
        """
        try:
            # Calcular porcentaje de validación adaptativo (IGUAL que OptimizationService)
            n_obs = len(serie_original)
            if n_obs >= 60:
                pct_validacion = 0.30
            elif n_obs >= 36:
                pct_validacion = 0.25
            else:
                pct_validacion = 0.20
            
            n_test = max(6, int(n_obs * pct_validacion))
            
            # Dividir en train/test
            train_original = serie_original[:-n_test]
            test_original = serie_original[-n_test:]
            
            if len(train_original) < 12:
                return None
            
            # Aplicar transformación a SAIDI
            self.scaler = None
            self.transformation_params = {}
            
            train_transformed, _ = self._apply_transformation(
                train_original.values, transformation
            )
            train_transformed_series = pd.Series(
                train_transformed, index=train_original.index
            )
            
            # CORREGIDO: Preparar exógenas SIN ESCALAR con validación estricta
            exog_train = None
            exog_test = None
            
            if exog_df is not None:
                try:
                    train_index = train_original.index
                    test_index = test_original.index
                    
                    # VALIDACIÓN 1: Verificar que exog_df contiene TODAS las fechas
                    missing_train = [idx for idx in train_index if idx not in exog_df.index]
                    missing_test = [idx for idx in test_index if idx not in exog_df.index]
                    
                    if missing_train or missing_test:
                        # Rechazar: faltan fechas
                        print(f"[METRICAS] Rechazado: faltan {len(missing_train)} fechas train, {len(missing_test)} test")
                        return None
                    
                    # VALIDACIÓN 2: Extraer subconjuntos con .loc (garantiza alineación)
                    exog_train = exog_df.loc[train_index].copy()
                    exog_test = exog_df.loc[test_index].copy()
                    
                    # VALIDACIÓN 3: Verificar que NO hay NaN
                    if exog_train.isnull().any().any() or exog_test.isnull().any().any():
                        print("[METRICAS] Rechazado: NaN en exógenas")
                        return None
                    
                    # VALIDACIÓN 4: Verificar dimensiones correctas
                    if len(exog_train) != len(train_original) or len(exog_test) != n_test:
                        print("[METRICAS] Rechazado: dimensiones incorrectas")
                        return None
                    
                    # VALIDACIÓN 5: Verificar que no hay infinitos
                    if np.isinf(exog_train.values).any() or np.isinf(exog_test.values).any():
                        print("[METRICAS] Rechazado: infinitos en exógenas")
                        return None
                    
                    # NOTA: NO ESCALAR - exog_train y exog_test permanecen en escala original
                    
                except Exception as e:
                    print(f"[METRICAS] Error preparando exógenas: {e}")
                    return None
            
            # Entrenar modelo con exógenas SIN ESCALAR
            try:
                model = SARIMAX(
                    train_transformed_series,
                    exog=exog_train,  # EN ESCALA ORIGINAL
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=True,
                    enforce_invertibility=True,
                )
                
                # Fit con maxiter limitado
                results = model.fit(disp=False, maxiter=50)
                
            except Exception as e:
                print(f"[METRICAS] Error en fit: {e}")
                return None
            
            # Predecir en test
            try:
                pred = results.get_forecast(steps=n_test, exog=exog_test)
                pred_mean_transformed = pred.predicted_mean
                
            except Exception as e:
                print(f"[METRICAS] Error en forecast: {e}")
                return None
            
            # Revertir transformación
            try:
                pred_mean_original = self._inverse_transformation(
                    pred_mean_transformed.values, transformation
                )
            except Exception as e:
                print(f"[METRICAS] Error invirtiendo transformación: {e}")
                return None
            
            # Calcular métricas en escala original
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
            
            # NUEVO: Score de estabilidad (como OptimizationService)
            stability_score = self._calculate_stability_score(
                test_values, pred_values, precision_final, mape
            )
            
            return {
                'rmse': rmse,
                'mae': mae,
                'mape': mape,
                'r2_score': r2_score,
                'precision_final': precision_final,
                'aic': results.aic,
                'bic': results.bic,
                'composite_score': composite_score,
                'n_params': complexity_penalty,
                'n_test': n_test,
                'stability_score': stability_score,
                'validation_pct': pct_validacion * 100,
                'exog_scaled': False  # CRÍTICO: Marcar que NO están escaladas
            }
            
        except Exception as e:
            print(f"Error calculando metricas: {e}")
            return None


    def _calculate_stability_score(self, 
                            actual_values: np.ndarray,  
                            predicted_values: np.ndarray,  
                            precision: float,
                            mape: float) -> float:
        """
        Calcular score de estabilidad del modelo
        COPIADO DE OptimizationService para consistencia
        
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
            if mean_abs_error > 1e-8:
                cv_error = std_error / mean_abs_error
                # Convertir a score (menor CV = mayor estabilidad)
                stability_cv = max(0, 100 * (1 - min(cv_error, 1)))
            else:
                # Si errores son muy pequeños, estabilidad neutral
                stability_cv = 50.0
            
            # Penalización adaptativa por MAPE
            if mape > 50:
                mape_penalty = 0.5  # Penalización fuerte
            elif mape > 30:
                mape_penalty = 0.7  # Penalización moderada
            else:
                mape_penalty = 1.0  # Sin penalización
            
            stability_cv = stability_cv * mape_penalty
            
            # Combinar estabilidad con precisión (60% estabilidad, 40% precisión)
            stability = (stability_cv * 0.6) + (precision * 0.4)
            
            return min(100.0, max(0.0, stability))
            
        except Exception:
            return 0.0

    def _generar_grafica(
        self,
        historico,
        pred_mean,
        faltantes,
        df,
        col_saidi,
        order,
        seasonal_order,
        metricas,
        transformation,
        exog_info=None,
        simulation_config=None,
        lower_bound=None,
        upper_bound=None,
    ):
        """Generar grafica de prediccion con intervalos de confianza y indicador de simulacion"""
        try:
            if historico.empty or pred_mean.empty:
                return None

            temp_dir = tempfile.gettempdir()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_path = os.path.join(temp_dir, f"saidi_prediction_{timestamp}.png")

            fig = plt.figure(figsize=(16, 10), dpi=100)

            # Grafica principal - datos historicos
            plt.plot(
                historico.index,
                historico[col_saidi],
                label="SAIDI Historico",
                color="blue",
                linewidth=3,
                marker="o",
                markersize=5,
            )

            if not historico.empty and len(pred_mean) > 0:
                ultimo_real_x = historico.index[-1]
                ultimo_real_y = historico[col_saidi].iloc[-1]

                # CORREGIDO: Conectar solo el primer punto, no toda la serie
                x_pred = [ultimo_real_x] + list(pred_mean.index)
                y_pred = [ultimo_real_y] + list(pred_mean.values)

                # Etiqueta segun si hay simulacion
                if simulation_config and simulation_config.get("enabled", False):
                    summary = simulation_config.get("summary", {})
                    pred_label = (
                        f"Prediccion SIMULADA: {summary.get('escenario', 'N/A')}"
                    )
                elif exog_info:
                    pred_label = "Prediccion CON variables exogenas"
                else:
                    pred_label = f"Prediccion ({transformation.upper()})"

                # Linea de prediccion - SOLO dibujar desde ultimo real hasta predicciones
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

                # CORREGIDO: Banda de intervalo de confianza SIN extender al ultimo real
                if lower_bound is not None and upper_bound is not None:
                    # NO incluir punto de conexion - solo las predicciones
                    plt.fill_between(
                        pred_mean.index,  # Solo fechas de prediccion
                        lower_bound.values,  # Limites directos
                        upper_bound.values,
                        color="orange",
                        alpha=0.25,
                        label="Intervalo de confianza 95%",
                        zorder=3,
                    )

                # Etiquetas de valores predichos
                for x, y in zip(pred_mean.index, pred_mean.values):
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

            # Linea divisoria entre historico y prediccion
            if not historico.empty:
                plt.axvline(
                    x=historico.index[-1],
                    color="gray",
                    linestyle="--",
                    alpha=0.8,
                    linewidth=2,
                )

            ax = plt.gca()
            x_min = historico.index[0] if not historico.empty else df.index[0]
            x_max = faltantes.index[-1] if not faltantes.empty else historico.index[-1]
            plt.xlim(x_min, x_max)

            meses_espanol = [
                "Ene",
                "Feb",
                "Mar",
                "Abr",
                "May",
                "Jun",
                "Jul",
                "Ago",
                "Sep",
                "Oct",
                "Nov",
                "Dic",
            ]

            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
            fechas_mensuales = pd.date_range(start=x_min, end=x_max, freq="MS")
            labels_mensuales = [
                f"{meses_espanol[f.month - 1]}\n{f.year}" for f in fechas_mensuales
            ]

            if len(fechas_mensuales) > 0:
                ax.set_xticks(fechas_mensuales)
                ax.set_xticklabels(
                    labels_mensuales, rotation=45, ha="right", fontsize=9
                )

            # Titulo con informacion de variables exogenas o simulacion
            exog_info_text = ""
            if simulation_config and simulation_config.get("enabled", False):
                summary = simulation_config.get("summary", {})
                escenario_name = summary.get("escenario", "Simulado")
                exog_info_text = f" [SIMULACION: {escenario_name}]"
            elif exog_info:
                vars_names = " + ".join([v["nombre"] for v in exog_info.values()])
                exog_info_text = f" [Con: {vars_names}]"

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

            plt.xlabel("Fecha", fontsize=12, weight="bold")
            plt.ylabel("SAIDI (minutos)", fontsize=12, weight="bold")

            plt.legend(fontsize=10, loc="upper left", frameon=True, shadow=True)
            plt.grid(True, alpha=0.4, linestyle="-", linewidth=0.8)

            plt.tight_layout()

            # Nota al pie con advertencia de simulacion
            footer_y = 0.01

            if simulation_config and simulation_config.get("enabled", False):
                summary = simulation_config.get("summary", {})
                footer_text = f"SIMULACION CLIMATICA APLICADA: {summary.get('escenario', 'N/A')} | "
                footer_text += (
                    f"Alcance: {summary.get('alcance_meses', 'N/A')} meses | "
                )
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
                    bbox=dict(
                        boxstyle="round,pad=0.5",
                        facecolor="#FFEBEE",
                        alpha=0.9,
                        edgecolor="#F44336",
                        linewidth=2,
                    ),
                )
                footer_y += 0.04

            if exog_info and not simulation_config:
                footer_text = "Con variables exogenas: "
                for var_code, var_data in exog_info.items():
                    footer_text += (
                        f"{var_data['nombre']} (r={var_data['correlacion']:.3f}) "
                    )
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
                    bbox=dict(
                        boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8
                    ),
                )

            plt.savefig(
                plot_path,
                dpi=100,
                bbox_inches="tight",
                facecolor="white",
                edgecolor="none",
            )
            plt.close(fig)

            self.plot_file_path = plot_path
            return plot_path

        except Exception as e:
            print(f"Error generando grafica: {e}")
            return None

    def cleanup_plot_file(self):
        """Limpiar archivo temporal de grafica"""
        if self.plot_file_path and os.path.exists(self.plot_file_path):
            try:
                os.remove(self.plot_file_path)
            except Exception as e:
                print(f"Error eliminando archivo temporal: {e}")
            finally:
                self.plot_file_path = None
