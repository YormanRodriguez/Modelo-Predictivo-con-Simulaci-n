# services/rolling_validation_service.py
"""
Servicio de Validación Temporal con Rolling Forecast para SAIDI
CORREGIDO: Manejo robusto de variables exógenas con baja cobertura
"""
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from scipy import stats
import os
import tempfile
from datetime import datetime
from typing import Optional, Dict, Any, Tuple
import warnings
warnings.filterwarnings('ignore')


class RollingValidationService:
    """Servicio de validación temporal avanzada para modelos SAIDI"""
    
    # Mapeo de regionales a transformaciones
    REGIONAL_TRANSFORMATIONS = {
        'SAIDI_O': 'boxcox',
        'SAIDI_C': 'original',
        'SAIDI_A': 'original',
        'SAIDI_P': 'boxcox',
        'SAIDI_T': 'sqrt',
        'SAIDI_Cens': 'original'
    }
    
    # Variables exógenas por regional
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

    REGIONAL_ORDERS = {
        'SAIDI_O': {
            'order': (3, 1, 6),
            'seasonal_order': (3, 1, 0, 12)
        },
        'SAIDI_C': {
            'order': (3, 1, 2),
            'seasonal_order': (1, 1, 2, 12)
        },
        'SAIDI_A': {
            'order': (2, 1, 3),
            'seasonal_order': (2, 1, 1, 12)
        },
        'SAIDI_P': {
            'order': (4, 1, 3),
            'seasonal_order': (1, 1, 4, 12)
        },
        'SAIDI_T': {
            'order': (3, 1, 3),
            'seasonal_order': (2, 1, 2, 12)
        },
        'SAIDI_Cens': {
            'order': (4, 1, 3),
            'seasonal_order': (1, 1, 4, 12)
        }
    }
    
    def __init__(self):
        self.default_order = (4, 1, 3)
        self.default_seasonal_order = (1, 1, 4, 12)
        self.plot_file_path = None
        self.scaler = None
        self.exog_scaler = None
        self.transformation_params = {}

    def load_optimized_config(self, regional_code: str) -> Optional[Dict[str, Any]]:
        """
        Cargar configuración optimizada para una regional desde archivo JSON.
        
        Args:
            regional_code: Código de la regional (ej: 'SAIDI_O')
        
        Returns:
            Dict con configuración óptima o None si no existe
        """
        
        # Ubicación del archivo de configuración
        config_file = Path(__file__).parent.parent / "config" / "optimized_models.json"
        
        # Validar existencia del archivo
        if not config_file.exists():
            print("[ROLLING_LOAD_CONFIG] No existe archivo de configuraciones optimizadas")
            return None
        
        try:
            # Cargar configuraciones
            with config_file.open(encoding="utf-8") as f:
                configs = json.load(f)
            
            # Buscar configuración de la regional
            if regional_code not in configs:
                print(f"[ROLLING_LOAD_CONFIG] No hay configuración optimizada para {regional_code}")
                return None
            
            config = configs[regional_code]
            
            print(f"[ROLLING_LOAD_CONFIG] Configuración cargada para {regional_code}")
            print(f"[ROLLING_LOAD_CONFIG]   Transformación: {config['transformation']}")
            print(f"[ROLLING_LOAD_CONFIG]   Order: {config['order']}")
            print(f"[ROLLING_LOAD_CONFIG]   Seasonal: {config['seasonal_order']}")
            print(f"[ROLLING_LOAD_CONFIG]   Precisión: {config['precision_final']:.1f}%")
            print(f"[ROLLING_LOAD_CONFIG]   Optimizado: {config['optimization_date']}")
            
            return config
            
        except (FileNotFoundError, json.JSONDecodeError, KeyError, OSError) as e:
            error_messages = {
                FileNotFoundError: "Archivo de configuración no encontrado",
                json.JSONDecodeError: f"Archivo JSON inválido: {e}",
                KeyError: f"Clave faltante en configuración: {e}",
                OSError: f"ERROR de E/S al leer archivo: {e}",
            }
            error_msg = error_messages.get(type(e), f"Error inesperado: {e}")
            print(f"[ROLLING_LOAD_CONFIG] ERROR: {error_msg}")
            return None

    def _get_correlation_for_var(self, var_code: str, regional_code: str) -> float:
        """
        Obtener correlación documentada de una variable específica
        
        Args:
            var_code: Código de la variable (ej: 'realfeel_min')
            regional_code: Código de la regional (ej: 'SAIDI_O')
        
        Returns:
            float: Correlación documentada o 0.0 si no existe
        """
        # Correlaciones REALES documentadas por regional
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

    def _get_orders_for_regional(self, regional_code: Optional[str]) -> Tuple[Tuple, Tuple]:
        """
        Obtener órdenes SARIMAX específicos para una regional.
        
        Prioriza configuración optimizada sobre defaults hardcodeados.
        
        Args:
            regional_code: Código de la regional (ej: 'SAIDI_O')
        
        Returns:
            Tuple de (order, seasonal_order) - Órdenes ARIMA y estacionales
        """
        if not regional_code:
            return self.default_order, self.default_seasonal_order
        
        # PRIORIDAD 1: Intentar cargar configuración optimizada
        optimized_config = self.load_optimized_config(regional_code)
        
        if optimized_config:
            order = tuple(optimized_config['order'])
            seasonal_order = tuple(optimized_config['seasonal_order'])
            
            print(f"[ROLLING_ORDERS] Usando parámetros OPTIMIZADOS para {regional_code}")
            print(f"[ROLLING_ORDERS]   Order: {order}")
            print(f"[ROLLING_ORDERS]   Seasonal: {seasonal_order}")
            print(f"[ROLLING_ORDERS]   Precisión documentada: {optimized_config['precision_final']:.1f}%")
            
            return order, seasonal_order
        
        # PRIORIDAD 2: Usar configuración hardcodeada
        if regional_code in self.REGIONAL_ORDERS:
            config = self.REGIONAL_ORDERS[regional_code]
            order = config['order']
            seasonal_order = config['seasonal_order']
            
            print(f"[ROLLING_ORDERS] Usando parámetros DEFAULT para {regional_code}")
            print(f"[ROLLING_ORDERS]   Order: {order}")
            print(f"[ROLLING_ORDERS]   Seasonal: {seasonal_order}")
            
            return order, seasonal_order
        
        # FALLBACK: Usar valores por defecto genéricos
        print(f"[ROLLING_ORDERS] Usando parámetros FALLBACK para {regional_code}")
        print(f"[ROLLING_ORDERS]   Order: {self.default_order}")
        print(f"[ROLLING_ORDERS]   Seasonal: {self.default_seasonal_order}")
        
        return self.default_order, self.default_seasonal_order
    
    def run_comprehensive_validation(self,
                             file_path: Optional[str] = None,
                             df_prepared: Optional[pd.DataFrame] = None,
                             order: Optional[Tuple] = None,
                             seasonal_order: Optional[Tuple] = None,
                             regional_code: Optional[str] = None,
                             climate_data: Optional[pd.DataFrame] = None,
                             validation_months: int = 6,
                             progress_callback=None,
                             log_callback=None) -> Dict[str, Any]:
        """
        Ejecutar análisis completo de validación temporal.
        
        Args:
            file_path: Ruta del archivo Excel SAIDI
            df_prepared: DataFrame SAIDI ya preparado
            order: Orden ARIMA (p, d, q) - opcional
            seasonal_order: Orden estacional (P, D, Q, s) - opcional
            regional_code: Código de la regional
            climate_data: DataFrame con datos climáticos mensuales
            validation_months: Cantidad de meses para validación
            progress_callback: Función para actualizar progreso
            log_callback: Función para logging
        
        Returns:
            Dict con resultados de validación completa

        """
        try:
            # Cargar configuración optimizada si existe
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
            
            # Resolver parámetros del modelo
            if order is None or seasonal_order is None:
                order_regional, seasonal_regional = self._get_orders_for_regional(regional_code)
                
                if order is None:
                    order = order_regional
                if seasonal_order is None:
                    seasonal_order = seasonal_regional
                
                if log_callback and regional_code and not optimized_config:
                    regional_nombre = {
                        'SAIDI_O': 'Ocaña',
                        'SAIDI_C': 'Cúcuta',
                        'SAIDI_A': 'Aguachica',
                        'SAIDI_P': 'Pamplona',
                        'SAIDI_T': 'Tibú',
                        'SAIDI_Cens': 'CENS'
                    }.get(regional_code, regional_code)
                    
                    log_callback(f"Usando parametros default para regional {regional_nombre}")
                    log_callback(f"   Order: {order}")
                    log_callback(f"   Seasonal Order: {seasonal_order}")
            
            transformation = self._get_transformation_for_regional(regional_code)
            
            if log_callback:
                log_callback("=" * 60)
                log_callback("INICIANDO VALIDACIÓN TEMPORAL COMPLETA")
                log_callback("=" * 60)
                log_callback(f"Regional: {regional_code}")
                log_callback(f"Transformación: {transformation.upper()}")
                log_callback(f"Parámetros: order={order}, seasonal_order={seasonal_order}")
                log_callback(f"Meses de validación: {validation_months}")
            
            if progress_callback:
                progress_callback(5, "Cargando y preparando datos...")
            
            # Cargar y preparar datos
            data_original, exog_df, exog_info = self._load_and_prepare_data(
                file_path, df_prepared, regional_code, climate_data, log_callback
            )
            
            if len(data_original) < 36:
                raise Exception("Se necesitan al menos 36 observaciones para validación temporal")
            
            # Aplicar transformación
            data_transformed, transform_info = self._apply_transformation(
                data_original.values, transformation
            )
            data_transformed_series = pd.Series(data_transformed, index=data_original.index)
            
            if log_callback:
                log_callback(f"Datos preparados: {len(data_original)} observaciones")
                log_callback(f"Transformación aplicada: {transform_info}")
            
            # ROLLING FORECAST
            if progress_callback:
                progress_callback(10, "Ejecutando Rolling Forecast...")
            
            rolling_results = self.run_rolling_forecast(
                data_original, data_transformed_series, exog_df,
                order, seasonal_order, transformation,
                validation_months, progress_callback, log_callback
            )
            
            # 2 TIME SERIES CROSS-VALIDATION
            if progress_callback:
                progress_callback(40, "Ejecutando Time Series CV...")
            
            cv_results = self.run_time_series_cv(
                data_original, data_transformed_series, exog_df,
                order, seasonal_order, transformation,
                progress_callback, log_callback
            )
            
            # ANÁLISIS DE ESTABILIDAD DE PARÁMETROS
            if progress_callback:
                progress_callback(60, "Analizando estabilidad de parámetros...")
            
            param_stability = self.analyze_parameter_stability(
                data_original, data_transformed_series, exog_df,
                order, seasonal_order, transformation,
                progress_callback, log_callback
            )
            
            # BACKTESTING MULTI-HORIZONTE
            if progress_callback:
                progress_callback(80, "Ejecutando backtesting...")
            
            backtesting_results = self.run_backtesting(
                data_original, data_transformed_series, exog_df,
                order, seasonal_order, transformation,
                progress_callback, log_callback
            )

            # DIAGNÓSTICO FINAL (sin split_precision)
            final_diagnosis = self._generate_final_diagnosis(
                rolling_results, cv_results, param_stability, backtesting_results
            )
            
            if progress_callback:
                progress_callback(95, "Generando gráficas...")
            
            # Generar gráficas
            plot_path = self._generate_comprehensive_plots(
                rolling_results, cv_results, param_stability, backtesting_results,
                final_diagnosis, order, seasonal_order, transformation, exog_info
            )
            
            if progress_callback:
                progress_callback(100, "Validación completada")
            
            if log_callback:
                log_callback("\n" + "=" * 60)
                log_callback("DIAGNÓSTICO FINAL")
                log_callback("=" * 60)
                log_callback(f"Calidad del Modelo: {final_diagnosis['model_quality']}")
                log_callback(f"Nivel de Confianza: {final_diagnosis['confidence_level']:.1f}%")
                log_callback(f"Recomendación: {final_diagnosis['recommendation']}")
                log_callback(f"\nPrecisión Rolling Forecast: {rolling_results['precision']:.1f}%")
                log_callback("(Validación temporal walk-forward - Gold standard)")
                
                if final_diagnosis['limitations']:
                    log_callback("\nLimitaciones identificadas:")
                    for lim in final_diagnosis['limitations']:
                        log_callback(f"  • {lim}")
            
            return {
                'success': True,
                'validation_analysis': {
                    'rolling_forecast': rolling_results,
                    'cross_validation': cv_results,
                    'parameter_stability': param_stability,
                    'backtesting': backtesting_results,
                    'final_diagnosis': final_diagnosis
                },
                'model_params': {
                    'order': order,
                    'seasonal_order': seasonal_order,
                    'transformation': transformation,
                    'regional_code': regional_code,
                    'with_exogenous': exog_df is not None
                },
                'exogenous_vars': exog_info,
                'plot_file': plot_path
            }
            
        except Exception as e:
            if log_callback:
                log_callback(f"ERROR: {str(e)}")
            raise Exception(f"Error en validación temporal: {str(e)}")
    
    def run_rolling_forecast(self,
                            data_original: pd.Series,
                            data_transformed: pd.Series,
                            exog_df: Optional[pd.DataFrame],
                            order: Tuple,
                            seasonal_order: Tuple,
                            transformation: str,
                            validation_months: int,
                            progress_callback=None,
                            log_callback=None) -> Dict[str, Any]:
        """Implementar Rolling Forecast (Walk-Forward Validation)."""
        if log_callback:
            log_callback("\n ROLLING FORECAST - Walk-Forward Validation")
            log_callback(f"Validando últimos {validation_months} meses")
        
        n_total = len(data_original)
        n_validation = min(validation_months, int(n_total * 0.20))
        n_train_initial = n_total - n_validation
        
        if n_train_initial < 24:
            raise Exception("Insuficientes datos para training (mínimo 24 meses)")
        
        monthly_predictions = []
        monthly_actuals = []
        monthly_errors = []
        monthly_dates = []
        
        # Iterar mes a mes
        for i in range(n_validation):
            if progress_callback:
                progress = 10 + int((i / n_validation) * 30)
                progress_callback(progress, f"Rolling forecast: mes {i+1}/{n_validation}")
            
            # Ventana de entrenamiento hasta t-1
            train_end_idx = n_train_initial + i
            train_data_trans = data_transformed.iloc[:train_end_idx]
            train_data_orig = data_original.iloc[:train_end_idx]
            
            # Definir pred_date ANTES del bloque if
            pred_date = data_original.index[train_end_idx]
            
            # Preparar exógenas para training y predicción
            exog_train = None
            exog_pred = None
            
            if exog_df is not None:
                exog_train = exog_df.loc[train_data_orig.index]
                
                # Verificar si hay datos exógenos reales para esa fecha
                if pred_date in exog_df.index:
                    exog_pred = exog_df.loc[[pred_date]]
                else:
                    # NO EXTRAPOLAR - usar promedio histórico del mismo mes
                    pred_month = pred_date.month
                    historical_same_month = exog_df[exog_df.index.month == pred_month]
                    
                    if len(historical_same_month) > 0:
                        exog_pred = pd.DataFrame(
                            [historical_same_month.mean()],
                            index=[pred_date],
                            columns=exog_df.columns
                        )
                        if log_callback and i == 0:
                            log_callback("i Usando promedios históricos para variables exógenas futuras")
                    else:
                        exog_pred = None
            
            # VALIDACIÓN: Verificar que no haya NaN en los datos
            if train_data_trans.isna().any() or (exog_train is not None and exog_train.isna().any().any()):
                if log_callback:
                    log_callback(f" Iteración {i+1} contiene NaN - omitida")
                continue
            
            try:
                # Parámetros más robustos
                model = SARIMAX(
                    train_data_trans,
                    exog=exog_train,
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                
                # Método de ajuste más estable
                results = model.fit(
                    disp=False,
                    method='lbfgs',
                    maxiter=100,
                    low_memory=True
                )
                
                # Predecir SOLO 1 mes adelante
                forecast = results.get_forecast(steps=1, exog=exog_pred)
                pred_transformed = forecast.predicted_mean.iloc[0]
                
                # Revertir transformación
                pred_original = self._inverse_transformation(
                    np.array([pred_transformed]), transformation
                )[0]
                
                # Valor real en t
                actual_original = data_original.iloc[train_end_idx]
                
                # Calcular error
                error = abs(pred_original - actual_original)
                error_pct = (error / actual_original * 100) if actual_original > 0 else 0
                
                monthly_predictions.append(pred_original)
                monthly_actuals.append(actual_original)
                monthly_errors.append(error)
                monthly_dates.append(pred_date)
                
                if log_callback and (i < 3 or i >= n_validation - 2):
                    log_callback(f"  Mes {i+1}: Pred={pred_original:.2f}, Real={actual_original:.2f}, Error={error_pct:.1f}%")
                
            except np.linalg.LinAlgError:
                if log_callback:
                    log_callback(f" Iteración {i+1} falló (matriz singular) - omitida")
                continue
            except Exception as e:
                if log_callback:
                    log_callback(f" Error en iteración {i+1}: {str(e)} - omitida")
                continue
        
        # Calcular métricas acumulativas
        if len(monthly_predictions) > 0:
            rmse = np.sqrt(mean_squared_error(monthly_actuals, monthly_predictions))
            mae = mean_absolute_error(monthly_actuals, monthly_predictions)
            
            mape = np.mean([abs(a - p) / a * 100 for a, p in zip(monthly_actuals, monthly_predictions) if a > 0])
            precision = max(0, min(100, (1 - mape / 100) * 100))
            
            # CAMBIO 1: Clasificar calidad (ajustado a MAPE real de SAIDI)
            # ANTES: rmse < 2.0 and precision >= 90 = EXCELENTE
            # DESPUÉS: Criterios más realistas basados en MAPE típico de series temporales complejas
            if rmse < 3.0 and precision >= 85:
                quality = "EXCELENTE"
            elif rmse < 4.0 and precision >= 78:
                quality = "BUENA"
            elif rmse < 5.5 and precision >= 70:
                quality = "REGULAR"
            else:
                quality = "MALA"
            
            return {
                'rmse': rmse,
                'mae': mae,
                'precision': precision,
                'mape': mape,
                'monthly_errors': monthly_errors,
                'monthly_predictions': monthly_predictions,
                'monthly_actuals': monthly_actuals,
                'monthly_dates': monthly_dates,
                'prediction_quality': quality,
                'n_predictions': len(monthly_predictions)
            }
        else:
            raise Exception("No se pudieron generar predicciones rolling")
    
    def run_time_series_cv(self,
                          data_original: pd.Series,
                          data_transformed: pd.Series,
                          exog_df: Optional[pd.DataFrame],
                          order: Tuple,
                          seasonal_order: Tuple,
                          transformation: str,
                          progress_callback=None,
                          log_callback=None) -> Dict[str, Any]:
        """Time Series Cross-Validation con splits temporales"""
        if log_callback:
            log_callback("\n TIME SERIES CROSS-VALIDATION")
        
        n_total = len(data_original)
        min_train = 24
        val_size = 3
        test_size = 3
        step = 1
        
        splits_details = []
        rmse_scores = []
        precision_scores = []
        
        # Calcular n_splits correctamente
        n_splits = 0
        temp_train_end = min_train
        while temp_train_end + val_size + test_size <= n_total:
            n_splits += 1
            temp_train_end += step
        
        if log_callback:
            log_callback(f"Configuración: Min train={min_train}, Val={val_size}, Test={test_size}")
            log_callback(f"Total datos={n_total}, Step={step}")
            log_callback(f"Splits a evaluar: {n_splits}")
            log_callback(f"Rango: train_end desde {min_train} hasta {min_train + (n_splits-1)*step}")
        
        current_train_end = min_train
        split_idx = 0
        failed_splits = 0
        max_iterations = n_splits + 10
        iteration_count = 0
        
        while current_train_end + val_size + test_size <= n_total:
            split_idx += 1
            iteration_count += 1
            
            # SEGURIDAD: Detectar bucle infinito
            if iteration_count > max_iterations:
                if log_callback:
                    log_callback(f" ALERTA: Detenido por seguridad tras {iteration_count} iteraciones")
                    log_callback(f"   Se esperaban {n_splits} splits pero se detectó bucle infinito")
                break
            
            if progress_callback:
                progress = 40 + int((min(split_idx, n_splits) / n_splits) * 20)
                progress_callback(progress, f"CV Split {split_idx}/{n_splits}")
            
            # Definir splits
            train_end = current_train_end
            val_end = train_end + val_size
            test_end = val_end + test_size
            
            train_trans = data_transformed.iloc[:train_end] 
            train_orig = data_original.iloc[:train_end]
            test_orig = data_original.iloc[val_end:test_end]
            
            # Exógenas
            exog_train = None
            exog_test = None
            if exog_df is not None:
                exog_train = exog_df.loc[train_orig.index]
                if test_orig.index[-1] <= exog_df.index.max():
                    exog_test = exog_df.loc[test_orig.index]
                else:
                    exog_test = None
            
            # VALIDACIÓN: Verificar que no haya NaN en los datos
            if train_trans.isna().any() or (exog_train is not None and exog_train.isna().any().any()):
                failed_splits += 1
                if log_callback and failed_splits <= 3:
                    log_callback(f" Split {split_idx} contiene NaN - omitido")
                current_train_end += step
                continue
            
            try:
                # Parámetros más robustos
                model = SARIMAX(
                    train_trans,
                    exog=exog_train,
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                
                # Método de ajuste más estable
                results = model.fit(
                    disp=False,
                    method='lbfgs',
                    maxiter=100,
                    low_memory=True
                )
                
                # Predecir test
                forecast = results.get_forecast(steps=test_size, exog=exog_test)
                pred_trans = forecast.predicted_mean.values
                pred_orig = self._inverse_transformation(pred_trans, transformation)
                
                # Métricas
                rmse = np.sqrt(mean_squared_error(test_orig.values, pred_orig))
                mape = np.mean(abs((test_orig.values - pred_orig) / test_orig.values)) * 100
                precision = max(0, min(100, (1 - mape / 100) * 100))
                
                rmse_scores.append(rmse)
                precision_scores.append(precision)
                
                splits_details.append({
                    'split': split_idx,
                    'train_size': train_end,
                    'test_size': test_size,
                    'rmse': rmse,
                    'precision': precision
                })
                
            except np.linalg.LinAlgError:
                failed_splits += 1
                if log_callback and failed_splits <= 3:
                    log_callback(f" Split {split_idx} falló (matriz singular) - omitido")
                continue
            except Exception as e:
                failed_splits += 1
                if log_callback and failed_splits <= 3:
                    log_callback(f" Split {split_idx} falló: {str(e)[:50]} - omitido")
                continue
            
            current_train_end += step
        
        if len(rmse_scores) < 3:
            raise Exception(f"Insuficientes splits exitosos: {len(rmse_scores)}/3 mínimo")
        
        # Mostrar resumen de splits fallidos
        if failed_splits > 0 and log_callback:
            log_callback(f"i Total de splits omitidos: {failed_splits}/{n_splits} ({failed_splits/n_splits*100:.1f}%)")
        
        # Estadísticas
        mean_rmse = np.mean(rmse_scores)
        std_rmse = np.std(rmse_scores)
        mean_precision = np.mean(precision_scores)
        precision_range = [min(precision_scores), max(precision_scores)]
        
        # Score de estabilidad (0-100)
        cv_rmse = (std_rmse / mean_rmse * 100) if mean_rmse > 0 else 100
        stability_score = max(0, min(100, 100 - cv_rmse))
        
        if log_callback:
            log_callback(f"  Mean RMSE: {mean_rmse:.4f} ± {std_rmse:.4f}")
            log_callback(f"  Mean Precision: {mean_precision:.1f}%")
            log_callback(f"  Stability Score: {stability_score:.1f}/100")
        
        return {
            'mean_rmse': mean_rmse,
            'std_rmse': std_rmse,
            'mean_precision': mean_precision,
            'precision_range': precision_range,
            'cv_stability_score': stability_score,
            'n_splits': len(rmse_scores),
            'n_failed': failed_splits,
            'splits_details': splits_details
        }
    
    def analyze_parameter_stability(self,
                               data_original: pd.Series,
                               data_transformed: pd.Series,
                               exog_df: Optional[pd.DataFrame],
                               order: Tuple,
                               seasonal_order: Tuple,
                               transformation: str,
                               progress_callback=None,
                               log_callback=None) -> Dict[str, Any]:
        """Analizar estabilidad de parámetros ARIMA/SARIMA - CORREGIDO"""
        if log_callback:
            log_callback("\n⚙ ANÁLISIS DE ESTABILIDAD DE PARÁMETROS")
        
        n_total = len(data_original)
        window_sizes = []
        
        # Ventanas crecientes desde 24 meses
        current = 24
        while current <= n_total:
            window_sizes.append(current)
            current += 6
        
        if window_sizes[-1] < n_total:
            window_sizes.append(n_total)
        
        if log_callback:
            log_callback(f"Ventanas a evaluar: {window_sizes}")
        
        params_evolution = {}
        window_sizes_success = []  # ← NUEVO: Solo ventanas exitosas
        failed_windows = 0
        
        for idx, window in enumerate(window_sizes):
            if progress_callback:
                progress = 60 + int((idx / len(window_sizes)) * 20)
                progress_callback(progress, f"Analizando ventana {window} meses")
            
            train_trans = data_transformed.iloc[:window]
            train_orig = data_original.iloc[:window]
            
            exog_train = None
            if exog_df is not None:
                exog_train = exog_df.loc[train_orig.index]
            
            # VALIDACIÓN: Verificar que no haya NaN
            if train_trans.isna().any() or (exog_train is not None and exog_train.isna().any().any()):
                failed_windows += 1
                if log_callback and failed_windows <= 2:
                    log_callback(f"⚠ Ventana {window} contiene NaN - omitida")
                continue
            
            try:
                model = SARIMAX(
                    train_trans,
                    exog=exog_train,
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                
                results = model.fit(
                    disp=False,
                    method='lbfgs',
                    maxiter=100,
                    low_memory=True
                )
                
                # Registrar ventana exitosa
                window_sizes_success.append(window)
                
                # Extraer parámetros
                params = results.params
                
                for param_name, param_value in params.items():
                    if param_name not in params_evolution:
                        params_evolution[param_name] = []
                    params_evolution[param_name].append(param_value)
                
            except (np.linalg.LinAlgError, Exception) as e:
                failed_windows += 1
                if log_callback and failed_windows <= 2:
                    log_callback(f"⚠ Ventana {window} falló: {str(e)[:50]} - omitida")
                continue
        
        if not params_evolution or len(window_sizes_success) < 2:
            raise Exception(f"Insuficientes ventanas exitosas: {len(window_sizes_success)}/2 mínimo")
        
        # ===== ANÁLISIS CORREGIDO CON FUNCIÓN LOGARÍTMICA SUAVIZADA =====
        param_analysis = {}
        unstable_params = []
        
        for param_name, values in params_evolution.items():
            if len(values) > 1:
                mean_val = np.mean(values)
                std_val = np.std(values)
                value_range = max(values) - min(values)
                
                # 
                if abs(mean_val) > 0.01:
                    cv = abs(std_val / mean_val)
                    # Usar log(1 + cv) para suavizar valores altos
                    # cv=0.5 → 85, cv=1.0 → 70, cv=2.0 → 50, cv=5.0 → 30
                    stability_score = max(0, min(100, 100 - 15 * np.log1p(cv)))
                else:
                    # Método basado en rango absoluto
                    if value_range < 0.2:
                        stability_score = 95
                    elif value_range < 0.5:
                        stability_score = 80
                    elif value_range < 1.0:
                        stability_score = 65
                    elif value_range < 2.0:
                        stability_score = 50
                    else:
                        stability_score = max(0, 50 - (value_range - 2.0) * 10)
                
                param_analysis[param_name] = {
                    'mean': mean_val,
                    'std': std_val,
                    'min': min(values),
                    'max': max(values),
                    'range': value_range,
                    'cv': abs(std_val / mean_val) if abs(mean_val) > 0.01 else None,
                    'stability_score': stability_score,
                    'values': values
                }
                
                # Criterio de inestabilidad MÁS REALISTA
                is_unstable = False
                
                if abs(mean_val) > 0.01:
                    cv = abs(std_val / mean_val)
                    # Inestable si CV > 1.5 Y score < 65
                    if cv > 1.5 and stability_score < 65:
                        is_unstable = True
                else:
                    # Inestable si rango > 1.5 Y score < 65
                    if value_range > 1.5 and stability_score < 65:
                        is_unstable = True
                
                if is_unstable:
                    unstable_params.append(param_name)
        
        # Score global: promedio ponderado por tipo de parámetro
        ar_scores = [p['stability_score'] for name, p in param_analysis.items() if 'ar.' in name.lower()]
        ma_scores = [p['stability_score'] for name, p in param_analysis.items() if 'ma.' in name.lower()]
        exog_scores = [p['stability_score'] for name, p in param_analysis.items() 
                       if 'ar.' not in name.lower() and 'ma.' not in name.lower() and 'sigma' not in name.lower()]
        
        # Ponderación: AR/MA más importantes (70%) que exógenas (30%)
        overall_stability = 0
        weight_sum = 0
        
        if ar_scores:
            overall_stability += np.mean(ar_scores) * 0.35
            weight_sum += 0.35
        
        if ma_scores:
            overall_stability += np.mean(ma_scores) * 0.35
            weight_sum += 0.35
        
        if exog_scores:
            overall_stability += np.mean(exog_scores) * 0.30
            weight_sum += 0.30
        
        if weight_sum > 0:
            overall_stability = overall_stability / weight_sum
        else:
            overall_stability = 0
        
        # Interpretación REALISTA
        if overall_stability >= 75:
            interpretation = "Los parámetros muestran ALTA estabilidad - Modelo robusto"
        elif overall_stability >= 65:
            interpretation = "Los parámetros muestran BUENA estabilidad - Modelo confiable"
        elif overall_stability >= 55:
            interpretation = "Los parámetros muestran estabilidad MODERADA - Aceptable para producción"
        else:
            interpretation = "Los parámetros muestran BAJA estabilidad - Revisar especificación"
        
        if log_callback:
            log_callback(f"  Overall Stability Score: {overall_stability:.1f}/100")
            log_callback(f"  {interpretation}")
            
            # Desglose por tipo
            if ar_scores:
                log_callback(f"  AR params: {np.mean(ar_scores):.1f}/100 (n={len(ar_scores)})")
            if ma_scores:
                log_callback(f"  MA params: {np.mean(ma_scores):.1f}/100 (n={len(ma_scores)})")
            if exog_scores:
                log_callback(f"  Exog params: {np.mean(exog_scores):.1f}/100 (n={len(exog_scores)})")
            
            if unstable_params:
                log_callback(f"  Parámetros críticos inestables ({len(unstable_params)}): {', '.join(unstable_params[:5])}")
            if failed_windows > 0:
                pct_failed = failed_windows / len(window_sizes) * 100
                log_callback(f"  Ventanas omitidas: {failed_windows}/{len(window_sizes)} ({pct_failed:.1f}%)")
        
        return {
            'overall_stability_score': overall_stability,
            'unstable_params': unstable_params,
            'parameter_details': param_analysis,
            'interpretation': interpretation,
            'window_sizes': window_sizes_success,
            'n_failed_windows': failed_windows,
            'param_group_scores': {
                'ar': np.mean(ar_scores) if ar_scores else None,
                'ma': np.mean(ma_scores) if ma_scores else None,
                'exog': np.mean(exog_scores) if exog_scores else None
            }
        }
    
    def run_backtesting(self,
                       data_original: pd.Series,
                       data_transformed: pd.Series,
                       exog_df: Optional[pd.DataFrame],
                       order: Tuple,
                       seasonal_order: Tuple,
                       transformation: str,
                       progress_callback=None,
                       log_callback=None) -> Dict[str, Any]:
        """Backtesting multi-horizonte"""
        if log_callback:
            log_callback("\n BACKTESTING MULTI-HORIZONTE")
        
        horizons = [1, 3, 6, 12]
        n_total = len(data_original)
        
        # Puntos de inicio cada 6 meses desde mes 24
        backtest_points = []
        start_point = 24
        while start_point + max(horizons) < n_total:
            backtest_points.append(start_point)
            start_point += 6
        
        if not backtest_points:
            backtest_points = [max(24, n_total - 12)]
        
        if log_callback:
            log_callback(f"Horizontes: {horizons}")
            log_callback(f"Puntos de evaluación: {len(backtest_points)}")
        
        metrics_by_horizon = {}
        
        for horizon in horizons:
            rmse_list = []
            precision_list = []
            mape_list = []
            failed_count = 0
            
            for idx, start_idx in enumerate(backtest_points):
                if start_idx + horizon >= n_total:
                    continue
                
                if progress_callback:
                    progress = 80 + int((idx / len(backtest_points)) * 15)
                    progress_callback(progress, f"Backtesting h={horizon}, punto {idx+1}/{len(backtest_points)}")
                
                train_trans = data_transformed.iloc[:start_idx]
                train_orig = data_original.iloc[:start_idx]
                test_orig = data_original.iloc[start_idx:start_idx + horizon]
                
                exog_train = None
                exog_test = None
                if exog_df is not None:
                    exog_train = exog_df.loc[train_orig.index]
                    if test_orig.index[-1] <= exog_df.index.max():
                        exog_test = exog_df.loc[test_orig.index]
                
                # VALIDACIÓN: Verificar que no haya NaN en los datos
                if train_trans.isna().any() or (exog_train is not None and exog_train.isna().any().any()):
                    failed_count += 1
                    continue
                
                try:
                    # Parámetros más robustos
                    model = SARIMAX(
                        train_trans,
                        exog=exog_train,
                        order=order,
                        seasonal_order=seasonal_order,
                        enforce_stationarity=False,
                        enforce_invertibility=False
                    )
                    
                    # Método de ajuste más estable
                    results = model.fit(
                        disp=False,
                        method='lbfgs',
                        maxiter=100,
                        low_memory=True
                    )
                    
                    forecast = results.get_forecast(steps=horizon, exog=exog_test)
                    pred_trans = forecast.predicted_mean.values
                    pred_orig = self._inverse_transformation(pred_trans, transformation)
                    
                    rmse = np.sqrt(mean_squared_error(test_orig.values, pred_orig))
                    mape = np.mean(abs((test_orig.values - pred_orig) / test_orig.values)) * 100
                    precision = max(0, min(100, (1 - mape / 100) * 100))
                    
                    rmse_list.append(rmse)
                    precision_list.append(precision)
                    mape_list.append(mape)
                    
                except (np.linalg.LinAlgError, Exception):
                    failed_count += 1
                    continue
            
            if rmse_list:
                metrics_by_horizon[horizon] = {
                    'rmse': np.mean(rmse_list),
                    'precision': np.mean(precision_list),
                    'mape': np.mean(mape_list),
                    'n_tests': len(rmse_list),
                    'n_failed': failed_count
                }
        
        # Calcular degradación
        if 1 in metrics_by_horizon and len(metrics_by_horizon) > 1:
            baseline_precision = metrics_by_horizon[1]['precision']
            max_horizon = max(metrics_by_horizon.keys())
            final_precision = metrics_by_horizon[max_horizon]['precision']
            
            degradation_rate = (final_precision - baseline_precision) / (max_horizon - 1)
        else:
            degradation_rate = 0
        
        # Horizonte óptimo (mejor balance precision/utility)
        optimal_horizon = 1
        best_score = 0
        for h, m in metrics_by_horizon.items():
            # Score = precision * factor de utilidad
            utility_factor = min(1.0, h / 6.0)
            score = m['precision'] * (0.7 + 0.3 * utility_factor)
            if score > best_score and m['precision'] >= 75:
                best_score = score
                optimal_horizon = h
        
        if log_callback:
            log_callback(f"  Horizonte óptimo: {optimal_horizon} meses")
            log_callback(f"  Degradación: {degradation_rate:.2f}% por mes")
            for h in sorted(metrics_by_horizon.keys()):
                m = metrics_by_horizon[h]
                log_callback(f"    H={h:2d}: Precisión={m['precision']:.1f}%, RMSE={m['rmse']:.3f}")
        
        return {
            'horizons': list(metrics_by_horizon.keys()),
            'metrics_by_horizon': metrics_by_horizon,
            'degradation_rate': degradation_rate,
            'optimal_horizon': optimal_horizon,
            'backtest_points': len(backtest_points)
        }

    def _generate_final_diagnosis(self,
                              rolling_results: Dict,
                              cv_results: Dict,
                              param_stability: Dict,
                              backtesting_results: Dict) -> Dict[str, Any]:
        """
        Generar diagnóstico final integrado
        
        Integra los resultados de las 4 metodologías principales:
        - Rolling Forecast (gold standard)
        - Cross-Validation
        - Parameter Stability
        - Backtesting Multi-Horizonte
        """
        # Criterios de calidad
        rolling_rmse = rolling_results['rmse']
        rolling_precision = rolling_results['precision']
        cv_stability = cv_results['cv_stability_score']
        param_stability_score = param_stability['overall_stability_score']
        degradation_rate = backtesting_results['degradation_rate']
        
        # Puntajes individuales (0-100)
        score_rolling = 100 if rolling_rmse < 2.0 else (100 - min(50, (rolling_rmse - 2.0) * 20))
        score_precision = rolling_precision
        score_cv = cv_stability
        score_params = param_stability_score
        score_degradation = max(0, 100 + degradation_rate * 20)
        
        # Puntaje global ponderado
        confidence_level = (
            score_rolling * 0.25 +
            score_precision * 0.25 +
            score_cv * 0.20 +
            score_params * 0.20 +
            score_degradation * 0.10
        )
        
        # CAMBIO 2: Clasificación de calidad (ajustado a realidad de series temporales complejas)
        # ANTES: confidence >= 85 and rmse < 3.0 and cv >= 85 and params >= 80 = EXCELENTE
        # DESPUÉS: Umbrales más realistas basados en MAPE típico de SAIDI (15-25%)
        if confidence_level >= 80 and rolling_rmse < 4.0 and cv_stability >= 80 and param_stability_score >= 75:
            model_quality = "EXCELENTE"
        elif confidence_level >= 72 and rolling_rmse < 5.0 and cv_stability >= 70 and param_stability_score >= 65:
            model_quality = "CONFIABLE"
        elif confidence_level >= 65 and rolling_rmse < 6.5:
            model_quality = "CUESTIONABLE"
        else:
            model_quality = "NO CONFIABLE"
        
        # Recomendación
        optimal_horizon = backtesting_results['optimal_horizon']
        
        if model_quality == "EXCELENTE":
            recommendation = f"Usar para pronósticos hasta {optimal_horizon} meses con alta confianza"
        elif model_quality == "CONFIABLE":
            recommendation = f"Usar para pronósticos hasta {optimal_horizon} meses con precaución moderada"
        elif model_quality == "CUESTIONABLE":
            recommendation = "Usar solo para pronósticos de corto plazo (1-2 meses) con monitoreo continuo"
        else:
            recommendation = "No recomendado para uso productivo - Revisar especificación del modelo"
        
        # CAMBIO 3: Limitaciones (umbrales más realistas)
        # ANTES: RMSE > 3.0, CV < 75, Params < 70, Degradation > 3.0, Precision < 80
        # DESPUÉS: Umbrales ajustados para evitar falsos positivos
        limitations = []
        
        if rolling_rmse > 5.0:
            limitations.append(f"RMSE elevado en rolling forecast ({rolling_rmse:.2f} min)")
        
        if cv_stability < 65:
            limitations.append(f"Estabilidad limitada en CV (Score: {cv_stability:.1f})")
        
        if param_stability_score < 60:
            limitations.append(f"Parámetros inestables (Score: {param_stability_score:.1f})")
        
        if abs(degradation_rate) > 4.0:
            limitations.append(f"Degradación notable después de {optimal_horizon} meses ({degradation_rate:.1f}% por mes)")
        
        if len(param_stability.get('unstable_params', [])) > 2:
            limitations.append(f"{len(param_stability['unstable_params'])} coeficientes inestables detectados")
        
        if rolling_precision < 70:
            limitations.append(f"Precisión promedio limitada ({rolling_precision:.1f}%)")
        
        return {
            'model_quality': model_quality,
            'confidence_level': confidence_level,
            'recommendation': recommendation,
            'limitations': limitations,
            'component_scores': {
                'rolling_forecast': score_rolling,
                'precision': score_precision,
                'cv_stability': score_cv,
                'parameter_stability': score_params,
                'degradation': score_degradation
            }
        }
    
    def _get_transformation_for_regional(self, regional_code: Optional[str]) -> str:
        """
        Obtener transformación para la regional.

        Prioriza configuración optimizada sobre defaults hardcodeados.

        Args:
            regional_code: Código de la regional (ej: 'SAIDI_O')

        Returns:
            str: Tipo de transformación ('original', 'boxcox', 'sqrt', etc.)

        """
        if not regional_code:
            return 'original'
        
        # PRIORIDAD 1: Intentar cargar configuración optimizada
        optimized_config = self.load_optimized_config(regional_code)
        
        if optimized_config:
            transformation = optimized_config.get('transformation', 'original')
            print(f"[ROLLING_TRANSFORMATION] Usando transformación OPTIMIZADA: {transformation}")
            return transformation
        
        # PRIORIDAD 2: Usar defaults hardcodeados
        if regional_code in self.REGIONAL_TRANSFORMATIONS:
            transformation = self.REGIONAL_TRANSFORMATIONS[regional_code]
            print(f"[ROLLING_TRANSFORMATION] Usando transformación DEFAULT: {transformation}")
            return transformation
        
        # FALLBACK: Original
        print("[ROLLING_TRANSFORMATION] Usando transformación FALLBACK: original")
        return 'original'

    def _load_and_prepare_data(self,
                            file_path: Optional[str],
                            df_prepared: Optional[pd.DataFrame],
                            regional_code: Optional[str],
                            climate_data: Optional[pd.DataFrame],
                            log_callback) -> Tuple[pd.Series, Optional[pd.DataFrame], Optional[Dict]]:
        """Cargar y preparar datos SAIDI + variables exógenas"""
        # Cargar datos SAIDI
        if df_prepared is not None:
            df = df_prepared.copy()
        elif file_path is not None:
            df = pd.read_excel(file_path, sheet_name="Hoja1")
        else:
            raise Exception("Debe proporcionar file_path o df_prepared")
        
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
        elif "SAIDI Histórico" in df.columns:
            col_saidi = "SAIDI Histórico"
        
        if col_saidi is None:
            raise Exception("No se encontró la columna SAIDI")
        
        data_original = df[df[col_saidi].notna()][col_saidi]
        
        # Preparar variables exógenas
        exog_df = None
        exog_info = None
        
        if climate_data is not None and not climate_data.empty:
            exog_df, exog_info = self._prepare_exogenous_variables(
                climate_data, df, regional_code, log_callback
            )
            
            if exog_df is not None:
                if not self._diagnose_exog_coverage(data_original, exog_df, log_callback):
                    if log_callback:
                        log_callback("=" * 60)
                        log_callback(" ADVERTENCIA: Cobertura insuficiente")
                        log_callback("Las variables exógenas serán DESACTIVADAS")
                        log_callback("=" * 60)
                    exog_df = None
                    exog_info = None
                else:
                    # Guardar scaler solo para compatibilidad (NO transformar)
                    self.exog_scaler = StandardScaler()
                    self.exog_scaler.fit(exog_df)
                    
                    if log_callback:
                        log_callback("Variables exógenas preparadas en ESCALA ORIGINAL")
                        log_callback("SARIMAX las normalizará internamente")
                        log_callback("(Escalado manual eliminado para evitar doble normalización)")
        
        return data_original, exog_df, exog_info

    def _diagnose_exog_coverage(self, 
                    serie_saidi: pd.Series, 
                    exog_df: pd.DataFrame,
                    log_callback) -> bool:
        """
        Diagnosticar cobertura temporal de variables exógenas
        COPIADO DE PredictionService/OptimizationService para consistencia
        
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
                log_callback("DIAGNÓSTICO DE COBERTURA EXÓGENA")
                log_callback("=" * 60)
                log_callback(f"SAIDI: {saidi_start.strftime('%Y-%m')} a {saidi_end.strftime('%Y-%m')} ({len(serie_saidi)} obs)")
                log_callback(f"EXOG:  {exog_start.strftime('%Y-%m')} a {exog_end.strftime('%Y-%m')} ({len(exog_df)} obs)")
            
            # 1 Verificar que los índices coinciden EXACTAMENTE
            if not exog_df.index.equals(serie_saidi.index):
                if log_callback:
                    log_callback("ADVERTENCIA: Índices no coinciden exactamente")
                
                # Verificar fechas faltantes
                missing_in_exog = [d for d in serie_saidi.index if d not in exog_df.index]
                
                if missing_in_exog:
                    pct_missing = len(missing_in_exog) / len(serie_saidi) * 100
                    
                    if log_callback:
                        log_callback(f"Fechas SAIDI faltantes en EXOG: {len(missing_in_exog)} ({pct_missing:.1f}%)")
                    
                    # CRÍTICO: Si falta >20% de fechas, rechazar
                    if pct_missing > 20:
                        if log_callback:
                            log_callback("ERROR CRÍTICO: >20% de fechas faltantes")
                            log_callback("Las variables exógenas NO cubren suficiente período histórico")
                        return False
            
            # 2 Verificar que NO hay NaN en ninguna columna
            if exog_df.isnull().any().any():
                nan_cols = exog_df.columns[exog_df.isnull().any()].tolist()
                
                if log_callback:
                    log_callback("ERROR: Columnas con NaN encontradas:")
                    for col in nan_cols:
                        nan_count = exog_df[col].isnull().sum()
                        pct_nan = (nan_count / len(exog_df)) * 100
                        log_callback(f"  - {col}: {nan_count} NaN ({pct_nan:.1f}%)")
                    log_callback("Variables exógenas deben estar completamente rellenas")
                
                return False
            
            # 3 Verificar valores infinitos
            if np.isinf(exog_df.values).any():
                if log_callback:
                    log_callback("ERROR: Variables exógenas contienen valores infinitos")
                return False
            
            # 4 Verificar que hay varianza en las variables
            zero_variance_vars = []
            for col in exog_df.columns:
                if exog_df[col].std() == 0:
                    zero_variance_vars.append(col)
            
            if zero_variance_vars:
                if log_callback:
                    log_callback("ADVERTENCIA: Variables con varianza cero:")
                    for var in zero_variance_vars:
                        log_callback(f"  - {var}")
                    log_callback("Estas variables no aportan información al modelo")
            
            if log_callback:
                log_callback("Cobertura temporal y calidad de datos OK")
                log_callback("=" * 60)
            
            return True
            
        except Exception as e:
            if log_callback:
                log_callback(f"ERROR durante diagnóstico: {e}")
            return False
    
    def _prepare_exogenous_variables(self,
                                    climate_data: pd.DataFrame,
                                    df_saidi: pd.DataFrame,
                                    regional_code: Optional[str],
                                    log_callback) -> Tuple[Optional[pd.DataFrame], Optional[Dict]]:
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
                log_callback(f" Preparando variables para {regional_code}")
                log_callback("   MODO: SIN ESCALADO (valores originales)")
            
            # Validar índice datetime
            if not isinstance(climate_data.index, pd.DatetimeIndex):
                fecha_col = None
                for col in ['fecha', 'Fecha', 'date', 'Date', 'month_date']:
                    if col in climate_data.columns:
                        fecha_col = col
                        break
                
                if fecha_col is None:
                    if log_callback:
                        log_callback(" ERROR: No se encontró columna de fecha válida")
                    return None, None
                
                try:
                    climate_data = climate_data.copy()
                    climate_data[fecha_col] = pd.to_datetime(climate_data[fecha_col])
                    climate_data = climate_data.set_index(fecha_col)
                except Exception as e:
                    if log_callback:
                        log_callback(f" ERROR convirtiendo índice: {str(e)}")
                    return None, None
            
            # Verificar que ahora es DatetimeIndex
            if not isinstance(climate_data.index, pd.DatetimeIndex):
                if log_callback:
                    log_callback("ERROR: Formato de fecha inválido")
                return None, None
            
            # Análisis de cobertura temporal
            historico = df_saidi[df_saidi['SAIDI'].notna() if 'SAIDI' in df_saidi.columns else df_saidi['SAIDI Histórico'].notna()]
            
            saidi_start = historico.index[0]
            saidi_end = historico.index[-1]
            clima_start = climate_data.index[0]
            clima_end = climate_data.index[-1]
            
            # Calcular periodo de overlap
            overlap_start = max(saidi_start, clima_start)
            overlap_end = min(saidi_end, clima_end)
            
            if overlap_start > overlap_end:
                if log_callback:
                    log_callback(" ERROR: Sin overlap entre SAIDI y CLIMA")
                return None, None
            
            overlap_mask = (historico.index >= overlap_start) & (historico.index <= overlap_end)
            overlap_months = overlap_mask.sum()
            
            # Validar overlap mínimo (12 meses)
            if overlap_months < 12:
                if log_callback:
                    log_callback(f" ERROR: Overlap insuficiente ({overlap_months} < 12 meses)")
                return None, None
            
            if log_callback:
                log_callback(f"   SAIDI: {saidi_start.strftime('%Y-%m')} a {saidi_end.strftime('%Y-%m')} ({len(historico)} meses)")
                log_callback(f"   CLIMA: {clima_start.strftime('%Y-%m')} a {clima_end.strftime('%Y-%m')} ({len(climate_data)} meses)")
                log_callback(f"   OVERLAP: {overlap_start.strftime('%Y-%m')} a {overlap_end.strftime('%Y-%m')} ({overlap_months} meses)")
            
            # Mapeo automático de columnas
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
                    log_callback(" ERROR: No se pudo mapear ninguna variable")
                return None, None
            
            # Preparación de variables SIN ESCALADO
            exog_df = pd.DataFrame(index=historico.index)
            exog_info = {}
            rejected_vars = []  
            
            for var_code, var_nombre in exog_vars_config.items():
                climate_col = climate_column_mapping.get(var_code)
                
                if not climate_col or climate_col not in climate_data.columns:
                    rejected_vars.append((var_code, "No encontrada en datos climáticos"))
                    continue
                
                try:
                    var_series = climate_data[climate_col].copy()
                    aligned_series = pd.Series(index=historico.index, dtype=float)
                    
                    # Llenar datos donde hay overlap REAL
                    for date in historico.index:
                        if date in var_series.index:
                            aligned_series[date] = var_series.loc[date]
                    
                    # VALIDACIÓN 1: Cobertura en overlap
                    overlap_data = aligned_series[overlap_mask]
                    datos_reales_overlap = overlap_data.notna().sum()
                    overlap_pct = (datos_reales_overlap / overlap_months) * 100
                    
                    # CRÍTICO: Rechazar si cobertura < 80%
                    if overlap_pct < 80:
                        rejected_vars.append((var_code, f"Cobertura {overlap_pct:.1f}% < 80%"))
                        if log_callback:
                            log_callback(f"RECHAZADA {var_code}: cobertura {overlap_pct:.1f}% < 80%")
                        continue
                    
                    # VALIDACIÓN 2: Varianza en overlap
                    var_std = overlap_data.std()
                    if pd.isna(var_std) or var_std == 0:
                        rejected_vars.append((var_code, "Varianza = 0"))
                        if log_callback:
                            log_callback(f" RECHAZADA {var_code}: varianza = 0")
                        continue
                    
                    # VALIDACIÓN 3: Cobertura en TODO el período histórico
                    coverage_total = aligned_series.notna().sum() / len(aligned_series) * 100
                    if coverage_total < 60:  # Al menos 60% de cobertura total
                        rejected_vars.append((var_code, f"Cobertura total {coverage_total:.1f}% < 60%"))
                        if log_callback:
                            log_callback(f" RECHAZADA {var_code}: cobertura total {coverage_total:.1f}% < 60%")
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
                        rejected_vars.append((var_code, f"{final_nan} NaN finales"))
                        if log_callback:
                            log_callback(f" RECHAZADA {var_code}: {final_nan} NaN finales")
                        continue
                    
                    # GUARDAR EN ESCALA ORIGINAL
                    exog_df[var_code] = aligned_series
                    
                    exog_info[var_code] = {
                        'nombre': var_nombre,
                        'columna_clima': climate_col,
                        'correlacion': self._get_correlation_for_var(var_code, regional_code),
                        'scaled': False,
                        'datos_reales_overlap': int(datos_reales_overlap),
                        'overlap_coverage_pct': float(overlap_pct),
                        'total_coverage_pct': float(coverage_total),  # ← NUEVO
                        'varianza_overlap': float(var_std)
                    }
                    
                    if log_callback:
                        log_callback(f" ACEPTADA {var_code} (overlap={overlap_pct:.1f}%, total={coverage_total:.1f}%, r={exog_info[var_code]['correlacion']:.3f})")
                        
                except Exception as e:
                    rejected_vars.append((var_code, f"Error: {str(e)[:50]}"))
                    if log_callback:
                        log_callback(f" ERROR {var_code}: {e}")
                    continue
            
            # VALIDACIÓN FINAL
            if exog_df.empty or exog_df.shape[1] == 0:
                if log_callback:
                    log_callback("=" * 60)
                    log_callback("ERROR: Ninguna variable aceptada")
                    log_callback(f"Variables rechazadas ({len(rejected_vars)}):")
                    for var, reason in rejected_vars:
                        log_callback(f"  - {var}: {reason}")
                    log_callback("=" * 60)
                return None, None
            
            if log_callback:
                log_callback("=" * 60)
                log_callback(f"Variables preparadas: {len(exog_df.columns)}/{len(exog_vars_config)}")
                log_callback("   ESCALA: ORIGINAL (sin StandardScaler)")
                if rejected_vars:
                    log_callback(f"   Variables rechazadas: {len(rejected_vars)}")
                    for var, reason in rejected_vars[:3]:  # Mostrar solo primeras 3
                        log_callback(f"     - {var}: {reason}")
                log_callback("=" * 60)
            
            return exog_df, exog_info if exog_info else None
            
        except Exception as e:
            if log_callback:
                log_callback(f"ERROR CRÍTICO: {str(e)}")
            return None, None
    
    def _align_exog_to_saidi(self,
                            exog_series: pd.DataFrame,
                            df_saidi: pd.DataFrame,
                            var_code: str,
                            log_callback) -> Optional[pd.Series]:
        """
        Alinear datos exógenos al índice de SAIDI
        
        Estrategia (IGUAL que prediction_service):
        - Para fechas con datos climáticos: usar el valor directo
        - Para fechas FUTURAS (después del último dato): usar último valor conocido
        - Para fechas PASADAS (antes del primer dato): usar primer valor conocido
        - Para fechas intermedias sin dato: interpolar linealmente
        """
        try:
            # Obtener las fechas del clima
            climate_dates = exog_series.index
            saidi_dates = df_saidi.index
            
            # Asegurar que ambos índices sean DatetimeIndex
            if not isinstance(climate_dates, pd.DatetimeIndex):
                climate_dates = pd.to_datetime(climate_dates)
            if not isinstance(saidi_dates, pd.DatetimeIndex):
                saidi_dates = pd.to_datetime(saidi_dates)
            
            # Crear serie resultado
            result = pd.Series(index=saidi_dates, dtype=float)
            
            # Obtener límites de datos climáticos
            min_climate_date = climate_dates.min()
            max_climate_date = climate_dates.max()
            first_known_value = exog_series.iloc[0].iloc[0]
            last_known_value = exog_series.iloc[-1].iloc[0]
            
            n_direct = 0
            n_past = 0
            n_future = 0
            n_interpolated = 0
            
            # Llenar valores según estrategia
            for date in saidi_dates:
                if date in climate_dates:
                    #  Caso 1: Dato climático directo
                    result[date] = exog_series.loc[date].iloc[0]
                    n_direct += 1
                    
                elif date < min_climate_date:
                    #  Caso 2: Fecha ANTES del primer dato → usar primer valor
                    result[date] = first_known_value
                    n_past += 1
                    
                elif date > max_climate_date:
                    #  Caso 3: Fecha DESPUÉS del último dato → usar último valor
                    result[date] = last_known_value
                    n_future += 1
                    
                else:
                    #  Caso 4: Fecha intermedia sin dato → marcar para interpolación
                    result[date] = np.nan
                    n_interpolated += 1
            
            # Interpolar valores intermedios (si los hay)
            if n_interpolated > 0:
                result = result.interpolate(method='linear', limit_direction='both')
            
            # Logging detallado
            if log_callback:
                log_callback(f"  - {var_code}: Alineación completada")
                log_callback(f"      ✓ {n_direct} valores directos del clima")
                if n_past > 0:
                    log_callback(f"      ← {n_past} valores pasados (usando primer valor: {first_known_value:.2f})")
                if n_future > 0:
                    log_callback(f"      → {n_future} valores futuros (usando último valor: {last_known_value:.2f})")
                if n_interpolated > 0:
                    log_callback(f"      ≈ {n_interpolated} valores interpolados linealmente")
            
            # Verificación final: NO debe haber NaN
            if result.isna().any():
                n_nan = result.isna().sum()
                if log_callback:
                    log_callback(f"       ADVERTENCIA: {n_nan} NaN detectados después de alineación")
                # Último recurso: rellenar con media
                result.fillna(result.mean(), inplace=True)
            
            return result
            
        except Exception as e:
            if log_callback:
                log_callback(f"   Error alineando variable {var_code}: {str(e)}")
                import traceback
                log_callback(traceback.format_exc())
            return None
    
    def _apply_transformation(self, data: np.ndarray, transformation_type: str) -> Tuple[np.ndarray, str]:
        """Aplicar transformación a los datos"""
        if transformation_type == 'original':
            return data, "Sin transformación"
        
        elif transformation_type == 'standard':
            self.scaler = StandardScaler()
            transformed = self.scaler.fit_transform(data.reshape(-1, 1)).flatten()
            return transformed, "StandardScaler"
        
        elif transformation_type == 'log':
            data_positive = np.maximum(data, 1e-10)
            transformed = np.log(data_positive)
            self.transformation_params['log_applied'] = True
            return transformed, "Log"
        
        elif transformation_type == 'boxcox':
            data_positive = np.maximum(data, 1e-10)
            transformed, lambda_param = stats.boxcox(data_positive)
            self.transformation_params['boxcox_lambda'] = lambda_param
            return transformed, f"Box-Cox (λ={lambda_param:.4f})"
        
        elif transformation_type == 'sqrt':
            data_positive = np.maximum(data, 0)
            transformed = np.sqrt(data_positive)
            self.transformation_params['sqrt_applied'] = True
            return transformed, "Sqrt"
        
        else:
            return data, "Sin transformación"
    
    def _inverse_transformation(self, data: np.ndarray, transformation_type: str) -> np.ndarray:
        """Revertir transformación"""
        if transformation_type == 'original':
            return data
        
        elif transformation_type == 'standard':
            if self.scaler is not None:
                return self.scaler.inverse_transform(data.reshape(-1, 1)).flatten()
            return data
        
        elif transformation_type == 'log':
            return np.exp(data)
        
        elif transformation_type == 'boxcox':
            lambda_param = self.transformation_params.get('boxcox_lambda', 0)
            if lambda_param == 0:
                return np.exp(data)
            else:
                return np.power(data * lambda_param + 1, 1 / lambda_param)
        
        elif transformation_type == 'sqrt':
            return np.power(data, 2)
        
        else:
            return data
    
    def _generate_comprehensive_plots(self,
                                 rolling_results: Dict,
                                 cv_results: Dict,
                                 param_stability: Dict,
                                 backtesting_results: Dict,
                                 final_diagnosis: Dict,
                                 order: Tuple,
                                 seasonal_order: Tuple,
                                 transformation: str,
                                 exog_info: Optional[Dict]) -> Optional[str]:
        """Generar panel de gráficas comprehensivo"""
        try:
            temp_dir = tempfile.gettempdir()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_path = os.path.join(temp_dir, f"saidi_rolling_validation_{timestamp}.png")
            
            plt.style.use('default')
            fig = plt.figure(figsize=(20, 14), dpi=100)
            
            # Panel 1: Rolling Forecast Performance
            ax1 = plt.subplot(2, 3, 1)
            
            dates = rolling_results['monthly_dates']
            actuals = rolling_results['monthly_actuals']
            predictions = rolling_results['monthly_predictions']
            errors = rolling_results['monthly_errors']
            
            # Convertir fechas a formato compatible con matplotlib
            if dates and len(dates) > 0:
                dates_converted = [pd.to_datetime(d).to_pydatetime() if isinstance(d, (pd.Timestamp, str)) else d for d in dates]
            else:
                raise ValueError("No hay fechas disponibles para graficar en Rolling Forecast")
            
            ax1.plot(dates_converted, actuals, 'o-', label='Real', color='blue', linewidth=2, markersize=6)
            ax1.plot(dates_converted, predictions, 's--', label='Predicho', color='red', linewidth=2, markersize=6)
            
            # Destacar errores grandes
            for i, (date, error, actual) in enumerate(zip(dates, errors, actuals)):
                error_pct = (error / actual * 100) if actual > 0 else 0
                if error_pct > 20:
                    ax1.scatter(date, predictions[i], color='orange', s=150, marker='X', 
                            edgecolors='darkred', linewidths=2, zorder=5)
            
            ax1.set_title('Rolling Forecast - Walk-Forward Validation', fontsize=13, fontweight='bold')
            ax1.set_xlabel('Fecha', fontsize=10)
            ax1.set_ylabel('SAIDI (minutos)', fontsize=10)
            ax1.legend(fontsize=9)
            ax1.grid(True, alpha=0.3)
            ax1.tick_params(axis='x', rotation=45)
            
            # Agregar métricas
            rmse = rolling_results['rmse']
            precision = rolling_results['precision']
            quality = rolling_results['prediction_quality']
            
            textstr = f"RMSE: {rmse:.2f} min\nPrecisión: {precision:.1f}%\nCalidad: {quality}"
            ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=9,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            # Panel 2: Time Series CV - Stability
            ax2 = plt.subplot(2, 3, 2)
            
            splits = cv_results['splits_details']
            rmse_vals = [s['rmse'] for s in splits]
            precision_vals = [s['precision'] for s in splits]
            
            bp = ax2.boxplot([rmse_vals, precision_vals], 
                            labels=['RMSE', 'Precisión (%)'],
                            patch_artist=True,
                            showmeans=True)
            
            colors = ['lightblue', 'lightgreen']
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
            
            ax2.set_title('Cross-Validation Stability', fontsize=13, fontweight='bold')
            ax2.set_ylabel('Valor', fontsize=10)
            ax2.grid(True, alpha=0.3, axis='y')
            
            cv_score = cv_results['cv_stability_score']
            n_failed = cv_results.get('n_failed', 0)
            textstr = f"Stability Score: {cv_score:.1f}/100\nSplits exitosos: {cv_results['n_splits']}"
            if n_failed > 0:
                textstr += f"\nSplits omitidos: {n_failed}"
            ax2.text(0.02, 0.98, textstr, transform=ax2.transAxes, fontsize=9,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
            
            # Panel 3: Parameter Stability
            ax3 = plt.subplot(2, 3, 3)
            
            param_details = param_stability['parameter_details']
            window_sizes = param_stability['window_sizes']
            
            # Seleccionar parámetros principales
            main_params = [p for p in param_details.keys() if any(x in p.lower() for x in ['ar.', 'ma.', 'sigma'])][:5]
            
            for param_name in main_params:
                if param_name in param_details:
                    values = param_details[param_name]['values']
                    if len(values) == len(window_sizes):
                        mean_val = param_details[param_name]['mean']
                        std_val = param_details[param_name]['std']
                        
                        ax3.plot(window_sizes, values, 'o-', label=param_name, linewidth=2, markersize=4)
                        
                        # Banda de confianza ±1σ
                        ax3.fill_between(window_sizes, 
                                        [mean_val - std_val] * len(window_sizes),
                                        [mean_val + std_val] * len(window_sizes),
                                        alpha=0.1)
            
            ax3.set_title('Parameter Stability Evolution', fontsize=13, fontweight='bold')
            ax3.set_xlabel('Tamaño de ventana (meses)', fontsize=10)
            ax3.set_ylabel('Valor del parámetro', fontsize=10)
            ax3.legend(fontsize=8, loc='best')
            ax3.grid(True, alpha=0.3)
            
            stability_score = param_stability['overall_stability_score']
            n_failed_windows = param_stability.get('n_failed_windows', 0)
            textstr = f"Stability Score: {stability_score:.1f}/100"
            if n_failed_windows > 0:
                textstr += f"\nVentanas omitidas: {n_failed_windows}"
            ax3.text(0.02, 0.98, textstr, transform=ax3.transAxes, fontsize=9,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.5))
            
            # Panel 4: Backtesting - Degradación por Horizonte
            ax4 = plt.subplot(2, 3, 4)
            
            horizons = backtesting_results['horizons']
            metrics_by_h = backtesting_results['metrics_by_horizon']
            
            precisions = [metrics_by_h[h]['precision'] for h in horizons]
            rmses = [metrics_by_h[h]['rmse'] for h in horizons]
            
            color = 'tab:blue'
            ax4.set_xlabel('Horizonte (meses)', fontsize=10)
            ax4.set_ylabel('Precisión (%)', color=color, fontsize=10)
            line1 = ax4.plot(horizons, precisions, 'o-', color=color, linewidth=2.5, 
                            markersize=8, label='Precisión')
            ax4.tick_params(axis='y', labelcolor=color)
            
            # Marcar horizonte óptimo
            optimal_h = backtesting_results['optimal_horizon']
            if optimal_h in horizons:
                opt_idx = horizons.index(optimal_h)
                ax4.scatter(optimal_h, precisions[opt_idx], color='green', s=250, 
                        marker='*', edgecolors='darkgreen', linewidths=2, zorder=5,
                        label=f'Óptimo: {optimal_h}m')
            
            # Zona roja (precisión < 75%)
            ax4.axhspan(0, 75, alpha=0.1, color='red', label='No confiable')
            ax4.axhline(y=75, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
            
            ax4_twin = ax4.twinx()
            color = 'tab:red'
            ax4_twin.set_ylabel('RMSE (min)', color=color, fontsize=10)
            line2 = ax4_twin.plot(horizons, rmses, 's--', color=color, linewidth=2, 
                                markersize=6, label='RMSE')
            ax4_twin.tick_params(axis='y', labelcolor=color)
            
            ax4.set_title('Backtesting - Degradación por Horizonte', fontsize=13, fontweight='bold')
            ax4.grid(True, alpha=0.3)
            
            lines = line1 + line2
            labels = [L.get_label() for L in lines]
            ax4.legend(lines, labels, fontsize=9, loc='lower left')
            
            # Panel 5: Diagnóstico Completo (SIN split_precision)
            ax5 = plt.subplot(2, 3, 5)
            ax5.axis('off')
            
            quality = final_diagnosis['model_quality']
            confidence = final_diagnosis['confidence_level']
            recommendation = final_diagnosis['recommendation']
            limitations = final_diagnosis['limitations']
            
            # Color según calidad
            if quality == "EXCELENTE":
                quality_color = 'green'
            elif quality == "CONFIABLE":
                quality_color = 'blue'
            elif quality == "CUESTIONABLE":
                quality_color = 'orange'
            else:
                quality_color = 'red'
            
            diagnosis_text = f"""
    DIAGNÓSTICO FINAL
    {'=' * 45}

    Calidad del Modelo: {quality}
    Nivel de Confianza: {confidence:.1f}%

    RECOMENDACIÓN:
    {recommendation}

    MÉTRICAS CLAVE:
    • Rolling Forecast RMSE: {rolling_results['rmse']:.2f} min
    • Precisión Rolling: {rolling_results['precision']:.1f}%
    • CV Stability Score: {cv_results['cv_stability_score']:.1f}/100
    • Parameter Stability: {param_stability['overall_stability_score']:.1f}/100
    • Horizonte óptimo: {backtesting_results['optimal_horizon']} meses
    • Degradación: {backtesting_results['degradation_rate']:.2f}% por mes

    NOTA: Rolling Forecast (walk-forward) es el gold
    standard para validación temporal de series de tiempo.
    Simula exactamente cómo funcionará el modelo en
    producción con datos futuros no vistos.

    """
            
            if limitations:
                diagnosis_text += "LIMITACIONES IDENTIFICADAS:\n"
                for lim in limitations[:4]:
                    diagnosis_text += f"• {lim}\n"
            
            diagnosis_text += f"\nModelo: SARIMAX{order}x{seasonal_order}"
            diagnosis_text += f"\nTransformación: {transformation.upper()}"
            
            if exog_info:
                diagnosis_text += f"\nVariables exógenas: {len(exog_info)}"
            
            ax5.text(0.05, 0.95, diagnosis_text, transform=ax5.transAxes,
                    fontsize=9, verticalalignment='top', family='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.4))
            
            # Indicador visual de calidad
            circle = plt.Circle((0.85, 0.5), 0.12, color=quality_color, alpha=0.7, transform=ax5.transAxes)
            ax5.add_patch(circle)
            ax5.text(0.85, 0.5, f'{confidence:.0f}', ha='center', va='center',
                    fontsize=24, fontweight='bold', color='white', transform=ax5.transAxes)
            
            # Panel 6: Scores por Componente
            ax6 = plt.subplot(2, 3, 6)
            
            component_scores = final_diagnosis['component_scores']
            components = list(component_scores.keys())
            scores = list(component_scores.values())
            
            # Nombres amigables
            friendly_names = {
                'rolling_forecast': 'Rolling\nForecast',
                'precision': 'Precisión',
                'cv_stability': 'CV\nStability',
                'parameter_stability': 'Param\nStability',
                'degradation': 'Degradación'
            }
            
            labels = [friendly_names.get(c, c) for c in components]
            
            colors_bars = ['green' if s >= 80 else 'orange' if s >= 65 else 'red' for s in scores]
            
            bars = ax6.barh(labels, scores, color=colors_bars, alpha=0.7, edgecolor='black')
            
            # Líneas de referencia
            ax6.axvline(x=85, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Excelente')
            ax6.axvline(x=70, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='Aceptable')
            
            ax6.set_title('Scores por Componente', fontsize=13, fontweight='bold')
            ax6.set_xlabel('Score (0-100)', fontsize=10)
            ax6.set_xlim(0, 100)
            ax6.legend(fontsize=8, loc='lower right')
            ax6.grid(True, alpha=0.3, axis='x')
            
            # Valores en barras
            for i, (bar, score) in enumerate(zip(bars, scores)):
                width = bar.get_width()
                ax6.text(width + 2, bar.get_y() + bar.get_height()/2, 
                        f'{score:.1f}', ha='left', va='center', fontsize=9, fontweight='bold')
            
            # Título general
            exog_suffix = f" [+{len(exog_info)} EXOG]" if exog_info else ""
            plt.suptitle(f'Validación Temporal Completa - SARIMAX{order}x{seasonal_order} + {transformation.upper()}{exog_suffix}',
                        fontsize=16, fontweight='bold', y=0.98)
            
            plt.tight_layout(rect=[0, 0.02, 1, 0.96])
            
            # Nota al pie
            footer_text = f'Generado: {datetime.now().strftime("%Y-%m-%d %H:%M")} | Rolling Forecast + CV + Parameter Stability + Backtesting'
            plt.figtext(0.5, 0.01, footer_text,
                    ha='center', fontsize=9, style='italic', color='darkblue',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.7))
            
            plt.savefig(plot_path, dpi=100, bbox_inches='tight', 
                    facecolor='white', edgecolor='none')
            plt.close(fig)
            
            self.plot_file_path = plot_path
            return plot_path
            
        except Exception as e:
            print(f"Error generando gráficas: {e}")
            import traceback
            traceback.print_exc()
            return None
        
    def cleanup_plot_file(self):
        """Limpiar archivo temporal de gráfica"""
        if self.plot_file_path and os.path.exists(self.plot_file_path):
            try:
                os.remove(self.plot_file_path)
            except Exception as e:
                print(f"Error eliminando archivo temporal: {e}")
            finally:
                self.plot_file_path = None