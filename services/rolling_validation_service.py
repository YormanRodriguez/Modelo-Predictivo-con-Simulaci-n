# services/rolling_validation_service.py
"""
Servicio de Validación Temporal con Rolling Forecast para SAIDI - Manejo robusto de errores de álgebra lineal
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from scipy import stats
import sys
import os
import tempfile
from datetime import datetime
from typing import Optional, Dict, Any, Tuple, List
from copy import deepcopy


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
        'SAIDI_O': {'temp_max': 'Temperatura máxima', 'humedad_avg': 'Humedad relativa', 'precip_total': 'Precipitación total'},
        'SAIDI_C': {'temp_max': 'Temperatura máxima', 'humedad_avg': 'Humedad relativa', 'precip_total': 'Precipitación total'},
        'SAIDI_A': {'temp_max': 'Temperatura máxima', 'humedad_avg': 'Humedad relativa', 'precip_total': 'Precipitación total'},
        'SAIDI_P': {'temp_max': 'Temperatura máxima', 'humedad_avg': 'Humedad relativa', 'precip_total': 'Precipitación total'},
        'SAIDI_T': {'temp_max': 'Temperatura máxima', 'humedad_avg': 'Humedad relativa', 'precip_total': 'Precipitación total'},
    }

    REGIONAL_ORDERS = {
        'SAIDI_O': {
            'order': (4, 1, 3),
            'seasonal_order': (1, 1, 4, 12)
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

    def _get_orders_for_regional(self, regional_code):
        """
        Obtener ordenes SARIMAX especificos para una regional
        
        Args:
            regional_code: Codigo de la regional (ej: 'SAIDI_O')
        
        Returns:
            tuple: (order, seasonal_order) - Ordenes ARIMA y estacionales
        """
        if regional_code and regional_code in self.REGIONAL_ORDERS:
            config = self.REGIONAL_ORDERS[regional_code]
            return config['order'], config['seasonal_order']
        else:
            # Fallback a valores por defecto si no hay configuracion especifica
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
        Ejecutar análisis completo de validación temporal
        """
        try:
            if order is None or seasonal_order is None:
                order_regional, seasonal_regional = self._get_orders_for_regional(regional_code)
                
                if order is None:
                    order = order_regional
                if seasonal_order is None:
                    seasonal_order = seasonal_regional
                
                if log_callback and regional_code:
                    regional_nombre = {
                        'SAIDI_O': 'Ocaña',
                        'SAIDI_C': 'Cúcuta',
                        'SAIDI_A': 'Aguachica',
                        'SAIDI_P': 'Pamplona',
                        'SAIDI_T': 'Tibú',
                        'SAIDI_Cens': 'CENS'
                    }.get(regional_code, regional_code)
                    
                    log_callback(f"Usando parametros optimizados para regional {regional_nombre}")
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
            
            # 1️ ROLLING FORECAST
            if progress_callback:
                progress_callback(10, "Ejecutando Rolling Forecast...")
            
            rolling_results = self.run_rolling_forecast(
                data_original, data_transformed_series, exog_df,
                order, seasonal_order, transformation,
                validation_months, progress_callback, log_callback
            )
            
            # 2️ TIME SERIES CROSS-VALIDATION
            if progress_callback:
                progress_callback(40, "Ejecutando Time Series CV...")
            
            cv_results = self.run_time_series_cv(
                data_original, data_transformed_series, exog_df,
                order, seasonal_order, transformation,
                progress_callback, log_callback
            )
            
            # 3️ ANÁLISIS DE ESTABILIDAD DE PARÁMETROS
            if progress_callback:
                progress_callback(60, "Analizando estabilidad de parámetros...")
            
            param_stability = self.analyze_parameter_stability(
                data_original, data_transformed_series, exog_df,
                order, seasonal_order, transformation,
                progress_callback, log_callback
            )
            
            # 4️ BACKTESTING MULTI-HORIZONTE
            if progress_callback:
                progress_callback(80, "Ejecutando backtesting...")
            
            backtesting_results = self.run_backtesting(
                data_original, data_transformed_series, exog_df,
                order, seasonal_order, transformation,
                progress_callback, log_callback
            )

            # 5️ PRECISIÓN SPLIT ÚNICO (Comparable con prediction_service)
            if progress_callback:
                progress_callback(85, "Calculando precisión comparable...")
            
            split_precision = self._calcular_precision_split_unico(
                data_original, data_transformed_series, exog_df,
                order, seasonal_order, transformation,
                log_callback
            )
            
            # DIAGNÓSTICO FINAL
            final_diagnosis = self._generate_final_diagnosis(
                rolling_results, cv_results, param_stability, backtesting_results,
                split_precision  
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
                log_callback("\nCOMPARACIÓN DE PRECISIONES:")
                log_callback(f"  • Precisión Rolling Forecast: {rolling_results['precision']:.1f}% (validación temporal estricta)")
                log_callback(f"  • Precisión Split Único: {split_precision['precision']:.1f}% (comparable con Predicción)")
                log_callback(f"    → La precisión rolling es más conservadora y robusta")
                log_callback(f"    → La precisión split único es la referencia del servicio de predicción")
                
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
        """
        Implementar Rolling Forecast (Walk-Forward Validation)
        """
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
            
            #  CORREGIDO: Definir pred_date ANTES del bloque if
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
                            log_callback(f"i Usando promedios históricos para variables exógenas futuras")
                    else:
                        exog_pred = None
            
            #  VALIDACIÓN: Verificar que no haya NaN en los datos
            if train_data_trans.isna().any() or (exog_train is not None and exog_train.isna().any().any()):
                if log_callback:
                    log_callback(f" Iteración {i+1} contiene NaN - omitida")
                continue
            
            try:
                #  CORREGIDO: Parámetros más robustos
                model = SARIMAX(
                    train_data_trans,
                    exog=exog_train,
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,  # Más flexible
                    enforce_invertibility=False  # Más flexible
                )
                
                #  CORREGIDO: Método de ajuste más estable
                results = model.fit(
                    disp=False,
                    method='lbfgs',  # Método más estable
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
                
            except np.linalg.LinAlgError as e:
                #  CORREGIDO: Manejo específico de errores de álgebra lineal
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
            
            # Clasificar calidad
            if rmse < 2.0 and precision >= 90:
                quality = "EXCELENTE"
            elif rmse < 3.0 and precision >= 85:
                quality = "BUENA"
            elif rmse < 4.5 and precision >= 75:
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
        """
        Time Series Cross-Validation con splits temporales
        """
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
        
        #  CORREGIDO: Calcular n_splits correctamente
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
        max_iterations = n_splits + 10  # Seguridad: máximo de iteraciones
        iteration_count = 0
        
        while current_train_end + val_size + test_size <= n_total:
            split_idx += 1
            iteration_count += 1
            
            #  SEGURIDAD: Detectar bucle infinito
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
            val_trans = data_transformed.iloc[train_end:val_end]
            test_trans = data_transformed.iloc[val_end:test_end]
            
            train_orig = data_original.iloc[:train_end]
            val_orig = data_original.iloc[train_end:val_end]
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
            
            #  VALIDACIÓN: Verificar que no haya NaN en los datos
            if train_trans.isna().any() or (exog_train is not None and exog_train.isna().any().any()):
                failed_splits += 1
                if log_callback and failed_splits <= 3:
                    log_callback(f" Split {split_idx} contiene NaN - omitido")
                #  CRÍTICO: Avanzar SIEMPRE
                current_train_end += step
                continue
            
            try:
                #  CORREGIDO: Parámetros más robustos
                model = SARIMAX(
                    train_trans,
                    exog=exog_train,
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                
                #  CORREGIDO: Método de ajuste más estable
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
                
            except np.linalg.LinAlgError as e:
                #  CORREGIDO: Manejo específico de errores de álgebra lineal
                failed_splits += 1
                if log_callback and failed_splits <= 3:  # Solo mostrar primeros 3 fallos
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
        """
        Analizar estabilidad de parámetros ARIMA/SARIMA
        """
        if log_callback:
            log_callback("\n ANÁLISIS DE ESTABILIDAD DE PARÁMETROS")
        
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
            
            #  VALIDACIÓN: Verificar que no haya NaN en los datos
            if train_trans.isna().any() or (exog_train is not None and exog_train.isna().any().any()):
                failed_windows += 1
                if log_callback and failed_windows <= 2:
                    log_callback(f" Ventana {window} contiene NaN - omitida")
                continue
            
            try:
                #  CORREGIDO: Parámetros más robustos
                model = SARIMAX(
                    train_trans,
                    exog=exog_train,
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                
                #  CORREGIDO: Método de ajuste más estable
                results = model.fit(
                    disp=False,
                    method='lbfgs',
                    maxiter=100,
                    low_memory=True
                )
                
                # Extraer parámetros
                params = results.params
                
                for param_name, param_value in params.items():
                    if param_name not in params_evolution:
                        params_evolution[param_name] = []
                    params_evolution[param_name].append(param_value)
                
            except np.linalg.LinAlgError:
                failed_windows += 1
                if log_callback and failed_windows <= 2:
                    log_callback(f" Ventana {window} falló (matriz singular) - omitida")
                continue
            except Exception as e:
                failed_windows += 1
                if log_callback and failed_windows <= 2:
                    log_callback(f" Ventana {window} falló - omitida")
                continue
        
        if not params_evolution:
            raise Exception("No se pudo analizar estabilidad de parámetros en ninguna ventana")
        
        # Analizar estabilidad de cada parámetro
        param_analysis = {}
        unstable_params = []
        
        for param_name, values in params_evolution.items():
            if len(values) > 1:
                mean_val = np.mean(values)
                std_val = np.std(values)
                
                # Score de estabilidad: mayor std = menor estabilidad
                if abs(mean_val) > 1e-6:
                    cv = abs(std_val / mean_val)
                    stability_score = max(0, min(100, 100 - (cv * 100)))
                else:
                    stability_score = 50
                
                param_analysis[param_name] = {
                    'mean': mean_val,
                    'std': std_val,
                    'stability_score': stability_score,
                    'values': values
                }
                
                if std_val > 0.15 or stability_score < 70:
                    unstable_params.append(param_name)
        
        # Score global
        if param_analysis:
            overall_stability = np.mean([p['stability_score'] for p in param_analysis.values()])
        else:
            overall_stability = 0
        
        # Interpretación
        if overall_stability >= 85:
            interpretation = "Los parámetros muestran ALTA estabilidad - Modelo robusto"
        elif overall_stability >= 75:
            interpretation = "Los parámetros muestran BUENA estabilidad - Modelo confiable"
        elif overall_stability >= 65:
            interpretation = "Los parámetros muestran estabilidad MODERADA"
        else:
            interpretation = "Los parámetros muestran BAJA estabilidad - Revisar especificación"
        
        if log_callback:
            log_callback(f"  Overall Stability Score: {overall_stability:.1f}/100")
            log_callback(f"  {interpretation}")
            if unstable_params:
                log_callback(f"  Parámetros inestables: {', '.join(unstable_params)}")
            if failed_windows > 0:
                log_callback(f"  Ventanas omitidas: {failed_windows}/{len(window_sizes)}")
        
        return {
            'overall_stability_score': overall_stability,
            'unstable_params': unstable_params,
            'parameter_details': param_analysis,
            'interpretation': interpretation,
            'window_sizes': window_sizes,
            'n_failed_windows': failed_windows
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
        """
        Backtesting multi-horizonte
        """
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
                
                test_trans = data_transformed.iloc[start_idx:start_idx + horizon]
                test_orig = data_original.iloc[start_idx:start_idx + horizon]
                
                exog_train = None
                exog_test = None
                if exog_df is not None:
                    exog_train = exog_df.loc[train_orig.index]
                    if test_orig.index[-1] <= exog_df.index.max():
                        exog_test = exog_df.loc[test_orig.index]
                
                #  VALIDACIÓN: Verificar que no haya NaN en los datos
                if train_trans.isna().any() or (exog_train is not None and exog_train.isna().any().any()):
                    failed_count += 1
                    continue
                
                try:
                    #  CORREGIDO: Parámetros más robustos
                    model = SARIMAX(
                        train_trans,
                        exog=exog_train,
                        order=order,
                        seasonal_order=seasonal_order,
                        enforce_stationarity=False,
                        enforce_invertibility=False
                    )
                    
                    #  CORREGIDO: Método de ajuste más estable
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
                    #  CORREGIDO: Silenciosamente omitir errores en backtesting
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
            utility_factor = min(1.0, h / 6.0)  # Preferir horizontes más largos
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
    
    def _calcular_precision_split_unico(self,
                                       data_original: pd.Series,
                                       data_transformed: pd.Series,
                                       exog_df: Optional[pd.DataFrame],
                                       order: Tuple,
                                       seasonal_order: Tuple,
                                       transformation: str,
                                       log_callback=None) -> Dict[str, Any]:
        """
        Calcular precisión con split único train/test (comparable con prediction_service)
        
        Usa el MISMO método que prediction_service._calcular_metricas_modelo()
        para obtener una métrica de precisión comparable.
        
        Returns:
            dict con: precision, mape, rmse, mae, r2_score
        """
        try:
            # Determinar porcentaje de validación (igual que prediction_service)
            if len(data_original) >= 60:
                pct_validacion = 0.30
            elif len(data_original) >= 36:
                pct_validacion = 0.25
            else:
                pct_validacion = 0.20
            
            n_test = max(6, int(len(data_original) * pct_validacion))
            
            # Split train/test
            train_original = data_original[:-n_test]
            test_original = data_original[-n_test:]
            
            train_transformed = data_transformed[:-n_test]
            
            # Preparar exógenas
            exog_train = None
            exog_test = None
            if exog_df is not None:
                exog_train = exog_df.loc[train_original.index]
                exog_test = exog_df.loc[test_original.index]
            
            # Entrenar modelo
            model = SARIMAX(
                train_transformed,
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
            
            # Predicción
            pred = results.get_forecast(steps=n_test, exog=exog_test)
            pred_mean_transformed = pred.predicted_mean.values
            
            # Revertir transformación
            pred_mean_original = self._inverse_transformation(
                pred_mean_transformed, transformation
            )
            
            # Calcular métricas
            test_values = test_original.values
            pred_values = pred_mean_original
            
            rmse = np.sqrt(mean_squared_error(test_values, pred_values))
            mae = np.mean(np.abs(test_values - pred_values))
            
            epsilon = 1e-8
            mape = np.mean(np.abs((test_values - pred_values) / 
                                (test_values + epsilon))) * 100
            
            ss_res = np.sum((test_values - pred_values) ** 2)
            ss_tot = np.sum((test_values - np.mean(test_values)) ** 2)
            r2_score = 1 - (ss_res / (ss_tot + epsilon))
            
            # Precisión final (mismo cálculo que prediction_service)
            precision = max(0, min(100, (1 - mape/100) * 100))
            
            if log_callback:
                log_callback(f"\n PRECISIÓN SPLIT ÚNICO (Referencia Comparable)")
                log_callback(f"Split: {len(train_original)} train / {n_test} test")
                log_callback(f"  Precisión: {precision:.1f}%")
                log_callback(f"  MAPE: {mape:.1f}%")
                log_callback(f"  RMSE: {rmse:.4f} min")
                log_callback(f"  R²: {r2_score:.4f}")
            
            return {
                'precision': precision,
                'mape': mape,
                'rmse': rmse,
                'mae': mae,
                'r2_score': r2_score,
                'n_test': n_test,
                'n_train': len(train_original)
            }
            
        except Exception as e:
            if log_callback:
                log_callback(f" Error calculando precisión split único: {str(e)}")
            return {
                'precision': 0.0,
                'mape': 100.0,
                'rmse': 0.0,
                'mae': 0.0,
                'r2_score': 0.0,
                'n_test': 0,
                'n_train': 0
            }

    def _generate_final_diagnosis(self,
                                  rolling_results: Dict,
                                  cv_results: Dict,
                                  param_stability: Dict,
                                  backtesting_results: Dict,
                                  split_precision: Dict) -> Dict[str, Any]: 
        """
        Generar diagnóstico final integrado
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
        score_degradation = max(0, 100 + degradation_rate * 20)  # degradation_rate es negativo
        
        # Puntaje global ponderado
        confidence_level = (
            score_rolling * 0.25 +
            score_precision * 0.25 +
            score_cv * 0.20 +
            score_params * 0.20 +
            score_degradation * 0.10
        )
        
        # Clasificación de calidad
        if confidence_level >= 85 and rolling_rmse < 3.0 and cv_stability >= 85 and param_stability_score >= 80:
            model_quality = "EXCELENTE"
        elif confidence_level >= 75 and rolling_rmse < 4.0 and cv_stability >= 75 and param_stability_score >= 70:
            model_quality = "CONFIABLE"
        elif confidence_level >= 65 and rolling_rmse < 5.0:
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
            recommendation = f"Usar solo para pronósticos de corto plazo (1-2 meses) con monitoreo continuo"
        else:
            recommendation = "No recomendado para uso productivo - Revisar especificación del modelo"
        
        # Limitaciones
        limitations = []
        
        if rolling_rmse > 3.0:
            limitations.append(f"RMSE elevado en rolling forecast ({rolling_rmse:.2f} min)")
        
        if cv_stability < 75:
            limitations.append(f"Estabilidad limitada en CV (Score: {cv_stability:.1f})")
        
        if param_stability_score < 70:
            limitations.append(f"Parámetros inestables (Score: {param_stability_score:.1f})")
        
        if abs(degradation_rate) > 3.0:
            limitations.append(f"Degradación notable después de {optimal_horizon} meses ({degradation_rate:.1f}% por mes)")
        
        if len(param_stability.get('unstable_params', [])) > 2:
            limitations.append(f"{len(param_stability['unstable_params'])} coeficientes inestables detectados")
        
        if rolling_precision < 80:
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
            },
            'split_precision': split_precision  
        }
    
    # ========== MÉTODOS AUXILIARES (continúan igual) ==========
    
    def _get_transformation_for_regional(self, regional_code: Optional[str]) -> str:
        """Obtener transformación para la regional"""
        if regional_code and regional_code in self.REGIONAL_TRANSFORMATIONS:
            return self.REGIONAL_TRANSFORMATIONS[regional_code]
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
                # Escalar variables exógenas
                self.exog_scaler = StandardScaler()
                exog_df_scaled = pd.DataFrame(
                    self.exog_scaler.fit_transform(exog_df),
                    index=exog_df.index,
                    columns=exog_df.columns
                )
                exog_df = exog_df_scaled
        
        return data_original, exog_df, exog_info
    
    def _prepare_exogenous_variables(self,
                                     climate_data: pd.DataFrame,
                                     df_saidi: pd.DataFrame,
                                     regional_code: Optional[str],
                                     log_callback) -> Tuple[Optional[pd.DataFrame], Optional[Dict]]:
        """
        Preparar variables exógenas climáticas
        
        Args:
            climate_data: DataFrame con datos climáticos mensuales
            df_saidi: DataFrame con datos SAIDI
            regional_code: Código de la regional (ej: 'SAIDI_O')
            log_callback: Función para logging
        
        Returns:
            Tuple: (exog_df, exog_info) o (None, None) si hay error
                - exog_df: DataFrame con variables exógenas alineadas (SIN ESCALAR)
                - exog_info: Dict con información de variables
        """
        try:
            if not regional_code or regional_code not in self.REGIONAL_EXOG_VARS:
                return None, None
            
            exog_vars_config = self.REGIONAL_EXOG_VARS[regional_code]
            
            climate_column_mapping = {
                'temp_max': 'temp_max',
                'humedad_avg': 'humedad_avg',
                'precip_total': 'precip_total'
            }
            
            exog_df = pd.DataFrame(index=df_saidi.index)
            exog_info = {}
            
            for var_code, var_nombre in exog_vars_config.items():
                climate_col = climate_column_mapping.get(var_code)
                
                if climate_col and climate_col in climate_data.columns:
                    var_series = climate_data[[climate_col]].copy()
                    var_series.columns = [var_code]
                    
                    exog_values = self._align_exog_to_saidi(
                        var_series, df_saidi, var_code, log_callback
                    )
                    
                    if exog_values is not None:
                        exog_df[var_code] = exog_values
                        exog_info[var_code] = {
                            'nombre': var_nombre,
                            'columna_clima': climate_col
                        }
            
            # Verificar cobertura antes de procesar
            if log_callback:
                log_callback(f"\n Variables exógenas preparadas: {len(exog_df.columns)}")
                for col in exog_df.columns:
                    n_total = len(exog_df)
                    n_valid = exog_df[col].notna().sum()
                    n_nan = n_total - n_valid
                    coverage_pct = (n_valid / n_total) * 100
                    log_callback(f"  - {col}: {n_valid}/{n_total} valores ({coverage_pct:.1f}% cobertura)")
                    if n_nan > 0:
                        log_callback(f"       {n_nan} NaN detectados - procesando...")
            
            # Eliminar columnas completamente vacías
            exog_df = exog_df.dropna(how='all', axis=1)

            # Verificación final
            if exog_df.isna().any().any():
                if log_callback:
                    log_callback(f"\n NaN residuales detectados - aplicando limpieza final...")
                
                # Forward fill, luego backward fill
                exog_df = exog_df.ffill()
                exog_df = exog_df.bfill()
                
                # Si aún hay NaN, rellenar con media de cada columna
                for col in exog_df.columns:
                    if exog_df[col].isna().any():
                        mean_value = exog_df[col].mean()
                        if pd.notna(mean_value):
                            exog_df[col].fillna(mean_value, inplace=True)
                            if log_callback:
                                log_callback(f"  - {col}: {exog_df[col].isna().sum()} NaN rellenados con media ({mean_value:.2f})")
                        else:
                            # Si ni siquiera hay media, usar 0
                            exog_df[col].fillna(0, inplace=True)
                            if log_callback:
                                log_callback(f"  - {col}: Sin valores válidos, rellenado con 0")
            
            # Verificación final: descartar columnas con demasiados NaN
            exog_df = exog_df.dropna(axis=1, how='any')
            
            if exog_df.empty:
                if log_callback:
                    log_callback(" Todas las variables exógenas contienen demasiados NaN - omitidas")
                return None, None
            
            if log_callback:
                log_callback(f"\n Variables exógenas finales: {len(exog_df.columns)}")
                log_callback(f"   Sin NaN - Listas para usar en validación")
            
            return exog_df, exog_info if exog_info else None
            
        except Exception as e:
            if log_callback:
                log_callback(f"Error preparando variables exógenas: {str(e)}")
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
            return transformed, f"StandardScaler"
        
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
        """
        Generar panel de gráficas comprehensivo
        """
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
            labels = [l.get_label() for l in lines]
            ax4.legend(lines, labels, fontsize=9, loc='lower left')
            
            # Panel 5: Diagnóstico Completo
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
• Precisión Rolling: {rolling_results['precision']:.1f}% (walk-forward)
• Precisión Split Único: {final_diagnosis['split_precision']['precision']:.1f}% [REF]
• CV Stability Score: {cv_results['cv_stability_score']:.1f}/100
• Parameter Stability: {param_stability['overall_stability_score']:.1f}/100
• Horizonte óptimo: {backtesting_results['optimal_horizon']} meses
• Degradación: {backtesting_results['degradation_rate']:.2f}% por mes

NOTA: La precisión split único es comparable con el
servicio de predicción. Rolling es más conservadora.

"""
            
            if limitations:
                diagnosis_text += "LIMITACIONES IDENTIFICADAS:\n"
                for lim in limitations[:4]:  # Máximo 4
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