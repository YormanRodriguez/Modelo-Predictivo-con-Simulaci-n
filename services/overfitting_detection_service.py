# services/overfitting_detection_service.py - MODERNIZADO
"""
Servicio de detección de overfitting con transformaciones por regional
ACTUALIZADO: Alineado con ValidationService y OptimizationService

CARACTERÍSTICAS:
- Soporte para variables exógenas climáticas (sin simulación)
- Transformaciones específicas por regional
- Métricas calculadas en escala original
- División: 70% Training, 15% Validation, 15% Test
- Precisión calculada como OptimizationService
- SIN intervalos de confianza
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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from scipy import stats
import sys
import os
import tempfile
from datetime import datetime
from typing import Optional, Dict, Any, Tuple

class OverfittingDetectionService:
    """Servicio para detectar overfitting en modelos SAIDI con transformaciones por regional"""
    
    # Mapeo de regionales a sus transformaciones óptimas (igual que otros servicios)
    REGIONAL_TRANSFORMATIONS = {
        'SAIDI_O': 'boxcox',      # Ocaña - BoxCox
        'SAIDI_C': 'original',    # Cúcuta - Original
        'SAIDI_A': 'original',    # Aguachica - Original
        'SAIDI_P': 'boxcox',      # Pamplona - Boxcox
        'SAIDI_T': 'sqrt',        # Tibú - Sqrt
        'SAIDI_Cens': 'original'  # Cens - Original
    }
    
    # Variables exógenas por regional (consistente con ValidationService)
    REGIONAL_EXOG_VARS = {
        'SAIDI_O': {
            'temp_max': 'Temperatura máxima',
            'humedad_avg': 'Humedad relativa',
            'precip_total': 'Precipitación total'
        },
        'SAIDI_C': {
            'temp_max': 'Temperatura máxima',
            'humedad_avg': 'Humedad relativa',
            'precip_total': 'Precipitación total'
        },
        'SAIDI_A': {
            'temp_max': 'Temperatura máxima',
            'humedad_avg': 'Humedad relativa',
            'precip_total': 'Precipitación total'
        },
        'SAIDI_P': {
            'temp_max': 'Temperatura máxima',
            'humedad_avg': 'Humedad relativa',
            'precip_total': 'Precipitación total'
        },
        'SAIDI_T': {
            'temp_max': 'Temperatura máxima',
            'humedad_avg': 'Humedad relativa',
            'precip_total': 'Precipitación total'
        },
    }
    
    def __init__(self):
        self.default_order = (4, 1, 3)
        self.default_seasonal_order = (1, 1, 4, 12)
        self.plot_file_path = None
        self.scaler = None
        self.exog_scaler = None
        self.transformation_params = {}
    
    def run_overfitting_detection(self, 
                                 file_path: Optional[str] = None, 
                                 df_prepared: Optional[pd.DataFrame] = None, 
                                 order: Optional[Tuple] = None, 
                                 seasonal_order: Optional[Tuple] = None, 
                                 regional_code: Optional[str] = None,
                                 climate_data: Optional[pd.DataFrame] = None,
                                 progress_callback = None, 
                                 log_callback = None) -> Dict[str, Any]:
        """
        Detectar overfitting con transformación específica por regional
        
        Args:
            file_path: Ruta del archivo Excel SAIDI
            df_prepared: DataFrame SAIDI preparado
            order: Orden ARIMA
            seasonal_order: Orden estacional
            regional_code: Código de la regional (e.g., 'SAIDI_C', 'SAIDI_O')
            climate_data: DataFrame con datos climáticos mensuales
            progress_callback: Función para actualizar progreso
            log_callback: Función para logging
        
        Returns:
            Diccionario con análisis de overfitting
        """
        try:
            if order is None:
                order = self.default_order
            if seasonal_order is None:
                seasonal_order = self.default_seasonal_order
            
            # Determinar transformación a usar
            transformation = self._get_transformation_for_regional(regional_code)
            
            if log_callback:
                log_callback(f"Iniciando detección de overfitting con order={order}, seasonal={seasonal_order}")
                log_callback(f"Regional: {regional_code} - Transformación: {transformation.upper()}")
            
            if progress_callback:
                progress_callback(10, "Cargando datos...")
            
            # Cargar datos SAIDI
            if df_prepared is not None:
                df = df_prepared.copy()
                if log_callback:
                    log_callback("Usando datos preparados del modelo")
            elif file_path is not None:
                df = pd.read_excel(file_path, sheet_name="Hoja1")
                if log_callback:
                    log_callback("Leyendo Excel en formato tradicional")
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
            
            # Obtener solo datos históricos (sin NaN)
            data_original = df[df[col_saidi].notna()][col_saidi]
            
            if len(data_original) < 24:
                raise Exception("Se necesitan al menos 24 observaciones para análisis de overfitting")
            
            if log_callback:
                log_callback(f"Datos históricos: {len(data_original)} observaciones")
                log_callback(f"Período: {data_original.index[0].strftime('%Y-%m')} a {data_original.index[-1].strftime('%Y-%m')}")
            
            if progress_callback:
                progress_callback(20, "Preparando variables exógenas...")
            
            # Preparar variables exógenas (si disponibles)
            exog_df = None
            exog_info = None
            
            if climate_data is not None and not climate_data.empty:
                exog_df, exog_info = self._prepare_exogenous_variables(
                    climate_data, df, regional_code, log_callback
                )
                
                if exog_df is not None:
                    if log_callback:
                        log_callback(f"✓ Variables exógenas disponibles: {len(exog_df.columns)}")
                        for var_code, var_data in exog_info.items():
                            log_callback(f"  • {var_data['nombre']}")
                    
                    # Escalar variables exógenas
                    self.exog_scaler = StandardScaler()
                    exog_df_scaled = pd.DataFrame(
                        self.exog_scaler.fit_transform(exog_df),
                        index=exog_df.index,
                        columns=exog_df.columns
                    )
                    exog_df = exog_df_scaled
                    if log_callback:
                        log_callback("Variables exógenas escaladas correctamente")
                else:
                    if log_callback:
                        log_callback("⚠ No se pudieron preparar variables exógenas")
            else:
                if log_callback:
                    log_callback("⚠ Sin datos climáticos - análisis sin variables exógenas")
            
            if progress_callback:
                progress_callback(30, f"Aplicando transformación {transformation.upper()}...")
            
            # Aplicar transformación
            data_values_original = data_original.values
            data_transformed, transform_info = self._apply_transformation(
                data_values_original, transformation
            )
            data_transformed_series = pd.Series(data_transformed, index=data_original.index)
            
            if log_callback:
                log_callback(f"Transformación aplicada: {transform_info}")
            
            # División: 70% train, 15% validation, 15% test
            n_total = len(data_transformed_series)
            n_train = int(n_total * 0.70)
            n_val = int(n_total * 0.15)
            
            if log_callback:
                log_callback(f"División de datos:")
                log_callback(f"  • Training: {n_train} observaciones (70%)")
                log_callback(f"  • Validation: {n_val} observaciones (15%)")
                log_callback(f"  • Test: {n_total - n_train - n_val} observaciones (15%)")
            
            # Dividir datos transformados
            train_transformed = data_transformed_series[:n_train]
            val_transformed = data_transformed_series[n_train:n_train + n_val]
            test_transformed = data_transformed_series[n_train + n_val:]
            
            # Dividir datos originales (para métricas finales)
            train_original = data_original[:n_train]
            val_original = data_original[n_train:n_train + n_val]
            test_original = data_original[n_train + n_val:]
            
            # Dividir variables exógenas si existen
            exog_train = None
            exog_val = None
            exog_test = None
            
            if exog_df is not None:
                exog_train = exog_df.loc[train_original.index]
                exog_val = exog_df.loc[val_original.index]
                exog_test = exog_df.loc[test_original.index]
                
                if log_callback:
                    log_callback(f"Variables exógenas divididas:")
                    log_callback(f"  • Train: {len(exog_train)} períodos")
                    log_callback(f"  • Val: {len(exog_val)} períodos")
                    log_callback(f"  • Test: {len(exog_test)} períodos")
            
            if progress_callback:
                progress_callback(40, "Entrenando modelo con datos de training...")
            
            # Entrenar modelo con datos transformados
            try:
                model = SARIMAX(
                    train_transformed,
                    exog=exog_train,
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=True,
                    enforce_invertibility=True
                )
                results = model.fit(disp=False)
                
                if log_callback:
                    log_callback(f"Modelo entrenado exitosamente con datos transformados")
                    log_callback(f"AIC: {results.aic:.2f}, BIC: {results.bic:.2f}")
                    if exog_train is not None:
                        log_callback(f"Modelo incluye {exog_train.shape[1]} variables exógenas")
                    
            except Exception as e:
                raise Exception(f"Error entrenando modelo: {str(e)}")
            
            if progress_callback:
                progress_callback(60, "Calculando métricas en cada conjunto...")
            
            # Calcular métricas en escala original (MODERNIZADO)
            metrics_train = self._calculate_metrics_original_scale(
                train_transformed, train_original, results, transformation, 
                "Training", None, None
            )
            metrics_val = self._calculate_metrics_original_scale(
                val_transformed, val_original, results, transformation, 
                "Validation", exog_val, val_original.index
            )
            metrics_test = self._calculate_metrics_original_scale(
                test_transformed, test_original, results, transformation, 
                "Test", exog_test, test_original.index
            )
            
            if log_callback:
                log_callback("\n" + "=" * 60)
                log_callback("MÉTRICAS EN ESCALA ORIGINAL (minutos SAIDI)")
                log_callback("=" * 60)
                
                for metrics, label in [(metrics_train, "TRAINING"), 
                                      (metrics_val, "VALIDATION"), 
                                      (metrics_test, "TEST")]:
                    log_callback(f"\n{label}:")
                    log_callback(f"  • RMSE: {metrics['rmse']:.4f} minutos")
                    log_callback(f"  • MAE: {metrics['mae']:.4f} minutos")
                    log_callback(f"  • R²: {metrics['r2']:.4f}")
                    log_callback(f"  • MAPE: {metrics['mape']:.2f}%")
                    log_callback(f"  • Precisión: {metrics['precision_final']:.1f}%")
            
            if progress_callback:
                progress_callback(80, "Analizando overfitting...")
            
            # Análisis de overfitting (MODERNIZADO)
            overfitting_analysis = self._analyze_overfitting(
                metrics_train, metrics_val, metrics_test, log_callback
            )
            
            if log_callback:
                log_callback("\n" + "=" * 60)
                log_callback("ANÁLISIS DE OVERFITTING")
                log_callback("=" * 60)
                log_callback(f"Estado: {overfitting_analysis['status']}")
                log_callback(f"Nivel: {overfitting_analysis['overfitting_level']}")
                log_callback(f"Score: {overfitting_analysis['overfitting_score']:.2f}/100")
                log_callback(f"Degradación Precisión Train→Test: {overfitting_analysis['precision_degradation']:.1f}%")
                log_callback(f"Degradación RMSE Train→Test: {overfitting_analysis['rmse_increase']:.1f}%")
                
                if overfitting_analysis['has_overfitting']:
                    log_callback("\n⚠️ OVERFITTING DETECTADO")
                    log_callback("Recomendaciones:")
                    for rec in overfitting_analysis['recommendations']:
                        log_callback(f"  • {rec}")
                else:
                    log_callback("\n✅ NO SE DETECTÓ OVERFITTING SIGNIFICATIVO")
                    log_callback("El modelo generaliza adecuadamente")
            
            if progress_callback:
                progress_callback(90, "Generando gráfica de análisis...")
            
            # Generar gráfica con valores en escala original
            plot_path = self._generate_overfitting_plot(
                train_original, val_original, test_original,
                train_transformed, val_transformed, test_transformed,
                results, metrics_train, metrics_val, metrics_test,
                overfitting_analysis, order, seasonal_order, transformation,
                exog_info
            )
            
            if progress_callback:
                progress_callback(100, "Análisis completado")
            
            return {
                'success': True,
                'overfitting_analysis': overfitting_analysis,
                'metrics': {
                    'train': metrics_train,
                    'validation': metrics_val,
                    'test': metrics_test
                },
                'model_params': {
                    'order': order,
                    'seasonal_order': seasonal_order,
                    'transformation': transformation,
                    'regional_code': regional_code,
                    'with_exogenous': exog_df is not None
                },
                'exogenous_vars': exog_info,
                'data_split': {
                    'n_train': n_train,
                    'n_val': n_val,
                    'n_test': len(test_original)
                },
                'plot_file': plot_path
            }
            
        except Exception as e:
            if log_callback:
                log_callback(f"ERROR: {str(e)}")
            raise Exception(f"Error en detección de overfitting: {str(e)}")
    
    def _get_transformation_for_regional(self, regional_code: Optional[str]) -> str:
        """Obtener la transformación correspondiente a la regional"""
        if regional_code and regional_code in self.REGIONAL_TRANSFORMATIONS:
            return self.REGIONAL_TRANSFORMATIONS[regional_code]
        return 'original'
    
    def _prepare_exogenous_variables(self,
                                     climate_data: pd.DataFrame,
                                     df_saidi: pd.DataFrame,
                                     regional_code: Optional[str],
                                     log_callback) -> Tuple[Optional[pd.DataFrame], Optional[Dict]]:
        """
        Preparar variables exógenas climáticas SIN ESCALAR
        (Copiado de ValidationService)
        """
        try:
            if climate_data is None or climate_data.empty:
                return None, None
            
            if not regional_code or regional_code not in self.REGIONAL_EXOG_VARS:
                if log_callback:
                    log_callback(f"Regional {regional_code} no tiene variables exógenas definidas")
                return None, None
            
            exog_vars_config = self.REGIONAL_EXOG_VARS[regional_code]
            
            climate_column_mapping = {
                'temp_max': 'temp_max',
                'humedad_avg': 'humedad_avg',
                'precip_total': 'precip_total'
            }
            
            # Crear DataFrame de variables exógenas SIN ESCALAR
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
                        
                        if log_callback:
                            log_callback(f"  ✓ {var_nombre} preparada")
            
            exog_df = exog_df.dropna(how='all')
            exog_df = exog_df.interpolate(method='linear', limit_direction='both')
            
            if exog_df.empty:
                return None, None
            
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
        (Copiado de ValidationService)
        """
        try:
            climate_dates = exog_series.index
            saidi_dates = df_saidi.index
            
            if not isinstance(climate_dates, pd.DatetimeIndex):
                climate_dates = pd.to_datetime(climate_dates)
            if not isinstance(saidi_dates, pd.DatetimeIndex):
                saidi_dates = pd.to_datetime(saidi_dates)
            
            result = pd.Series(index=saidi_dates, dtype=float)
            
            for date in saidi_dates:
                if date in climate_dates:
                    result[date] = exog_series.loc[date].iloc[0]
            
            max_climate_date = climate_dates.max()
            future_indices = saidi_dates > max_climate_date
            
            if future_indices.any():
                last_known_value = exog_series.iloc[-1].iloc[0]
                result.loc[future_indices] = last_known_value
            
            min_climate_date = climate_dates.min()
            past_indices = saidi_dates < min_climate_date
            
            if past_indices.any():
                first_known_value = exog_series.iloc[0].iloc[0]
                result.loc[past_indices] = first_known_value
            
            return result
            
        except Exception as e:
            if log_callback:
                log_callback(f"Error alineando variable {var_code}: {str(e)}")
            return None
    
    def _apply_transformation(self, data: np.ndarray, transformation_type: str) -> Tuple[np.ndarray, str]:
        """Aplicar transformación a los datos"""
        if transformation_type == 'original':
            return data, "Sin transformación (datos originales)"
        
        elif transformation_type == 'standard':
            self.scaler = StandardScaler()
            transformed = self.scaler.fit_transform(data.reshape(-1, 1)).flatten()
            return transformed, f"StandardScaler (media={self.scaler.mean_[0]:.2f}, std={np.sqrt(self.scaler.var_[0]):.2f})"
        
        elif transformation_type == 'log':
            data_positive = np.maximum(data, 1e-10)
            transformed = np.log(data_positive)
            self.transformation_params['log_applied'] = True
            return transformed, "Transformación logarítmica (log)"
        
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
            return data, "Sin transformación (tipo desconocido)"
    
    def _inverse_transformation(self, data: np.ndarray, transformation_type: str) -> np.ndarray:
        """Revertir transformación a escala original"""
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
    
    def _calculate_metrics_original_scale(self, 
                                         data_transformed: pd.Series, 
                                         data_original: pd.Series, 
                                         model_results, 
                                         transformation: str, 
                                         label: str,
                                         exog_data: Optional[pd.DataFrame] = None,
                                         prediction_index: Optional[pd.DatetimeIndex] = None) -> Dict[str, float]:
        """
        Calcular métricas en escala original después de revertir transformación
        MODERNIZADO: Soporte para variables exógenas y precisión como OptimizationService
        """
        try:
            # Para Training: usar fittedvalues (in-sample)
            if label == "Training":
                fitted_transformed = model_results.fittedvalues
                fitted_transformed = fitted_transformed.reindex(data_transformed.index)
            
            # Para Validation/Test: usar get_forecast con exog
            else:
                if prediction_index is None:
                    prediction_index = data_transformed.index
                
                steps = len(prediction_index)
                pred = model_results.get_forecast(steps=steps, exog=exog_data)
                fitted_transformed = pred.predicted_mean
            
            # Verificar que no haya NaN
            if fitted_transformed.isna().any():
                raise ValueError(f"Predicciones contienen NaN para {label}")
            
            # Revertir a escala original
            fitted_original = self._inverse_transformation(
                fitted_transformed.values, transformation
            )
            
            # Verificar dimensiones
            if len(fitted_original) != len(data_original):
                raise ValueError(f"Dimensiones no coinciden: {len(fitted_original)} vs {len(data_original)}")
            
            # Calcular métricas en escala original (MODERNIZADO)
            rmse = np.sqrt(mean_squared_error(data_original.values, fitted_original))
            mae = mean_absolute_error(data_original.values, fitted_original)
            r2 = r2_score(data_original.values, fitted_original)
            
            epsilon = 1e-8
            mape = np.mean(np.abs((data_original.values - fitted_original) / 
                                 (data_original.values + epsilon))) * 100
            
            # Calcular precisión como OptimizationService
            precision_final = max(0.0, min(100.0, (1 - mape/100) * 100))
            
            if np.isnan(precision_final) or np.isinf(precision_final):
                precision_final = 0.0
            
            return {
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'mape': mape,
                'precision_final': precision_final,
                'fitted_values': fitted_original
            }
            
        except Exception as e:
            print(f"ERROR calculando métricas para {label}: {e}")
            import traceback
            traceback.print_exc()
            return {
                'rmse': float('inf'),
                'mae': float('inf'),
                'r2': -float('inf'),
                'mape': 100.0,
                'precision_final': 0.0,
                'fitted_values': None
            }
    
    def _analyze_overfitting(self, 
                            metrics_train: Dict, 
                            metrics_val: Dict, 
                            metrics_test: Dict, 
                            log_callback = None) -> Dict[str, Any]:
        """
        Analizar presencia y severidad de overfitting
        MODERNIZADO: Usa precisión en lugar de solo R² y RMSE
        """
        
        # Extraer métricas clave
        precision_train = metrics_train['precision_final']
        precision_val = metrics_val['precision_final']
        precision_test = metrics_test['precision_final']
        
        r2_train = metrics_train['r2']
        r2_val = metrics_val['r2']
        r2_test = metrics_test['r2']
        
        rmse_train = metrics_train['rmse']
        rmse_test = metrics_test['rmse']
        
        # Degradación porcentual de precisión
        if precision_train > 0:
            precision_degradation = ((precision_train - precision_test) / precision_train) * 100
        else:
            precision_degradation = 100.0
        
        # Degradación de R²
        if abs(r2_train) > 1e-8:
            r2_degradation = ((r2_train - r2_test) / max(abs(r2_train), 1e-8)) * 100
        else:
            r2_degradation = 0.0
        
        # Aumento de RMSE
        if rmse_train > 1e-8:
            rmse_increase = ((rmse_test - rmse_train) / rmse_train) * 100
        else:
            rmse_increase = 0.0
        
        # Score de overfitting (0-100, mayor = más overfitting)
        overfitting_score = 0
        
        # Factor 1: Degradación de Precisión (peso 40%)
        precision_diff = max(0, precision_degradation)
        overfitting_score += min(precision_diff * 0.4, 40)
        
        # Factor 2: Aumento de RMSE (peso 30%)
        overfitting_score += min(max(0, rmse_increase) * 0.3, 30)
        
        # Factor 3: Validación intermedia (peso 30%)
        if precision_val < precision_test:
            val_penalty = abs(precision_val - precision_test)
            overfitting_score += min(val_penalty * 0.3, 30)
        
        overfitting_score = min(overfitting_score, 100)
        
        # Clasificación
        if overfitting_score < 10:
            level = "NULO"
            status = "✅ Modelo generaliza excelentemente"
            has_overfitting = False
        elif overfitting_score < 20:
            level = "MÍNIMO"
            status = "✅ Overfitting negligible"
            has_overfitting = False
        elif overfitting_score < 35:
            level = "MODERADO"
            status = "⚠️ Overfitting moderado detectado"
            has_overfitting = True
        elif overfitting_score < 50:
            level = "ALTO"
            status = "❌ Overfitting significativo"
            has_overfitting = True
        else:
            level = "CRÍTICO"
            status = "❌ Overfitting severo - modelo no confiable"
            has_overfitting = True
        
        # Recomendaciones
        recommendations = []
        if has_overfitting:
            if overfitting_score > 35:
                recommendations.append("Considerar modelo más simple (reducir orden)")
                recommendations.append("Aumentar datos de entrenamiento si es posible")
            if rmse_increase > 30:
                recommendations.append("El RMSE aumenta significativamente en test")
            if precision_degradation > 20:
                recommendations.append("La precisión se degrada notablemente en test")
            if r2_degradation > 20:
                recommendations.append("Considerar regularización o validación cruzada")
            recommendations.append("Evaluar transformación alternativa de datos")
        
        return {
            'overfitting_score': overfitting_score,
            'overfitting_level': level,
            'status': status,
            'has_overfitting': has_overfitting,
            'precision_degradation': precision_degradation,
            'r2_degradation': r2_degradation,
            'rmse_increase': rmse_increase,
            'metrics_comparison': {
                'precision': {'train': precision_train, 'val': precision_val, 'test': precision_test},
                'r2': {'train': r2_train, 'val': r2_val, 'test': r2_test},
                'rmse': {'train': rmse_train, 'test': rmse_test}
            },
            'recommendations': recommendations
        }
    
    def _generate_overfitting_plot(self, 
                                   train_orig: pd.Series, 
                                   val_orig: pd.Series, 
                                   test_orig: pd.Series,
                                   train_trans: pd.Series, 
                                   val_trans: pd.Series, 
                                   test_trans: pd.Series,
                                   model_results, 
                                   metrics_train: Dict, 
                                   metrics_val: Dict, 
                                   metrics_test: Dict, 
                                   overfitting_analysis: Dict, 
                                   order: Tuple, 
                                   seasonal_order: Tuple, 
                                   transformation: str,
                                   exog_info: Optional[Dict] = None) -> Optional[str]:
        """
        Generar gráfica de análisis de overfitting con valores en escala original
        MODERNIZADO: Incluye precisión y soporte para variables exógenas
        """
        try:
            temp_dir = tempfile.gettempdir()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_path = os.path.join(temp_dir, f"saidi_overfitting_{timestamp}.png")
            
            plt.style.use('default')
            fig = plt.figure(figsize=(18, 12), dpi=100)
            
            # Plot principal: Valores reales vs fitted (ESCALA ORIGINAL)
            ax1 = plt.subplot(2, 2, 1)
            
            # Datos reales
            ax1.plot(train_orig.index, train_orig.values, 'o-', 
                    label='Training (Real)', color='blue', linewidth=2, markersize=4)
            ax1.plot(val_orig.index, val_orig.values, 'o-', 
                    label='Validation (Real)', color='green', linewidth=2, markersize=4)
            ax1.plot(test_orig.index, test_orig.values, 'o-', 
                    label='Test (Real)', color='red', linewidth=2, markersize=4)
            
            # Predicciones en escala original
            if metrics_train['fitted_values'] is not None:
                ax1.plot(train_orig.index, metrics_train['fitted_values'], '--', 
                        label='Fitted Training', color='cyan', linewidth=2, alpha=0.7)
            if metrics_val['fitted_values'] is not None:
                ax1.plot(val_orig.index, metrics_val['fitted_values'], '--', 
                        label='Fitted Validation', color='lime', linewidth=2, alpha=0.7)
            if metrics_test['fitted_values'] is not None:
                ax1.plot(test_orig.index, metrics_test['fitted_values'], '--', 
                        label='Fitted Test', color='orange', linewidth=2, alpha=0.7)
            
            ax1.axvline(x=train_orig.index[-1], color='gray', linestyle=':', 
                       linewidth=2, alpha=0.5, label='Train|Val')
            ax1.axvline(x=val_orig.index[-1], color='gray', linestyle='-.', 
                       linewidth=2, alpha=0.5, label='Val|Test')
            
            exog_label = " [+EXOG]" if exog_info else ""
            ax1.set_title(f'Real vs Predicción (Escala Original - minutos SAIDI){exog_label}', 
                         fontsize=14, fontweight='bold')
            ax1.set_xlabel('Fecha', fontsize=11)
            ax1.set_ylabel('SAIDI (minutos)', fontsize=11)
            ax1.legend(fontsize=9, loc='best')
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Comparación de métricas por conjunto (MODERNIZADO con Precisión)
            ax2 = plt.subplot(2, 2, 2)
            
            conjuntos = ['Training', 'Validation', 'Test']
            precision_values = [metrics_train['precision_final'], 
                               metrics_val['precision_final'], 
                               metrics_test['precision_final']]
            rmse_values = [metrics_train['rmse'], metrics_val['rmse'], metrics_test['rmse']]
            
            x_pos = np.arange(len(conjuntos))
            width = 0.35
            
            # Normalizar RMSE para visualización
            max_rmse = max(rmse_values)
            rmse_norm = [r/max_rmse for r in rmse_values]
            
            # Normalizar Precisión a escala 0-1
            precision_norm = [p/100 for p in precision_values]
            
            bars1 = ax2.bar(x_pos - width/2, precision_norm, width, 
                           label='Precisión (/100)', color='steelblue', alpha=0.8)
            bars2 = ax2.bar(x_pos + width/2, rmse_norm, width, 
                           label='RMSE (norm)', color='coral', alpha=0.8)
            
            ax2.set_title('Comparación de Métricas por Conjunto', 
                         fontsize=14, fontweight='bold')
            ax2.set_ylabel('Valor', fontsize=11)
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(conjuntos)
            ax2.legend(fontsize=9)
            ax2.grid(True, alpha=0.3, axis='y')
            
            # Añadir valores sobre las barras
            for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
                # Precisión
                height1 = bar1.get_height()
                ax2.text(bar1.get_x() + bar1.get_width()/2., height1,
                        f'{precision_values[i]:.1f}%', ha='center', va='bottom', fontsize=8)
                # RMSE
                height2 = bar2.get_height()
                ax2.text(bar2.get_x() + bar2.get_width()/2., height2,
                        f'{rmse_values[i]:.3f}', ha='center', va='bottom', fontsize=8)
            
            # Plot 3: Residuos por conjunto (ESCALA ORIGINAL)
            ax3 = plt.subplot(2, 2, 3)
            
            if metrics_train['fitted_values'] is not None:
                residuals_train = train_orig.values - metrics_train['fitted_values']
                ax3.scatter(train_orig.index, residuals_train, 
                           alpha=0.6, s=40, label='Training', color='blue')
            
            if metrics_val['fitted_values'] is not None:
                residuals_val = val_orig.values - metrics_val['fitted_values']
                ax3.scatter(val_orig.index, residuals_val, 
                           alpha=0.6, s=40, label='Validation', color='green')
            
            if metrics_test['fitted_values'] is not None:
                residuals_test = test_orig.values - metrics_test['fitted_values']
                ax3.scatter(test_orig.index, residuals_test, 
                           alpha=0.6, s=40, label='Test', color='red')
            
            ax3.axhline(y=0, color='black', linestyle='-', linewidth=1.5, alpha=0.5)
            ax3.set_title('Residuos por Conjunto (minutos SAIDI)', 
                         fontsize=14, fontweight='bold')
            ax3.set_xlabel('Fecha', fontsize=11)
            ax3.set_ylabel('Residuo (minutos)', fontsize=11)
            ax3.legend(fontsize=9)
            ax3.grid(True, alpha=0.3)
            
            # Plot 4: Panel de diagnóstico (MODERNIZADO)
            ax4 = plt.subplot(2, 2, 4)
            ax4.axis('off')
            
            score = overfitting_analysis['overfitting_score']
            level = overfitting_analysis['overfitting_level']
            
            # Color según severidad
            if score < 20:
                color = 'green'
            elif score < 35:
                color = 'orange'
            else:
                color = 'red'
            
            info_text = f"""
ANÁLISIS DE OVERFITTING
{'=' * 50}

Score de Overfitting: {score:.2f}/100
Nivel: {level}
Status: {overfitting_analysis['status']}

MÉTRICAS (Escala Original - minutos SAIDI)
{'=' * 50}

Training:
  • RMSE: {metrics_train['rmse']:.4f} min
  • Precisión: {metrics_train['precision_final']:.1f}%
  • R²: {metrics_train['r2']:.4f}
  • MAPE: {metrics_train['mape']:.2f}%

Validation:
  • RMSE: {metrics_val['rmse']:.4f} min
  • Precisión: {metrics_val['precision_final']:.1f}%
  • R²: {metrics_val['r2']:.4f}
  • MAPE: {metrics_val['mape']:.2f}%

Test:
  • RMSE: {metrics_test['rmse']:.4f} min
  • Precisión: {metrics_test['precision_final']:.1f}%
  • R²: {metrics_test['r2']:.4f}
  • MAPE: {metrics_test['mape']:.2f}%

DEGRADACIÓN
{'=' * 50}
Precisión Train→Test: {overfitting_analysis['precision_degradation']:.1f}%
R² Train→Test: {overfitting_analysis['r2_degradation']:.1f}%
RMSE Aumento: {overfitting_analysis['rmse_increase']:.1f}%

Transformación: {transformation.upper()}
Modelo: SARIMAX{order}x{seasonal_order}
"""
            
            if exog_info:
                info_text += f"\nVariables Exógenas: {len(exog_info)}"
            
            ax4.text(0.05, 0.95, info_text, transform=ax4.transAxes,
                    fontsize=10, verticalalignment='top', family='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
            
            # Indicador visual de overfitting
            circle = plt.Circle((0.85, 0.5), 0.08, color=color, alpha=0.6)
            ax4.add_patch(circle)
            ax4.text(0.85, 0.5, f'{score:.0f}', ha='center', va='center',
                    fontsize=20, fontweight='bold', color='white')
            
            # Título general
            title_suffix = " [+EXOG]" if exog_info else ""
            plt.suptitle(f'Análisis de Overfitting - SARIMAX{order}x{seasonal_order} + {transformation.upper()}{title_suffix}',
                        fontsize=16, fontweight='bold', y=0.98)
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.96])
            
            # Nota al pie
            footer_text = 'Todas las métricas calculadas en escala original (minutos SAIDI)'
            if exog_info:
                footer_text += f' | {len(exog_info)} variables exógenas incluidas'
            
            plt.figtext(0.5, 0.01, footer_text,
                       ha='center', fontsize=10, style='italic', color='darkblue',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.7))
            
            plt.savefig(plot_path, dpi=100, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            plt.close(fig)
            
            self.plot_file_path = plot_path
            return plot_path
            
        except Exception as e:
            print(f"Error generando gráfica de overfitting: {e}")
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