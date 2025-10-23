# services/improved_model_service.py - Servicio con transformaciones por regional
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.stats import boxcox
from itertools import product
import tempfile
import os
from datetime import datetime

class ImprovedModelService:
    """Servicio con técnicas avanzadas para reducir overfitting + transformaciones por regional"""
    
    # NUEVO: Mapeo de transformaciones por regional (idéntico a prediction_service)
    REGIONAL_TRANSFORMATIONS = {
        'SAIDI_O': 'original',      # Ocaña - Original
        'SAIDI_C': 'log',            # Cúcuta - Log
        'SAIDI_A': 'original',       # Aguachica - Original
        'SAIDI_P': 'log',            # Pamplona - Log
        'SAIDI_T': 'boxcox',         # Tibú - BoxCox
        'SAIDI_Cens': 'log'          # Cens - Log
    }
    
    def __init__(self):
        self.plot_files = []
        self.scaler = None
        self.transformation_params = {}
    
    def _get_transformation_for_regional(self, regional_code):
        """Obtener la transformación correspondiente a la regional"""
        if regional_code and regional_code in self.REGIONAL_TRANSFORMATIONS:
            return self.REGIONAL_TRANSFORMATIONS[regional_code]
        return 'original'  # Default
    
    def _apply_transformation(self, data, transformation_type):
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
        
        else:
            return data, "Sin transformación (tipo desconocido)"
    
    def _inverse_transformation(self, data, transformation_type):
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
        
        else:
            return data
    
    def run_cross_validation(self, file_path=None, df_prepared=None, order=None, seasonal_order=None, 
                           n_splits=5, regional_code=None, progress_callback=None, log_callback=None):
        """
        Cross-validation temporal con transformaciones por regional
        
        Args:
            regional_code: Código de la regional (e.g., 'SAIDI_C', 'SAIDI_O')
        """
        try:
            # NUEVO: Determinar transformación
            transformation = self._get_transformation_for_regional(regional_code)
            
            if log_callback:
                log_callback("=" * 80)
                log_callback("CROSS-VALIDATION TEMPORAL CON VENTANAS DESLIZANTES")
                log_callback("=" * 80)
                if regional_code:
                    log_callback(f"Regional: {regional_code} - Transformación: {transformation.upper()}")
            
            if progress_callback:
                progress_callback(10, "Cargando datos...")
            
            if df_prepared is not None:
                df = df_prepared.copy()
                if log_callback:
                    log_callback("Usando datos preparados del modelo")
            elif file_path is not None:
                df = pd.read_excel(file_path, sheet_name="Hoja1")
            else:
                raise Exception("Debe proporcionar file_path o df_prepared")
            
            if not isinstance(df.index, pd.DatetimeIndex):
                if "Fecha" in df.columns:
                    df["Fecha"] = pd.to_datetime(df["Fecha"])
                    df.set_index("Fecha", inplace=True)
                else:
                    df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
                    df.set_index(df.columns[0], inplace=True)
            
            col_saidi = None
            if "SAIDI" in df.columns:
                col_saidi = "SAIDI"
            elif "SAIDI Histórico" in df.columns:
                col_saidi = "SAIDI Histórico"
            
            if col_saidi is None:
                raise Exception("No se encontró la columna SAIDI")
            
            historico_original = df[df[col_saidi].notna()][col_saidi]
            
            if len(historico_original) < 30:
                raise Exception("Se necesitan al menos 30 observaciones para CV")
            
            if log_callback:
                log_callback(f"Datos históricos: {len(historico_original)} observaciones")
                log_callback(f"Configuración CV: {n_splits} folds con walk-forward validation")
            
            # NUEVO: Aplicar transformación a datos históricos
            if progress_callback:
                progress_callback(15, f"Aplicando transformación {transformation.upper()}...")
            
            historico_values_original = historico_original.values
            historico_transformed, transform_info = self._apply_transformation(
                historico_values_original, transformation
            )
            historico_transformed_series = pd.Series(historico_transformed, index=historico_original.index)
            
            if log_callback:
                log_callback(f"Transformación aplicada: {transform_info}")
            
            if progress_callback:
                progress_callback(20, "Configurando folds temporales...")
            
            n_total = len(historico_transformed_series)
            min_train = int(n_total * 0.5)
            test_size = max(6, int(n_total * 0.15))
            
            fold_results = []
            fold_rmse = []
            fold_mape = []
            fold_r2 = []
            
            for fold in range(n_splits):
                if progress_callback:
                    progress = 20 + (fold / n_splits) * 60
                    progress_callback(int(progress), f"Evaluando fold {fold+1}/{n_splits}...")
                
                train_end = min_train + fold * test_size
                test_end = min(train_end + test_size, n_total)
                
                if test_end > n_total or (test_end - train_end) < 6:
                    break
                
                # NUEVO: Trabajar con datos transformados
                train_data_transformed = historico_transformed_series[:train_end]
                test_data_transformed = historico_transformed_series[train_end:test_end]
                
                # Datos originales para cálculo de métricas
                train_data_original = historico_original[:train_end]
                test_data_original = historico_original[train_end:test_end]
                
                if log_callback:
                    log_callback(f"\nFold {fold+1}:")
                    log_callback(f"  Train: {len(train_data_transformed)} obs ({train_data_transformed.index[0].strftime('%Y-%m')} a {train_data_transformed.index[-1].strftime('%Y-%m')})")
                    log_callback(f"  Test:  {len(test_data_transformed)} obs ({test_data_transformed.index[0].strftime('%Y-%m')} a {test_data_transformed.index[-1].strftime('%Y-%m')})")
                
                try:
                    # Entrenar con datos transformados
                    model = SARIMAX(
                        train_data_transformed,
                        order=order,
                        seasonal_order=seasonal_order,
                        enforce_stationarity=True,
                        enforce_invertibility=True
                    )
                    results = model.fit(disp=False, maxiter=100)
                    
                    # Predecir en escala transformada
                    pred = results.get_forecast(steps=len(test_data_transformed))
                    pred_mean_transformed = pred.predicted_mean
                    
                    # NUEVO: Revertir a escala original para métricas
                    pred_mean_original = self._inverse_transformation(
                        pred_mean_transformed.values, transformation
                    )
                    
                    # Calcular métricas en escala original
                    rmse = np.sqrt(mean_squared_error(test_data_original, pred_mean_original))
                    mae = np.mean(np.abs(test_data_original - pred_mean_original))
                    
                    epsilon = 1e-8
                    mape = np.mean(np.abs((test_data_original - pred_mean_original) / (test_data_original + epsilon))) * 100
                    
                    ss_res = np.sum((test_data_original - pred_mean_original) ** 2)
                    ss_tot = np.sum((test_data_original - np.mean(test_data_original)) ** 2)
                    r2 = 1 - (ss_res / (ss_tot + epsilon))
                    
                    fold_results.append({
                        'fold': fold + 1,
                        'rmse': rmse,
                        'mae': mae,
                        'mape': mape,
                        'r2': r2,
                        'train_size': len(train_data_transformed),
                        'test_size': len(test_data_transformed)
                    })
                    
                    fold_rmse.append(rmse)
                    fold_mape.append(mape)
                    fold_r2.append(r2)
                    
                    if log_callback:
                        log_callback(f"  RMSE: {rmse:.4f} min | MAPE: {mape:.1f}% | R²: {r2:.3f} (escala original)")
                    
                except Exception as e:
                    if log_callback:
                        log_callback(f"  Error en fold {fold+1}: {str(e)}")
                    continue
            
            if len(fold_results) == 0:
                raise Exception("No se pudo completar ningún fold de CV")
            
            if progress_callback:
                progress_callback(85, "Calculando estadísticas de CV...")
            
            mean_rmse = np.mean(fold_rmse)
            std_rmse = np.std(fold_rmse)
            mean_mape = np.mean(fold_mape)
            std_mape = np.std(fold_mape)
            mean_r2 = np.mean(fold_r2)
            std_r2 = np.std(fold_r2)
            
            cv_rmse = (std_rmse / mean_rmse) if mean_rmse > 0 else 0
            stability_score = max(0, 100 - (cv_rmse * 100))
            
            if log_callback:
                log_callback("\n" + "=" * 80)
                log_callback("RESULTADOS DE CROSS-VALIDATION (Escala Original)")
                log_callback("=" * 80)
                log_callback(f"Folds completados: {len(fold_results)}")
                log_callback(f"\nRMSE: {mean_rmse:.4f} ± {std_rmse:.4f} minutos")
                log_callback(f"MAPE: {mean_mape:.1f}% ± {std_mape:.1f}%")
                log_callback(f"R²:   {mean_r2:.3f} ± {std_r2:.3f}")
                log_callback(f"\nScore de Estabilidad: {stability_score:.2f}/100")
                
                if stability_score >= 80:
                    log_callback("  ✅ EXCELENTE - Modelo muy estable")
                elif stability_score >= 60:
                    log_callback("  ✅ BUENO - Modelo estable")
                elif stability_score >= 40:
                    log_callback("  ⚠️ MODERADO - Estabilidad aceptable")
                else:
                    log_callback("  ❌ BAJO - Modelo inestable entre folds")
                
                log_callback("=" * 80)
            
            if progress_callback:
                progress_callback(95, "Generando gráfica de CV...")
            
            plot_path = self._generate_cv_plot(fold_results, order, seasonal_order, 
                                               mean_rmse, std_rmse, stability_score, transformation)
            
            if progress_callback:
                progress_callback(100, "Cross-validation completada")
            
            return {
                'success': True,
                'cv_results': {
                    'mean_rmse': mean_rmse,
                    'std_rmse': std_rmse,
                    'mean_mape': mean_mape,
                    'std_mape': std_mape,
                    'mean_r2': mean_r2,
                    'std_r2': std_r2,
                    'fold_scores': fold_results,
                    'stability_score': stability_score,
                    'transformation': transformation
                },
                'plot_files': {
                    'cv_plot': plot_path
                }
            }
            
        except Exception as e:
            if log_callback:
                log_callback(f"ERROR en CV: {str(e)}")
            raise Exception(f"Error en cross-validation: {str(e)}")
    
    def find_best_simple_model(self, file_path=None, df_prepared=None, 
                               regional_code=None, progress_callback=None, log_callback=None):
        """
        Búsqueda de modelo simple con transformaciones por regional
        
        Args:
            regional_code: Código de la regional
        """
        try:
            # NUEVO: Determinar transformación
            transformation = self._get_transformation_for_regional(regional_code)
            
            if log_callback:
                log_callback("=" * 80)
                log_callback("BÚSQUEDA DE MODELO SIMPLE ÓPTIMO")
                log_callback("=" * 80)
                if regional_code:
                    log_callback(f"Regional: {regional_code} - Transformación: {transformation.upper()}")
            
            if progress_callback:
                progress_callback(10, "Cargando datos...")
            
            if df_prepared is not None:
                df = df_prepared.copy()
            elif file_path is not None:
                df = pd.read_excel(file_path, sheet_name="Hoja1")
            else:
                raise Exception("Debe proporcionar file_path o df_prepared")
            
            if not isinstance(df.index, pd.DatetimeIndex):
                if "Fecha" in df.columns:
                    df["Fecha"] = pd.to_datetime(df["Fecha"])
                    df.set_index("Fecha", inplace=True)
                else:
                    df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
                    df.set_index(df.columns[0], inplace=True)
            
            col_saidi = "SAIDI" if "SAIDI" in df.columns else "SAIDI Histórico"
            historico_original = df[df[col_saidi].notna()][col_saidi]
            
            if log_callback:
                log_callback(f"Datos históricos: {len(historico_original)} observaciones")
            
            # NUEVO: Aplicar transformación
            if progress_callback:
                progress_callback(15, f"Aplicando transformación {transformation.upper()}...")
            
            historico_values_original = historico_original.values
            historico_transformed, transform_info = self._apply_transformation(
                historico_values_original, transformation
            )
            historico_transformed_series = pd.Series(historico_transformed, index=historico_original.index)
            
            if log_callback:
                log_callback(f"Transformación aplicada: {transform_info}")
                log_callback("Criterios de selección:")
                log_callback("  • Minimizar AIC/BIC")
                log_callback("  • Penalizar complejidad")
                log_callback("  • R² > 0.5")
                log_callback("  • Overfitting score < 20")
            
            p_range = range(0, 3)
            d_range = range(0, 2)
            q_range = range(0, 3)
            P_range = range(0, 2)
            D_range = range(0, 2)
            Q_range = range(0, 2)
            s = 12
            
            total_combinations = len(p_range) * len(d_range) * len(q_range) * len(P_range) * len(D_range) * len(Q_range)
            
            n_test = max(12, int(len(historico_original) * 0.20))
            
            # Datos transformados para entrenamiento
            train_data_transformed = historico_transformed_series[:-n_test]
            test_data_transformed = historico_transformed_series[-n_test:]
            
            # Datos originales para métricas
            train_data_original = historico_original[:-n_test]
            test_data_original = historico_original[-n_test:]
            
            models = []
            current = 0
            
            for p, d, q in product(p_range, d_range, q_range):
                for P, D, Q in product(P_range, D_range, Q_range):
                    current += 1
                    
                    if progress_callback:
                        progress = 15 + (current / total_combinations) * 70
                        progress_callback(int(progress), f"Evaluando {current}/{total_combinations}")
                    
                    order = (p, d, q)
                    seasonal_order = (P, D, Q, s)
                    complexity = p + d + q + P + D + Q
                    
                    if complexity > 6:
                        continue
                    
                    try:
                        # Entrenar con datos transformados
                        model_train = SARIMAX(
                            train_data_transformed,
                            order=order,
                            seasonal_order=seasonal_order,
                            enforce_stationarity=True,
                            enforce_invertibility=True
                        )
                        results_train = model_train.fit(disp=False, maxiter=100)
                        
                        # Predicciones transformadas
                        train_pred_transformed = results_train.fittedvalues
                        pred_test = results_train.get_forecast(steps=n_test)
                        test_pred_transformed = pred_test.predicted_mean
                        
                        # NUEVO: Revertir a escala original
                        train_pred_original = self._inverse_transformation(
                            train_pred_transformed.values, transformation
                        )
                        test_pred_original = self._inverse_transformation(
                            test_pred_transformed.values, transformation
                        )
                        
                        # Calcular métricas en escala original
                        train_rmse = np.sqrt(mean_squared_error(train_data_original, train_pred_original))
                        test_rmse = np.sqrt(mean_squared_error(test_data_original, test_pred_original))
                        
                        epsilon = 1e-8
                        train_mape = np.mean(np.abs((train_data_original - train_pred_original) / (train_data_original + epsilon))) * 100
                        test_mape = np.mean(np.abs((test_data_original - test_pred_original) / (test_data_original + epsilon))) * 100
                        
                        ss_res_train = np.sum((train_data_original - train_pred_original) ** 2)
                        ss_tot_train = np.sum((train_data_original - np.mean(train_data_original)) ** 2)
                        r2_train = 1 - (ss_res_train / (ss_tot_train + epsilon))
                        
                        ss_res_test = np.sum((test_data_original - test_pred_original) ** 2)
                        ss_tot_test = np.sum((test_data_original - np.mean(test_data_original)) ** 2)
                        r2_test = 1 - (ss_res_test / (ss_tot_test + epsilon))
                        
                        rmse_diff = abs(((test_rmse - train_rmse) / train_rmse) * 100)
                        mape_diff = abs(((test_mape - train_mape) / (train_mape + epsilon)) * 100)
                        overfitting_score = (rmse_diff + mape_diff) / 2
                        
                        if r2_train < 0.5 or r2_test < 0.5:
                            continue
                        if overfitting_score > 20:
                            continue
                        
                        aic = results_train.aic
                        bic = results_train.bic
                        
                        composite_score = (
                            aic * 0.3 + 
                            bic * 0.3 + 
                            overfitting_score * 0.2 + 
                            complexity * 10 * 0.2
                        )
                        
                        models.append({
                            'order': order,
                            'seasonal_order': seasonal_order,
                            'complexity': complexity,
                            'aic': aic,
                            'bic': bic,
                            'r2_train': r2_train,
                            'r2_test': r2_test,
                            'train_rmse': train_rmse,
                            'test_rmse': test_rmse,
                            'train_mape': train_mape,
                            'test_mape': test_mape,
                            'overfitting_score': overfitting_score,
                            'composite_score': composite_score
                        })
                        
                    except:
                        continue
            
            if len(models) == 0:
                raise Exception("No se encontraron modelos que cumplan los criterios")
            
            models.sort(key=lambda x: x['composite_score'])
            top_models = models[:5]
            
            if log_callback:
                log_callback("\n" + "=" * 80)
                log_callback("TOP 5 MODELOS SIMPLES (Métricas en Escala Original)")
                log_callback("=" * 80)
                
                for i, m in enumerate(top_models, 1):
                    log_callback(f"\n#{i} - order={m['order']}, seasonal={m['seasonal_order']}")
                    log_callback(f"  Complejidad: {m['complexity']} parámetros")
                    log_callback(f"  AIC: {m['aic']:.1f} | BIC: {m['bic']:.1f}")
                    log_callback(f"  R² Train: {m['r2_train']:.3f} | R² Test: {m['r2_test']:.3f}")
                    log_callback(f"  RMSE Train: {m['train_rmse']:.4f} | Test: {m['test_rmse']:.4f} min")
                    log_callback(f"  Overfitting Score: {m['overfitting_score']:.2f}/100")
                
                log_callback("\n" + "=" * 80)
            
            if progress_callback:
                progress_callback(95, "Generando gráfica comparativa...")
            
            plot_path = self._generate_model_comparison_plot(top_models, transformation)
            
            if progress_callback:
                progress_callback(100, "Búsqueda completada")
            
            return {
                'success': True,
                'best_model': top_models[0],
                'top_models': top_models,
                'transformation': transformation,
                'plot_files': {
                    'model_comparison_plot': plot_path
                }
            }
            
        except Exception as e:
            if log_callback:
                log_callback(f"ERROR en búsqueda: {str(e)}")
            raise Exception(f"Error en búsqueda de modelo: {str(e)}")
    
    def compare_transformations(self, file_path=None, df_prepared=None, order=None, seasonal_order=None,
                               regional_code=None, progress_callback=None, log_callback=None):
        """
        Comparar transformaciones (incluye la asignada a la regional)
        
        Args:
            regional_code: Código de la regional
        """
        try:
            # NUEVO: Obtener transformación asignada
            assigned_transformation = self._get_transformation_for_regional(regional_code)
            
            if log_callback:
                log_callback("=" * 80)
                log_callback("COMPARACIÓN DE TRANSFORMACIONES DE DATOS")
                log_callback("=" * 80)
                if regional_code:
                    log_callback(f"Regional: {regional_code}")
                    log_callback(f"Transformación asignada actual: {assigned_transformation.upper()}")
            
            if progress_callback:
                progress_callback(10, "Cargando datos...")
            
            if df_prepared is not None:
                df = df_prepared.copy()
            elif file_path is not None:
                df = pd.read_excel(file_path, sheet_name="Hoja1")
            else:
                raise Exception("Debe proporcionar file_path o df_prepared")
            
            if not isinstance(df.index, pd.DatetimeIndex):
                if "Fecha" in df.columns:
                    df["Fecha"] = pd.to_datetime(df["Fecha"])
                    df.set_index("Fecha", inplace=True)
                else:
                    df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
                    df.set_index(df.columns[0], inplace=True)
            
            col_saidi = "SAIDI" if "SAIDI" in df.columns else "SAIDI Histórico"
            historico_original = df[df[col_saidi].notna()][col_saidi]
            
            if log_callback:
                log_callback(f"Datos originales: {len(historico_original)} observaciones")
            
            n_test = max(12, int(len(historico_original) * 0.20))
            
            transformations = {}
            
            # Evaluar todas las transformaciones
            if progress_callback:
                progress_callback(20, "Evaluando datos originales...")
            
            if log_callback:
                log_callback("\n1. Evaluando datos ORIGINALES...")
            
            result_original = self._evaluate_transformation(
                historico_original, order, seasonal_order, n_test, "original"
            )
            transformations['original'] = result_original
            
            if log_callback:
                marker = "★" if assigned_transformation == 'original' else "○"
                log_callback(f"  {marker} Overfitting Score: {result_original['overfitting_score']:.2f}")
            
            # StandardScaler
            if progress_callback:
                progress_callback(40, "Evaluando normalización...")
            
            if log_callback:
                log_callback("\n2. Evaluando datos NORMALIZADOS (StandardScaler)...")
            
            scaler = StandardScaler()
            historico_scaled = pd.Series(
                scaler.fit_transform(historico_original.values.reshape(-1, 1)).flatten(),
                index=historico_original.index
            )
            
            result_scaled = self._evaluate_transformation(
                historico_scaled, order, seasonal_order, n_test, "standard"
            )
            transformations['standard'] = result_scaled
            
            if log_callback:
                marker = "★" if assigned_transformation == 'standard' else "○"
                log_callback(f"  {marker} Overfitting Score: {result_scaled['overfitting_score']:.2f}")
            
            # Log
            if progress_callback:
                progress_callback(60, "Evaluando transformación logarítmica...")
            
            if log_callback:
                log_callback("\n3. Evaluando transformación LOGARÍTMICA...")
            
            if (historico_original > 0).all():
                historico_log = pd.Series(
                    np.log(historico_original.values),
                    index=historico_original.index
                )
                
                result_log = self._evaluate_transformation(
                    historico_log, order, seasonal_order, n_test, "log"
                )
                transformations['log'] = result_log
                
                if log_callback:
                    marker = "★" if assigned_transformation == 'log' else "○"
                    log_callback(f"  {marker} Overfitting Score: {result_log['overfitting_score']:.2f}")
            else:
                if log_callback:
                    log_callback("  OMITIDA - Datos contienen valores no positivos")
            
            # Box-Cox
            if progress_callback:
                progress_callback(80, "Evaluando transformación Box-Cox...")
            
            if log_callback:
                log_callback("\n4. Evaluando transformación BOX-COX...")
            
            if (historico_original > 0).all():
                try:
                    historico_boxcox, lambda_param = boxcox(historico_original.values)
                    historico_boxcox = pd.Series(historico_boxcox, index=historico_original.index)
                    
                    result_boxcox = self._evaluate_transformation(
                        historico_boxcox, order, seasonal_order, n_test, "boxcox"
                    )
                    result_boxcox['lambda'] = lambda_param
                    transformations['boxcox'] = result_boxcox
                    
                    if log_callback:
                        marker = "★" if assigned_transformation == 'boxcox' else "○"
                        log_callback(f"  {marker} Lambda: {lambda_param:.4f}")
                        log_callback(f"  {marker} Overfitting Score: {result_boxcox['overfitting_score']:.2f}")
                except:
                    if log_callback:
                        log_callback("  ERROR - No se pudo aplicar Box-Cox")
            else:
                if log_callback:
                    log_callback("  OMITIDA - Datos contienen valores no positivos")
            
            best_method = min(transformations.keys(), 
                            key=lambda k: transformations[k]['overfitting_score'])
            best_result = transformations[best_method]
            
            original_score = transformations['original']['overfitting_score']
            best_score = best_result['overfitting_score']
            improvement = ((original_score - best_score) / original_score) * 100
            
            if log_callback:
                log_callback("\n" + "=" * 80)
                log_callback("RESULTADOS DE COMPARACIÓN (Escala Original)")
                log_callback("=" * 80)
                log_callback(f"Transformación asignada actualmente: {assigned_transformation.upper()}")
                log_callback(f"Mejor transformación encontrada: {best_method.upper()}")
                log_callback(f"Overfitting Score mejor: {best_score:.2f}/100")
                log_callback(f"Mejora respecto a original: {improvement:.1f}%")
                log_callback(f"R² Train: {best_result['r2_train']:.3f} | Test: {best_result['r2_test']:.3f}")
                
                if best_method != assigned_transformation:
                    log_callback(f"\n⚠️ NOTA: La transformación óptima ({best_method.upper()}) difiere de la asignada ({assigned_transformation.upper()})")
                else:
                    log_callback(f"\n✅ La transformación asignada es óptima para esta regional")
                
                log_callback("=" * 80)
            
            if progress_callback:
                progress_callback(95, "Generando gráfica comparativa...")
            
            plot_path = self._generate_transformation_plot(transformations, best_method, assigned_transformation)
            
            if progress_callback:
                progress_callback(100, "Comparación completada")
            
            return {
                'success': True,
                'best_transformation': {
                    'method': best_method,
                    'overfitting_improvement': improvement,
                    'final_overfitting_score': best_score,
                    'r2_train': best_result['r2_train'],
                    'r2_test': best_result['r2_test']
                },
                'assigned_transformation': assigned_transformation,
                'all_transformations': transformations,
                'plot_files': {
                    'transformation_plot': plot_path
                }
            }
            
        except Exception as e:
            if log_callback:
                log_callback(f"ERROR en transformaciones: {str(e)}")
            raise Exception(f"Error en comparación de transformaciones: {str(e)}")
    
    def _evaluate_transformation(self, data, order, seasonal_order, n_test, method_name):
        """Evaluar una transformación (métricas en escala original para transformaciones aplicadas)"""
        try:
            train_data = data[:-n_test]
            test_data = data[-n_test:]
            
            model = SARIMAX(
                train_data,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=True,
                enforce_invertibility=True
            )
            results = model.fit(disp=False, maxiter=100)
            
            train_pred = results.fittedvalues
            train_rmse = np.sqrt(mean_squared_error(train_data, train_pred))
            
            epsilon = 1e-8
            train_mape = np.mean(np.abs((train_data - train_pred) / (train_data + epsilon))) * 100
            
            ss_res_train = np.sum((train_data - train_pred) ** 2)
            ss_tot_train = np.sum((train_data - np.mean(train_data)) ** 2)
            r2_train = 1 - (ss_res_train / (ss_tot_train + epsilon))
            
            pred = results.get_forecast(steps=n_test)
            test_pred = pred.predicted_mean
            test_rmse = np.sqrt(mean_squared_error(test_data, test_pred))
            test_mape = np.mean(np.abs((test_data - test_pred) / (test_data + epsilon))) * 100
            
            ss_res_test = np.sum((test_data - test_pred) ** 2)
            ss_tot_test = np.sum((test_data - np.mean(test_data)) ** 2)
            r2_test = 1 - (ss_res_test / (ss_tot_test + epsilon))
            
            rmse_diff = abs(((test_rmse - train_rmse) / train_rmse) * 100)
            mape_diff = abs(((test_mape - train_mape) / (train_mape + epsilon)) * 100)
            r2_diff = abs((r2_train - r2_test) * 100)
            
            overfitting_score = (rmse_diff + mape_diff + r2_diff) / 3
            
            return {
                'method': method_name,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'train_mape': train_mape,
                'test_mape': test_mape,
                'r2_train': r2_train,
                'r2_test': r2_test,
                'overfitting_score': overfitting_score,
                'aic': results.aic,
                'bic': results.bic
            }
            
        except Exception as e:
            return {
                'method': method_name,
                'train_rmse': float('inf'),
                'test_rmse': float('inf'),
                'train_mape': 100,
                'test_mape': 100,
                'r2_train': -1,
                'r2_test': -1,
                'overfitting_score': 100,
                'aic': float('inf'),
                'bic': float('inf')
            }
    
    def _generate_cv_plot(self, fold_results, order, seasonal_order, mean_rmse, std_rmse, stability_score, transformation):
        """Generar gráfica de cross-validation con información de transformación"""
        try:
            temp_dir = tempfile.gettempdir()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_path = os.path.join(temp_dir, f"cv_analysis_{timestamp}.png")
            
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            folds = [r['fold'] for r in fold_results]
            rmse_values = [r['rmse'] for r in fold_results]
            
            axes[0, 0].bar(folds, rmse_values, color='steelblue', alpha=0.7)
            axes[0, 0].axhline(y=mean_rmse, color='red', linestyle='--', linewidth=2, label=f'Media: {mean_rmse:.4f}')
            axes[0, 0].axhline(y=mean_rmse + std_rmse, color='orange', linestyle=':', alpha=0.7, label=f'±1σ')
            axes[0, 0].axhline(y=mean_rmse - std_rmse, color='orange', linestyle=':', alpha=0.7)
            axes[0, 0].set_xlabel('Fold', fontweight='bold')
            axes[0, 0].set_ylabel('RMSE (minutos)', fontweight='bold')
            axes[0, 0].set_title('RMSE por Fold - Escala Original', fontsize=14, fontweight='bold')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            mape_values = [r['mape'] for r in fold_results]
            axes[0, 1].plot(folds, mape_values, marker='o', linewidth=2, markersize=8, color='coral')
            axes[0, 1].fill_between(folds, mape_values, alpha=0.3, color='coral')
            axes[0, 1].set_xlabel('Fold', fontweight='bold')
            axes[0, 1].set_ylabel('MAPE (%)', fontweight='bold')
            axes[0, 1].set_title('MAPE por Fold - Escala Original', fontsize=14, fontweight='bold')
            axes[0, 1].grid(True, alpha=0.3)
            
            r2_values = [r['r2'] for r in fold_results]
            axes[1, 0].scatter(folds, r2_values, s=150, c=r2_values, cmap='RdYlGn', edgecolors='black', linewidth=1.5)
            axes[1, 0].axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Umbral mínimo (0.5)')
            axes[1, 0].set_xlabel('Fold', fontweight='bold')
            axes[1, 0].set_ylabel('R²', fontweight='bold')
            axes[1, 0].set_title('R² por Fold - Escala Original', fontsize=14, fontweight='bold')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            axes[1, 1].axis('off')
            
            stability_color = 'green' if stability_score >= 80 else 'orange' if stability_score >= 60 else 'red'
            
            summary_text = f"""
CROSS-VALIDATION TEMPORAL
Modelo: SARIMAX{order}x{seasonal_order}
Transformación: {transformation.upper()}

ESTADÍSTICAS AGREGADAS:
- Folds completados: {len(fold_results)}
- RMSE: {mean_rmse:.4f} ± {std_rmse:.4f} min
- MAPE: {np.mean(mape_values):.2f}% ± {np.std(mape_values):.2f}%
- R²: {np.mean(r2_values):.3f} ± {np.std(r2_values):.3f}

ESTABILIDAD DEL MODELO:
- Score: {stability_score:.2f}/100
- CV RMSE: {(std_rmse/mean_rmse)*100:.2f}%

INTERPRETACIÓN:
"""
            if stability_score >= 80:
                summary_text += "✅ EXCELENTE - Modelo muy estable\n"
            elif stability_score >= 60:
                summary_text += "✅ BUENO - Modelo estable\n"
            elif stability_score >= 40:
                summary_text += "⚠️ MODERADO - Estabilidad aceptable\n"
            else:
                summary_text += "❌ BAJO - Modelo inestable\n"
            
            summary_text += "\nMétricas calculadas en escala original\nWalk-Forward Validation temporal"
            
            axes[1, 1].text(0.1, 0.9, summary_text, transform=axes[1, 1].transAxes,
                          fontsize=11, verticalalignment='top', family='monospace',
                          bbox=dict(boxstyle='round,pad=1', facecolor=stability_color, 
                                   alpha=0.2, edgecolor=stability_color, linewidth=2))
            
            plt.suptitle(f'Análisis de Cross-Validation Temporal - {transformation.upper()}', 
                        fontsize=16, fontweight='bold', y=0.995)
            plt.tight_layout()
            plt.savefig(plot_path, dpi=100, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            
            self.plot_files.append(plot_path)
            return plot_path
            
        except Exception as e:
            print(f"Error generando gráfica CV: {e}")
            return None
    
    def _generate_model_comparison_plot(self, top_models, transformation):
        """Generar gráfica comparativa de modelos con información de transformación"""
        try:
            temp_dir = tempfile.gettempdir()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_path = os.path.join(temp_dir, f"model_comparison_{timestamp}.png")
            
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            models_labels = [f"M{i+1}" for i in range(len(top_models))]
            
            aic_values = [m['aic'] for m in top_models]
            bic_values = [m['bic'] for m in top_models]
            
            x = np.arange(len(models_labels))
            width = 0.35
            
            axes[0, 0].bar(x - width/2, aic_values, width, label='AIC', color='skyblue')
            axes[0, 0].bar(x + width/2, bic_values, width, label='BIC', color='lightcoral')
            axes[0, 0].set_xlabel('Modelo', fontweight='bold')
            axes[0, 0].set_ylabel('Criterio de Información', fontweight='bold')
            axes[0, 0].set_title('AIC y BIC (menor es mejor)', fontsize=14, fontweight='bold')
            axes[0, 0].set_xticks(x)
            axes[0, 0].set_xticklabels(models_labels)
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3, axis='y')
            
            overfitting_scores = [m['overfitting_score'] for m in top_models]
            colors = ['green' if s < 10 else 'orange' if s < 20 else 'red' for s in overfitting_scores]
            
            axes[0, 1].bar(models_labels, overfitting_scores, color=colors, alpha=0.7)
            axes[0, 1].axhline(y=10, color='green', linestyle='--', alpha=0.5, label='Umbral óptimo')
            axes[0, 1].axhline(y=20, color='red', linestyle='--', alpha=0.5, label='Umbral máximo')
            axes[0, 1].set_xlabel('Modelo', fontweight='bold')
            axes[0, 1].set_ylabel('Overfitting Score', fontweight='bold')
            axes[0, 1].set_title('Score de Overfitting (menor es mejor)', fontsize=14, fontweight='bold')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3, axis='y')
            
            r2_train = [m['r2_train'] for m in top_models]
            r2_test = [m['r2_test'] for m in top_models]
            
            axes[1, 0].plot(models_labels, r2_train, marker='o', linewidth=2, markersize=10, 
                          label='R² Train', color='blue')
            axes[1, 0].plot(models_labels, r2_test, marker='s', linewidth=2, markersize=10, 
                          label='R² Test', color='red')
            axes[1, 0].axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
            axes[1, 0].set_xlabel('Modelo', fontweight='bold')
            axes[1, 0].set_ylabel('R²', fontweight='bold')
            axes[1, 0].set_title('R² Training vs Test - Escala Original', fontsize=14, fontweight='bold')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            axes[1, 1].axis('off')
            
            details_text = f"TOP 5 MODELOS SIMPLES\nTransformación: {transformation.upper()}\n\n"
            for i, m in enumerate(top_models, 1):
                details_text += f"M{i}: order={m['order']}, seasonal={m['seasonal_order']}\n"
                details_text += f"    Complejidad: {m['complexity']} params\n"
                details_text += f"    Overfitting: {m['overfitting_score']:.2f}\n"
                details_text += f"    AIC: {m['aic']:.1f} | BIC: {m['bic']:.1f}\n\n"
            
            details_text += f"MEJOR MODELO: M1\n"
            details_text += f"Balance óptimo entre simplicidad,\n"
            details_text += f"ajuste y generalización\n\n"
            details_text += f"Métricas en escala original (minutos)"
            
            axes[1, 1].text(0.05, 0.95, details_text, transform=axes[1, 1].transAxes,
                          fontsize=10, verticalalignment='top', family='monospace',
                          bbox=dict(boxstyle='round,pad=0.8', facecolor='lightgreen', 
                                   alpha=0.3, edgecolor='green', linewidth=2))
            
            plt.suptitle(f'Comparación de Modelos Simples - {transformation.upper()}', 
                        fontsize=16, fontweight='bold', y=0.995)
            plt.tight_layout()
            plt.savefig(plot_path, dpi=100, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            
            self.plot_files.append(plot_path)
            return plot_path
            
        except Exception as e:
            print(f"Error generando gráfica de comparación: {e}")
            return None
    
    def _generate_transformation_plot(self, transformations, best_method, assigned_method):
        """Generar gráfica comparativa de transformaciones con asignación actual"""
        try:
            temp_dir = tempfile.gettempdir()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_path = os.path.join(temp_dir, f"transformation_comparison_{timestamp}.png")
            
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            methods = list(transformations.keys())
            method_labels = [m.upper() for m in methods]
            
            overfitting_scores = [transformations[m]['overfitting_score'] for m in methods]
            colors = []
            for m in methods:
                if m == best_method and m == assigned_method:
                    colors.append('gold')  # Mejor Y asignado
                elif m == best_method:
                    colors.append('lightgreen')  # Mejor pero no asignado
                elif m == assigned_method:
                    colors.append('lightblue')  # Asignado pero no mejor
                else:
                    colors.append('lightgray')
            
            bars = axes[0, 0].bar(method_labels, overfitting_scores, color=colors, 
                                 edgecolor='black', linewidth=1.5, alpha=0.8)
            
            best_idx = methods.index(best_method)
            bars[best_idx].set_edgecolor('green')
            bars[best_idx].set_linewidth(3)
            
            if assigned_method != best_method and assigned_method in methods:
                assigned_idx = methods.index(assigned_method)
                bars[assigned_idx].set_edgecolor('blue')
                bars[assigned_idx].set_linewidth(3)
                bars[assigned_idx].set_linestyle('--')
            
            axes[0, 0].set_ylabel('Overfitting Score', fontweight='bold')
            axes[0, 0].set_title('Overfitting Score por Transformación (menor es mejor)', 
                               fontsize=14, fontweight='bold')
            axes[0, 0].grid(True, alpha=0.3, axis='y')
            
            for i, (bar, score) in enumerate(zip(bars, overfitting_scores)):
                height = bar.get_height()
                axes[0, 0].text(bar.get_x() + bar.get_width()/2., height,
                              f'{score:.2f}', ha='center', va='bottom', fontweight='bold')
            
            r2_train = [transformations[m]['r2_train'] for m in methods]
            r2_test = [transformations[m]['r2_test'] for m in methods]
            
            x = np.arange(len(method_labels))
            width = 0.35
            
            axes[0, 1].bar(x - width/2, r2_train, width, label='R² Train', color='steelblue')
            axes[0, 1].bar(x + width/2, r2_test, width, label='R² Test', color='coral')
            axes[0, 1].axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
            axes[0, 1].set_ylabel('R²', fontweight='bold')
            axes[0, 1].set_title('R² Training vs Test por Transformación', fontsize=14, fontweight='bold')
            axes[0, 1].set_xticks(x)
            axes[0, 1].set_xticklabels(method_labels)
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3, axis='y')
            
            train_rmse = [transformations[m]['train_rmse'] for m in methods]
            test_rmse = [transformations[m]['test_rmse'] for m in methods]
            
            axes[1, 0].plot(method_labels, train_rmse, marker='o', linewidth=2, 
                          markersize=10, label='RMSE Train', color='blue')
            axes[1, 0].plot(method_labels, test_rmse, marker='s', linewidth=2, 
                          markersize=10, label='RMSE Test', color='red')
            axes[1, 0].set_ylabel('RMSE', fontweight='bold')
            axes[1, 0].set_title('RMSE Training vs Test', fontsize=14, fontweight='bold')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            axes[1, 1].axis('off')
            
            best_result = transformations[best_method]
            original_score = transformations['original']['overfitting_score']
            improvement = ((original_score - best_result['overfitting_score']) / original_score) * 100
            
            summary_text = f"""
COMPARACIÓN DE TRANSFORMACIONES

ASIGNADA ACTUALMENTE:
- Método: {assigned_method.upper()}
- Score: {transformations[assigned_method]['overfitting_score']:.2f}

MEJOR ENCONTRADA:
- Método: {best_method.upper()}
- Overfitting Score: {best_result['overfitting_score']:.2f}/100
- Mejora vs original: {improvement:+.1f}%

MÉTRICAS DEL MEJOR:
- R² Train: {best_result['r2_train']:.3f}
- R² Test: {best_result['r2_test']:.3f}
- RMSE Train: {best_result['train_rmse']:.4f}
- RMSE Test: {best_result['test_rmse']:.4f}

TODAS LAS TRANSFORMACIONES:
"""
            for m in methods:
                score = transformations[m]['overfitting_score']
                if m == best_method and m == assigned_method:
                    marker = "★✓"
                elif m == best_method:
                    marker = "★"
                elif m == assigned_method:
                    marker = "✓"
                else:
                    marker = "○"
                summary_text += f"{marker} {m.upper()}: {score:.2f}\n"
            
            if best_method != assigned_method:
                summary_text += f"\n⚠️ Considerar cambiar a {best_method.upper()}"
            else:
                summary_text += f"\n✅ Asignación actual es óptima"
            
            axes[1, 1].text(0.05, 0.95, summary_text, transform=axes[1, 1].transAxes,
                          fontsize=10, verticalalignment='top', family='monospace',
                          bbox=dict(boxstyle='round,pad=0.8', facecolor='lightyellow', 
                                   alpha=0.8, edgecolor='orange', linewidth=2))
            
            plt.suptitle('Comparación de Transformaciones de Datos', fontsize=16, fontweight='bold', y=0.995)
            plt.tight_layout()
            plt.savefig(plot_path, dpi=100, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            
            self.plot_files.append(plot_path)
            return plot_path
            
        except Exception as e:
            print(f"Error generando gráfica de transformaciones: {e}")
            return None
    
    def cleanup_plot_files(self):
        """Limpiar archivos temporales de gráficas"""
        for plot_path in self.plot_files:
            if plot_path and os.path.exists(plot_path):
                try:
                    os.remove(plot_path)
                except Exception as e:
                    print(f"Error eliminando archivo: {e}")
        self.plot_files = []