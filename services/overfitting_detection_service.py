# services/overfitting_detection_service.py - Servicio de detección de overfitting con transformaciones por regional
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from scipy import stats
import sys
import os
import tempfile
from datetime import datetime

class OverfittingDetectionService:
    """Servicio para detectar overfitting en modelos SAIDI con transformaciones por regional"""
    
    # NUEVO: Mapeo de regionales a sus transformaciones óptimas (igual que prediction_service)
    REGIONAL_TRANSFORMATIONS = {
        'SAIDI_O': 'boxcox',      # Ocaña - BoxCox
        'SAIDI_C': 'original',            # Cúcuta - Original
        'SAIDI_A': 'original',       # Aguachica - Original
        'SAIDI_P': 'boxcox',            # Pamplona - Boxcox
        'SAIDI_T': 'sqrt',         # Tibú - Sqrt
        'SAIDI_Cens': 'original'          # Cens - Original
    }
    
    def __init__(self):
        self.default_order = (3, 0, 3)
        self.default_seasonal_order = (3, 1, 3, 12)
        self.plot_file_path = None
        self.scaler = None
        self.transformation_params = {}
    
    def run_overfitting_detection(self, file_path=None, df_prepared=None, order=None, 
                                 seasonal_order=None, regional_code=None, 
                                 progress_callback=None, log_callback=None):
        """
        Detectar overfitting con transformación específica por regional
        
        Args:
            regional_code: Código de la regional (e.g., 'SAIDI_C', 'SAIDI_O')
        """
        try:
            if order is None:
                order = self.default_order
            if seasonal_order is None:
                seasonal_order = self.default_seasonal_order
            
            # NUEVO: Determinar transformación a usar
            transformation = self._get_transformation_for_regional(regional_code)
            
            if log_callback:
                log_callback(f"Iniciando detección de overfitting con order={order}, seasonal={seasonal_order}")
                log_callback(f"Regional: {regional_code} - Transformación: {transformation.upper()}")
            
            if progress_callback:
                progress_callback(10, "Cargando datos...")
            
            # Cargar datos
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
                progress_callback(20, f"Aplicando transformación {transformation.upper()}...")
            
            # NUEVO: Aplicar transformación
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
            
            if progress_callback:
                progress_callback(40, "Entrenando modelo con datos de training...")
            
            # Entrenar modelo con datos transformados
            try:
                model = SARIMAX(
                    train_transformed,
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=True,
                    enforce_invertibility=True
                )
                results = model.fit(disp=False)
                
                if log_callback:
                    log_callback(f"Modelo entrenado exitosamente con datos transformados")
                    log_callback(f"AIC: {results.aic:.2f}, BIC: {results.bic:.2f}")
                    
            except Exception as e:
                raise Exception(f"Error entrenando modelo: {str(e)}")
            
            if progress_callback:
                progress_callback(60, "Calculando métricas en cada conjunto...")
            
            # NUEVO: Calcular métricas en escala original
            metrics_train = self._calculate_metrics_original_scale(
                train_transformed, train_original, results, transformation, "Training"
            )
            metrics_val = self._calculate_metrics_original_scale(
                val_transformed, val_original, results, transformation, "Validation"
            )
            metrics_test = self._calculate_metrics_original_scale(
                test_transformed, test_original, results, transformation, "Test"
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
            
            if progress_callback:
                progress_callback(80, "Analizando overfitting...")
            
            # Análisis de overfitting
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
                log_callback(f"Degradación Train→Test: {overfitting_analysis['degradation_pct']:.1f}%")
                
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
                overfitting_analysis, order, seasonal_order, transformation
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
                    'regional_code': regional_code
                },
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
    
    def _get_transformation_for_regional(self, regional_code):
        """Obtener la transformación correspondiente a la regional"""
        if regional_code and regional_code in self.REGIONAL_TRANSFORMATIONS:
            return self.REGIONAL_TRANSFORMATIONS[regional_code]
        return 'original'
    
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
    
    def _calculate_metrics_original_scale(self, data_transformed, data_original, 
                                         model_results, transformation, label):
        """Calcular métricas en escala original después de revertir transformación"""
        try:
            # Para Training: usar fittedvalues (in-sample)
            if label == "Training":
                fitted_transformed = model_results.fittedvalues
                # Alinear con los datos de training
                fitted_transformed = fitted_transformed.reindex(data_transformed.index)
            
            # Para Validation/Test: usar get_prediction con índices específicos
            else:
                # Predecir específicamente para este rango de fechas
                pred = model_results.get_prediction(
                    start=data_transformed.index[0],
                    end=data_transformed.index[-1]
                )
                fitted_transformed = pred.predicted_mean
            
            # Verificar que no haya NaN
            if fitted_transformed.isna().any():
                raise ValueError(f"Predicciones contienen NaN para {label}")
            
            # NUEVO: Revertir a escala original
            fitted_original = self._inverse_transformation(
                fitted_transformed.values, transformation
            )
            
            # Verificar dimensiones
            if len(fitted_original) != len(data_original):
                raise ValueError(f"Dimensiones no coinciden: {len(fitted_original)} vs {len(data_original)}")
            
            # Calcular métricas en escala original
            rmse = np.sqrt(mean_squared_error(data_original, fitted_original))
            mae = mean_absolute_error(data_original, fitted_original)
            r2 = r2_score(data_original, fitted_original)
            
            epsilon = 1e-8
            mape = np.mean(np.abs((data_original - fitted_original) / 
                                 (data_original.values + epsilon))) * 100
            
            return {
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'mape': mape,
                'fitted_values': fitted_original  # En escala original
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
                'fitted_values': None
            }
    
    def _analyze_overfitting(self, metrics_train, metrics_val, metrics_test, log_callback=None):
        """Analizar presencia y severidad de overfitting"""
        
        # Calcular degradación de rendimiento
        r2_train = metrics_train['r2']
        r2_val = metrics_val['r2']
        r2_test = metrics_test['r2']
        
        rmse_train = metrics_train['rmse']
        rmse_test = metrics_test['rmse']
        
        # Degradación porcentual
        r2_degradation = ((r2_train - r2_test) / max(abs(r2_train), 1e-8)) * 100
        rmse_increase = ((rmse_test - rmse_train) / max(rmse_train, 1e-8)) * 100
        
        # Score de overfitting (0-100, mayor = más overfitting)
        overfitting_score = 0
        
        # Factor 1: Diferencia de R² (peso 40%)
        r2_diff = abs(r2_train - r2_test)
        overfitting_score += min(r2_diff * 100, 40)
        
        # Factor 2: Aumento de RMSE (peso 30%)
        overfitting_score += min(rmse_increase * 0.3, 30)
        
        # Factor 3: Validación intermedia (peso 30%)
        if r2_val < r2_test:
            val_penalty = abs(r2_val - r2_test) * 100
            overfitting_score += min(val_penalty, 30)
        
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
            if r2_degradation > 20:
                recommendations.append("Considerar regularización o validación cruzada")
            recommendations.append("Evaluar transformación alternativa de datos")
        
        return {
            'overfitting_score': overfitting_score,
            'overfitting_level': level,
            'status': status,
            'has_overfitting': has_overfitting,
            'r2_degradation': r2_degradation,
            'rmse_increase': rmse_increase,
            'degradation_pct': r2_degradation,
            'metrics_comparison': {
                'r2': {'train': r2_train, 'val': r2_val, 'test': r2_test},
                'rmse': {'train': rmse_train, 'test': rmse_test}
            },
            'recommendations': recommendations
        }
    
    def _generate_overfitting_plot(self, train_orig, val_orig, test_orig,
                                   train_trans, val_trans, test_trans,
                                   model_results, metrics_train, metrics_val, 
                                   metrics_test, overfitting_analysis, 
                                   order, seasonal_order, transformation):
        """Generar gráfica de análisis de overfitting con valores en escala original"""
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
            
            ax1.set_title(f'Real vs Predicción (Escala Original - minutos SAIDI)', 
                         fontsize=14, fontweight='bold')
            ax1.set_xlabel('Fecha', fontsize=11)
            ax1.set_ylabel('SAIDI (minutos)', fontsize=11)
            ax1.legend(fontsize=9, loc='best')
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Comparación de métricas por conjunto
            ax2 = plt.subplot(2, 2, 2)
            
            conjuntos = ['Training', 'Validation', 'Test']
            r2_values = [metrics_train['r2'], metrics_val['r2'], metrics_test['r2']]
            rmse_values = [metrics_train['rmse'], metrics_val['rmse'], metrics_test['rmse']]
            
            x_pos = np.arange(len(conjuntos))
            width = 0.35
            
            # Normalizar RMSE para visualización
            max_rmse = max(rmse_values)
            rmse_norm = [r/max_rmse for r in rmse_values]
            
            bars1 = ax2.bar(x_pos - width/2, r2_values, width, 
                           label='R² Score', color='steelblue', alpha=0.8)
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
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.3f}', ha='center', va='bottom', fontsize=8)
            
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
            
            # Plot 4: Panel de diagnóstico
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
  • R²: {metrics_train['r2']:.4f}
  • MAPE: {metrics_train['mape']:.2f}%

Validation:
  • RMSE: {metrics_val['rmse']:.4f} min
  • R²: {metrics_val['r2']:.4f}
  • MAPE: {metrics_val['mape']:.2f}%

Test:
  • RMSE: {metrics_test['rmse']:.4f} min
  • R²: {metrics_test['r2']:.4f}
  • MAPE: {metrics_test['mape']:.2f}%

DEGRADACIÓN
{'=' * 50}
R² Train→Test: {overfitting_analysis['r2_degradation']:.1f}%
RMSE Aumento: {overfitting_analysis['rmse_increase']:.1f}%

Transformación: {transformation.upper()}
Modelo: SARIMAX{order}x{seasonal_order}
"""
            
            ax4.text(0.05, 0.95, info_text, transform=ax4.transAxes,
                    fontsize=10, verticalalignment='top', family='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
            
            # Indicador visual de overfitting
            circle = plt.Circle((0.85, 0.5), 0.08, color=color, alpha=0.6)
            ax4.add_patch(circle)
            ax4.text(0.85, 0.5, f'{score:.0f}', ha='center', va='center',
                    fontsize=20, fontweight='bold', color='white')
            
            # Título general
            plt.suptitle(f'Análisis de Overfitting - SARIMAX{order}x{seasonal_order} + {transformation.upper()}',
                        fontsize=16, fontweight='bold', y=0.98)
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.96])
            
            # Nota al pie
            plt.figtext(0.5, 0.01, 
                       'Todas las métricas calculadas en escala original (minutos SAIDI)',
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