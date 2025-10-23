# services/validation_service.py - Servicio de validación con transformaciones por regional
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from scipy import stats
import sys
import os
import tempfile
from datetime import datetime

class ValidationService:
    """Servicio para validar modelos SARIMAX con transformaciones por regional"""
    
    # NUEVO: Mapeo de regionales a sus transformaciones óptimas
    REGIONAL_TRANSFORMATIONS = {
        'SAIDI_O': 'boxcox',      # Ocaña - Boxcox
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
        self.transformation_params = {}  # Para guardar parámetros de transformación
    
    def run_validation(self, file_path=None, df_prepared=None, order=None, seasonal_order=None,
                      regional_code=None, progress_callback=None, log_callback=None):
        """
        Ejecutar validación del modelo SARIMAX con transformación específica por regional
        
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
                log_callback(f"Iniciando validación con parámetros: order={order}, seasonal_order={seasonal_order}")
                log_callback(f"Regional: {regional_code} - Transformación: {transformation.upper()}")
                
            if progress_callback:
                progress_callback(10, "Cargando datos...")
            
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
            
            if log_callback:
                log_callback(f"Columnas encontradas: {df.columns.tolist()}")
            
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
            
            historico = df[df[col_saidi].notna()]
            
            if len(historico) < 12:
                raise Exception("Se necesitan al menos 12 observaciones históricas para la validación")
            
            if log_callback:
                log_callback(f"Dataset: {len(historico)} observaciones")
                log_callback(f"Período: {historico.index[0].strftime('%Y-%m')} a {historico.index[-1].strftime('%Y-%m')}")
            
            if progress_callback:
                progress_callback(25, "Dividiendo datos para validación...")
            
            # Determinar porcentaje de validación
            if len(historico) >= 60:
                pct_validacion = 0.30
            elif len(historico) >= 36:
                pct_validacion = 0.25
            else:
                pct_validacion = 0.20
                
            n_test = max(6, int(len(historico) * pct_validacion))
            datos_entrenamiento_original = historico[col_saidi][:-n_test]
            datos_validacion_original = historico[col_saidi][-n_test:]
            
            if log_callback:
                log_callback(f"División: {len(datos_entrenamiento_original)} datos entrenamiento, {len(datos_validacion_original)} datos validación")
                log_callback(f"Porcentaje validación: {pct_validacion*100:.0f}%")
            
            # NUEVO: Aplicar transformación según regional
            if progress_callback:
                progress_callback(40, f"Aplicando transformación {transformation.upper()}...")
            
            train_transformed, transform_info = self._apply_transformation(
                datos_entrenamiento_original.values, transformation
            )
            datos_entrenamiento_transformed = pd.Series(train_transformed, index=datos_entrenamiento_original.index)
            
            if log_callback:
                log_callback(f"Transformación aplicada: {transform_info}")
            
            if progress_callback:
                progress_callback(50, "Entrenando modelo SARIMAX con datos transformados...")
            
            # Entrenar modelo con datos TRANSFORMADOS
            try:
                model = SARIMAX(
                    datos_entrenamiento_transformed,
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=True,
                    enforce_invertibility=True
                )
                results = model.fit(disp=False)
                
                if log_callback:
                    log_callback(f"Modelo SARIMAX ajustado con transformación {transformation.upper()}")
                    
            except Exception as e:
                raise Exception(f"Error ajustando modelo: {str(e)}")
            
            if progress_callback:
                progress_callback(75, "Generando predicciones de validación...")
            
            # Predecir en escala TRANSFORMADA
            try:
                pred = results.get_forecast(steps=n_test)
                predicciones_transformed = pred.predicted_mean
                
                # NUEVO: Revertir predicciones a escala ORIGINAL
                predicciones_original = self._inverse_transformation(
                    predicciones_transformed.values, transformation
                )
                predicciones_validacion = pd.Series(predicciones_original, index=predicciones_transformed.index)
                
                if log_callback:
                    log_callback(f"Predicciones generadas y revertidas a escala original para {len(predicciones_validacion)} períodos")
                    
            except Exception as e:
                raise Exception(f"Error generando predicciones: {str(e)}")
            
            if progress_callback:
                progress_callback(90, "Calculando métricas de validación...")
            
            # Calcular métricas en escala ORIGINAL
            metricas = self._calcular_metricas_validacion(datos_validacion_original.values, predicciones_validacion.values)
            
            if log_callback:
                log_callback("=" * 60)
                log_callback("MÉTRICAS DEL MODELO (Escala Original)")
                log_callback("=" * 60)
                log_callback(f"RMSE: {metricas['rmse']:.4f} minutos")
                log_callback(f"MAE: {metricas['mae']:.4f} minutos")
                log_callback(f"MAPE: {metricas['mape']:.1f}%")
                log_callback(f"R²: {metricas['r2_score']:.3f}")
                log_callback(f"PRECISIÓN FINAL: {metricas['precision_final']:.1f}%")
                log_callback(f"├─ Componente MAPE: {metricas['precision_mape']:.1f}%")
                log_callback(f"├─ Componente R²: {metricas['precision_r2']:.1f}%") 
                log_callback(f"└─ Componente RMSE: {metricas['precision_rmse']:.1f}%")
                
                precision = metricas['precision_final']
                if precision >= 90:
                    interpretacion = "EXCELENTE - Predicciones muy confiables"
                elif precision >= 80:
                    interpretacion = "BUENO - Predicciones confiables"
                elif precision >= 70:
                    interpretacion = "ACEPTABLE - Predicciones moderadamente confiables"
                elif precision >= 60:
                    interpretacion = "REGULAR - Usar con precaución"
                else:
                    interpretacion = "BAJO - Modelo poco confiable"
                    
                log_callback(f"INTERPRETACIÓN: {interpretacion}")
                log_callback("=" * 60)
            
            if progress_callback:
                progress_callback(95, "Generando gráfica de validación...")
            
            # Generar gráfica con datos en escala ORIGINAL
            plot_path = self._generar_grafica_validacion(
                datos_entrenamiento_original, datos_validacion_original, predicciones_validacion,
                col_saidi, order, seasonal_order, metricas, pct_validacion, transformation
            )
            
            if progress_callback:
                progress_callback(100, "Validación completada exitosamente")
            
            return {
                'success': True,
                'metrics': metricas,
                'model_params': {
                    'order': order,
                    'seasonal_order': seasonal_order,
                    'transformation': transformation,
                    'regional_code': regional_code
                },
                'training_count': len(datos_entrenamiento_original),
                'validation_count': len(datos_validacion_original),
                'validation_percentage': pct_validacion * 100,
                'training_period': {
                    'start': datos_entrenamiento_original.index[0].strftime('%Y-%m'),
                    'end': datos_entrenamiento_original.index[-1].strftime('%Y-%m')
                },
                'validation_period': {
                    'start': datos_validacion_original.index[0].strftime('%Y-%m'),
                    'end': datos_validacion_original.index[-1].strftime('%Y-%m')
                },
                'plot_file': plot_path
            }
            
        except Exception as e:
            if log_callback:
                log_callback(f"ERROR: {str(e)}")
            raise Exception(f"Error en validación: {str(e)}")
    
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
            # Asegurar valores positivos
            data_positive = np.maximum(data, 1e-10)
            transformed = np.log(data_positive)
            self.transformation_params['log_applied'] = True
            return transformed, "Transformación logarítmica (log)"
        
        elif transformation_type == 'boxcox':
            # Asegurar valores positivos
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
    
    def _calcular_metricas_validacion(self, datos_reales, predicciones):
        """Calcular métricas de validación en ESCALA ORIGINAL"""
        rmse = np.sqrt(mean_squared_error(datos_reales, predicciones))
        mae = np.mean(np.abs(datos_reales - predicciones))
        
        epsilon = 1e-8
        mape = np.mean(np.abs((datos_reales - predicciones) / (datos_reales + epsilon))) * 100
        
        ss_res = np.sum((datos_reales - predicciones) ** 2)
        ss_tot = np.sum((datos_reales - np.mean(datos_reales)) ** 2)
        r2_score = 1 - (ss_res / (ss_tot + epsilon))
        
        precision_mape = max(0, 100 - mape)
        precision_r2 = max(0, r2_score * 100)
        mean_actual = np.mean(datos_reales)
        precision_rmse = max(0, (1 - rmse/mean_actual) * 100)
        
        precision_final = (precision_mape * 0.4 + precision_r2 * 0.4 + precision_rmse * 0.2)
        precision_final = max(0, min(100, precision_final))
        
        return {
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'r2_score': r2_score,
            'precision_mape': precision_mape,
            'precision_r2': precision_r2,
            'precision_rmse': precision_rmse,
            'precision_final': precision_final
        }
    
    def _generar_grafica_validacion(self, datos_entrenamiento, datos_validacion, 
                                   predicciones_validacion, col_saidi, order, 
                                   seasonal_order, metricas, pct_validacion, transformation):
        """Generar gráfica de validación (datos en escala original)"""
        try:
            if datos_entrenamiento.empty or datos_validacion.empty or predicciones_validacion.empty:
                return None
                
            temp_dir = tempfile.gettempdir()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_path = os.path.join(temp_dir, f"saidi_validation_{timestamp}.png")
            
            plt.style.use('default')
            fig = plt.figure(figsize=(16, 10), dpi=100)
            
            plt.plot(datos_entrenamiento.index, datos_entrenamiento.values, 
                    label=f"Datos de Entrenamiento ({100-int(pct_validacion*100)}% - {len(datos_entrenamiento)} obs.)", 
                    color="blue", linewidth=3, marker='o', markersize=5)
            
            ultimo_punto_entrenamiento = datos_entrenamiento.iloc[-1]
            fecha_ultimo_entrenamiento = datos_entrenamiento.index[-1]
            
            fechas_validacion_extendidas = [fecha_ultimo_entrenamiento] + list(datos_validacion.index)
            valores_validacion_extendidos = [ultimo_punto_entrenamiento] + list(datos_validacion.values)
            valores_prediccion_extendidos = [ultimo_punto_entrenamiento] + list(predicciones_validacion.values)
            
            plt.plot(fechas_validacion_extendidas, valores_validacion_extendidos, 
                    label=f"Datos Reales de Validación ({int(pct_validacion*100)}% - {len(datos_validacion)} obs.)", 
                    color="navy", linewidth=3, linestyle=':', marker='s', markersize=7)
            
            plt.plot(fechas_validacion_extendidas, valores_prediccion_extendidos, 
                    label=f"Predicciones del Modelo ({transformation.upper()})", 
                    color="orange", linewidth=3, marker='^', markersize=7)
            
            for x, y in zip(datos_entrenamiento.index, datos_entrenamiento.values):
                plt.text(x, y+0.3, f"{y:.1f}", color="blue", fontsize=8, 
                        ha='center', va='bottom', rotation=0, alpha=0.9, weight='bold')
            
            for x, y in zip(datos_validacion.index, datos_validacion.values):
                plt.text(x, y+0.4, f"{y:.1f}", color="navy", fontsize=9, 
                        ha='center', va='bottom', rotation=0, weight='bold')
            
            for x, y in zip(predicciones_validacion.index, predicciones_validacion.values):
                plt.text(x, y-0.5, f"{y:.1f}", color="orange", fontsize=9, 
                        ha='center', va='top', rotation=0, weight='bold')
            
            plt.fill_between(fechas_validacion_extendidas, 
                            valores_validacion_extendidos, 
                            valores_prediccion_extendidos,
                            alpha=0.2, color='red', 
                            label='Área de Error')
            
            if not datos_entrenamiento.empty:
                separacion_x = datos_entrenamiento.index[-1]
                plt.axvline(x=separacion_x, color='gray', linestyle='--', alpha=0.8, linewidth=2)
                
                y_limits = plt.ylim()
                y_pos = y_limits[1] * 0.75
                plt.text(separacion_x, y_pos, 'División\nEntrenamiento/Validación', 
                        ha='center', va='center', color='gray', fontsize=10, weight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.9, edgecolor='gray'))
            
            info_metricas = (f"MÉTRICAS\n"
                            f"RMSE: {metricas['rmse']:.3f} | MAE: {metricas['mae']:.3f}\n"
                            f"MAPE: {metricas['mape']:.1f}% | R²: {metricas['r2_score']:.3f}\n"
                            f"Precisión: {metricas['precision_final']:.1f}%")
            
            plt.text(0.01, 0.24, info_metricas, transform=plt.gca().transAxes, 
                    fontsize=10, verticalalignment='top', 
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.9, edgecolor='navy'))
            
            info_componentes = (f"COMPONENTES PRECISIÓN\n"
                               f"MAPE: {metricas['precision_mape']:.1f}% | R²: {metricas['precision_r2']:.1f}%\n"
                               f"RMSE: {metricas['precision_rmse']:.1f}% | Formula: 0.4+0.4+0.2")
            
            plt.text(0.01, 0.09, info_componentes, transform=plt.gca().transAxes, 
                    fontsize=9, verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='wheat', alpha=0.9, edgecolor='orange'))
            
            info_parametros = (f"PARÁMETROS M4 + {transformation.upper()}\n"
                              f"order = {order} | seasonal = {seasonal_order}\n"
                              f"Train: {len(datos_entrenamiento)} | Valid: {len(datos_validacion)}")
            
            plt.text(0.985, 0.08, info_parametros, transform=plt.gca().transAxes, 
                    fontsize=9, verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='lightgreen', alpha=0.9, edgecolor='green'))
            
            precision = metricas['precision_final']
            if precision >= 90:
                interpretacion = "EXCELENTE"
                color_interp = "green"
            elif precision >= 80:
                interpretacion = "BUENO" 
                color_interp = "limegreen"
            elif precision >= 70:
                interpretacion = "ACEPTABLE"
                color_interp = "orange"
            elif precision >= 60:
                interpretacion = "REGULAR"
                color_interp = "red"
            else:
                interpretacion = "BAJO"
                color_interp = "darkred"
            
            plt.text(0.985, 0.97, f"{interpretacion}\n{precision:.1f}%", 
                    transform=plt.gca().transAxes, fontsize=12, weight='bold',
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor=color_interp, alpha=0.8, edgecolor='black'),
                    color='black')
            
            ax = plt.gca()
            
            if not datos_entrenamiento.empty and not datos_validacion.empty:
                x_min = datos_entrenamiento.index[0]
                x_max = datos_validacion.index[-1]
            elif not datos_entrenamiento.empty:
                x_min = datos_entrenamiento.index[0]
                x_max = datos_entrenamiento.index[-1]
            else:
                all_dates = list(datos_entrenamiento.index) + list(datos_validacion.index)
                x_min = min(all_dates)
                x_max = max(all_dates)
                
            plt.xlim(x_min, x_max)
            
            all_values = list(datos_entrenamiento.values) + list(datos_validacion.values) + list(predicciones_validacion.values)
            y_min = min(all_values) * 0.92
            y_max = max(all_values) * 1.08
            plt.ylim(y_min, y_max)
            
            meses_espanol = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 
                            'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
            
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
            
            fechas_mensuales = pd.date_range(start=x_min, end=x_max, freq='MS')
            labels_mensuales = []
            for fecha in fechas_mensuales:
                mes_nombre = meses_espanol[fecha.month - 1]
                if fecha.month == 1 or len(fechas_mensuales) <= 12:
                    labels_mensuales.append(f"{mes_nombre}\n{fecha.year}")
                else:
                    labels_mensuales.append(mes_nombre)
            
            if len(fechas_mensuales) > 0:
                ax.set_xticks(fechas_mensuales)
                ax.set_xticklabels(labels_mensuales, rotation=45, ha='right', fontsize=9)
            
            plt.title(f"Validación Modelo M4: SARIMAX{order}x{seasonal_order} + {transformation.upper()}", 
                     fontsize=18, fontweight='bold', pad=25)
            
            plt.xlabel("Fecha", fontsize=14, weight='bold')
            plt.ylabel("SAIDI (minutos)", fontsize=14, weight='bold')
            
            plt.legend(fontsize=11, loc='upper center', bbox_to_anchor=(0.25, -0.08), 
                      ncol=2, frameon=True, shadow=True, fancybox=True)
            
            plt.grid(True, alpha=0.4, linestyle='-', linewidth=0.8)
            
            plt.tight_layout()
            plt.subplots_adjust(top=0.93, bottom=0.35, left=0.038, right=0.787)
            
            plt.figtext(0.5, 0.02, 
                       f"Transformación: {transformation.upper()} - Métricas en escala original", 
                       ha='center', fontsize=12, style='italic', color='darkblue', weight='bold',
                       bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.8))
            
            plt.savefig(plot_path, dpi=100, bbox_inches='tight', facecolor='white', edgecolor='none')
            plt.close(fig)
            
            self.plot_file_path = plot_path
            return plot_path
            
        except Exception as e:
            print(f"Error generando gráfica de validación: {e}")
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