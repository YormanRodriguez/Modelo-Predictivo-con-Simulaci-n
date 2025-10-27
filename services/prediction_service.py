# services/prediction_service.py - VERSION CON VARIABLES EXOGENAS, SIMULACION E INTERVALOS DE CONFIANZA CORREGIDOS
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
from services.climate_simulation_service import ClimateSimulationService
from services.uncertainty_service import UncertaintyService 
from services.export_service import ExportService

class PredictionService:
    """Servicio para generar predicciones SAIDI con variables exogenas climaticas, simulacion e intervalos de confianza"""
    
    # Mapeo de regionales a sus transformaciones optimas
    REGIONAL_TRANSFORMATIONS = {
        'SAIDI_O': 'boxcox',      # Ocana - Box-Cox
        'SAIDI_C': 'original',    # Cucuta - Original
        'SAIDI_A': 'original',    # Aguachica - Original
        'SAIDI_P': 'boxcox',      # Pamplona - Box-Cox
        'SAIDI_T': 'sqrt',        # Tibu - Sqrt
        'SAIDI_Cens': 'original'  # Cens - Original
    }
    
    # Variables exogenas por regional (usando columnas procesadas por ClimateModel)
    REGIONAL_EXOG_VARS = {
        'SAIDI_O': {
            'temp_max': 'Temperatura maxima',         # Alta correlacion con SAIDI
            'humedad_avg': 'Humedad relativa',        # Correlacionada con fallas
            'precip_total': 'Precipitacion total'     # Tormentas causan interrupciones
        },
        'SAIDI_C': {
            'temp_max': 'Temperatura maxima',
            'humedad_avg': 'Humedad relativa',
            'precip_total': 'Precipitacion total'
        },
        'SAIDI_A': {
            'temp_max': 'Temperatura maxima',
            'humedad_avg': 'Humedad relativa',
            'precip_total': 'Precipitacion total'
        },
        'SAIDI_P': {
            'temp_max': 'Temperatura maxima',
            'humedad_avg': 'Humedad relativa',
            'precip_total': 'Precipitacion total'
        },
        'SAIDI_T': {
            'temp_max': 'Temperatura maxima',
            'humedad_avg': 'Humedad relativa',
            'precip_total': 'Precipitacion total'
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
    
    def export_predictions(self, predictions_dict, regional_code, regional_nombre, 
                      output_dir=None, include_intervals=True, model_params=None,
                      metrics=None):
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
                model_info['metrics'] = metrics
            
            # Llamar al servicio de exportacion
            filepath = self.export_service.export_predictions_to_excel(
                predictions_dict=predictions_dict,
                regional_code=regional_code,
                regional_nombre=regional_nombre,
                output_dir=output_dir,
                include_confidence_intervals=include_intervals,
                model_info=model_info
            )
            
            return filepath
            
        except Exception as e:
            print(f"Error exportando predicciones: {str(e)}")
            return None

    
    def run_prediction(self, file_path=None, df_prepared=None, order=None, seasonal_order=None, 
                    regional_code=None, climate_data=None, simulation_config=None,
                    progress_callback=None, log_callback=None):
        """
        Ejecutar prediccion SAIDI con variables exogenas climaticas, simulacion e intervalos de confianza
        
        Args:
            file_path: Ruta del archivo SAIDI Excel
            df_prepared: DataFrame de SAIDI ya preparado
            order: Orden ARIMA
            seasonal_order: Orden estacional ARIMA
            regional_code: Codigo de la regional
            climate_data: DataFrame con datos climaticos mensuales
            simulation_config: Configuracion de simulacion climatica (opcional)
            progress_callback: Funcion para actualizar progreso
            log_callback: Funcion para loguear mensajes
        """
        try:
            if order is None:
                order = self.default_order
            if seasonal_order is None:
                seasonal_order = self.default_seasonal_order
            
            # Determinar transformacion segun regional
            transformation = self._get_transformation_for_regional(regional_code)
            
            if log_callback:
                order_str = str(order)
                seasonal_str = str(seasonal_order)
                log_callback(f"Iniciando prediccion con parametros: order={order_str}, seasonal_order={seasonal_str}")
                log_callback(f"Regional: {regional_code} - Transformacion: {transformation.upper()}")
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
            
            # Preparar variables exogenas
            exog_df = None
            exog_info = None
            simulation_applied = False

            if climate_data is not None and not climate_data.empty:
                exog_df, exog_info = self._prepare_exogenous_variables(
                    climate_data, df, regional_code, log_callback
                )
                
                if exog_df is not None:
                    if log_callback:
                        log_callback(f"Variables exogenas disponibles: {len(exog_df.columns)}")
                    
                    # CRITICO: NO ESCALAR AQUI si hay simulacion
                    # Verificar si hay simulacion climatica
                    if simulation_config and simulation_config.get('enabled', False):
                        if log_callback:
                            log_callback("=" * 60)
                            log_callback("SIMULACION CLIMATICA ACTIVADA")
                            log_callback("=" * 60)
                            log_callback("Variables exogenas SIN ESCALAR (para simulacion)")
                        simulation_applied = True
                        # NO ESCALAR - guardar scaler pero no aplicar aun
                        self.exog_scaler = StandardScaler()
                        self.exog_scaler.fit(exog_df)  # Solo FIT, no transform
                        # exog_df permanece SIN ESCALAR para entrenamiento con simulacion
                    else:
                        # Sin simulacion: escalar normalmente
                        if log_callback:
                            log_callback("Escalando variables exogenas...")
                        self.exog_scaler = StandardScaler()
                        exog_df_scaled = pd.DataFrame(
                            self.exog_scaler.fit_transform(exog_df),
                            index=exog_df.index,
                            columns=exog_df.columns
                        )
                        exog_df = exog_df_scaled
                        if log_callback:
                            log_callback(f"Variables exogenas escaladas correctamente")
                else:
                    if log_callback:
                        log_callback("No se pudieron preparar variables exogenas, continuando sin ellas")
            else:
                if log_callback:
                    log_callback("No hay datos climaticos disponibles, prediccion sin variables exogenas")
            
            if progress_callback:
                progress_callback(25, "Procesando datos historicos...")
            
            # Separar datos historicos y faltantes
            faltantes = df[df[col_saidi].isna()]
            historico = df[df[col_saidi].notna()]
            
            if faltantes.empty:
                if log_callback:
                    log_callback("No hay meses faltantes, creando predicciones futuras")
                ultimo_mes = historico.index[-1]
                fechas_futuras = pd.date_range(start=ultimo_mes + pd.DateOffset(months=1), periods=6, freq='MS')
                for fecha in fechas_futuras:
                    df.loc[fecha, col_saidi] = np.nan
                faltantes = df[df[col_saidi].isna()]
            
            if log_callback:
                log_callback(f"Datos historicos SAIDI: {len(historico)} observaciones")
                log_callback(f"Meses a predecir: {len(faltantes)} observaciones")
                log_callback(f"Periodo historico: {historico.index[0].strftime('%Y-%m')} a {historico.index[-1].strftime('%Y-%m')}")
            
            if progress_callback:
                progress_callback(30, f"Aplicando transformacion {transformation.upper()}...")
            
            # Aplicar transformacion
            historico_values_original = historico[col_saidi].values
            historico_transformed, transform_info = self._apply_transformation(
                historico_values_original, transformation
            )
            historico_transformed_series = pd.Series(historico_transformed, index=historico.index)
            
            if log_callback:
                log_callback(f"Transformacion aplicada: {transform_info}")
            
            if progress_callback:
                progress_callback(40, "Calculando metricas del modelo...")
            
            # Calcular metricas
            historico_original_series = pd.Series(historico_values_original, index=historico.index)
            metricas = self._calcular_metricas_modelo(
                historico_original_series, order, seasonal_order, transformation, exog_df
            )
            
            if metricas and log_callback:
                log_callback(f"Metricas del modelo (en escala original):")
                log_callback(f"  - RMSE: {metricas['rmse']:.4f} minutos")
                log_callback(f"  - MAE: {metricas['mae']:.4f} minutos")
                log_callback(f"  - MAPE: {metricas['mape']:.1f}%")
                log_callback(f"  - R2 Score: {metricas['r2_score']:.4f}")
                log_callback(f"  - Precision Final: {metricas['precision_final']:.1f}%")
                if exog_df is not None:
                    log_callback(f"   Con variables exogenas")
            
            if progress_callback:
                progress_callback(60, "Ajustando modelo SARIMAX...")
            
            # Ajustar modelo con o sin variables exogenas
            try:
                # Filtrar exog_df para que coincida con historico
                exog_train = None
                if exog_df is not None:
                    exog_train = exog_df.loc[historico.index]
                    if log_callback:
                        log_callback(f"Variables exogenas de entrenamiento: {len(exog_train)} periodos")
                        if simulation_applied:
                            log_callback("ADVERTENCIA: USANDO DATOS SIN ESCALAR para entrenamiento (se escalaran en forecast)")
                
                model = SARIMAX(
                    historico_transformed_series,
                    exog=exog_train,
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=True,
                    enforce_invertibility=True
                )
                results = model.fit(disp=False)
                
                if log_callback:
                    log_callback(f"Modelo SARIMAX ajustado correctamente")
                    if exog_train is not None:
                        log_callback(f"  Variables exogenas incluidas: {exog_train.shape[1]}")
                        if simulation_applied:
                            log_callback(f"  Modo: Entrenamiento con datos SIN ESCALAR")
                    
            except Exception as e:
                raise Exception(f"Error ajustando modelo: {str(e)}")
            
            if progress_callback:
                progress_callback(80, "Generando predicciones con intervalos de confianza...")
            
            # Generar predicciones con intervalos de confianza OPTIMIZADOS
            try:
                # Preparar variables exogenas para prediccion
                exog_forecast = None
                exog_forecast_original = None  # Guardar version sin escalar
                
                if exog_df is not None:
                    # SIEMPRE extender SIN ESCALAR primero
                    exog_forecast_original = self._extend_exogenous_for_forecast(
                        exog_df, faltantes.index, log_callback, 
                        unscale=simulation_applied
                    )
                    
                    # Aplicar simulacion SI esta habilitada (trabaja con valores SIN ESCALAR)
                    if simulation_applied and simulation_config:
                        exog_forecast = self._apply_climate_simulation(
                            exog_forecast_original, simulation_config, log_callback
                        )
                        # _apply_climate_simulation ya retorna ESCALADO
                    else:
                        # Si no hay simulacion, escalar aqui
                        if self.exog_scaler is not None:
                            exog_forecast = pd.DataFrame(
                                self.exog_scaler.transform(exog_forecast_original),
                                index=exog_forecast_original.index,
                                columns=exog_forecast_original.columns
                            )
                
                # OPTIMIZADO: Calcular intervalos de forma más eficiente
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
                
                # AJUSTAR intervalos de forma RAZONABLE
                # Los intervalos paramétricos del modelo ya son conservadores al 95%
                # Solo necesitamos un ajuste MUY leve
                adjustment_factor = 1.0
                
                if metricas:
                    # Ajuste SUTIL basado en precisión histórica
                    # Un buen modelo (MAPE < 15%) casi no necesita ajuste
                    # Un modelo regular necesita un pequeño margen adicional
                    if metricas['mape'] > 30:
                        adjustment_factor = 1.15  # +15% más ancho (modelo pobre)
                    elif metricas['mape'] > 20:
                        adjustment_factor = 1.10  # +10% más ancho (modelo regular)
                    elif metricas['mape'] > 15:
                        adjustment_factor = 1.05  # +5% más ancho (modelo bueno)
                    else:
                        adjustment_factor = 1.02  # +2% más ancho (modelo excelente)
                    
                    if log_callback:
                        log_callback(f"  Factor de ajuste de intervalos: {adjustment_factor:.2f}x (basado en MAPE={metricas['mape']:.1f}%)")
                
                # Aplicar ajuste conservador
                for i in range(len(pred_mean_original)):
                    center = pred_mean_original[i]
                    half_width = (upper_bound_original[i] - lower_bound_original[i]) / 2
                    adjusted_half_width = half_width * adjustment_factor
                    
                    lower_bound_original[i] = max(0, center - adjusted_half_width)
                    upper_bound_original[i] = center + adjusted_half_width
                
                # Si hay variables exógenas, añadir incertidumbre MUY MODERADA
                if exog_forecast_original is not None and climate_data is not None:
                    try:
                        # Incertidumbre de exógenas: MÁXIMO 5% adicional
                        # Esto reconoce que las variables climáticas tienen incertidumbre
                        # pero no debe dominar el intervalo
                        exog_uncertainty_pct = 0.03  # 3% base (muy conservador)
                        
                        # Calcular variabilidad histórica de variables climáticas
                        max_cv = 0
                        for col in exog_forecast_original.columns:
                            if col in climate_data.columns:
                                col_std = climate_data[col].std()
                                col_mean = climate_data[col].mean()
                                if col_mean > 0:
                                    cv = col_std / col_mean  # Coeficiente de variación
                                    max_cv = max(max_cv, cv)
                        
                        # Limitar influencia de CV a máximo 5% adicional
                        exog_uncertainty_pct = min(0.05, 0.03 + max_cv * 0.1)
                        
                        # Expandir intervalos muy moderadamente
                        exog_factor = 1.0 + exog_uncertainty_pct
                        
                        if log_callback:
                            log_callback(f"  Incertidumbre de variables exógenas: +{exog_uncertainty_pct*100:.1f}%")
                        
                        for i in range(len(pred_mean_original)):
                            center = pred_mean_original[i]
                            half_width = (upper_bound_original[i] - lower_bound_original[i]) / 2
                            adjusted_half_width = half_width * exog_factor
                            
                            lower_bound_original[i] = max(0, center - adjusted_half_width)
                            upper_bound_original[i] = center + adjusted_half_width
                    
                    except Exception as e:
                        if log_callback:
                            log_callback(f"  No se pudo ajustar por incertidumbre de exógenas: {str(e)}")
                
                # VALIDACIÓN FINAL: Asegurar que los intervalos sean RAZONABLES
                # Los intervalos NO deben ser más anchos que ±50% del valor predicho
                # Esto evita intervalos absurdos que no aportan información
                for i in range(len(pred_mean_original)):
                    center = pred_mean_original[i]
                    if center > 0:
                        max_reasonable_width = center * 0.50  # ±50% máximo
                        
                        current_lower = lower_bound_original[i]
                        current_upper = upper_bound_original[i]
                        
                        # Si el intervalo es demasiado ancho, recortarlo
                        if (center - current_lower) > max_reasonable_width:
                            lower_bound_original[i] = max(0, center - max_reasonable_width)
                        
                        if (current_upper - center) > max_reasonable_width:
                            upper_bound_original[i] = center + max_reasonable_width
                
                if log_callback:
                    # Calcular ancho promedio de intervalos como % de la predicción
                    avg_width_pct = 0
                    for i in range(len(pred_mean_original)):
                        if pred_mean_original[i] > 0:
                            width = upper_bound_original[i] - lower_bound_original[i]
                            width_pct = (width / (2 * pred_mean_original[i])) * 100
                            avg_width_pct += width_pct
                    avg_width_pct /= len(pred_mean_original)
                    log_callback(f"  Ancho promedio de intervalos: ±{avg_width_pct:.0f}% del valor predicho")
                
                # Calcular margen de error (desviación estándar aproximada)
                z_score = stats.norm.ppf(0.975)  # Para 95% de confianza
                margin_error = (upper_bound_original - lower_bound_original) / (2 * z_score)
                
                # Crear Series con indices correctos
                pred_mean = pd.Series(pred_mean_original, index=faltantes.index)
                lower_bound = pd.Series(lower_bound_original, index=faltantes.index)
                upper_bound = pd.Series(upper_bound_original, index=faltantes.index)
                margin_error_series = pd.Series(margin_error, index=faltantes.index)
                
                # Crear resultado compatible con formato esperado
                uncertainty_result = {
                    'predictions': pred_mean_original,
                    'lower_bound': lower_bound_original,
                    'upper_bound': upper_bound_original,
                    'margin_error': margin_error,
                    'method': 'parametric_adjusted',
                    'confidence_level': 0.95
                }
                
                if log_callback:
                    log_callback(f"Predicciones con intervalos de confianza generadas para {len(pred_mean)} periodos")
                    log_callback(f"  Metodo: Intervalos paramétricos ajustados")
                    if exog_forecast is not None:
                        if simulation_applied:
                            log_callback(f"  Usando variables exogenas SIMULADAS")
                        else:
                            log_callback(f"  Usando variables exogenas proyectadas")
                    log_callback(f"  Intervalos de confianza: 95%")
                    avg_margin_pct = np.mean(margin_error / pred_mean_original) * 100
                    log_callback(f"  Margen de error promedio: ±{avg_margin_pct:.1f}%")
                    if simulation_applied:
                        log_callback(f"  NOTA: El margen refleja incertidumbre estadistica del modelo, no del escenario simulado")
                    
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
                    log_callback("PREDICCION SAIDI CON SIMULACION CLIMATICA E INTERVALOS DE CONFIANZA")
                else:
                    log_callback("RESUMEN DE PREDICCIONES CON VARIABLES EXOGENAS E INTERVALOS")
                log_callback("=" * 60)
                
                if simulation_applied and simulation_config:
                    summary = simulation_config.get('summary', {})
                    log_callback(f"Escenario simulado: {summary.get('escenario', 'N/A')}")
                    log_callback(f"Dias simulados: {summary.get('dias_simulados', 'N/A')}")
                    log_callback(f"Alcance: {summary.get('alcance_meses', 'N/A')} meses")
                    log_callback("\nCambios aplicados a variables:")
                    for var, change in summary.get('percentage_changes', {}).items():
                        log_callback(f"  - {var}: {change:+.1f}%")
                    log_callback("")
                
                log_callback(f"Predicciones generadas: {len(pred_mean)}")
                log_callback(f"Periodo: {faltantes.index[0].strftime('%Y-%m')} a {faltantes.index[-1].strftime('%Y-%m')}")
                log_callback(f"Metodo de intervalos: {uncertainty_result['method']}")
                log_callback("Valores predichos con intervalos de confianza (95%):")
                
                for fecha, valor, inferior, superior, margen in zip(
                    faltantes.index, pred_mean, lower_bound, upper_bound, margin_error_series
                ):
                    # Calcular margenes asimetricos para mostrar
                    margen_sup = superior - valor
                    margen_inf = valor - inferior
                    margen_pct = (margen / valor * 100) if valor > 0 else 0
                    
                    log_callback(
                        f"  • {fecha.strftime('%Y-%m')}: {valor:.2f} min "
                        f"[IC: {inferior:.2f} - {superior:.2f}] "
                        f"(+{margen_sup:.2f}/-{margen_inf:.2f} | ±{margen_pct:.0f}%)"
                    )
                      
                if exog_info:
                    log_callback("\nVariables exogenas utilizadas:")
                    for var_code, var_info in exog_info.items():
                        log_callback(f"  - {var_info['nombre']}: correlacion {var_info['correlacion']}")
                
                if simulation_applied:
                    log_callback("\n NOTA: Esta es una proyeccion hipotetica basada en simulacion climatica.")
                    log_callback("   La precision del modelo base se mantiene, pero los resultados")
                    log_callback("   dependen del realismo del escenario simulado.")
                    log_callback("   Los intervalos de confianza reflejan incertidumbre estadistica,")
                    log_callback("   no incertidumbre sobre el escenario climatico simulado.")
                
                log_callback("=" * 60)
            
            # Generar grafica con intervalos de confianza
            plot_path = self._generar_grafica(
                historico, pred_mean, faltantes, df, col_saidi, 
                order, seasonal_order, metricas, transformation, exog_info,
                simulation_config if simulation_applied else None,
                lower_bound, upper_bound
            )
            
            if progress_callback:
                progress_callback(100, "Prediccion completada")
            
            # Preparar resultados con intervalos de confianza
            predicciones_dict = {}
            for fecha, valor, inferior, superior, margen in zip(
                faltantes.index, pred_mean, lower_bound, upper_bound, margin_error_series
            ):
                predicciones_dict[fecha.strftime('%Y-%m')] = {
                    'valor_predicho': float(valor),
                    'limite_inferior': float(inferior),
                    'limite_superior': float(superior),
                    'margen_error': float(margen)
                }
            
            return {
                'success': True,
                'predictions': predicciones_dict,
                'metrics': metricas,
                'model_params': {
                    'order': list(order),
                    'seasonal_order': list(seasonal_order),
                    'transformation': transformation,
                    'regional_code': regional_code,
                    'with_exogenous': exog_df is not None,
                    'with_simulation': simulation_applied,
                    'confidence_level': 0.95,
                    'uncertainty_method': uncertainty_result['method']
                },
                'exogenous_vars': exog_info,
                'simulation_config': simulation_config if simulation_applied else None,
                'historical_count': len(historico),
                'prediction_count': len(pred_mean),
                'plot_file': plot_path,
                'export_service': self.export_service  
            }
            
        except Exception as e:
            if log_callback:
                log_callback(f"ERROR: {str(e)}")
            raise Exception(f"Error en prediccion: {str(e)}")
    
    def _apply_climate_simulation(self, exog_forecast_original, simulation_config, log_callback=None):
        """
        Aplicar simulacion climatica a variables SIN ESCALAR
        Luego escalar el resultado
        
        Args:
            exog_forecast_original: DataFrame SIN ESCALAR con variables exogenas
            simulation_config: Configuracion de simulacion
            log_callback: Funcion para logging
        
        Returns:
            DataFrame ESCALADO con simulacion aplicada
        """
        try:
            if not simulation_config.get('enabled', False):
                # Si no hay simulacion, solo escalar y retornar
                if self.exog_scaler is not None:
                    return pd.DataFrame(
                        self.exog_scaler.transform(exog_forecast_original),
                        index=exog_forecast_original.index,
                        columns=exog_forecast_original.columns
                    )
                return exog_forecast_original
            
            if log_callback:
                log_callback("Aplicando simulacion climatica a variables exogenas...")
                log_callback("   Entrada: valores originales SIN ESCALAR")
            
            escenario = simulation_config['escenario']
            slider_adjustment = simulation_config['slider_adjustment']
            dias_base = simulation_config['dias_base']
            alcance_meses = simulation_config['alcance_meses']
            percentiles = simulation_config['percentiles']
            regional_code = simulation_config['regional_code']
            
            # Aplicar simulacion a valores SIN ESCALAR
            exog_simulated = self.simulation_service.apply_simulation(
                exog_forecast=exog_forecast_original,
                escenario=escenario,
                slider_adjustment=slider_adjustment,
                dias_base=dias_base,
                alcance_meses=alcance_meses,
                percentiles=percentiles,
                regional_code=regional_code
            )
            
            if log_callback:
                log_callback(f"   Simulacion aplicada a {alcance_meses} mes(es)")
                log_callback(f"   Escenario: {escenario}")
                log_callback(f"   Ajuste: {slider_adjustment:+d} dias sobre base de {dias_base}")
                
                # Mostrar cambios en el primer mes
                if alcance_meses >= 1:
                    log_callback("   Cambios en primer mes:")
                    for col in exog_simulated.columns:
                        original_val = exog_forecast_original.iloc[0][col]
                        simulated_val = exog_simulated.iloc[0][col]
                        change_pct = ((simulated_val - original_val) / original_val) * 100
                        log_callback(f"     - {col}: {original_val:.2f} -> {simulated_val:.2f} ({change_pct:+.1f}%)")
            
            # ESCALAR despues de simular
            if self.exog_scaler is not None:
                exog_simulated_scaled = pd.DataFrame(
                    self.exog_scaler.transform(exog_simulated),
                    index=exog_simulated.index,
                    columns=exog_simulated.columns
                )
                
                if log_callback:
                    log_callback("   Variables simuladas escaladas para el modelo")
                    log_callback("   Salida: valores ESCALADOS listos para prediccion")
                
                return exog_simulated_scaled
            
            if log_callback:
                log_callback("   ADVERTENCIA: No hay scaler, retornando simulacion sin escalar")
            
            return exog_simulated
            
        except Exception as e:
            if log_callback:
                log_callback(f"ERROR aplicando simulacion: {str(e)}")
                import traceback
                log_callback(traceback.format_exc())
            # En caso de error, solo escalar y retornar sin simulacion
            if self.exog_scaler is not None:
                return pd.DataFrame(
                    self.exog_scaler.transform(exog_forecast_original),
                    index=exog_forecast_original.index,
                    columns=exog_forecast_original.columns
                )
            return exog_forecast_original
    
    def _prepare_exogenous_variables(self, climate_data, df_saidi, regional_code, log_callback=None):
        """
        Preparar variables exogenas climaticas SIN ESCALAR
        El escalado se hace despues en el flujo principal
        """
        try:
            if climate_data is None or climate_data.empty:
                return None, None
            
            if regional_code not in self.REGIONAL_EXOG_VARS:
                if log_callback:
                    log_callback(f"Regional {regional_code} no tiene variables exogenas definidas")
                return None, None
            
            exog_vars_config = self.REGIONAL_EXOG_VARS[regional_code]
            
            climate_column_mapping = {
                'temp_max': 'temp_max',
                'humedad_avg': 'humedad_avg',
                'precip_total': 'precip_total'
            }
            
            # Crear DataFrame de variables exogenas SIN ESCALAR
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
                            'columna_clima': climate_col,
                            'correlacion': self._get_correlation_for_var(var_code, regional_code)
                        }
                        
                        if log_callback:
                            log_callback(f"  {var_nombre} preparada")
            
            exog_df = exog_df.dropna(how='all')
            exog_df = exog_df.interpolate(method='linear', limit_direction='both')
            
            if exog_df.empty:
                return None, None
            
            # NO ESCALAR AQUI - se escalara despues en el flujo principal
            return exog_df, exog_info if exog_info else None
            
        except Exception as e:
            if log_callback:
                log_callback(f"Error preparando variables exogenas: {str(e)}")
            return None, None
    
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
                    log_callback(f"  - {var_code}: {n_future} valores futuros proyectados desde ultima observacion")
            
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
                    log_callback(f"  - {var_code}: {n_past} valores pasados proyectados desde primera observacion")
            
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
    
    def _extend_exogenous_for_forecast(self, exog_df, forecast_dates, log_callback=None, unscale=False):
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
                    columns=exog_df.columns
                )
                if log_callback:
                    log_callback("  Des-escalando variables para simulacion...")
            else:
                # Ya estan sin escalar o no hay scaler
                exog_df_original = exog_df.copy()
            
            # Crear forecast con valores originales (sin escalar)
            exog_forecast = pd.DataFrame(index=forecast_dates, columns=exog_df_original.columns)
            
            for col in exog_df_original.columns:
                last_value = exog_df_original[col].iloc[-1]
                exog_forecast[col] = last_value
            
            if log_callback:
                log_callback(f"  Variables extendidas: {len(forecast_dates)} periodos (sin escalar)")
            
            return exog_forecast  # Retorna SIN ESCALAR
            
        except Exception as e:
            if log_callback:
                log_callback(f"ERROR extendiendo variables exogenas: {str(e)}")
            return None
    
    def _get_correlation_for_var(self, var_code, regional_code):
        """Obtener la correlacion documentada de una variable"""
        correlations = {
            'SAIDI_O': {
                'temp_max': 0.450,
                'humedad_avg': 0.380,
                'precip_total': 0.420
            },
            'SAIDI_C': {
                'temp_max': 0.450,
                'humedad_avg': 0.380,
                'precip_total': 0.420
            },
            'SAIDI_A': {
                'temp_max': 0.450,
                'humedad_avg': 0.380,
                'precip_total': 0.420
            },
            'SAIDI_P': {
                'temp_max': 0.450,
                'humedad_avg': 0.380,
                'precip_total': 0.420
            },
            'SAIDI_T': {
                'temp_max': 0.450,
                'humedad_avg': 0.380,
                'precip_total': 0.420
            }
        }
        
        if regional_code in correlations and var_code in correlations[regional_code]:
            return correlations[regional_code][var_code]
        return 0.0
    
    def _get_transformation_for_regional(self, regional_code):
        """Obtener transformacion para la regional"""
        if regional_code and regional_code in self.REGIONAL_TRANSFORMATIONS:
            return self.REGIONAL_TRANSFORMATIONS[regional_code]
        return 'original'
    
    def _apply_transformation(self, data, transformation_type):
        """Aplicar transformacion a los datos"""
        if transformation_type == 'original':
            return data, "Sin transformacion (datos originales)"
        
        elif transformation_type == 'standard':
            self.scaler = StandardScaler()
            transformed = self.scaler.fit_transform(data.reshape(-1, 1)).flatten()
            return transformed, f"StandardScaler"
        
        elif transformation_type == 'log':
            data_positive = np.maximum(data, 1e-10)
            transformed = np.log(data_positive)
            self.transformation_params['log_applied'] = True
            return transformed, "Transformacion logaritmica"
        
        elif transformation_type == 'boxcox':
            data_positive = np.maximum(data, 1e-10)
            transformed, lambda_param = stats.boxcox(data_positive)
            self.transformation_params['boxcox_lambda'] = lambda_param
            return transformed, f"Box-Cox (lambda={lambda_param:.4f})"
        
        elif transformation_type == 'sqrt':
            data_positive = np.maximum(data, 1e-10)
            transformed = np.sqrt(data_positive)
            return transformed, "Transformacion raiz cuadrada"
        
        else:
            return data, "Sin transformacion"
    
    def _inverse_transformation(self, data, transformation_type):
        """Revertir transformacion a escala original"""
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
    
    def _calcular_metricas_modelo(self, serie_original, order, seasonal_order, 
                                transformation, exog_df=None):
        try:
            if len(serie_original) >= 60:
                pct_validacion = 0.30
            elif len(serie_original) >= 36:
                pct_validacion = 0.25
            else:
                pct_validacion = 0.20
            
            n_test = max(6, int(len(serie_original) * pct_validacion))
            
            train_original = serie_original[:-n_test]
            test_original = serie_original[-n_test:]
            
            # Aplicar transformacion
            train_transformed, _ = self._apply_transformation(train_original.values, transformation)
            train_transformed_series = pd.Series(train_transformed, index=train_original.index)
            
            # Preparar exogenas
            exog_train = None
            if exog_df is not None:
                exog_train = exog_df.loc[train_original.index]
            
            # Entrenar modelo
            model = SARIMAX(
                train_transformed_series,
                exog=exog_train,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=True,
                enforce_invertibility=True
            )
            results = model.fit(disp=False)
            
            # Prediccion
            exog_test = None
            if exog_df is not None:
                exog_test = exog_df.loc[test_original.index]
            
            pred = results.get_forecast(steps=n_test, exog=exog_test)
            pred_mean_transformed = pred.predicted_mean
            
            # Revertir transformacion
            pred_mean_original = self._inverse_transformation(
                pred_mean_transformed.values, transformation
            )
            
            # Convertir a numpy arrays
            test_values = test_original.values
            pred_values = pred_mean_original
            
            # Calcular metricas con arrays
            rmse = np.sqrt(mean_squared_error(test_values, pred_values))
            mae = np.mean(np.abs(test_values - pred_values))
            
            epsilon = 1e-8
            mape = np.mean(np.abs((test_values - pred_values) / 
                                (test_values + epsilon))) * 100
            
            ss_res = np.sum((test_values - pred_values) ** 2)
            ss_tot = np.sum((test_values - np.mean(test_values)) ** 2)
            r2_score = 1 - (ss_res / (ss_tot + epsilon))
            
            precision_final = max(0, min(100, (1 - mape/100) * 100))
            
            return {
                'rmse': rmse,
                'mae': mae,
                'mape': mape,
                'r2_score': r2_score,
                'precision_final': precision_final,
                'aic': results.aic,
                'bic': results.bic,
                'n_test': n_test
            }
            
        except Exception as e:
            print(f"Error calculando metricas: {e}")
            return None
    
    def _generar_grafica(self, historico, pred_mean, faltantes, df, col_saidi, 
                        order, seasonal_order, metricas, transformation, exog_info=None,
                        simulation_config=None, lower_bound=None, upper_bound=None):
        """Generar grafica de prediccion con intervalos de confianza y indicador de simulacion"""
        try:
            if historico.empty or pred_mean.empty:
                return None
            
            temp_dir = tempfile.gettempdir()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_path = os.path.join(temp_dir, f"saidi_prediction_{timestamp}.png")
            
            fig = plt.figure(figsize=(16, 10), dpi=100)
            
            # Grafica principal - datos historicos
            plt.plot(historico.index, historico[col_saidi], label="SAIDI Historico", 
                    color="blue", linewidth=3, marker='o', markersize=5)
            
            if not historico.empty and len(pred_mean) > 0:
                ultimo_real_x = historico.index[-1]
                ultimo_real_y = historico[col_saidi].iloc[-1]
                
                # CORREGIDO: Conectar solo el primer punto, no toda la serie
                x_pred = [ultimo_real_x] + list(pred_mean.index)
                y_pred = [ultimo_real_y] + list(pred_mean.values)
                
                # Etiqueta segun si hay simulacion
                if simulation_config and simulation_config.get('enabled', False):
                    summary = simulation_config.get('summary', {})
                    pred_label = f"Prediccion SIMULADA: {summary.get('escenario', 'N/A')}"
                elif exog_info:
                    pred_label = "Prediccion CON variables exogenas"
                else:
                    pred_label = f"Prediccion ({transformation.upper()})"
                
                # Linea de prediccion - SOLO dibujar desde ultimo real hasta predicciones
                plt.plot(x_pred, y_pred, label=pred_label, 
                        color="orange", linewidth=3, marker='^', markersize=7, zorder=5)
                
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
                        zorder=3
                    )
                
                # Etiquetas de valores predichos
                for x, y in zip(pred_mean.index, pred_mean.values):
                    plt.text(x, y+0.4, f"{y:.1f}", color="orange", fontsize=9, 
                            ha='center', va='bottom', weight='bold')
            
            # Linea divisoria entre historico y prediccion
            if not historico.empty:
                plt.axvline(x=historico.index[-1], color='gray', linestyle='--', alpha=0.8, linewidth=2)
            
            ax = plt.gca()
            x_min = historico.index[0] if not historico.empty else df.index[0]
            x_max = faltantes.index[-1] if not faltantes.empty else historico.index[-1]
            plt.xlim(x_min, x_max)
            
            meses_espanol = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 
                            'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
            
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
            fechas_mensuales = pd.date_range(start=x_min, end=x_max, freq='MS')
            labels_mensuales = [f"{meses_espanol[f.month-1]}\n{f.year}" for f in fechas_mensuales]
            
            if len(fechas_mensuales) > 0:
                ax.set_xticks(fechas_mensuales)
                ax.set_xticklabels(labels_mensuales, rotation=45, ha='right', fontsize=9)
            
            # Titulo con informacion de variables exogenas o simulacion
            exog_info_text = ""
            if simulation_config and simulation_config.get('enabled', False):
                summary = simulation_config.get('summary', {})
                escenario_name = summary.get('escenario', 'Simulado')
                exog_info_text = f" [SIMULACION: {escenario_name}]"
            elif exog_info:
                vars_names = " + ".join([v['nombre'] for v in exog_info.values()])
                exog_info_text = f" [Con: {vars_names}]"
            
            precision_text = f" - Precision: {metricas['precision_final']:.1f}%" if metricas else ""
            order_str = f"({order[0]},{order[1]},{order[2]})"
            seasonal_str = f"({seasonal_order[0]},{seasonal_order[1]},{seasonal_order[2]},{seasonal_order[3]})"
            plt.title(f"SAIDI: Prediccion SARIMAX{order_str}x{seasonal_str} + {transformation.upper()}{exog_info_text}{precision_text}", 
                     fontsize=16, fontweight='bold', pad=20)
            
            plt.xlabel("Fecha", fontsize=12, weight='bold')
            plt.ylabel("SAIDI (minutos)", fontsize=12, weight='bold')
            
            plt.legend(fontsize=10, loc='upper left', frameon=True, shadow=True)
            plt.grid(True, alpha=0.4, linestyle='-', linewidth=0.8)
            
            plt.tight_layout()
            
            # Nota al pie con advertencia de simulacion
            footer_y = 0.01
            
            if simulation_config and simulation_config.get('enabled', False):
                summary = simulation_config.get('summary', {})
                footer_text = f"SIMULACION CLIMATICA APLICADA: {summary.get('escenario', 'N/A')} | "
                footer_text += f"Alcance: {summary.get('alcance_meses', 'N/A')} meses | "
                footer_text += f"Intervalos reflejan incertidumbre estadistica, no del escenario simulado"
                
                plt.figtext(0.5, footer_y, footer_text, ha='center', fontsize=9, 
                           style='italic', color='darkred', weight='bold',
                           bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFEBEE', 
                                    alpha=0.9, edgecolor='#F44336', linewidth=2))
                footer_y += 0.04
            
            if exog_info and not simulation_config:
                footer_text = "Con variables exogenas: "
                for var_code, var_data in exog_info.items():
                    footer_text += f"{var_data['nombre']} (r={var_data['correlacion']:.3f}) "
                footer_text += " | Intervalos de confianza: 95%"
                plt.figtext(0.5, footer_y, footer_text, ha='center', fontsize=9, 
                           style='italic', color='darkblue', weight='bold',
                           bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))
            
            plt.savefig(plot_path, dpi=100, bbox_inches='tight', facecolor='white', edgecolor='none')
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