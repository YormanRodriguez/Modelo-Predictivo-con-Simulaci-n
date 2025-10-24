# services/validation_service.py
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
from typing import Optional, Dict, Any, Tuple
from services.climate_simulation_service import ClimateSimulationService

class ValidationService:
    """
    Servicio para validar modelos SARIMAX con transformaciones por regional
    
    ACTUALIZACIONES:
    - Soporte para variables exogenas climaticas
    - SOPORTE PARA SIMULACION CLIMATICA (como PredictionService)
    - SIN intervalos de confianza
    - Calculo de precision IDENTICO a OptimizationService
    - Metricas completamente alineadas
    - Validacion consistente (20-30% test)
    """
    
    # Mapeo de regionales a sus transformaciones optimas
    REGIONAL_TRANSFORMATIONS = {
        'SAIDI_O': 'boxcox',      # Ocana - Boxcox
        'SAIDI_C': 'original',    # Cucuta - Original
        'SAIDI_A': 'original',    # Aguachica - Original
        'SAIDI_P': 'boxcox',      # Pamplona - Boxcox
        'SAIDI_T': 'sqrt',        # Tibu - Sqrt
        'SAIDI_Cens': 'original'  # Cens - Original
    }
    
    # Variables exogenas por regional (consistente con OptimizationService)
    REGIONAL_EXOG_VARS = {
        'SAIDI_O': {
            'temp_max': 'Temperatura maxima',
            'humedad_avg': 'Humedad relativa',
            'precip_total': 'Precipitacion total'
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
    
    def run_validation(self, 
                      file_path: Optional[str] = None, 
                      df_prepared: Optional[pd.DataFrame] = None, 
                      order: Optional[Tuple] = None, 
                      seasonal_order: Optional[Tuple] = None,
                      regional_code: Optional[str] = None, 
                      climate_data: Optional[pd.DataFrame] = None,
                      simulation_config: Optional[Dict] = None,  
                      progress_callback = None, 
                      log_callback = None) -> Dict[str, Any]:
        """
        Ejecutar validacion del modelo SARIMAX con transformacion especifica por regional
        
        Args:
            file_path: Ruta del archivo Excel SAIDI
            df_prepared: DataFrame SAIDI preparado
            order: Orden ARIMA
            seasonal_order: Orden estacional
            regional_code: Codigo de la regional (e.g., 'SAIDI_C', 'SAIDI_O')
            climate_data: DataFrame con datos climaticos mensuales
            simulation_config: Configuracion de simulacion climatica (NUEVO)
            progress_callback: Funcion para actualizar progreso
            log_callback: Funcion para logging
        
        Returns:
            Diccionario con resultados de validacion
        """
        try:
            if order is None:
                order = self.default_order
            if seasonal_order is None:
                seasonal_order = self.default_seasonal_order
            
            # Determinar transformacion a usar
            transformation = self._get_transformation_for_regional(regional_code)
            
            # Detectar si hay simulacion
            simulation_applied = simulation_config and simulation_config.get('enabled', False)
            
            if log_callback:
                log_callback(f"Iniciando validacion con parametros: order={order}, seasonal_order={seasonal_order}")
                log_callback(f"Regional: {regional_code} - Transformacion: {transformation.upper()}")
                
                if simulation_applied:
                    log_callback("=" * 60)
                    log_callback("  VALIDACIÓN CON SIMULACIÓN CLIMÁTICA")
                    log_callback("=" * 60)
                    
                    summary = simulation_config.get('summary', {})
                    log_callback(f" Escenario: {summary.get('escenario', 'N/A')}")
                    log_callback(f" Alcance: {summary.get('alcance_meses', 'N/A')} meses")
                    log_callback(f" Días base: {summary.get('dias_base', 'N/A')}")
                    log_callback(f" Ajuste: {summary.get('slider_adjustment', 0):+d} días")
                    log_callback(f" Total días simulados: {summary.get('dias_simulados', 'N/A')}")
                    
                    # Mostrar variables que se modificarán
                    log_callback("\n Variables a modificar:")
                    changes = summary.get('percentage_changes', {})
                    var_names = {
                        'temp_max': 'Temperatura máxima',
                        'humedad_avg': 'Humedad relativa',
                        'precip_total': 'Precipitación total'
                    }
                    for var, change_pct in changes.items():
                        var_name = var_names.get(var, var)
                        arrow = "↑" if change_pct > 0 else "↓" if change_pct < 0 else "→"
                        log_callback(f"   {arrow} {var_name}: {change_pct:+.1f}%")
                    
                    log_callback("")
                    log_callback("  NOTA: Validación bajo condiciones climáticas HIPOTÉTICAS")
                    log_callback("=" * 60)
                else:
                    log_callback("Modo: Validacion estandar (sin simulacion)")
                
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
            
            if log_callback:
                log_callback(f"Columnas encontradas: {df.columns.tolist()}")
            
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
            
            historico = df[df[col_saidi].notna()]
            
            if len(historico) < 12:
                raise Exception("Se necesitan al menos 12 observaciones historicas para la validacion")
            
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
                    climate_data, df, regional_code, log_callback
                )
                
                if exog_df is not None:
                    if log_callback:
                        log_callback(f"Variables exogenas disponibles: {len(exog_df.columns)}")
                        for var_code, var_data in exog_info.items():
                            log_callback(f"  - {var_data['nombre']}")
                    
                    # CRITICO: NO ESCALAR si hay simulacion
                    if simulation_applied:
                        if log_callback:
                            log_callback("Variables exogenas SIN ESCALAR (para simulacion)")
                        # Solo FIT del scaler, no transform
                        self.exog_scaler = StandardScaler()
                        self.exog_scaler.fit(exog_df)
                        # exog_df permanece SIN ESCALAR
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
                            log_callback("Variables exogenas escaladas correctamente")
                else:
                    if log_callback:
                        log_callback("No se pudieron preparar variables exogenas, continuando sin ellas")
            else:
                if log_callback:
                    log_callback("No hay datos climaticos disponibles, validacion sin variables exogenas")
            
            if progress_callback:
                progress_callback(30, "Dividiendo datos para validacion...")
            
            # Determinar porcentaje de validacion (IDENTICO a OptimizationService)
            n_obs = len(historico)
            if n_obs >= 60:
                pct_validacion = 0.30
            elif n_obs >= 36:
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
                datos_entrenamiento_original.values, transformation
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
                exog_train = exog_df.loc[datos_entrenamiento_original.index]
                
                # AQUI ESTA LA DIFERENCIA: Si hay simulacion, aplicarla a exog_test
                if simulation_applied:
                    if log_callback:
                        log_callback("Preparando variables exogenas para validacion con simulacion...")
                    
                    # Obtener variables SIN ESCALAR para el periodo de validacion
                    exog_test_original = exog_df.loc[datos_validacion_original.index]
                    
                    # Aplicar simulacion (retorna ESCALADO)
                    exog_test = self._apply_climate_simulation(
                        exog_test_original, simulation_config, log_callback
                    )
                    
                    if log_callback:
                        summary = simulation_config.get('summary', {})
                        log_callback(f"Simulacion aplicada a periodo de validacion:")
                        log_callback(f"  - Escenario: {summary.get('escenario', 'N/A')}")
                        log_callback(f"  - Periodos afectados: {len(exog_test)}")
                else:
                    # Sin simulacion: usar directamente
                    exog_test = exog_df.loc[datos_validacion_original.index]
                
                if log_callback:
                    log_callback(f"Variables exogenas de entrenamiento: {len(exog_train)} periodos")
                    log_callback(f"Variables exogenas de validacion: {len(exog_test)} periodos")
                    if simulation_applied:
                        log_callback("  (con simulacion climatica aplicada)")
            
            # Entrenar modelo con datos TRANSFORMADOS
            try:
                model = SARIMAX(
                    datos_entrenamiento_transformed,
                    exog=exog_train,
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=True,
                    enforce_invertibility=True
                )
                results = model.fit(disp=False)
                
                if log_callback:
                    log_callback(f"Modelo SARIMAX ajustado con transformacion {transformation.upper()}")
                    if exog_train is not None:
                        log_callback(f"Modelo incluye {exog_train.shape[1]} variables exogenas")
                    
            except Exception as e:
                raise Exception(f"Error ajustando modelo: {str(e)}")
            
            if progress_callback:
                progress_callback(70, "Generando predicciones de validacion...")
            
            # Predecir en escala TRANSFORMADA
            try:
                pred = results.get_forecast(steps=n_test, exog=exog_test)
                predicciones_transformed = pred.predicted_mean
                
                # Revertir predicciones a escala ORIGINAL
                predicciones_original = self._inverse_transformation(
                    predicciones_transformed.values, transformation
                )
                
                predicciones_validacion = pd.Series(predicciones_original, index=predicciones_transformed.index)
                
                if log_callback:
                    log_callback(f"Predicciones generadas y revertidas a escala original para {len(predicciones_validacion)} periodos")
                    if simulation_applied:
                        log_callback("  (basadas en condiciones climaticas simuladas)")
                    
            except Exception as e:
                raise Exception(f"Error generando predicciones: {str(e)}")
            
            if progress_callback:
                progress_callback(85, "Calculando metricas de validacion...")
            
            # Calcular metricas IDENTICO a OptimizationService
            metricas = self._calcular_metricas_validacion_optimized(
                datos_validacion_original.values,
                predicciones_original,
                order,
                seasonal_order,
                transformation,
                exog_df is not None,
                pct_validacion,
                n_test
            )
            
            # Calcular complejidad del modelo (IDENTICO a OptimizationService)
            complexity_penalty = sum(order) + sum(seasonal_order[:3])
            composite_score = metricas['rmse'] + (complexity_penalty * 0.05)
            
            # Calcular estabilidad (IDENTICO a OptimizationService)
            stability_score = self._calculate_stability_numpy(
                datos_validacion_original.values,
                predicciones_original,
                metricas['precision_final'],
                metricas['mape']
            )
            
            # Agregar métricas adicionales
            metricas['composite_score'] = composite_score
            metricas['stability_score'] = stability_score
            metricas['complexity'] = complexity_penalty
            
            if log_callback:
                log_callback("=" * 60)
                if simulation_applied:
                    log_callback("METRICAS DE VALIDACION CON SIMULACION CLIMATICA")
                    summary = simulation_config.get('summary', {})
                    log_callback(f"Escenario: {summary.get('escenario', 'N/A')}")
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
                
                precision = metricas['precision_final']
                if precision >= 60:
                    interpretacion = "EXCELENTE - Predicciones muy confiables"
                elif precision >= 40:
                    interpretacion = "BUENO - Predicciones confiables"
                elif precision >= 20:
                    interpretacion = "ACEPTABLE - Predicciones moderadamente confiables"
                else:
                    interpretacion = "LIMITADO - Modelo poco confiable"
                    
                log_callback(f"INTERPRETACION: {interpretacion}")
                log_callback(f"Validacion: {pct_validacion*100:.0f}% de datos como test ({n_test} meses)")
                
                if simulation_applied:
                    log_callback("")
                    log_callback("NOTA: Metricas bajo condiciones climaticas simuladas")
                    log_callback("      Los valores reales pueden diferir si el clima no sigue el escenario")
                
                if exog_info:
                    log_callback("\nVariables exogenas utilizadas en validacion:")
                    for var_code, var_data in exog_info.items():
                        log_callback(f"  - {var_data['nombre']}")
                
                log_callback("=" * 60)
            
            if progress_callback:
                progress_callback(95, "Generando grafica de validacion...")
            
            # Generar grafica con datos en escala ORIGINAL
            plot_path = self._generar_grafica_validacion(
                datos_entrenamiento_original, 
                datos_validacion_original, 
                predicciones_validacion,
                col_saidi, 
                order, 
                seasonal_order, 
                metricas, 
                pct_validacion, 
                transformation,
                exog_info,
                simulation_config if simulation_applied else None  # NUEVO
            )
            
            if progress_callback:
                progress_callback(100, "Validacion completada exitosamente")
            
            return {
                'success': True,
                'metrics': metricas,
                'model_params': {
                    'order': order,
                    'seasonal_order': seasonal_order,
                    'transformation': transformation,
                    'regional_code': regional_code,
                    'with_exogenous': exog_df is not None,
                    'with_simulation': simulation_applied,  # NUEVO
                    'complexity': complexity_penalty
                },
                'predictions': {
                    'mean': predicciones_validacion.to_dict()
                },
                'exogenous_vars': exog_info,
                'simulation_config': simulation_config if simulation_applied else None,  # NUEVO
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
            raise Exception(f"Error en validacion: {str(e)}")
    
    def _apply_climate_simulation(self, exog_forecast_original, simulation_config, log_callback=None):
        """
        Aplicar simulacion climatica a variables SIN ESCALAR (IDENTICO a PredictionService)
        Luego escalar el resultado
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
                    log_callback("   Salida: valores ESCALADOS listos para validacion")
                
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
    
    def _get_transformation_for_regional(self, regional_code: Optional[str]) -> str:
        """Obtener la transformacion correspondiente a la regional"""
        if regional_code and regional_code in self.REGIONAL_TRANSFORMATIONS:
            return self.REGIONAL_TRANSFORMATIONS[regional_code]
        return 'original'
    
    def _prepare_exogenous_variables(self,
                                     climate_data: pd.DataFrame,
                                     df_saidi: pd.DataFrame,
                                     regional_code: Optional[str],
                                     log_callback) -> Tuple[Optional[pd.DataFrame], Optional[Dict]]:
        """
        Preparar variables exogenas climaticas SIN ESCALAR
        (IDENTICO a OptimizationService)
        """
        try:
            if climate_data is None or climate_data.empty:
                return None, None
            
            if not regional_code or regional_code not in self.REGIONAL_EXOG_VARS:
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
                            'columna_clima': climate_col
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
    
    def _align_exog_to_saidi(self,
                            exog_series: pd.DataFrame,
                            df_saidi: pd.DataFrame,
                            var_code: str,
                            log_callback) -> Optional[pd.Series]:
        """
        Alinear datos exogenos al indice de SAIDI
        (IDENTICO a OptimizationService)
        """
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
            
            return result
            
        except Exception as e:
            if log_callback:
                log_callback(f"Error alineando variable {var_code}: {str(e)}")
            return None
    
    def _apply_transformation(self, data: np.ndarray, transformation_type: str) -> Tuple[np.ndarray, str]:
        """Aplicar transformacion a los datos (IDENTICO a OptimizationService)"""
        if transformation_type == 'original':
            return data, "Sin transformacion (datos originales)"
        
        elif transformation_type == 'standard':
            self.scaler = StandardScaler()
            transformed = self.scaler.fit_transform(data.reshape(-1, 1)).flatten()
            return transformed, f"StandardScaler (media={self.scaler.mean_[0]:.2f}, std={np.sqrt(self.scaler.var_[0]):.2f})"
        
        elif transformation_type == 'log':
            data_positive = np.maximum(data, 1e-10)
            transformed = np.log(data_positive)
            self.transformation_params['log_applied'] = True
            return transformed, "Transformacion logaritmica (log)"
        
        elif transformation_type == 'boxcox':
            data_positive = np.maximum(data, 1e-10)
            transformed, lambda_param = stats.boxcox(data_positive)
            self.transformation_params['boxcox_lambda'] = lambda_param
            return transformed, f"Box-Cox (lambda={lambda_param:.4f})"
        
        elif transformation_type == 'sqrt':
            data_positive = np.maximum(data, 0)
            transformed = np.sqrt(data_positive)
            self.transformation_params['sqrt_applied'] = True
            return transformed, "Sqrt"
        
        else:
            return data, "Sin transformacion (tipo desconocido)"
    
    def _inverse_transformation(self, data: np.ndarray, transformation_type: str) -> np.ndarray:
        """Revertir transformacion a escala original (IDENTICO a OptimizationService)"""
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
    
    def _calcular_metricas_validacion_optimized(self,
                                               test_values: np.ndarray,
                                               pred_values: np.ndarray,
                                               order: Tuple,
                                               seasonal_order: Tuple,
                                               transformation: str,
                                               with_exogenous: bool,
                                               pct_validacion: float,
                                               n_test: int) -> Dict[str, float]:
        """
        Calcular metricas de validacion IDENTICO a OptimizationService
        
        Esta implementación garantiza que:
        - La precisión se calcula exactamente igual
        - Todas las métricas usan numpy arrays
        - El resultado es consistente con la optimización
        """
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
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'r2_score': r2_score,
            'precision_final': precision_final,
            'n_test': n_test,
            'validation_pct': pct_validacion * 100
        }
    
    def _calculate_stability_numpy(self, 
                                   actual_values: np.ndarray,  
                                   predicted_values: np.ndarray,  
                                   precision: float,
                                   mape: float) -> float:
        """
        Calcular score de estabilidad (IDENTICO a OptimizationService)
        """
        try:
            errors = actual_values - predicted_values
            
            mean_abs_error = np.mean(np.abs(errors))
            std_error = np.std(errors)
            
            if mean_abs_error > 1e-8:
                cv_error = std_error / mean_abs_error
                stability_cv = max(0, 100 * (1 - min(cv_error, 1)))
            else:
                stability_cv = 50.0
            
            # Penalización por MAPE alto
            if mape > 50:
                mape_penalty = 0.5
            elif mape > 30:
                mape_penalty = 0.7
            else:
                mape_penalty = 1.0
            
            stability_cv = stability_cv * mape_penalty
            
            # Combinar con precisión
            stability = (stability_cv * 0.6) + (precision * 0.4)
            
            return min(100.0, max(0.0, stability))
            
        except Exception:
            return 0.0
    
    def _generar_grafica_validacion(self, 
                                   datos_entrenamiento: pd.Series, 
                                   datos_validacion: pd.Series, 
                                   predicciones_validacion: pd.Series,
                                   col_saidi: str, 
                                   order: Tuple, 
                                   seasonal_order: Tuple, 
                                   metricas: Dict, 
                                   pct_validacion: float,
                                   transformation: str,
                                   exog_info: Optional[Dict] = None,
                                   simulation_config: Optional[Dict] = None) -> Optional[str]:
        """
        Generar grafica de validacion con metricas alineadas y soporte para simulacion
        """
        try:
            if datos_entrenamiento.empty or datos_validacion.empty or predicciones_validacion.empty:
                return None
                
            temp_dir = tempfile.gettempdir()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_path = os.path.join(temp_dir, f"saidi_validation_{timestamp}.png")
            
            plt.style.use('default')
            fig = plt.figure(figsize=(16, 10), dpi=100)
            
            # Detectar si hay simulacion
            simulation_applied = simulation_config and simulation_config.get('enabled', False)
            
            # Grafica principal - datos de entrenamiento
            plt.plot(datos_entrenamiento.index, datos_entrenamiento.values, 
                    label=f"Datos de Entrenamiento ({100-int(pct_validacion*100)}% - {len(datos_entrenamiento)} obs.)", 
                    color="blue", linewidth=3, marker='o', markersize=5)
            
            ultimo_punto_entrenamiento = datos_entrenamiento.iloc[-1]
            fecha_ultimo_entrenamiento = datos_entrenamiento.index[-1]
            
            # Conectar entrenamiento con validacion
            fechas_validacion_extendidas = [fecha_ultimo_entrenamiento] + list(datos_validacion.index)
            valores_validacion_extendidos = [ultimo_punto_entrenamiento] + list(datos_validacion.values)
            valores_prediccion_extendidos = [ultimo_punto_entrenamiento] + list(predicciones_validacion.values)
            
            # Datos reales de validacion
            plt.plot(fechas_validacion_extendidas, valores_validacion_extendidos, 
                    label=f"Datos Reales de Validacion ({int(pct_validacion*100)}% - {len(datos_validacion)} obs.)", 
                    color="navy", linewidth=3, linestyle=':', marker='s', markersize=7)
            
            # Predicciones del modelo con etiqueta segun simulacion
            if simulation_applied:
                summary = simulation_config.get('summary', {})
                exog_label = f" [SIMULADO: {summary.get('escenario', 'N/A')}]"
                pred_color = "red"
            elif exog_info:
                exog_label = " [+EXOG]"
                pred_color = "orange"
            else:
                exog_label = ""
                pred_color = "orange"
            
            plt.plot(fechas_validacion_extendidas, valores_prediccion_extendidos, 
                    label=f"Predicciones del Modelo ({transformation.upper()}){exog_label}", 
                    color=pred_color, linewidth=3, marker='^', markersize=7, zorder=5)
            
            # Etiquetas de valores - datos de entrenamiento
            for x, y in zip(datos_entrenamiento.index, datos_entrenamiento.values):
                plt.text(x, y+0.3, f"{y:.1f}", color="blue", fontsize=8, 
                        ha='center', va='bottom', rotation=0, alpha=0.9, weight='bold')
            
            # Etiquetas de valores - datos reales de validacion
            for x, y in zip(datos_validacion.index, datos_validacion.values):
                plt.text(x, y+0.4, f"{y:.1f}", color="navy", fontsize=9, 
                        ha='center', va='bottom', rotation=0, weight='bold')
            
            # Etiquetas de valores - predicciones
            for x, y in zip(predicciones_validacion.index, predicciones_validacion.values):
                plt.text(x, y-0.5, f"{y:.1f}", color=pred_color, fontsize=9, 
                        ha='center', va='top', rotation=0, weight='bold')
            
            # Area de error entre real y prediccion
            plt.fill_between(fechas_validacion_extendidas, 
                            valores_validacion_extendidos, 
                            valores_prediccion_extendidos,
                            alpha=0.2, color='red', 
                            label='Area de Error')
            
            # Linea divisoria entre entrenamiento y validacion
            if not datos_entrenamiento.empty:
                separacion_x = datos_entrenamiento.index[-1]
                plt.axvline(x=separacion_x, color='gray', linestyle='--', alpha=0.8, linewidth=2)
                
                y_limits = plt.ylim()
                y_pos = y_limits[1] * 0.75
                plt.text(separacion_x, y_pos, 'Division\nEntrenamiento/Validacion', 
                        ha='center', va='center', color='gray', fontsize=10, weight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.9, edgecolor='gray'))
            
            # Cuadro de metricas (ACTUALIZADO con nuevas métricas)
            info_metricas = (f"METRICAS VALIDACION\n"
                            f"RMSE: {metricas['rmse']:.3f} | MAE: {metricas['mae']:.3f}\n"
                            f"MAPE: {metricas['mape']:.1f}% | R2: {metricas['r2_score']:.3f}\n"
                            f"Precision: {metricas['precision_final']:.1f}%")
            
            plt.text(0.01, 0.24, info_metricas, transform=plt.gca().transAxes, 
                    fontsize=10, verticalalignment='top', 
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.9, edgecolor='navy'))
            
            # Cuadro de estabilidad y complejidad (NUEVO)
            stability = metricas.get('stability_score', 0)
            complexity = metricas.get('complexity', 0)
            composite = metricas.get('composite_score', 0)
            
            info_estabilidad = (f"ESTABILIDAD & COMPLEJIDAD\n"
                               f"Stability Score: {stability:.1f}/100\n"
                               f"Complejidad: {complexity} params\n"
                               f"Composite Score: {composite:.3f}")
            
            plt.text(0.01, 0.09, info_estabilidad, transform=plt.gca().transAxes, 
                    fontsize=9, verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='wheat', alpha=0.9, edgecolor='orange'))
            
            # Cuadro de parametros del modelo con info de simulacion
            info_parametros = (f"PARAMETROS + {transformation.upper()}\n"
                              f"order = {order} | seasonal = {seasonal_order}\n"
                              f"Train: {len(datos_entrenamiento)} | Valid: {len(datos_validacion)}")
            
            if simulation_applied:
                summary = simulation_config.get('summary', {})
                info_parametros += f"\nSIMULACION: {summary.get('escenario', 'N/A')}"
                info_parametros += f"\nAlcance: {summary.get('alcance_meses', 'N/A')} meses"
            elif exog_info:
                info_parametros += f"\nVariables exogenas: {len(exog_info)}"
            
            plt.text(0.985, 0.08, info_parametros, transform=plt.gca().transAxes, 
                    fontsize=9, verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='lightgreen', alpha=0.9, edgecolor='green'))
            
            # Indicador de calidad (ACTUALIZADO con umbrales de OptimizationService)
            precision = metricas['precision_final']
            if precision >= 60:
                interpretacion = "EXCELENTE"
                color_interp = "green"
            elif precision >= 40:
                interpretacion = "BUENO" 
                color_interp = "limegreen"
            elif precision >= 20:
                interpretacion = "ACEPTABLE"
                color_interp = "orange"
            else:
                interpretacion = "LIMITADO"
                color_interp = "red"
            
            # Si hay simulacion, agregar indicador
            if simulation_applied:
                interpretacion += "\n[SIMULADO]"
            
            plt.text(0.985, 0.97, f"{interpretacion}\n{precision:.1f}%", 
                    transform=plt.gca().transAxes, fontsize=12, weight='bold',
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor=color_interp, alpha=0.8, edgecolor='black'),
                    color='black')
            
            # Configurar ejes
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
            
            # Configurar etiquetas de fechas
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
            
            # Titulo con info de simulacion
            # Titulo con info de simulacion
            title_text = f"Validacion Modelo: SARIMAX{order}x{seasonal_order} + {transformation.upper()}"
            if simulation_applied:
                summary = simulation_config.get('summary', {})
                escenario_name = summary.get('escenario', 'N/A').upper()
                title_text += f" [ SIMULACIÓN: {escenario_name}]"
            elif exog_info:
                title_text += " [+EXOG]"
            
            plt.title(title_text, fontsize=18, fontweight='bold', pad=25)
            
            plt.xlabel("Fecha", fontsize=14, weight='bold')
            plt.ylabel("SAIDI (minutos)", fontsize=14, weight='bold')
            
            plt.legend(fontsize=11, loc='upper center', bbox_to_anchor=(0.25, -0.08), 
                      ncol=2, frameon=True, shadow=True, fancybox=True)
            
            plt.grid(True, alpha=0.4, linestyle='-', linewidth=0.8)
            
            plt.tight_layout()
            plt.subplots_adjust(top=0.93, bottom=0.35, left=0.038, right=0.787)
            
            # Nota al pie 
            footer_text = f"Transformacion: {transformation.upper()} - Precision calculada como OptimizationService"
            footer_color = 'lightyellow'
            footer_edge_color = 'darkblue'
            footer_text_color = 'darkblue'
            
            if simulation_applied:
                summary = simulation_config.get('summary', {})
                escenario = summary.get('escenario', 'N/A').upper()
                dias = summary.get('dias_simulados', 'N/A')
                
                footer_text = f"  VALIDACIÓN CON SIMULACIÓN CLIMÁTICA - Escenario: {escenario} ({dias} días)"
                footer_text += " | Métricas bajo condiciones HIPOTÉTICAS"
                footer_color = '#FFEBEE'
                footer_edge_color = '#F44336'
                footer_text_color = 'darkred'
            elif exog_info:
                footer_text += f" - Con {len(exog_info)} variables exogenas"
            
            footer_text += f" - Validacion: {metricas['validation_pct']:.0f}% test"
            
            plt.figtext(0.5, 0.02, footer_text, 
                       ha='center', fontsize=12, style='italic', color=footer_text_color, weight='bold',
                       bbox=dict(boxstyle='round,pad=0.4', facecolor=footer_color, alpha=0.8, 
                                edgecolor=footer_edge_color, linewidth=2 if simulation_applied else 1))
            
            plt.savefig(plot_path, dpi=100, bbox_inches='tight', facecolor='white', edgecolor='none')
            plt.close(fig)
            
            self.plot_file_path = plot_path
            return plot_path
            
        except Exception as e:
            print(f"Error generando grafica de validacion: {e}")
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