# model/climate_model.py - Modelo para datos climáticos mensuales
import pandas as pd
import numpy as np
import os
from typing import Optional, Dict, Any, List
from PyQt6.QtCore import QObject, pyqtSignal

class ClimateModel(QObject):
    """Modelo para manejar datos climáticos mensuales de múltiples regionales"""
    
    # Señales
    climate_data_loaded = pyqtSignal(dict)  # Regional y sus datos cargados
    all_climate_loaded = pyqtSignal(dict)   # Todas las regionales cargadas
    status_changed = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    
    # Mapeo de regionales (sin CENS)
    REGIONALES = {
        'SAIDI_C': 'Cúcuta',
        'SAIDI_O': 'Ocaña',
        'SAIDI_A': 'Aguachica',
        'SAIDI_P': 'Pamplona',
        'SAIDI_T': 'Tibú'
    }
    
    # Columnas climáticas relevantes (ESTRUCTURA MENSUAL)
    CLIMATE_COLUMNS = {
        # Identificación temporal
        'date': 'year_month',
        'year': 'year',
        'month': 'month',
        'valid_days': 'valid_days',
        'total_records': 'total_records',
        
        # Temperatura
        'temperature_avg': 'wswdat_temp_c_monthly_avg',
        'temperature_min': 'wswdat_temp_c_monthly_min',
        'temperature_max': 'wswdat_temp_c_monthly_max',
        'temp_days': 'wswdat_temp_c_days_with_data',
        
        # Humedad
        'humidity_avg': 'wswdat_relative_humidity_monthly_avg',
        'humidity_days': 'wswdat_relative_humidity_days_with_data',
        
        # Punto de rocío
        'dewpoint_avg': 'wswdat_dewpoint_c_monthly_avg',
        'dewpoint_min': 'wswdat_dewpoint_c_monthly_min',
        'dewpoint_max': 'wswdat_dewpoint_c_monthly_max',
        'dewpoint_days': 'wswdat_dewpoint_c_days_with_data',
        
        # Presión atmosférica
        'pressure_rel_avg': 'wswdat_pressure_rel_hpa_monthly_avg',
        'pressure_abs_avg': 'wswdat_pressure_abs_hpa_monthly_avg',
        'pressure_rel_days': 'wswdat_pressure_rel_hpa_days_with_data',
        'pressure_abs_days': 'wswdat_pressure_abs_hpa_days_with_data',
        
        # Precipitación
        'precipitation_total': 'wswdat_precip_today_mm_monthly_total',
        'precipitation_avg_daily': 'wswdat_precip_today_mm_monthly_avg_daily',
        'precipitation_max_daily': 'wswdat_precip_today_mm_max_daily',
        'days_with_rain': 'wswdat_precip_today_mm_days_with_precip',
        'precip_total_days': 'wswdat_precip_today_mm_total_days_measured',
        'precip_rate_avg': 'wswdat_precip_rate_mmh_monthly_avg_rate',
        'precip_rate_max': 'wswdat_precip_rate_mmh_monthly_max_rate',
        'precip_rate_total': 'wswdat_precip_rate_mmh_monthly_total_integrated',
        
        # Evapotranspiración
        'eto_avg': 'wswdat_eto_mm_monthly_avg',
        'eto_days': 'wswdat_eto_mm_days_with_data',
        
        # Radiación solar
        'solar_rad_avg': 'wswdat_solar_rad_wm2_monthly_avg',
        'solar_rad_max': 'wswdat_solar_rad_wm2_monthly_max',
        'solar_rad_days': 'wswdat_solar_rad_wm2_days_with_data',
        
        # Iluminancia
        'illuminance_avg': 'wswdat_illuminance_lux_monthly_avg',
        'illuminance_max': 'wswdat_illuminance_lux_monthly_max',
        'illuminance_days': 'wswdat_illuminance_lux_days_with_data',
        
        # Índice UV
        'uv_index_avg': 'wswdat_uv_index_monthly_avg',
        'uv_index_max': 'wswdat_uv_index_monthly_max',
        'uv_index_days': 'wswdat_uv_index_days_with_data',
        
        # Índice de calor
        'heat_index_avg': 'wswdat_heat_index_c_monthly_avg',
        'heat_index_min': 'wswdat_heat_index_c_monthly_min',
        'heat_index_max': 'wswdat_heat_index_c_monthly_max',
        'heat_index_days': 'wswdat_heat_index_c_days_with_data',
        
        # Sensación térmica (por viento)
        'windchill_avg': 'wswdat_windchill_c_monthly_avg',
        'windchill_min': 'wswdat_windchill_c_monthly_min',
        'windchill_max': 'wswdat_windchill_c_monthly_max',
        'windchill_days': 'wswdat_windchill_c_days_with_data',
        
        # Temperatura aparente (RealFeel)
        'realfeel_avg': 'wswdat_realfeel_c_monthly_avg',
        'realfeel_min': 'wswdat_realfeel_c_monthly_min',
        'realfeel_max': 'wswdat_realfeel_c_monthly_max',
        'realfeel_days': 'wswdat_realfeel_c_days_with_data',
        
        # Viento
        'wind_speed_avg': 'wswdat_wind_speed_kmh_monthly_avg',
        'wind_speed_max': 'wswdat_wind_speed_kmh_monthly_max',
        'wind_speed_days': 'wswdat_wind_speed_kmh_days_with_data',
        'wind_gust_avg': 'wswdat_wind_gust_kmh_monthly_avg',
        'wind_gust_max': 'wswdat_wind_gust_kmh_monthly_max',
        'wind_gust_days': 'wswdat_wind_gust_kmh_days_with_data',
        'wind_dir_avg': 'wswdat_wind_degrees_monthly_avg',
        'wind_dir_max': 'wswdat_wind_degrees_monthly_max',
        'wind_dir_days': 'wswdat_wind_degrees_days_with_data',
    }
    
    def __init__(self):
        super().__init__()
        self._climate_data = {}  # Diccionario: regional_code -> DataFrame
        self._file_paths = {}    # Diccionario: regional_code -> file_path
        self._climate_info = {}  # Diccionario: regional_code -> info
        
    def load_climate_file(self, regional_code: str, file_path: str) -> bool:
        """
        Cargar archivo climático mensual para una regional específica
        Soporta CSV y Excel (.xlsx, .xls)
        
        Args:
            regional_code: Código de regional (SAIDI_C, SAIDI_O, etc.)
            file_path: Ruta del archivo CSV o Excel
        """
        try:
            if regional_code not in self.REGIONALES:
                error_msg = f"Regional no válida: {regional_code}"
                self.error_occurred.emit(error_msg)
                return False
            
            regional_nombre = self.REGIONALES[regional_code]
            self.status_changed.emit(f"Cargando datos climáticos de {regional_nombre}...")
            
            # Verificar que el archivo existe
            if not os.path.exists(file_path):
                error_msg = f"Archivo no encontrado: {file_path}"
                self.error_occurred.emit(error_msg)
                return False
            
            print(f"[DEBUG CLIMATE] Cargando: {file_path}")
            
            # Detectar tipo de archivo y leer
            file_ext = os.path.splitext(file_path)[1].lower()
            
            try:
                if file_ext in ['.xlsx', '.xls']:
                    # Leer Excel
                    print(f"[DEBUG CLIMATE] Leyendo archivo Excel: {file_ext}")
                    df = pd.read_excel(file_path)
                    print(f"[DEBUG CLIMATE] Excel leído: {len(df)} registros mensuales")
                    
                elif file_ext == '.csv':
                    # Leer CSV
                    print(f"[DEBUG CLIMATE] Leyendo archivo CSV")
                    df = pd.read_csv(
                        file_path,
                        sep=";",
                        decimal=",",
                        na_values=["NULL", "null", ""],
                        low_memory=False
                    )
                    print(f"[DEBUG CLIMATE] CSV leído: {len(df)} registros mensuales")
                    
                else:
                    error_msg = f"Formato de archivo no soportado: {file_ext}. Use .xlsx, .xls o .csv"
                    self.error_occurred.emit(error_msg)
                    return False
                
            except Exception as e:
                error_msg = f"Error leyendo archivo: {str(e)}"
                self.error_occurred.emit(error_msg)
                return False
            
            # Mostrar columnas detectadas
            print(f"[DEBUG CLIMATE] Columnas en archivo: {list(df.columns)[:10]}...")
            
            # Validar estructura
            validation = self._validate_climate_structure(df, regional_nombre)
            
            if not validation['valid']:
                self.error_occurred.emit(validation['error'])
                return False
            
            # Procesar y limpiar datos
            df_processed = self._process_climate_data(df, regional_code)
            
            if df_processed is None or df_processed.empty:
                error_msg = f"No hay datos válidos en el archivo de {regional_nombre}"
                self.error_occurred.emit(error_msg)
                return False
            
            # Guardar datos procesados
            self._climate_data[regional_code] = df_processed
            self._file_paths[regional_code] = file_path
            
            # Generar información de resumen
            info = self._generate_climate_info(df_processed, regional_code, file_path)
            self._climate_info[regional_code] = info
            
            print(f"[DEBUG CLIMATE] {regional_nombre} cargada: {len(df_processed)} meses válidos")
            
            # Emitir señal de carga exitosa
            self.climate_data_loaded.emit(info)
            self.status_changed.emit(f"Datos climáticos de {regional_nombre} cargados")
            
            # Verificar si ya están todas las regionales
            if len(self._climate_data) == len(self.REGIONALES):
                self._emit_all_loaded()
            
            return True
            
        except Exception as e:
            error_msg = f"Error inesperado cargando clima de {regional_code}: {str(e)}"
            print(f"[DEBUG CLIMATE ERROR] {error_msg}")
            self.error_occurred.emit(error_msg)
            return False
    
    def _validate_climate_structure(self, df: pd.DataFrame, regional_nombre: str) -> Dict[str, Any]:
        """Validar estructura del archivo climático mensual"""
        try:
            if df.empty:
                return {'valid': False, 'error': f'Archivo de {regional_nombre} está vacío'}
            
            # Verificar columna de fecha (year_month)
            date_col = self.CLIMATE_COLUMNS['date']
            if date_col not in df.columns:
                return {
                    'valid': False, 
                    'error': f'Columna de fecha "{date_col}" no encontrada en {regional_nombre}. '
                            f'Columnas disponibles: {", ".join(df.columns[:5])}...'
                }
            
            # Verificar al menos algunas columnas climáticas importantes
            required_cols = [
                self.CLIMATE_COLUMNS['temperature_avg'],
                self.CLIMATE_COLUMNS['humidity_avg'],
                self.CLIMATE_COLUMNS['precipitation_total']
            ]
            
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                return {
                    'valid': False,
                    'error': f'Columnas críticas faltantes en {regional_nombre}: {", ".join(missing_cols)}'
                }
            
            # Validar formato de fechas year_month
            try:
                # Intentar parsear como YYYY-MM
                pd.to_datetime(df[date_col], format='%Y-%m', errors='coerce')
            except:
                return {
                    'valid': False,
                    'error': f'Columna "{date_col}" en {regional_nombre} no tiene formato válido (esperado: YYYY-MM)'
                }
            
            # Verificar que hay datos suficientes
            if len(df) < 6:
                return {
                    'valid': False,
                    'error': f'{regional_nombre} tiene muy pocos datos mensuales ({len(df)}). Se requieren al menos 6 meses.'
                }
            
            return {'valid': True}
            
        except Exception as e:
            return {'valid': False, 'error': f'Error validando estructura: {str(e)}'}
    
    def _process_climate_data(self, df: pd.DataFrame, regional_code: str) -> Optional[pd.DataFrame]:
        """
        Procesar y limpiar datos climáticos mensuales.
        VERSIÓN CORREGIDA: Extrae TODAS las variables definidas en CLIMATE_COLUMNS
        """
        try:
            df_clean = df.copy()
            
            # Convertir fecha year_month a datetime
            date_col = self.CLIMATE_COLUMNS['date']
            df_clean['fecha'] = pd.to_datetime(df_clean[date_col], format='%Y-%m', errors='coerce')
            
            # Eliminar filas sin fecha válida
            df_clean = df_clean.dropna(subset=['fecha'])
            
            if df_clean.empty:
                print("[DEBUG CLIMATE ERROR] No hay fechas válidas después de conversión")
                return None
            
            # NUEVO: Extraer automáticamente TODAS las variables disponibles
            columns_to_extract = {}
            
            # Columnas a excluir (temporales y metadata)
            excluded_keys = ['date', 'year', 'month', 'valid_days', 'total_records']
            
            # Iterar sobre TODAS las variables climáticas definidas
            for var_key, var_column in self.CLIMATE_COLUMNS.items():
                # Saltar columnas excluidas
                if var_key in excluded_keys:
                    continue
                
                # Si la columna existe en el DataFrame, mapearla
                if var_column in df_clean.columns:
                    columns_to_extract[var_column] = var_key
                else:
                    # Log para debugging (opcional, puede ser verbose)
                    # print(f"[DEBUG] Variable {var_key} ({var_column}) no disponible en {regional_code}")
                    pass
            
            # Crear DataFrame final
            df_final = pd.DataFrame()
            df_final['fecha'] = df_clean['fecha']
            
            # Agregar year y month
            df_final['year'] = df_final['fecha'].dt.year
            df_final['month'] = df_final['fecha'].dt.month
            
            # Convertir y agregar todas las columnas numéricas
            for original_col, new_col in columns_to_extract.items():
                df_final[new_col] = pd.to_numeric(df_clean[original_col], errors='coerce')
            
            # Agregar identificador de regional
            df_final['regional'] = regional_code
            
            # Ordenar por fecha
            df_final = df_final.sort_values('fecha').reset_index(drop=True)
            
            # NUEVO: Estadísticas detalladas de extracción
            total_possible = len(self.CLIMATE_COLUMNS) - len(excluded_keys)
            total_extracted = len(columns_to_extract)
            extraction_rate = (total_extracted / total_possible * 100) if total_possible > 0 else 0
            
            print(f"[DEBUG CLIMATE] Datos procesados: {len(df_final)} meses")
            print(f"[DEBUG CLIMATE] Variables extraídas: {total_extracted}/{total_possible} ({extraction_rate:.1f}%)")
            print(f"[DEBUG CLIMATE] Rango temporal: {df_final['fecha'].min()} a {df_final['fecha'].max()}")
            
            # NUEVO: Validación de variables críticas para el analizador
            critical_vars = [
                'temperature_avg', 'humidity_avg', 'precipitation_total',
                'dewpoint_avg', 'pressure_rel_avg', 'wind_speed_avg'
            ]
            
            missing_critical = [var for var in critical_vars if var not in df_final.columns]
            if missing_critical:
                print(f"[WARNING] Variables críticas faltantes: {', '.join(missing_critical)}")
            
            # NUEVO: Reporte de categorías de variables
            var_categories = {
                'Temperatura': [col for col in df_final.columns if 'temp' in col or 'heat' in col or 'windchill' in col or 'realfeel' in col],
                'Humedad': [col for col in df_final.columns if 'humidity' in col or 'dewpoint' in col],
                'Precipitación': [col for col in df_final.columns if 'precip' in col or 'rain' in col],
                'Presión': [col for col in df_final.columns if 'pressure' in col],
                'Viento': [col for col in df_final.columns if 'wind' in col],
                'Radiación': [col for col in df_final.columns if 'solar' in col or 'uv' in col or 'illuminance' in col],
                'Evapotranspiración': [col for col in df_final.columns if 'eto' in col],
                'Días_medición': [col for col in df_final.columns if 'days' in col]
            }
            
            print(f"[DEBUG CLIMATE] Variables por categoría:")
            for category, vars_list in var_categories.items():
                if vars_list:
                    print(f"  - {category}: {len(vars_list)} variables")
            
            return df_final
            
        except Exception as e:
            print(f"[DEBUG CLIMATE ERROR] Error procesando datos: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _generate_climate_info(self, df: pd.DataFrame, regional_code: str, file_path: str) -> Dict[str, Any]:
        """Generar información resumida de datos climáticos mensuales"""
        regional_nombre = self.REGIONALES[regional_code]
        
        # Calcular completitud de cada variable
        completeness = {}
        for col in df.columns:
            if col not in ['fecha', 'year', 'month', 'regional']:
                valid_count = df[col].notna().sum()
                total_count = len(df)
                completeness[col] = {
                    'valid': valid_count,
                    'total': total_count,
                    'percentage': (valid_count / total_count * 100) if total_count > 0 else 0
                }
        
        info = {
            'regional_code': regional_code,
            'regional_name': regional_nombre,
            'file_name': os.path.basename(file_path),
            'file_path': file_path,
            'total_records': len(df),
            'date_range': {
                'start': df['fecha'].min(),
                'end': df['fecha'].max(),
                'span_months': len(df),
                'span_years': (df['fecha'].max().year - df['fecha'].min().year) + 1
            },
            'variables': [col for col in df.columns if col not in ['fecha', 'year', 'month', 'regional']],
            'completeness': completeness,
            'avg_completeness': np.mean([c['percentage'] for c in completeness.values()]) if completeness else 0
        }
        
        return info
    
    def _emit_all_loaded(self):
        """Emitir señal cuando todas las regionales están cargadas"""
        summary = {
            'total_regionales': len(self._climate_data),
            'regionales_loaded': list(self._climate_data.keys()),
            'climate_info': self._climate_info.copy()
        }
        self.all_climate_loaded.emit(summary)
        self.status_changed.emit("Todos los datos climáticos cargados correctamente")
    
    def get_climate_data(self, regional_code: str) -> Optional[pd.DataFrame]:
        """Obtener datos climáticos de una regional específica"""
        if regional_code in self._climate_data:
            return self._climate_data[regional_code].copy()
        return None
    
    #def get_climate_data(self, regional_code: str) -> Optional[pd.DataFrame]:
    #    """Obtener datos climáticos con índice DatetimeIndex garantizado"""
    #    if regional_code not in self.climate_data_store:
    #        return None
    #    df = self.climate_data_store[regional_code].copy()
    #    # ASEGURAR DatetimeIndex
    #    if not isinstance(df.index, pd.DatetimeIndex):
    #        # Buscar columna de fecha
    #        for col in ['fecha', 'Fecha', 'date', 'Date', 'month_date']:
    #            if col in df.columns:
    #                df[col] = pd.to_datetime(df[col])
    #                df = df.set_index(col)
    #                break    
    #    return df
    
    def get_all_climate_data(self) -> Dict[str, pd.DataFrame]:
        """Obtener todos los datos climáticos cargados"""
        return {k: v.copy() for k, v in self._climate_data.items()}
    
    def is_regional_loaded(self, regional_code: str) -> bool:
        """Verificar si una regional tiene datos climáticos cargados"""
        return regional_code in self._climate_data
    
    def are_all_loaded(self) -> bool:
        """Verificar si todas las regionales tienen datos climáticos"""
        return len(self._climate_data) == len(self.REGIONALES)
    
    def get_loaded_regionales(self) -> List[str]:
        """Obtener lista de regionales con datos cargados"""
        return list(self._climate_data.keys())
    
    def get_missing_regionales(self) -> List[str]:
        """Obtener lista de regionales sin datos cargados"""
        return [code for code in self.REGIONALES.keys() if code not in self._climate_data]
    
    def get_climate_info(self, regional_code: str) -> Optional[Dict[str, Any]]:
        """Obtener información de una regional específica"""
        return self._climate_info.get(regional_code)
    
    def clear_regional_data(self, regional_code: str):
        """Limpiar datos de una regional específica"""
        if regional_code in self._climate_data:
            del self._climate_data[regional_code]
            del self._file_paths[regional_code]
            del self._climate_info[regional_code]
            
            regional_nombre = self.REGIONALES[regional_code]
            self.status_changed.emit(f"Datos de {regional_nombre} eliminados")
            print(f"[DEBUG CLIMATE] Datos de {regional_nombre} limpiados")
    
    def clear_all_data(self):
        """Limpiar todos los datos climáticos"""
        self._climate_data.clear()
        self._file_paths.clear()
        self._climate_info.clear()
        
        self.status_changed.emit("Todos los datos climáticos eliminados")
        print("[DEBUG CLIMATE] Todos los datos climáticos limpiados")
    
    def export_summary_report(self) -> str:
        """Generar reporte de resumen de datos climáticos"""
        report = "=" * 60 + "\n"
        report += "RESUMEN DE DATOS CLIMÁTICOS MENSUALES CARGADOS\n"
        report += "=" * 60 + "\n\n"
        
        for regional_code in self.REGIONALES.keys():
            if regional_code in self._climate_info:
                info = self._climate_info[regional_code]
                report += f"Regional: {info['regional_name']} ({regional_code})\n"
                report += f"  Archivo: {info['file_name']}\n"
                report += f"  Registros mensuales: {info['total_records']}\n"
                report += f"  Período: {info['date_range']['start'].strftime('%Y-%m')} a "
                report += f"{info['date_range']['end'].strftime('%Y-%m')}\n"
                report += f"  Duración: {info['date_range']['span_months']} meses "
                report += f"({info['date_range']['span_years']} años)\n"
                report += f"  Variables: {len(info['variables'])}\n"
                report += f"  Completitud promedio: {info['avg_completeness']:.1f}%\n"
                report += "\n"
            else:
                report += f"Regional: {self.REGIONALES[regional_code]} ({regional_code})\n"
                report += "   NO CARGADA\n\n"
        
        return report
    
    # ========================================================================
    # MÉTODOS NUEVOS: Validación de compatibilidad con analizador
    # ========================================================================
    
    def validate_analyzer_compatibility(self, regional_code: str) -> Dict[str, Any]:
        """
        Validar que las variables extraídas son compatibles con el analizador de correlación.
        
        Returns:
            Dict con estadísticas de compatibilidad
        """
        if regional_code not in self._climate_data:
            return {'error': f'Regional {regional_code} no tiene datos cargados'}
        
        df = self._climate_data[regional_code]
        
        # Variables que el analizador espera (lista completa del SAIDIClimateAnalyzer)
        analyzer_expected_vars = {
            # Temperatura
            'temperature_avg', 'temperature_min', 'temperature_max', 'temp_days',
            
            # Humedad
            'humidity_avg', 'humidity_days',
            
            # Punto de rocío
            'dewpoint_avg', 'dewpoint_min', 'dewpoint_max', 'dewpoint_days',
            
            # Presión atmosférica
            'pressure_rel_avg', 'pressure_abs_avg', 'pressure_rel_days', 'pressure_abs_days',
            
            # Precipitación
            'precipitation_total', 'precipitation_avg_daily', 'precipitation_max_daily',
            'days_with_rain', 'precip_total_days', 'precip_rate_avg', 'precip_rate_max', 'precip_rate_total',
            
            # Evapotranspiración
            'eto_avg', 'eto_days',
            
            # Radiación solar
            'solar_rad_avg', 'solar_rad_max', 'solar_rad_days',
            
            # Iluminancia
            'illuminance_avg', 'illuminance_max', 'illuminance_days',
            
            # Índice UV
            'uv_index_avg', 'uv_index_max', 'uv_index_days',
            
            # Índice de calor
            'heat_index_avg', 'heat_index_min', 'heat_index_max', 'heat_index_days',
            
            # Sensación térmica (windchill)
            'windchill_avg', 'windchill_min', 'windchill_max', 'windchill_days',
            
            # Temperatura aparente (RealFeel)
            'realfeel_avg', 'realfeel_min', 'realfeel_max', 'realfeel_days',
            
            # Viento
            'wind_speed_avg', 'wind_speed_max', 'wind_speed_days',
            'wind_gust_avg', 'wind_gust_max', 'wind_gust_days',
            'wind_dir_avg', 'wind_dir_max', 'wind_dir_days'
        }
        
        # Variables realmente disponibles (excluyendo metadata)
        available_vars = set(df.columns) - {'fecha', 'year', 'month', 'regional'}
        
        # Calcular compatibilidad
        present_vars = analyzer_expected_vars & available_vars
        missing_vars = analyzer_expected_vars - available_vars
        extra_vars = available_vars - analyzer_expected_vars
        
        # Calcular completitud por categoría
        category_completeness = {}
        
        categories = {
            'Temperatura': ['temperature_', 'temp_', 'heat_index_', 'windchill_', 'realfeel_'],
            'Humedad': ['humidity_', 'dewpoint_'],
            'Precipitación': ['precipitation_', 'precip_', 'days_with_rain'],
            'Presión': ['pressure_'],
            'Viento': ['wind_'],
            'Radiación': ['solar_rad_', 'uv_index_', 'illuminance_'],
            'Evapotranspiración': ['eto_']
        }
        
        for category, prefixes in categories.items():
            expected = [v for v in analyzer_expected_vars if any(v.startswith(p) for p in prefixes)]
            available = [v for v in present_vars if any(v.startswith(p) for p in prefixes)]
            
            if expected:
                completeness = len(available) / len(expected) * 100
                category_completeness[category] = {
                    'expected': len(expected),
                    'available': len(available),
                    'completeness_pct': completeness
                }
        
        compatibility_score = (len(present_vars) / len(analyzer_expected_vars) * 100) if analyzer_expected_vars else 0
        
        result = {
            'regional_code': regional_code,
            'regional_name': self.REGIONALES[regional_code],
            'compatibility_score': compatibility_score,
            'total_expected': len(analyzer_expected_vars),
            'total_available': len(present_vars),
            'total_missing': len(missing_vars),
            'present_vars': sorted(list(present_vars)),
            'missing_vars': sorted(list(missing_vars)),
            'extra_vars': sorted(list(extra_vars)),
            'category_completeness': category_completeness
        }
        
        return result
    
    def print_compatibility_report(self):
        """Imprimir reporte de compatibilidad con analizador para todas las regionales cargadas"""
        
        print("\n" + "="*80)
        print("REPORTE DE COMPATIBILIDAD CON ANALIZADOR DE CORRELACIÓN")
        print("="*80)
        
        for regional_code in self.get_loaded_regionales():
            validation = self.validate_analyzer_compatibility(regional_code)
            
            print(f"\n{'='*60}")
            print(f"Regional: {validation['regional_name']} ({regional_code})")
            print(f"{'='*60}")
            print(f"Compatibilidad: {validation['compatibility_score']:.1f}%")
            print(f"Variables disponibles: {validation['total_available']}/{validation['total_expected']}")
            
            if validation['compatibility_score'] >= 90:
                status = " EXCELENTE"
            elif validation['compatibility_score'] >= 75:
                status = " BUENA"
            elif validation['compatibility_score'] >= 60:
                status = " ACEPTABLE"
            else:
                status = " INSUFICIENTE"
            
            print(f"Estado: {status}")
            
            print(f"\nCompletitud por categoría:")
            for category, stats in validation['category_completeness'].items():
                completeness = stats['completeness_pct']
                indicator = "✅" if completeness >= 80 else "⚠️" if completeness >= 50 else "❌"
                print(f"  {indicator} {category}: {stats['available']}/{stats['expected']} ({completeness:.0f}%)")
            
            if validation['missing_vars']:
                print(f"\n Variables faltantes críticas ({len(validation['missing_vars'])}):")
                # Mostrar solo las primeras 10 para no saturar
                for var in validation['missing_vars'][:10]:
                    print(f"  - {var}")
                if len(validation['missing_vars']) > 10:
                    print(f"  ... y {len(validation['missing_vars']) - 10} más")
        
        print("\n" + "="*80)
    
    def get_compatibility_summary(self) -> Dict[str, Any]:
        """
        Obtener resumen de compatibilidad de todas las regionales cargadas
        
        Returns:
            Dict con resumen general de compatibilidad
        """
        summary = {
            'regionales': {},
            'overall_stats': {
                'total_regionales': len(self.get_loaded_regionales()),
                'excellent': 0,  # >= 90%
                'good': 0,       # >= 75%
                'acceptable': 0, # >= 60%
                'insufficient': 0 # < 60%
            }
        }
        
        for regional_code in self.get_loaded_regionales():
            validation = self.validate_analyzer_compatibility(regional_code)
            
            score = validation['compatibility_score']
            summary['regionales'][regional_code] = {
                'name': validation['regional_name'],
                'score': score,
                'available': validation['total_available'],
                'expected': validation['total_expected'],
                'missing': validation['total_missing']
            }
            
            # Clasificar
            if score >= 90:
                summary['overall_stats']['excellent'] += 1
            elif score >= 75:
                summary['overall_stats']['good'] += 1
            elif score >= 60:
                summary['overall_stats']['acceptable'] += 1
            else:
                summary['overall_stats']['insufficient'] += 1
        
        return summary
    
    def export_compatibility_report(self, output_path: str = "climate_compatibility_report.txt") -> bool:
        """
        Exportar reporte de compatibilidad a archivo de texto
        
        Args:
            output_path: Ruta del archivo de salida
            
        Returns:
            True si se exportó exitosamente
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("="*80 + "\n")
                f.write("REPORTE DE COMPATIBILIDAD CON ANALIZADOR DE CORRELACIÓN\n")
                f.write("="*80 + "\n\n")
                
                summary = self.get_compatibility_summary()
                
                f.write("RESUMEN GENERAL\n")
                f.write("-" * 40 + "\n")
                f.write(f"Total regionales cargadas: {summary['overall_stats']['total_regionales']}\n")
                f.write(f"Compatibilidad excelente (>=90%): {summary['overall_stats']['excellent']}\n")
                f.write(f"Compatibilidad buena (>=75%): {summary['overall_stats']['good']}\n")
                f.write(f"Compatibilidad aceptable (>=60%): {summary['overall_stats']['acceptable']}\n")
                f.write(f"Compatibilidad insuficiente (<60%): {summary['overall_stats']['insufficient']}\n")
                f.write("\n")
                
                for regional_code in self.get_loaded_regionales():
                    validation = self.validate_analyzer_compatibility(regional_code)
                    
                    f.write("="*60 + "\n")
                    f.write(f"Regional: {validation['regional_name']} ({regional_code})\n")
                    f.write("="*60 + "\n")
                    f.write(f"Compatibilidad: {validation['compatibility_score']:.1f}%\n")
                    f.write(f"Variables disponibles: {validation['total_available']}/{validation['total_expected']}\n")
                    
                    if validation['compatibility_score'] >= 90:
                        status = "EXCELENTE"
                    elif validation['compatibility_score'] >= 75:
                        status = "BUENA"
                    elif validation['compatibility_score'] >= 60:
                        status = "ACEPTABLE"
                    else:
                        status = "INSUFICIENTE"
                    
                    f.write(f"Estado: {status}\n\n")
                    
                    f.write("Completitud por categoría:\n")
                    for category, stats in validation['category_completeness'].items():
                        completeness = stats['completeness_pct']
                        f.write(f"  - {category}: {stats['available']}/{stats['expected']} ({completeness:.0f}%)\n")
                    
                    if validation['missing_vars']:
                        f.write(f"\nVariables faltantes ({len(validation['missing_vars'])}):\n")
                        for var in validation['missing_vars']:
                            f.write(f"  - {var}\n")
                    
                    f.write("\n")
            
            print(f"[INFO] Reporte de compatibilidad exportado a: {output_path}")
            return True
            
        except Exception as e:
            print(f"[ERROR] Error exportando reporte: {e}")
            return False