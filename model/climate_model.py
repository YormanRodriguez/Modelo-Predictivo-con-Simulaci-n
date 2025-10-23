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
        'date': 'year_month',  # Columna principal de fecha (YYYY-MM)
        'year': 'year',
        'month': 'month',
        
        # Temperatura (promedios mensuales)
        'temperature_avg': 'wswdat_temp_c_monthly_avg',
        'temperature_min': 'wswdat_temp_c_monthly_min',
        'temperature_max': 'wswdat_temp_c_monthly_max',
        
        # Humedad relativa
        'humidity_avg': 'wswdat_relative_humidity_monthly_avg',
        
        # Precipitación
        'precipitation_total': 'wswdat_precip_today_mm_monthly_total',
        'precipitation_avg_daily': 'wswdat_precip_today_mm_monthly_avg_daily',
        'precipitation_max_daily': 'wswdat_precip_today_mm_max_daily',
        'days_with_rain': 'wswdat_precip_today_mm_days_with_precip',
        
        # Presión atmosférica
        'pressure_rel_avg': 'wswdat_pressure_rel_hpa_monthly_avg',
        'pressure_abs_avg': 'wswdat_pressure_abs_hpa_monthly_avg',
        
        # Viento
        'wind_speed_avg': 'wswdat_wind_speed_kmh_monthly_avg',
        'wind_speed_max': 'wswdat_wind_speed_kmh_monthly_max',
        'wind_gust_avg': 'wswdat_wind_gust_kmh_monthly_avg',
        'wind_gust_max': 'wswdat_wind_gust_kmh_monthly_max',
        
        # Radiación solar
        'solar_rad_avg': 'wswdat_solar_rad_wm2_monthly_avg',
        'solar_rad_max': 'wswdat_solar_rad_wm2_monthly_max',
        
        # Índice UV
        'uv_index_avg': 'wswdat_uv_index_monthly_avg',
        'uv_index_max': 'wswdat_uv_index_monthly_max',
        
        # Punto de rocío
        'dewpoint_avg': 'wswdat_dewpoint_c_monthly_avg',
        
        # Índice de calor
        'heat_index_avg': 'wswdat_heat_index_c_monthly_avg',
        
        # Evapotranspiración
        'eto_avg': 'wswdat_eto_mm_monthly_avg'
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
        """Procesar y limpiar datos climáticos mensuales"""
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
            
            # Seleccionar y renombrar columnas relevantes
            columns_to_extract = {}
            
            # Temperatura
            if self.CLIMATE_COLUMNS['temperature_avg'] in df_clean.columns:
                columns_to_extract[self.CLIMATE_COLUMNS['temperature_avg']] = 'temp_avg'
            if self.CLIMATE_COLUMNS['temperature_min'] in df_clean.columns:
                columns_to_extract[self.CLIMATE_COLUMNS['temperature_min']] = 'temp_min'
            if self.CLIMATE_COLUMNS['temperature_max'] in df_clean.columns:
                columns_to_extract[self.CLIMATE_COLUMNS['temperature_max']] = 'temp_max'
            
            # Humedad
            if self.CLIMATE_COLUMNS['humidity_avg'] in df_clean.columns:
                columns_to_extract[self.CLIMATE_COLUMNS['humidity_avg']] = 'humedad_avg'
            
            # Precipitación
            if self.CLIMATE_COLUMNS['precipitation_total'] in df_clean.columns:
                columns_to_extract[self.CLIMATE_COLUMNS['precipitation_total']] = 'precip_total'
            if self.CLIMATE_COLUMNS['precipitation_avg_daily'] in df_clean.columns:
                columns_to_extract[self.CLIMATE_COLUMNS['precipitation_avg_daily']] = 'precip_avg_daily'
            if self.CLIMATE_COLUMNS['days_with_rain'] in df_clean.columns:
                columns_to_extract[self.CLIMATE_COLUMNS['days_with_rain']] = 'dias_lluvia'
            
            # Presión
            if self.CLIMATE_COLUMNS['pressure_rel_avg'] in df_clean.columns:
                columns_to_extract[self.CLIMATE_COLUMNS['pressure_rel_avg']] = 'presion_avg'
            
            # Viento
            if self.CLIMATE_COLUMNS['wind_speed_avg'] in df_clean.columns:
                columns_to_extract[self.CLIMATE_COLUMNS['wind_speed_avg']] = 'viento_avg'
            if self.CLIMATE_COLUMNS['wind_speed_max'] in df_clean.columns:
                columns_to_extract[self.CLIMATE_COLUMNS['wind_speed_max']] = 'viento_max'
            
            # Radiación solar
            if self.CLIMATE_COLUMNS['solar_rad_avg'] in df_clean.columns:
                columns_to_extract[self.CLIMATE_COLUMNS['solar_rad_avg']] = 'radiacion_solar_avg'
            
            # Índice UV
            if self.CLIMATE_COLUMNS['uv_index_avg'] in df_clean.columns:
                columns_to_extract[self.CLIMATE_COLUMNS['uv_index_avg']] = 'uv_index_avg'
            
            # Evapotranspiración
            if self.CLIMATE_COLUMNS['eto_avg'] in df_clean.columns:
                columns_to_extract[self.CLIMATE_COLUMNS['eto_avg']] = 'eto_avg'
            
            # Crear DataFrame final con columnas seleccionadas
            df_final = pd.DataFrame()
            df_final['fecha'] = df_clean['fecha']
            
            # Agregar year y month
            df_final['year'] = df_final['fecha'].dt.year
            df_final['month'] = df_final['fecha'].dt.month
            
            # Convertir columnas numéricas
            for original_col, new_col in columns_to_extract.items():
                df_final[new_col] = pd.to_numeric(df_clean[original_col], errors='coerce')
            
            # Agregar identificador de regional
            df_final['regional'] = regional_code
            
            # Ordenar por fecha
            df_final = df_final.sort_values('fecha').reset_index(drop=True)
            
            print(f"[DEBUG CLIMATE] Datos procesados: {len(df_final)} meses")
            print(f"[DEBUG CLIMATE] Variables extraídas: {list(df_final.columns)}")
            print(f"[DEBUG CLIMATE] Rango temporal: {df_final['fecha'].min()} a {df_final['fecha'].max()}")
            
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