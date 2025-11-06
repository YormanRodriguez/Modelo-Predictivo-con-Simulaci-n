# model/excel_model.py - Modelo de datos Excel con soporte regional
import pandas as pd
import os
from typing import Optional, Dict, Any, List
from PyQt6.QtCore import QObject, pyqtSignal

class ExcelModel(QObject):
    """Modelo para manejar datos de Excel con soporte para regionales"""
    
    # Señales
    data_loaded = pyqtSignal(dict)  # Información del archivo cargado
    status_changed = pyqtSignal(str)  # Cambios de estado
    error_occurred = pyqtSignal(str)  # Errores
    regionales_detected = pyqtSignal(list)  # Regionales detectadas
    
    # Mapeo de códigos de columnas a nombres de regionales
    REGIONAL_MAPPING = {
        'SAIDI_C': 'Cúcuta',
        'SAIDI_O': 'Ocaña',
        'SAIDI_A': 'Aguachica',
        'SAIDI_P': 'Pamplona',
        'SAIDI_T': 'Tibú',
        'SAIDI_Cens': 'Empresa General'
    }
    
    def __init__(self):
        super().__init__()
        self._excel_data = None
        self._file_path = None
        self._validated = False
        self._file_info = {}
        self._is_regional_format = False  #indica si es formato regional
        self._available_regionales = []  #lista de regionales disponibles
        self._selected_regional = None  #regional seleccionada
        
    def load_excel_file(self, file_path: str) -> bool:
        """Cargar archivo Excel desde una ruta específica"""
        try:
            self.status_changed.emit("Cargando archivo Excel...")
            
            # Verificar que el archivo existe
            if not os.path.exists(file_path):
                error_msg = f"Archivo no encontrado: {file_path}"
                self.error_occurred.emit(error_msg)
                return False
                
            # Verificar extensión
            file_ext = os.path.splitext(file_path)[1].lower()
            if file_ext not in ['.xlsx', '.xls']:
                error_msg = f"Formato de archivo no válido: {file_ext}"
                self.error_occurred.emit(error_msg)
                return False
                
            print(f"[DEBUG] Cargando archivo: {file_path}")
            
            # Intentar leer el archivo
            try:
                # Primero intentar leer la Hoja1, si no existe, leer la primera hoja
                try:
                    df = pd.read_excel(file_path, sheet_name="Hoja1")
                    print("[DEBUG] Hoja 'Hoja1' leída exitosamente")
                except (FileNotFoundError, pd.errors.EmptyDataError):
                    df = pd.read_excel(file_path, sheet_name=0)
                    print("[DEBUG] Primera hoja leída exitosamente")
                    
            except Exception as e:
                error_msg = f"Error al leer Excel: {str(e)}"
                self.error_occurred.emit(error_msg)
                return False
            
            print(f"[DEBUG] Dimensiones: {df.shape[0]} filas x {df.shape[1]} columnas")
            print(f"[DEBUG] Columnas detectadas: {list(df.columns)}")
            
            # Detectar formato (regional vs tradicional)
            is_regional = self._detect_regional_format(df)
            print(f"[DEBUG] Formato regional detectado: {is_regional}")
            
            # Validar estructura según el formato
            if is_regional:
                validation_result = self._validate_regional_structure(df)
            else:
                validation_result = self._validate_excel_structure(df, file_path)
            
            if validation_result['valid']:
                self._excel_data = df
                self._file_path = file_path
                self._validated = True
                self._is_regional_format = is_regional
                
                # Si es formato regional, detectar regionales disponibles
                if is_regional:
                    self._available_regionales = validation_result['regionales']
                    print(f"[DEBUG] Regionales disponibles: {self._available_regionales}")
                    self.regionales_detected.emit(self._available_regionales)
                else:
                    self._available_regionales = []
                
                # Generar información del archivo
                self._file_info = self._generate_file_info()
                
                print("[DEBUG] Archivo Excel cargado y validado exitosamente")
                
                # Emitir señal de datos cargados
                self.data_loaded.emit(self._file_info)
                self.status_changed.emit("Archivo Excel cargado exitosamente")
                
                return True
            else:
                error_msg = f"Error de validación: {validation_result['error']}"
                self.error_occurred.emit(error_msg)
                return False
                
        except Exception as e:
            error_msg = f"Error inesperado al cargar Excel: {str(e)}"
            print(f"[DEBUG ERROR] {error_msg}")
            self.error_occurred.emit(error_msg)
            return False
    
    def _detect_regional_format(self, df: pd.DataFrame) -> bool:
        """Detectar si el Excel tiene formato regional (year-month + SAIDI_X)"""
        try:
            # Verificar si existe columna 'year-month'
            has_year_month = 'year-month' in df.columns
            
            # Verificar si existen columnas SAIDI_X
            saidi_cols = [col for col in df.columns if col.startswith('SAIDI_')]
            has_saidi_regional = len(saidi_cols) > 0
            
            is_regional = has_year_month and has_saidi_regional
            
            print(f"[DEBUG] Columna 'year-month': {has_year_month}")
            print(f"[DEBUG] Columnas SAIDI_X encontradas: {saidi_cols}")
            
            return is_regional
            
        except Exception as e:
            print(f"[DEBUG ERROR] Error detectando formato: {e}")
            return False
    
    def _validate_regional_structure(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validar estructura de Excel con formato regional"""
        try:
            print("[DEBUG] Validando estructura regional...")
            
            # Verificar que no esté vacío
            if df.empty:
                return {'valid': False, 'error': 'El archivo está vacío'}
            
            # Verificar columna year-month
            if 'year-month' not in df.columns:
                return {'valid': False, 'error': 'No se encontró columna "year-month"'}
            
            # Validar formato de fechas en year-month
            try:
                df['year-month'] = pd.to_datetime(df['year-month'], format='%Y-%m')
                print("[DEBUG] Columna 'year-month' validada como fechas")
            except ValueError:
                return {'valid': False, 'error': 'La columna "year-month" no tiene formato válido (esperado: YYYY-MM)'}
            
            # Detectar columnas SAIDI regionales
            saidi_columns = [col for col in df.columns if col.startswith('SAIDI_')]
            
            if len(saidi_columns) == 0:
                return {'valid': False, 'error': 'No se encontraron columnas SAIDI_X'}
            
            print(f"[DEBUG] Columnas SAIDI detectadas: {saidi_columns}")
            
            # Validar que las columnas SAIDI sean numéricas
            for col in saidi_columns:
                try:
                    pd.to_numeric(df[col], errors='coerce')
                except ValueError:
                    return {'valid': False, 'error': f'La columna "{col}" no contiene valores numéricos válidos'}

            # Mapear códigos a nombres de regionales
            regionales_disponibles = []
            for col in saidi_columns:
                nombre_regional = self.REGIONAL_MAPPING.get(col, col)
                regionales_disponibles.append({
                    'codigo': col,
                    'nombre': nombre_regional
                })
            
            print(f"[DEBUG] Regionales mapeadas: {regionales_disponibles}")
            
            # Verificar datos históricos (al menos una regional con 12+ observaciones)
            min_datos = 12
            regionales_validas = []
            for col in saidi_columns:
                try:
                    datos_no_nulos = df[col].notna().sum()
                    if datos_no_nulos >= min_datos:
                        regionales_validas.append(col)
                except (KeyError, AttributeError):  # Removida la variable 'e' no utilizada
                    print(f"[DEBUG ERROR] Error al procesar columna {col}")
                    continue

            if len(regionales_validas) == 0:
                return {
                    'valid': False, 
                    'error': f'Ninguna regional tiene al menos {min_datos} observaciones válidas'
                }
            
            print(f"[DEBUG] Regionales con datos suficientes: {regionales_validas}")
            
            return {
                'valid': True,
                'regionales': regionales_disponibles,
                'fecha_col': 'year-month',
                'saidi_columns': saidi_columns,
                'regionales_validas': regionales_validas
            }
            
        except Exception as e:
            print(f"[DEBUG ERROR] Error en validación regional: {e}")
            return {'valid': False, 'error': f'Error durante la validación: {str(e)}'}
    
    def _validate_excel_structure(self, df: pd.DataFrame, file_path: str) -> Dict[str, Any]:
        """Validar que el Excel tenga la estructura correcta para SAIDI tradicional"""
        try:
            print("[DEBUG] Validando estructura tradicional...")
            
            # Verificar que no esté vacío
            if df.empty:
                return {'valid': False, 'error': 'El archivo está vacío'}
            
            if len(df.columns) < 2:
                return {'valid': False, 'error': 'Se necesitan al menos 2 columnas (Fecha y SAIDI)'}
            
            # Verificar columna de fecha (primera columna)
            fecha_col = df.columns[0]
            try:
                pd.to_datetime(df.iloc[:, 0])
                print(f"[DEBUG] Columna '{fecha_col}' validada como fechas")
            except (ValueError, TypeError):
                return {'valid': False, 'error': f'La primera columna "{fecha_col}" no contiene fechas válidas'}
            
            # Verificar columna SAIDI
            saidi_found = False
            saidi_col = None
            
            for col in df.columns[1:]:
                if 'SAIDI' in str(col).upper():
                    saidi_col = col
                    saidi_found = True
                    break
            
            if not saidi_found:
                return {'valid': False, 'error': 'No se encontró una columna con "SAIDI" en el nombre'}
            
            print(f"[DEBUG] Columna SAIDI detectada: '{saidi_col}'")
            
            # Verificar que la columna SAIDI tenga datos numéricos
            saidi_data = df[saidi_col].dropna()
            if saidi_data.empty:
                return {'valid': False, 'error': f'La columna "{saidi_col}" no contiene datos válidos'}
            
            try:
                pd.to_numeric(saidi_data)
            except (ValueError, TypeError):
                return {'valid': False, 'error': f'La columna "{saidi_col}" no contiene valores numéricos válidos'}
            
            # Verificar que haya suficientes datos históricos
            datos_historicos = len(saidi_data)
            if datos_historicos < 12:
                return {'valid': False, 'error': f'Se necesitan al menos 12 observaciones históricas, se encontraron {datos_historicos}'}
            
            print(f"[DEBUG] Datos históricos validados: {datos_historicos} observaciones")
            
            return {
                'valid': True, 
                'fecha_col': fecha_col,
                'saidi_col': saidi_col,
                'datos_historicos': datos_historicos
            }
            
        except Exception as e:
            print(f"[DEBUG ERROR] Error en validación tradicional: {e}")
            return {'valid': False, 'error': f'Error durante la validación: {str(e)}'}
    
    def set_selected_regional(self, regional_codigo: str) -> bool:
        """Seleccionar una regional específica para trabajar"""
        if not self._is_regional_format:
            print("[DEBUG] No es formato regional, ignorando selección")
            return False
        
        if regional_codigo not in [r['codigo'] for r in self._available_regionales]:
            print(f"[DEBUG ERROR] Regional '{regional_codigo}' no disponible")
            return False
        
        self._selected_regional = regional_codigo
        print(f"[DEBUG] Regional seleccionada: {regional_codigo} ({self.REGIONAL_MAPPING.get(regional_codigo, regional_codigo)})")
        return True
    
    def get_selected_regional(self) -> Optional[str]:
        """Obtener la regional actualmente seleccionada"""
        return self._selected_regional
    
    def get_available_regionales(self) -> List[Dict[str, str]]:
        """Obtener lista de regionales disponibles"""
        return self._available_regionales.copy()
    
    def is_regional_format(self) -> bool:
        """Verificar si el archivo cargado es formato regional"""
        return self._is_regional_format
    
    def _generate_file_info(self) -> Dict[str, Any]:
        """Generar información resumida del Excel cargado"""
        if not self.is_excel_loaded():
            return {'loaded': False}
        
        df = self._excel_data
        info = {
            'loaded': True,
            'file_name': os.path.basename(self._file_path),
            'file_path': self._file_path,
            'rows': df.shape[0],
            'columns': df.shape[1],
            'column_names': list(df.columns),
            'is_regional_format': self._is_regional_format
        }
        
        if self._is_regional_format:
            info['available_regionales'] = self._available_regionales
            info['has_saidi'] = True
            info['date_range'] = {
                'start': str(df['year-month'].min()) if 'year-month' in df.columns else None,
                'end': str(df['year-month'].max()) if 'year-month' in df.columns else None
            }
        else:
            info['has_saidi'] = any('SAIDI' in str(col).upper() for col in df.columns)
            info['date_range'] = {
                'start': str(df.iloc[:, 0].min()) if not df.empty else None,
                'end': str(df.iloc[:, 0].max()) if not df.empty else None
            }
        
        return info
    
    def is_excel_loaded(self) -> bool:
        """Verificar si hay un Excel cargado y validado"""
        return self._validated and self._excel_data is not None
    
    def get_excel_data(self) -> Optional[pd.DataFrame]:
        """Obtener los datos del Excel cargado (formato original)"""
        if self.is_excel_loaded():
            return self._excel_data.copy()
        return None
    
    def get_excel_data_for_analysis(self) -> Optional[pd.DataFrame]:
        """
        Obtener datos preparados para análisis (formato unificado)
        Convierte formato regional a formato tradicional si es necesario
        """
        if not self.is_excel_loaded():
            return None
        
        df = self._excel_data.copy()
        
        if self._is_regional_format:
            # Si es formato regional, necesitamos una regional seleccionada
            if not self._selected_regional:
                print("[DEBUG WARNING] No hay regional seleccionada")
                return None
            
            # Convertir a formato tradicional
            df_analysis = pd.DataFrame()
            df_analysis['Fecha'] = pd.to_datetime(df['year-month'])
            df_analysis['SAIDI'] = df[self._selected_regional]
            
            print(f"[DEBUG] Datos preparados para análisis de regional: {self._selected_regional}")
            print(f"[DEBUG] Observaciones: {df_analysis['SAIDI'].notna().sum()} no nulas de {len(df_analysis)} totales")
            
            return df_analysis
        else:
            # Ya está en formato tradicional
            return df
    
    def get_file_path(self) -> Optional[str]:
        """Obtener la ruta del archivo Excel cargado"""
        return self._file_path if self.is_excel_loaded() else None
    
    def get_file_name(self) -> Optional[str]:
        """Obtener solo el nombre del archivo Excel cargado"""
        if self._file_path:
            return os.path.basename(self._file_path)
        return None
    
    def get_file_info(self) -> Dict[str, Any]:
        """Obtener información del archivo cargado"""
        return self._file_info.copy() if self._file_info else {'loaded': False}
    
    def clear_excel(self):
        """Limpiar los datos cargados"""
        self._excel_data = None
        self._file_path = None
        self._validated = False
        self._file_info = {}
        self._is_regional_format = False
        self._available_regionales = []
        self._selected_regional = None
        
        self.status_changed.emit("Datos limpiados")
        print("[DEBUG] Excel data cleared")
    
    def get_saidi_column(self) -> Optional[str]:
        """Obtener el nombre de la columna SAIDI"""
        if not self.is_excel_loaded():
            return None
        
        if self._is_regional_format:
            return self._selected_regional if self._selected_regional else None
        else:
            df = self._excel_data
            for col in df.columns:
                if 'SAIDI' in str(col).upper():
                    return col
        return None
    
    def get_date_column(self) -> Optional[str]:
        """Obtener el nombre de la columna de fecha"""
        if not self.is_excel_loaded():
            return None
        
        if self._is_regional_format:
            return 'year-month'
        else:
            df = self._excel_data
            if "Fecha" in df.columns:
                return "Fecha"
            else:
                return df.columns[0]
    
    def get_historical_data(self) -> Optional[pd.DataFrame]:
        """Obtener solo los datos históricos (sin NaN)"""
        df = self.get_excel_data_for_analysis()
        if df is None:
            return None
        
        saidi_col = 'SAIDI'  # Siempre 'SAIDI' después de get_excel_data_for_analysis
        date_col = 'Fecha'  # Siempre 'Fecha' después de get_excel_data_for_analysis
        
        if saidi_col not in df.columns:
            return None
        
        df[date_col] = pd.to_datetime(df[date_col])
        df.set_index(date_col, inplace=True)
        
        return df[df[saidi_col].notna()]
    
    def get_missing_data(self) -> Optional[pd.DataFrame]:
        """Obtener datos faltantes (NaN)"""
        df = self.get_excel_data_for_analysis()
        if df is None:
            return None
        
        saidi_col = 'SAIDI'
        date_col = 'Fecha'
        
        if saidi_col not in df.columns:
            return None
        
        df[date_col] = pd.to_datetime(df[date_col])
        df.set_index(date_col, inplace=True)
        
        return df[df[saidi_col].isna()]