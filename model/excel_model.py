# model/excel_model.py - Modelo de datos Excel con soporte regional

from pathlib import Path
from typing import Any, ClassVar

import pandas as pd
from PyQt6.QtCore import QObject, pyqtSignal


class ExcelModel(QObject):
    """Modelo para manejar datos de Excel con soporte para regionales."""

    # Señales
    data_loaded = pyqtSignal(dict)  # Información del archivo cargado
    status_changed = pyqtSignal(str)  # Cambios de estado
    error_occurred = pyqtSignal(str)  # Errores
    regionales_detected = pyqtSignal(list)  # Regionales detectadas

    # Mapeo de códigos de columnas a nombres de regionales
    REGIONAL_MAPPING: ClassVar[dict[str, str]] = {
        "SAIDI_C": "Cúcuta",
        "SAIDI_O": "Ocaña",
        "SAIDI_A": "Aguachica",
        "SAIDI_P": "Pamplona",
        "SAIDI_T": "Tibú",
        "SAIDI_Cens": "Empresa General",
    }

    def __init__(self):
        super().__init__()
        self._excel_data = None
        self._file_path = None
        self._validated = False
        self._file_info = {}
        self._is_regional_format = False  # indica si es formato regional
        self._available_regionales = []  # lista de regionales disponibles
        self._selected_regional = None  # regional seleccionada

    def load_excel_file(self, file_path: str) -> bool:
        """Cargar archivo Excel desde una ruta específica."""
        try:
            self.status_changed.emit("Cargando archivo Excel...")

            # Validar archivo
            if not self._validate_file_path(file_path):
                return False

            print(f"[DEBUG] Cargando archivo: {file_path}")

            # Leer archivo Excel
            df = self._read_excel_file(file_path)
            if df is None:
                return False

            print(f"[DEBUG] Dimensiones: {df.shape[0]} filas x {df.shape[1]} columnas")
            print(f"[DEBUG] Columnas detectadas: {list(df.columns)}")

            # Procesar y validar datos
            return self._process_and_validate_data(df, file_path)

        except (KeyError, AttributeError, ValueError, TypeError) as e:
            error_msg = f"Error al procesar datos del Excel: {e!s}"
            print(f"[DEBUG ERROR] {error_msg}")
            self.error_occurred.emit(error_msg)
            return False

    def _validate_file_path(self, file_path: str) -> bool:
        """Validar que el archivo existe y tiene la extensión correcta."""
        path = Path(file_path)

        # Verificar que el archivo existe
        if not path.exists():
            error_msg = f"Archivo no encontrado: {file_path}"
            self.error_occurred.emit(error_msg)
            return False

        # Verificar extensión
        file_ext = path.suffix.lower()
        if file_ext not in [".xlsx", ".xls"]:
            error_msg = f"Formato de archivo no válido: {file_ext}"
            self.error_occurred.emit(error_msg)
            return False

        return True

    def _read_excel_file(self, file_path: str) -> pd.DataFrame | None:
        """Leer archivo Excel, intentando primero 'Hoja1' y luego la primera hoja."""
        try:
            # Primero intentar leer la Hoja1, si no existe, leer la primera hoja
            try:
                df = pd.read_excel(file_path, sheet_name="Hoja1")
                print("[DEBUG] Hoja 'Hoja1' leída exitosamente")
            except ValueError:  # ValueError se lanza cuando la hoja no existe
                df = pd.read_excel(file_path, sheet_name=0)
                print("[DEBUG] Primera hoja leída exitosamente")

        except (FileNotFoundError, PermissionError, pd.errors.EmptyDataError, OSError) as e:
            error_msg = f"Error al leer Excel: {e!s}"
            self.error_occurred.emit(error_msg)
            return None
        else:
            return df

    def _process_and_validate_data(self, df: pd.DataFrame, file_path: str) -> bool:
        """Procesar y validar los datos del DataFrame."""
        # Detectar formato (regional vs tradicional)
        is_regional = self._detect_regional_format(df)
        print(f"[DEBUG] Formato regional detectado: {is_regional}")

        # Validar estructura según el formato
        validation_result = self._get_validation_result(df, is_regional=is_regional)

        if not validation_result["valid"]:
            error_msg = f"Error de validación: {validation_result['error']}"
            self.error_occurred.emit(error_msg)
            return False

        # Guardar datos validados
        self._save_validated_data(df, file_path, validation_result, is_regional=is_regional)
        return True

    def _get_validation_result(self, df: pd.DataFrame, *, is_regional: bool) -> dict[str, Any]:
        """Obtener resultado de validación según el formato."""
        if is_regional:
            return self._validate_regional_structure(df)
        return self._validate_excel_structure(df)

    def _detect_regional_format(self, df: pd.DataFrame) -> bool:
        """Detectar si el DataFrame tiene formato regional."""
        # Verificar si existe la columna year-month
        has_year_month = "year-month" in df.columns
        # Verificar si existen columnas SAIDI_X
        has_saidi_columns = any(col.startswith("SAIDI_") for col in df.columns)
        return has_year_month and has_saidi_columns

    def _save_validated_data(
        self, df: pd.DataFrame, file_path: str, validation_result: dict[str, Any], *, is_regional: bool,
    ) -> None:
        """Guardar datos validados y emitir señales correspondientes."""
        self._excel_data = df
        self._file_path = file_path
        self._validated = True
        self._is_regional_format = is_regional

        # Si es formato regional, detectar regionales disponibles
        if is_regional:
            self._available_regionales = validation_result["regionales"]
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

    def _validate_regional_structure(self, df: pd.DataFrame) -> dict[str, Any]:
        """Validar estructura de Excel con formato regional."""
        try:
            print("[DEBUG] Validando estructura regional...")

            # Validaciones básicas
            basic_validation = self._validate_regional_basic_structure(df)
            if not basic_validation["valid"]:
                return basic_validation

            # Detectar y validar columnas SAIDI
            saidi_validation = self._validate_regional_saidi_columns(df)
            if not saidi_validation["valid"]:
                return saidi_validation

            # Validar datos históricos suficientes
            historical_validation = self._validate_regional_historical_data(
                df, saidi_validation["saidi_columns"],
            )
            if not historical_validation["valid"]:
                return historical_validation

            return {
                "valid": True,
                "regionales": saidi_validation["regionales"],
                "fecha_col": "year-month",
                "saidi_columns": saidi_validation["saidi_columns"],
                "regionales_validas": historical_validation["regionales_validas"],
            }

        except (KeyError, AttributeError, TypeError) as e:
            print(f"[DEBUG ERROR] Error en validación regional: {e}")
            return {"valid": False, "error": f"Error durante la validación: {e!s}"}

    def _validate_regional_basic_structure(self, df: pd.DataFrame) -> dict[str, Any]:
        """Validar estructura básica del formato regional."""
        if df.empty:
            return {"valid": False, "error": "El archivo está vacío"}

        if "year-month" not in df.columns:
            return {"valid": False, "error": 'No se encontró columna "year-month"'}

        try:
            pd.to_datetime(df["year-month"], format="%Y-%m")
            print("[DEBUG] Columna 'year-month' validada como fechas")
        except ValueError:
            return {
                "valid": False,
                "error": 'La columna "year-month" no tiene formato válido (esperado: YYYY-MM)',
            }

        return {"valid": True}

    def _validate_regional_saidi_columns(self, df: pd.DataFrame) -> dict[str, Any]:
        """Detectar y validar columnas SAIDI regionales."""
        saidi_columns = [col for col in df.columns if col.startswith("SAIDI_")]

        if len(saidi_columns) == 0:
            return {"valid": False, "error": "No se encontraron columnas SAIDI_X"}

        print(f"[DEBUG] Columnas SAIDI detectadas: {saidi_columns}")

        # Validar que las columnas SAIDI sean numéricas
        for col in saidi_columns:
            try:
                pd.to_numeric(df[col], errors="coerce")
            except ValueError:
                return {
                    "valid": False,
                    "error": f'La columna "{col}" no contiene valores numéricos válidos',
                }

        # Mapear códigos a nombres de regionales
        regionales_disponibles = [
            {"codigo": col, "nombre": self.REGIONAL_MAPPING.get(col, col)}
            for col in saidi_columns
        ]

        print(f"[DEBUG] Regionales mapeadas: {regionales_disponibles}")

        return {
            "valid": True,
            "saidi_columns": saidi_columns,
            "regionales": regionales_disponibles,
        }

    def _validate_regional_historical_data(
        self, df: pd.DataFrame, saidi_columns: list[str],
    ) -> dict[str, Any]:
        """Validar que haya suficientes datos históricos en las columnas SAIDI."""
        min_datos = 12
        regionales_validas = []

        for col in saidi_columns:
            try:
                datos_no_nulos = df[col].notna().sum()
                if datos_no_nulos >= min_datos:
                    regionales_validas.append(col)
            except (KeyError, AttributeError):
                print(f"[DEBUG ERROR] Error al procesar columna {col}")
                continue

        if len(regionales_validas) == 0:
            return {
                "valid": False,
                "error": f"Ninguna regional tiene al menos {min_datos} observaciones válidas",
            }

        print(f"[DEBUG] Regionales con datos suficientes: {regionales_validas}")

        return {"valid": True, "regionales_validas": regionales_validas}

    def _validate_excel_structure(self, df: pd.DataFrame) -> dict[str, Any]:
        """Validar que el Excel tenga la estructura correcta para SAIDI tradicional."""
        try:
            print("[DEBUG] Validando estructura tradicional...")

            # Validar estructura básica
            basic_validation = self._validate_traditional_basic_structure(df)
            if not basic_validation["valid"]:
                return basic_validation

            # Validar columna de fecha
            fecha_col = df.columns[0]
            date_validation = self._validate_date_column(df, fecha_col)
            if not date_validation["valid"]:
                return date_validation

            # Validar columna SAIDI
            saidi_validation = self._validate_traditional_saidi_column(df)
            if not saidi_validation["valid"]:
                return saidi_validation

            saidi_col = saidi_validation["saidi_col"]

            # Validar datos históricos
            historical_validation = self._validate_traditional_historical_data(df, saidi_col)
            if not historical_validation["valid"]:
                return historical_validation

            print(f"[DEBUG] Datos históricos validados: {historical_validation['datos_historicos']} observaciones")

            return {
                "valid": True,
                "fecha_col": fecha_col,
                "saidi_col": saidi_col,
                "datos_historicos": historical_validation["datos_historicos"],
            }

        except (KeyError, AttributeError, TypeError, IndexError) as e:
            print(f"[DEBUG ERROR] Error en validación tradicional: {e}")
            return {"valid": False, "error": f"Error durante la validación: {e!s}"}

    def _validate_traditional_basic_structure(self, df: pd.DataFrame) -> dict[str, Any]:
        """Validar estructura básica del formato tradicional."""
        if df.empty:
            return {"valid": False, "error": "El archivo está vacío"}

        columnas_minimas = 2
        if len(df.columns) < columnas_minimas:
            return {"valid": False, "error": "Se necesitan al menos 2 columnas (Fecha y SAIDI)"}

        return {"valid": True}

    def _validate_date_column(self, df: pd.DataFrame, fecha_col: str) -> dict[str, Any]:
        """Validar que la columna de fecha contenga fechas válidas."""
        try:
            pd.to_datetime(df.iloc[:, 0])
        except (ValueError, TypeError):
            return {
                "valid": False,
                "error": f'La primera columna "{fecha_col}" no contiene fechas válidas',
            }
        else:
            print(f"[DEBUG] Columna '{fecha_col}' validada como fechas")
            return {"valid": True}

    def _validate_traditional_saidi_column(self, df: pd.DataFrame) -> dict[str, Any]:
        """Buscar y validar la columna SAIDI en formato tradicional."""
        saidi_col = self._find_saidi_column(df)

        if saidi_col is None:
            return {"valid": False, "error": 'No se encontró una columna con "SAIDI" en el nombre'}

        print(f"[DEBUG] Columna SAIDI detectada: '{saidi_col}'")

        # Verificar que la columna SAIDI tenga datos numéricos
        saidi_data = df[saidi_col].dropna()
        if saidi_data.empty:
            return {"valid": False, "error": f'La columna "{saidi_col}" no contiene datos válidos'}

        try:
            pd.to_numeric(saidi_data)
        except (ValueError, TypeError):
            return {
                "valid": False,
                "error": f'La columna "{saidi_col}" no contiene valores numéricos válidos',
            }

        return {"valid": True, "saidi_col": saidi_col}

    def _find_saidi_column(self, df: pd.DataFrame) -> str | None:
        """Buscar la columna que contiene 'SAIDI' en su nombre."""
        for col in df.columns[1:]:
            if "SAIDI" in str(col).upper():
                return col
        return None

    def _validate_traditional_historical_data(
        self, df: pd.DataFrame, saidi_col: str,
    ) -> dict[str, Any]:
        """Validar que haya suficientes datos históricos."""
        saidi_data = df[saidi_col].dropna()
        datos_historicos = len(saidi_data)
        meses_minimos = 12

        if datos_historicos < meses_minimos:
            return {
                "valid": False,
                "error": f"Se necesitan al menos {meses_minimos} observaciones históricas, se encontraron {datos_historicos}",
            }

        return {"valid": True, "datos_historicos": datos_historicos}

    def set_selected_regional(self, regional_codigo: str) -> bool:
        """Seleccionar una regional específica para trabajar."""
        if not self._is_regional_format:
            print("[DEBUG] No es formato regional, ignorando selección")
            return False

        if regional_codigo not in [r["codigo"] for r in self._available_regionales]:
            print(f"[DEBUG ERROR] Regional '{regional_codigo}' no disponible")
            return False

        self._selected_regional = regional_codigo
        print(f"[DEBUG] Regional seleccionada: {regional_codigo} ({self.REGIONAL_MAPPING.get(regional_codigo, regional_codigo)})")
        return True

    def get_selected_regional(self) -> str | None:
        """Obtener la regional actualmente seleccionad."""
        return self._selected_regional

    def get_available_regionales(self) -> list[dict[str, str]]:
        """Obtener lista de regionales disponibles."""
        return self._available_regionales.copy()

    def is_regional_format(self) -> bool:
        """Verificar si el archivo cargado es formato regional."""
        return self._is_regional_format

    def _generate_file_info(self) -> dict[str, Any]:
        """Generar información resumida del Excel cargado."""
        if not self.is_excel_loaded():
            return {"loaded": False}

        df = self._excel_data

        # Usar Path en lugar de os.path.basename
        file_path = Path(self._file_path)

        info = {
            "loaded": True,
            "file_name": file_path.name,  # Reemplaza os.path.basename()
            "file_path": self._file_path,
            "rows": df.shape[0],
            "columns": df.shape[1],
            "column_names": list(df.columns),
            "is_regional_format": self._is_regional_format,
        }

        if self._is_regional_format:
            info["available_regionales"] = self._available_regionales
            info["has_saidi"] = True
            info["date_range"] = {
                "start": str(df["year-month"].min()) if "year-month" in df.columns else None,
                "end": str(df["year-month"].max()) if "year-month" in df.columns else None,
            }
        else:
            info["has_saidi"] = any("SAIDI" in str(col).upper() for col in df.columns)
            info["date_range"] = {
                "start": str(df.iloc[:, 0].min()) if not df.empty else None,
                "end": str(df.iloc[:, 0].max()) if not df.empty else None,
            }

        return info

    def is_excel_loaded(self) -> bool:
        """Verificar si hay un Excel cargado y validado."""
        return self._validated and self._excel_data is not None

    def get_excel_data(self) -> pd.DataFrame | None:
        """Obtener los datos del Excel cargado (formato original)."""
        if self.is_excel_loaded():
            return self._excel_data.copy()
        return None

    def get_excel_data_for_analysis(self) -> pd.DataFrame | None:
        """Obtener datos preparados para análisis (formato unificado) Convierte formato regional a formato tradicional si es necesario."""
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
            df_analysis["Fecha"] = pd.to_datetime(df["year-month"])
            df_analysis["SAIDI"] = df[self._selected_regional]

            print(f"[DEBUG] Datos preparados para análisis de regional: {self._selected_regional}")
            print(f"[DEBUG] Observaciones: {df_analysis['SAIDI'].notna().sum()} no nulas de {len(df_analysis)} totales")

            return df_analysis
        # Ya está en formato tradicional
        return df

    def get_file_path(self) -> str | None:
        """Obtener la ruta del archivo Excel cargado."""
        return self._file_path if self.is_excel_loaded() else None

    def get_file_name(self) -> str | None:
        """Obtener solo el nombre del archivo Excel cargado."""
        if self._file_path:
            return Path(self._file_path).name
        return None

    def get_file_info(self) -> dict[str, Any]:
        """Obtener información del archivo cargado."""
        return self._file_info.copy() if self._file_info else {"loaded": False}

    def clear_excel(self):
        """Limpiar los datos cargados."""
        self._excel_data = None
        self._file_path = None
        self._validated = False
        self._file_info = {}
        self._is_regional_format = False
        self._available_regionales = []
        self._selected_regional = None

        self.status_changed.emit("Datos limpiados")
        print("[DEBUG] Excel data cleared")

    def get_saidi_column(self) -> str | None:
        """Obtener el nombre de la columna SAIDI."""
        if not self.is_excel_loaded():
            return None

        if self._is_regional_format:
            return self._selected_regional if self._selected_regional else None
        df = self._excel_data
        for col in df.columns:
            if "SAIDI" in str(col).upper():
                return col
        return None

    def get_date_column(self) -> str | None:
        """Obtener el nombre de la columna de fecha."""
        if not self.is_excel_loaded():
            return None

        if self._is_regional_format:
            return "year-month"
        df = self._excel_data
        if "Fecha" in df.columns:
            return "Fecha"
        return df.columns[0]

    def get_historical_data(self) -> pd.DataFrame | None:
        """Obtener solo los datos históricos (sin NaN)."""
        df = self.get_excel_data_for_analysis()
        if df is None:
            return None

        saidi_col = "SAIDI"  # Siempre 'SAIDI' después de get_excel_data_for_analysis
        date_col = "Fecha"  # Siempre 'Fecha' después de get_excel_data_for_analysis

        if saidi_col not in df.columns:
            return None

        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col)

        return df[df[saidi_col].notna()]

    def get_missing_data(self) -> pd.DataFrame | None:
        """Obtener datos faltantes (NaN)."""
        df = self.get_excel_data_for_analysis()
        if df is None:
            return None

        saidi_col = "SAIDI"
        date_col = "Fecha"

        if saidi_col not in df.columns:
            return None

        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col)

        return df[df[saidi_col].isna()]
