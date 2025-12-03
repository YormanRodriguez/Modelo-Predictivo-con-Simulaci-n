# services/validation_service.py
import json

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

import os
import tempfile
import traceback
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX

from services.climate_simulation_service import ClimateSimulationService

warnings.filterwarnings("ignore")

class ValidationService:
    """Servicio para validar modelos SARIMAX con transformaciones por regional."""

    # Mapeo de regionales a sus transformaciones optimas
    REGIONAL_TRANSFORMATIONS = {
        "SAIDI_O": "original",
        "SAIDI_C": "original",
        "SAIDI_A": "original",
        "SAIDI_P": "boxcox",
        "SAIDI_T": "sqrt",
        "SAIDI_Cens": "original",
    }

    # Variables exogenas por regional
    REGIONAL_EXOG_VARS = {
    "SAIDI_O": {  # Ocaña - 7 variables correlacionadas
        "realfeel_min": "Temperatura aparente mínima",           # r=0.689 *** FUERTE
        "windchill_avg": "Sensación térmica promedio",          # r=0.520 ** MODERADA-FUERTE
        "dewpoint_avg": "Punto de rocío promedio",              # r=0.470 ** MODERADA-FUERTE
        "windchill_max": "Sensación térmica máxima",            # r=0.464 ** MODERADA-FUERTE
        "dewpoint_min": "Punto de rocío mínimo",                # r=0.456 ** MODERADA-FUERTE
        "precipitation_max_daily": "Precipitación máxima diaria", # r=0.452
        "precipitation_avg_daily": "Precipitación promedio diaria", # r=0.438
    },

    "SAIDI_C": {  # Cúcuta - 4 variables correlacionadas
        "realfeel_avg": "Temperatura aparente promedio",        # r=0.573 ** MODERADA-FUERTE
        "pressure_rel_avg": "Presión relativa promedio",        # r=-0.358 (negativa)
        "wind_speed_max": "Velocidad máxima del viento",        # r=0.356
        "pressure_abs_avg": "Presión absoluta promedio",        # r=-0.356 (negativa)
    },

    "SAIDI_T": {  # Tibú - 8 variables correlacionadas
        "realfeel_avg": "Temperatura aparente promedio",        # r=0.906 *** MUY FUERTE
        "wind_dir_avg": "Dirección promedio del viento",        # r=-0.400 (negativa)
        "uv_index_avg": "Índice UV promedio",                   # r=0.385
        "heat_index_avg": "Índice de calor promedio",           # r=0.363
        "temperature_min": "Temperatura mínima",                # r=0.352
        "windchill_min": "Sensación térmica mínima",            # r=0.340
        "temperature_avg": "Temperatura promedio",              # r=0.338
        "pressure_rel_avg": "Presión relativa promedio",        # r=-0.330 (negativa)
    },

    "SAIDI_A": {  # Aguachica - 2 variables correlacionadas
        "uv_index_max": "Índice UV máximo",                     # r=0.664 *** FUERTE
        "days_with_rain": "Días con lluvia",                    # r=0.535 ** MODERADA-FUERTE
    },

    "SAIDI_P": {  # Pamplona - 3 variables correlacionadas
        "precipitation_total": "Precipitación total",           # r=0.577 ** MODERADA-FUERTE
        "precipitation_avg_daily": "Precipitación promedio diaria", # r=0.552
        "realfeel_min": "Temperatura aparente mínima",          # r=0.344
    },
}

    REGIONAL_ORDERS = {
        "SAIDI_O": {
            "order": (3, 1, 6),
            "seasonal_order": (3, 1, 0, 12),
        },
        "SAIDI_C": {
            "order": (3, 1, 2),
            "seasonal_order": (1, 1, 2, 12),
        },
        "SAIDI_A": {
            "order": (2, 1, 3),
            "seasonal_order": (2, 1, 1, 12),
        },
        "SAIDI_P": {
            "order": (4, 1, 3),
            "seasonal_order": (1, 1, 4, 12),
        },
        "SAIDI_T": {
            "order": (3, 1, 3),
            "seasonal_order": (2, 1, 2, 12),
        },
        "SAIDI_Cens": {
            "order": (4, 1, 3),
            "seasonal_order": (1, 1, 4, 12),
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

    def load_optimized_config(self, regional_code: str) -> dict[str, Any] | None:
        """
        Cargar configuración optimizada para una regional.

        Lee el archivo JSON generado por OptimizationService y retorna
        los mejores parámetros encontrados previamente.

        Args:
            regional_code: Código de la regional (ej: 'SAIDI_O')

        Returns:
            Dict con configuración óptima o None si no existe

        """
        try:

            # Ubicación del archivo de configuración
            config_file = Path(__file__).parent.parent / "config" / "optimized_models.json"

            if not config_file.exists():
                print("[LOAD_CONFIG] No existe archivo de configuraciones optimizadas")
                return None

            # Cargar configuraciones
            with open(config_file, encoding="utf-8") as f:
                configs = json.load(f)

            # Buscar configuración de la regional
            if regional_code not in configs:
                print(f"[LOAD_CONFIG] No hay configuración optimizada para {regional_code}")
                return None

            config = configs[regional_code]

            print(f"[LOAD_CONFIG] ✓ Configuración cargada para {regional_code}")
            print(f"[LOAD_CONFIG]   Transformación: {config['transformation']}")
            print(f"[LOAD_CONFIG]   Order: {config['order']}")
            print(f"[LOAD_CONFIG]   Seasonal: {config['seasonal_order']}")
            print(f"[LOAD_CONFIG]   Precisión: {config['precision_final']:.1f}%")
            print(f"[LOAD_CONFIG]   Optimizado: {config['optimization_date']}")

            return config

        except Exception as e:
            print(f"[LOAD_CONFIG] ERROR cargando configuración: {e}")
            return None

    def _get_orders_for_regional(self, regional_code):
        """
        MÉTODO ACTUALIZADO: Obtener órdenes SARIMAX específicos para una regional.

        Prioriza configuración optimizada sobre defaults hardcodeados.

        Args:
            regional_code: Código de la regional (ej: 'SAIDI_O')

        Returns:
            tuple: (order, seasonal_order) - Órdenes ARIMA y estacionales

        """
        if not regional_code:
            return self.default_order, self.default_seasonal_order

        # PRIORIDAD 1: Intentar cargar configuración optimizada
        optimized_config = self.load_optimized_config(regional_code)

        if optimized_config:
            order = tuple(optimized_config["order"])
            seasonal_order = tuple(optimized_config["seasonal_order"])

            print(f"[ORDERS] Usando parámetros OPTIMIZADOS para {regional_code}")
            print(f"[ORDERS]   Order: {order}")
            print(f"[ORDERS]   Seasonal: {seasonal_order}")
            print(f"[ORDERS]   Precisión documentada: {optimized_config['precision_final']:.1f}%")

            return order, seasonal_order

        # PRIORIDAD 2: Usar configuración hardcodeada
        if regional_code in self.REGIONAL_ORDERS:
            config = self.REGIONAL_ORDERS[regional_code]
            order = config["order"]
            seasonal_order = config["seasonal_order"]

            print(f"[ORDERS] Usando parámetros DEFAULT para {regional_code}")
            print(f"[ORDERS]   Order: {order}")
            print(f"[ORDERS]   Seasonal: {seasonal_order}")

            return order, seasonal_order

        # FALLBACK: Usar valores por defecto genéricos
        print(f"[ORDERS] Usando parámetros FALLBACK para {regional_code}")
        print(f"[ORDERS]   Order: {self.default_order}")
        print(f"[ORDERS]   Seasonal: {self.default_seasonal_order}")

        return self.default_order, self.default_seasonal_order

    def run_validation(self,
                  file_path: str | None = None,
                  df_prepared: pd.DataFrame | None = None,
                  order: tuple | None = None,
                  seasonal_order: tuple | None = None,
                  regional_code: str | None = None,
                  climate_data: pd.DataFrame | None = None,
                  simulation_config: dict | None = None,
                  progress_callback = None,
                  log_callback = None) -> dict[str, Any]:
        """
        Ejecutar validacion del modelo SARIMAX con transformacion especifica por regional.

        Carga automáticamente parámetros optimizados si existen

        Args:
            file_path: Ruta del archivo SAIDI Excel
            df_prepared: DataFrame de SAIDI ya preparado
            order: Orden ARIMA (opcional - si None usa el optimizado/default de la regional)
            seasonal_order: Orden estacional ARIMA (opcional - si None usa el optimizado/default)
            regional_code: Código de la regional
            climate_data: DataFrame con datos climáticos mensuales
            simulation_config: Configuración de simulación climática (opcional)
            progress_callback: Función para actualizar progreso
            log_callback: Función para loguear mensajes

        Returns:
            Diccionario con resultados de validacion

        """
        try:
            # ========== NUEVO: CARGAR CONFIGURACIÓN OPTIMIZADA ==========
            optimized_config = None

            if regional_code:
                optimized_config = self.load_optimized_config(regional_code)

                if optimized_config and log_callback:
                    log_callback("=" * 80)
                    log_callback("⚙️  USANDO CONFIGURACIÓN OPTIMIZADA")
                    log_callback("=" * 80)
                    log_callback(f"Regional: {regional_code}")
                    log_callback(f"Transformación: {optimized_config['transformation'].upper()}")
                    log_callback(f"Order: {optimized_config['order']}")
                    log_callback(f"Seasonal: {optimized_config['seasonal_order']}")
                    log_callback(f"Precisión documentada: {optimized_config['precision_final']:.1f}%")
                    log_callback(f"Optimizado en: {optimized_config['optimization_date']}")
                    log_callback("=" * 80)

            # Obtener parámetros (prioriza optimizados > hardcoded > default)
            if order is None or seasonal_order is None:
                order_regional, seasonal_regional = self._get_orders_for_regional(regional_code)

                if order is None:
                    order = order_regional
                if seasonal_order is None:
                    seasonal_order = seasonal_regional

                if log_callback and regional_code and not optimized_config:
                    regional_nombre = {
                        "SAIDI_O": "Ocaña",
                        "SAIDI_C": "Cúcuta",
                        "SAIDI_A": "Aguachica",
                        "SAIDI_P": "Pamplona",
                        "SAIDI_T": "Tibú",
                        "SAIDI_Cens": "CENS",
                    }.get(regional_code, regional_code)

                    log_callback(f"Usando parametros default para regional {regional_nombre}")
                    log_callback(f"   Order: {order}")
                    log_callback(f"   Seasonal Order: {seasonal_order}")

            # Determinar transformacion (prioriza optimizada)
            transformation = self._get_transformation_for_regional(regional_code)

            # Detectar si hay simulacion
            simulation_applied = simulation_config and simulation_config.get("enabled", False)

            if log_callback:
                log_callback(f"Iniciando validacion con parametros: order={order}, seasonal_order={seasonal_order}")
                log_callback(f"Regional: {regional_code} - Transformacion: {transformation.upper()}")

                if simulation_applied:
                    log_callback("=" * 60)
                    log_callback("🌦️ VALIDACIÓN CON SIMULACIÓN CLIMÁTICA")
                    log_callback("=" * 60)

                    summary = simulation_config.get("summary", {})
                    log_callback(f"Escenario: {summary.get('escenario', 'N/A')}")
                    log_callback(f"Alcance: {simulation_config.get('alcance_meses', 'N/A')} meses")
                    log_callback(f"Días base: {simulation_config.get('dias_base', 'N/A')}")
                    log_callback(f"Ajuste: {simulation_config.get('slider_adjustment', 0):+d} días")
                    log_callback(f"Total días simulados: {summary.get('dias_simulados', 'N/A')}")

                    # Mostrar variables que se modificarán
                    if "variables_afectadas" in summary:
                        log_callback("\nVariables climáticas a modificar:")
                        vars_afectadas = summary["variables_afectadas"]
                        for var_code, var_info in vars_afectadas.items():
                            change = var_info.get("cambio_porcentual", 0)
                            arrow = "↑" if change > 0 else "↓" if change < 0 else "→"
                            log_callback(f"   {arrow} {var_info['nombre']}: {change:+.1f}%")

                    log_callback("")
                    log_callback("NOTA: Validación bajo condiciones climáticas HIPOTÉTICAS")
                    log_callback("Las métricas reflejan el comportamiento del modelo")
                    log_callback("bajo el escenario simulado, NO el clima real histórico")
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

            meses=12
            if len(historico) < meses:
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
                    climate_data, df, regional_code, log_callback,
                )

                if exog_df is not None:
                    if log_callback:
                        log_callback(f"Variables exogenas disponibles: {len(exog_df.columns)}")

                    # ========== NUEVO: Validar cobertura como OptimizationService ==========
                    if not self._diagnose_exog_coverage(historico[col_saidi], exog_df, log_callback):
                        if log_callback:
                            log_callback("=" * 60)
                            log_callback("ADVERTENCIA: Cobertura insuficiente")
                            log_callback("Las variables exogenas seran DESACTIVADAS")
                            log_callback("=" * 60)
                        exog_df = None
                        exog_info = None
                    else:
                        # Solo si pasa la validación, continuar con las exógenas
                        if log_callback:
                            for var_code, var_data in exog_info.items():
                                log_callback(f"  - {var_data['nombre']}")

                        # CRÍTICO: NO ESCALAR AQUÍ
                        # SARIMAX normaliza internamente las variables exógenas
                        # Escalar manualmente causa DOBLE ESCALADO y pérdida de precisión

                        if simulation_applied:
                            # [Código de simulación - se omite por ahora]
                            pass
                        else:
                            # ========== CAMBIO CRÍTICO: Sin simulación tampoco escalar ==========
                            if log_callback:
                                log_callback("Variables exogenas en escala ORIGINAL")
                                log_callback("SARIMAX las normalizara internamente")

                            # Guardar scaler solo para compatibilidad, pero NO transformar
                            self.exog_scaler = StandardScaler()
                            self.exog_scaler.fit(exog_df)  # Solo FIT, NO transform
                            # exog_df permanece completamente SIN ESCALAR
                elif log_callback:
                    log_callback("No se pudieron preparar variables exogenas, continuando sin ellas")
            elif log_callback:
                log_callback("No hay datos climaticos disponibles, validacion sin variables exogenas")

            if progress_callback:
                progress_callback(30, "Dividiendo datos para validacion...")

            # Determinar porcentaje de validacion (IDENTICO a OptimizationService)
            n_obervaciones_mayor_60=60
            n_obervaciones_mayor_36=36
            n_obs = len(historico)
            if n_obs >= n_obervaciones_mayor_60:
                pct_validacion = 0.30
            elif n_obs >= n_obervaciones_mayor_36:
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
                datos_entrenamiento_original.values, transformation,
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
                try:
                    # Extraer variables para entrenamiento (sin escalar)
                    exog_train = exog_df.loc[datos_entrenamiento_original.index]

                    if simulation_applied:
                        if log_callback:
                            log_callback("=" * 60)
                            log_callback("PREPARANDO VALIDACIÓN CON SIMULACIÓN CLIMÁTICA")
                            log_callback("=" * 60)

                        # Obtener variables SIN ESCALAR para el periodo de validación
                        exog_test_original = exog_df.loc[datos_validacion_original.index]

                        if log_callback:
                            log_callback("Variables de validación ANTES de simulación:")
                            log_callback(f"  Periodo: {exog_test_original.index[0].strftime('%Y-%m')} a {exog_test_original.index[-1].strftime('%Y-%m')}")
                            log_callback(f"  Variables: {len(exog_test_original.columns)}")
                            log_callback(f"  Shape: {exog_test_original.shape}")

                        # Aplicar simulación (retorna en escala original)
                        exog_test = self._apply_climate_simulation(
                            exog_test_original, simulation_config, log_callback,
                        )

                        if log_callback:
                            summary = simulation_config.get("summary", {})
                            log_callback("\n✓ Simulación aplicada a periodo de validación:")
                            log_callback(f"  - Escenario: {summary.get('escenario', 'N/A')}")
                            log_callback(f"  - Alcance: {simulation_config.get('alcance_meses', 'N/A')} meses")
                            log_callback(f"  - Días simulados: {summary.get('dias_simulados', 'N/A')}")
                            log_callback(f"  - Periodos afectados: {len(exog_test)}")
                            log_callback("=" * 60)
                    else:
                        # Sin simulación: usar directamente (sin escalar)
                        exog_test = exog_df.loc[datos_validacion_original.index]

                        if log_callback:
                            log_callback("Variables de validación SIN simulación (escala original)")

                    # Validación de dimensiones
                    if log_callback:
                        log_callback("\nVariables exógenas preparadas:")
                        log_callback(f"  - Entrenamiento: {len(exog_train)} periodos x {exog_train.shape[1]} variables")
                        log_callback(f"  - Validación: {len(exog_test)} periodos x {exog_test.shape[1]} variables")
                        if simulation_applied:
                            log_callback("  - Modo: CON SIMULACIÓN CLIMÁTICA")
                        else:
                            log_callback("  - Modo: SIN SIMULACIÓN")
                        log_callback("  - Escala: ORIGINAL (SARIMAX normaliza internamente)")

                except Exception as e:
                    if log_callback:
                        log_callback(f"ERROR preparando variables exógenas: {e!s}")
                        log_callback(traceback.format_exc())

                    # Fallback: desactivar exógenas
                    exog_train = None
                    exog_test = None

                    if log_callback:
                        log_callback("ADVERTENCIA: Variables exógenas desactivadas por error")

            # Entrenar modelo con datos TRANSFORMADOS
            try:
                model = SARIMAX(
                    datos_entrenamiento_transformed,
                    exog=exog_train,
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=True,
                    enforce_invertibility=True,
                )
                results = model.fit(disp=False)

                if log_callback:
                    log_callback(f"Modelo SARIMAX ajustado con transformacion {transformation.upper()}")
                    if exog_train is not None:
                        log_callback(f"Modelo incluye {exog_train.shape[1]} variables exogenas")

            except Exception as e:
                raise Exception(f"Error ajustando modelo: {e!s}")

            if progress_callback:
                progress_callback(70, "Generando predicciones de validacion...")

            # Predecir en escala TRANSFORMADA
            try:
                pred = results.get_forecast(steps=n_test, exog=exog_test)
                predicciones_transformed = pred.predicted_mean

                # Revertir predicciones a escala ORIGINAL
                predicciones_original = self._inverse_transformation(
                    predicciones_transformed.values, transformation,
                )

                predicciones_validacion = pd.Series(predicciones_original, index=predicciones_transformed.index)

                if log_callback:
                    log_callback(f"Predicciones generadas y revertidas a escala original para {len(predicciones_validacion)} periodos")
                    if simulation_applied:
                        log_callback("  (basadas en condiciones climaticas simuladas)")

            except Exception as e:
                raise Exception(f"Error generando predicciones: {e!s}")

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
                n_test,
            )

            # Calcular complejidad del modelo (IDENTICO a OptimizationService)
            complexity_penalty = sum(order) + sum(seasonal_order[:3])
            composite_score = metricas["rmse"] + (complexity_penalty * 0.05)

            # Calcular estabilidad (IDENTICO a OptimizationService)
            stability_score = self._calculate_stability_numpy(
                datos_validacion_original.values,
                predicciones_original,
                metricas["precision_final"],
                metricas["mape"],
            )

            # Agregar métricas adicionales
            metricas["composite_score"] = composite_score
            metricas["stability_score"] = stability_score
            metricas["complexity"] = complexity_penalty

            if log_callback:
                log_callback("=" * 60)
                if simulation_applied:
                    log_callback("METRICAS DE VALIDACION CON SIMULACION CLIMATICA")
                    summary = simulation_config.get("summary", {})
                    log_callback(f"Escenario simulado: {summary.get('escenario', 'N/A')}")
                    log_callback(f"Alcance: {simulation_config.get('alcance_meses', 'N/A')} meses")
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

                # Interpretación de precisión
                precision_excelente=60
                precision_buena=40
                precision_aceptable=20
                precision = metricas["precision_final"]
                if precision >= precision_excelente:
                    interpretacion = "EXCELENTE - Predicciones muy confiables"
                elif precision >= precision_buena:
                    interpretacion = "BUENO - Predicciones confiables"
                elif precision >= precision_aceptable:
                    interpretacion = "ACEPTABLE - Predicciones moderadamente confiables"
                else:
                    interpretacion = "LIMITADO - Modelo poco confiable"

                log_callback(f"INTERPRETACION: {interpretacion}")
                log_callback(f"Validacion: {pct_validacion*100:.0f}% de datos como test ({n_test} meses)")

                if simulation_applied:
                    log_callback("")
                    log_callback(" ADVERTENCIA: Métricas bajo condiciones climáticas SIMULADAS")
                    log_callback("   Los valores reflejan el desempeño del modelo bajo el escenario:")
                    log_callback(f"   '{summary.get('escenario', 'N/A')}' con {summary.get('dias_simulados', 'N/A')} días")
                    log_callback("   Los valores reales pueden DIFERIR significativamente")
                    log_callback("   si el clima no sigue este patrón hipotético")

                if exog_info:
                    log_callback("\nVariables exogenas utilizadas en validacion:")
                    for var_code, var_data in exog_info.items():
                        correlacion_str = f" (r={var_data['correlacion']:.3f})" if var_data.get("correlacion", 0) != 0 else ""
                        log_callback(f"  - {var_data['nombre']}{correlacion_str}")

                    if simulation_applied:
                        log_callback("Estas variables fueron MODIFICADAS según el escenario simulado")

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
                simulation_config if simulation_applied else None,
            )

            if progress_callback:
                progress_callback(100, "Validacion completada exitosamente")

            return {
                "success": True,
                "metrics": metricas,
                "model_params": {
                    "order": order,
                    "seasonal_order": seasonal_order,
                    "transformation": transformation,
                    "regional_code": regional_code,
                    "with_exogenous": exog_df is not None,
                    "with_simulation": simulation_applied,
                    "complexity": complexity_penalty,
                },
                "predictions": {
                    "mean": predicciones_validacion.to_dict(),
                },
                "exogenous_vars": exog_info,
                "simulation_config": simulation_config if simulation_applied else None,
                "training_count": len(datos_entrenamiento_original),
                "validation_count": len(datos_validacion_original),
                "validation_percentage": pct_validacion * 100,
                "training_period": {
                    "start": datos_entrenamiento_original.index[0].strftime("%Y-%m"),
                    "end": datos_entrenamiento_original.index[-1].strftime("%Y-%m"),
                },
                "validation_period": {
                    "start": datos_validacion_original.index[0].strftime("%Y-%m"),
                    "end": datos_validacion_original.index[-1].strftime("%Y-%m"),
                },
                "plot_file": plot_path,
            }

        except Exception as e:
            if log_callback:
                log_callback(f"ERROR: {e!s}")
            raise Exception(f"Error en validacion: {e!s}")

    def _apply_climate_simulation(
    self, exog_forecast_original, simulation_config, log_callback=None,
    ):
        """
        Aplicar simulación climática a variables SIN ESCALAR.

        Args:
            exog_forecast_original: DataFrame SIN ESCALAR con variables exógenas
            simulation_config: Dict con configuración de simulación
            log_callback: Función para logging

        Returns:
            DataFrame con simulación aplicada (sin escalar para SARIMAX)

        """
        try:
            # Validar que la simulación esté habilitada
            if not simulation_config.get("enabled", False):
                if log_callback:
                    log_callback("Simulación NO habilitada, usando valores originales")
                return exog_forecast_original

            if log_callback:
                log_callback("=" * 60)
                log_callback("APLICANDO SIMULACIÓN CLIMÁTICA EN VALIDACIÓN")
                log_callback("=" * 60)
                log_callback("   Entrada: valores originales SIN ESCALAR")

            escenario = simulation_config.get("scenario_name",
                        simulation_config.get("escenario", "condiciones_normales"))

            # Extraer configuración del slider
            slider_adjustment = simulation_config.get("slider_adjustment", 0)
            dias_base = simulation_config.get("dias_base", 30)
            alcance_meses = simulation_config.get("alcance_meses", 3)
            percentiles = simulation_config.get("percentiles", {})
            regional_code = simulation_config.get("regional_code", "SAIDI_O")

            # Validar que tengamos los datos necesarios
            if not percentiles:
                if log_callback:
                    log_callback("ERROR: No hay percentiles disponibles para simulación")
                    log_callback("Usando valores originales sin simulación")
                return exog_forecast_original

            if log_callback:
                log_callback(f"   Escenario: {escenario}")
                log_callback(f"   Regional: {regional_code}")
                log_callback(f"   Slider: {slider_adjustment:+d} días sobre base de {dias_base}")
                log_callback(f"   Alcance: {alcance_meses} mes(es)")

            # Calcular factor de intensidad
            dias_simulados = dias_base + slider_adjustment
            intensity_adjustment = dias_simulados / dias_base if dias_base > 0 else 1.0

            if log_callback:
                log_callback(f"   Intensidad calculada: {intensity_adjustment:.2f}x")

            # ========== APLICAR SIMULACIÓN ==========
            try:
                exog_simulated = self.simulation_service.apply_simulation(
                    exog_forecast=exog_forecast_original,
                    scenario_name=escenario,
                    intensity_adjustment=intensity_adjustment,
                    alcance_meses=alcance_meses,
                    percentiles=percentiles,
                    regional_code=regional_code,
                )

                if log_callback:
                    log_callback("   ✓ Simulación aplicada correctamente en validación")

                    # Mostrar cambios en el primer mes
                    if alcance_meses >= 1 and len(exog_simulated) > 0:
                        log_callback("\n Cambios en primer mes (validación):")
                        for col in exog_simulated.columns:
                            try:
                                original_val = exog_forecast_original.iloc[0][col]
                                simulated_val = exog_simulated.iloc[0][col]

                                if original_val != 0:
                                    change_pct = ((simulated_val - original_val) / original_val) * 100
                                else:
                                    change_pct = 0

                                log_callback(
                                    f"     - {col}: {original_val:.2f} → {simulated_val:.2f} "
                                    f"({change_pct:+.1f}%)",
                                )
                            except Exception as e:
                                log_callback(f"     - {col}: Error mostrando cambio: {e}")

                    log_callback("\n   Salida: valores SIMULADOS (escala original)")
                    log_callback("=" * 60)

                return exog_simulated

            except Exception as sim_error:
                if log_callback:
                    log_callback(f"ERROR en apply_simulation: {sim_error!s}")
                    log_callback(traceback.format_exc())

                # Fallback: retornar valores originales
                if log_callback:
                    log_callback("FALLBACK: Usando valores originales sin simulación")
                return exog_forecast_original

        except Exception as e:
            if log_callback:
                log_callback(f"ERROR CRÍTICO en _apply_climate_simulation: {e!s}")
                log_callback(traceback.format_exc())

            # En caso de error total, retornar valores originales
            return exog_forecast_original

    def _get_transformation_for_regional(self, regional_code: str | None) -> str:
        """
        Obtener transformación para la regional.

        Prioriza configuración optimizada sobre defaults hardcodeados.

        Args:
            regional_code: Código de la regional (ej: 'SAIDI_O')

        Returns:
            str: Tipo de transformación a aplicar

        """
        if not regional_code:
            return "original"

        # PRIORIDAD 1: Intentar cargar configuración optimizada
        optimized_config = self.load_optimized_config(regional_code)

        if optimized_config:
            transformation = optimized_config.get("transformation", "original")
            print(f"[TRANSFORMATION] Usando transformación OPTIMIZADA: {transformation}")
            return transformation

        # PRIORIDAD 2: Usar defaults hardcodeados
        if regional_code in self.REGIONAL_TRANSFORMATIONS:
            transformation = self.REGIONAL_TRANSFORMATIONS[regional_code]
            print(f"[TRANSFORMATION] Usando transformación DEFAULT: {transformation}")
            return transformation

        # FALLBACK: Original
        print("[TRANSFORMATION] Usando transformación FALLBACK: original")
        return "original"

    def _prepare_exogenous_variables(self,
                                 climate_data: pd.DataFrame,
                                 df_saidi: pd.DataFrame,
                                 regional_code: str | None,
                                 log_callback) -> tuple[pd.DataFrame | None, dict | None]:
        """
        Preparar variables exógenas climáticas SIN ESCALAR.

        CAMBIOS CRÍTICOS:
        1. Mapeo de columnas con coincidencia parcial (no solo exacta)
        2. Validación de cobertura temporal (80% mínimo en overlap)
        3. Validación de varianza no-cero en overlap
        4. Relleno inteligente: forward-fill + backward-fill (máx 3) + media
        5. RETORNA EN ESCALA ORIGINAL (sin StandardScaler)

        Args:
            climate_data: DataFrame con datos climáticos mensuales
            df_saidi: DataFrame SAIDI completo
            regional_code: Código de la regional
            log_callback: Función para logging

        Returns:
            Tuple de (exog_df, exog_info) o (None, None) si falla
            - exog_df: DataFrame EN ESCALA ORIGINAL
            - exog_info: Dict con metadata de cada variable

        """
        try:
            # Validaciones iniciales
            if climate_data is None or climate_data.empty:
                if log_callback:
                    log_callback("Sin datos climáticos disponibles")
                return None, None

            if not regional_code or regional_code not in self.REGIONAL_EXOG_VARS:
                if log_callback:
                    log_callback(f"Regional {regional_code} sin variables definidas")
                return None, None

            if log_callback:
                log_callback(f"Preparando variables para {regional_code}")
                log_callback("MODO: SIN ESCALADO (valores originales)")

            # ========== VALIDAR ÍNDICE DATETIME ==========
            if not isinstance(climate_data.index, pd.DatetimeIndex):
                # Buscar columna de fecha
                fecha_col = None
                for col in ["fecha", "Fecha", "date", "Date", "month_date"]:
                    if col in climate_data.columns:
                        fecha_col = col
                        break

                if fecha_col is None:
                    if log_callback:
                        log_callback("ERROR: No se encontró columna de fecha válida")
                    return None, None

                try:
                    climate_data = climate_data.copy()
                    climate_data[fecha_col] = pd.to_datetime(climate_data[fecha_col])
                    climate_data = climate_data.set_index(fecha_col)
                except Exception as e:
                    if log_callback:
                        log_callback(f"ERROR convirtiendo índice: {e!s}")
                    return None, None

            # Verificar que ahora es DatetimeIndex
            if not isinstance(climate_data.index, pd.DatetimeIndex):
                if log_callback:
                    log_callback("ERROR: Formato de fecha inválido")
                return None, None

            # ========== ANÁLISIS DE COBERTURA TEMPORAL ==========
            historico = df_saidi[df_saidi["SAIDI"].notna() if "SAIDI" in df_saidi.columns else df_saidi["SAIDI Historico"].notna()]

            saidi_start = historico.index[0]
            saidi_end = historico.index[-1]
            clima_start = climate_data.index[0]
            clima_end = climate_data.index[-1]

            # Calcular periodo de overlap
            overlap_start = max(saidi_start, clima_start)
            overlap_end = min(saidi_end, clima_end)

            if overlap_start > overlap_end:
                if log_callback:
                    log_callback("ERROR: Sin overlap entre SAIDI y CLIMA")
                return None, None

            overlap_mask = (historico.index >= overlap_start) & (historico.index <= overlap_end)
            overlap_months = overlap_mask.sum()

            # Validar overlap mínimo (12 meses)
            meses=12
            if overlap_months < meses:
                if log_callback:
                    log_callback(f"ERROR: Overlap insuficiente ({overlap_months} < 12 meses)")
                return None, None

            if log_callback:
                log_callback(f"SAIDI: {saidi_start.strftime('%Y-%m')} a {saidi_end.strftime('%Y-%m')} ({len(historico)} meses)")
                log_callback(f"CLIMA: {clima_start.strftime('%Y-%m')} a {clima_end.strftime('%Y-%m')} ({len(climate_data)} meses)")
                log_callback(f"OVERLAP: {overlap_start.strftime('%Y-%m')} a {overlap_end.strftime('%Y-%m')} ({overlap_months} meses)")

            # ========== MAPEO AUTOMÁTICO DE COLUMNAS ==========
            exog_vars_config = self.REGIONAL_EXOG_VARS[regional_code]

            # Normalizar nombres disponibles
            available_cols_normalized = {}
            for col in climate_data.columns:
                normalized = col.lower().strip().replace(" ", "_").replace("-", "_")
                available_cols_normalized[normalized] = col

            # Mapear cada variable con búsqueda flexible
            climate_column_mapping = {}

            mejor_modelo = 2
            for var_code in exog_vars_config.keys():
                var_normalized = var_code.lower().strip()

                # Intento 1: Coincidencia exacta
                if var_normalized in available_cols_normalized:
                    climate_column_mapping[var_code] = available_cols_normalized[var_normalized]
                    continue

                # Intento 2: Coincidencia parcial (al menos 2 partes)
                var_parts = var_normalized.split("_")
                best_match = None
                best_match_score = 0

                for norm_col, orig_col in available_cols_normalized.items():
                    matches = sum(1 for part in var_parts if part in norm_col)
                    if matches > best_match_score:
                        best_match_score = matches
                        best_match = orig_col

                if best_match_score >= mejor_modelo:
                    climate_column_mapping[var_code] = best_match

            if not climate_column_mapping:
                if log_callback:
                    log_callback("ERROR: No se pudo mapear ninguna variable")
                return None, None

            # ========== PREPARACIÓN DE VARIABLES SIN ESCALADO ==========
            exog_df = pd.DataFrame(index=historico.index)
            exog_info = {}
            cobertura_minima=80

            for var_code, var_nombre in exog_vars_config.items():
                climate_col = climate_column_mapping.get(var_code)

                if not climate_col or climate_col not in climate_data.columns:
                    continue

                try:
                    # Extraer serie del clima
                    var_series = climate_data[climate_col].copy()

                    # Crear serie alineada (inicialmente vacía)
                    aligned_series = pd.Series(index=historico.index, dtype=float)

                    # Llenar datos donde hay overlap REAL
                    for date in historico.index:
                        if date in var_series.index:
                            aligned_series[date] = var_series.loc[date]

                    # VALIDACIÓN: Cobertura en overlap
                    overlap_data = aligned_series[overlap_mask]
                    datos_reales_overlap = overlap_data.notna().sum()
                    overlap_pct = (datos_reales_overlap / overlap_months) * 100

                    # RECHAZAR si cobertura < 80%
                    if overlap_pct < cobertura_minima:
                        if log_callback:
                            log_callback(f"X RECHAZADA {var_code}: cobertura {overlap_pct:.1f}% < 80%")
                        continue

                    # VALIDACIÓN: Varianza en overlap
                    var_std = overlap_data.std()

                    if pd.isna(var_std) or var_std == 0:
                        if log_callback:
                            log_callback(f"RECHAZADA {var_code}: varianza = 0")
                        continue

                    # Forward-fill para fechas futuras
                    aligned_series = aligned_series.fillna(method="ffill")

                    # Backward-fill (máx 3 meses) para fechas pasadas
                    aligned_series = aligned_series.fillna(method="bfill", limit=3)

                    # Si AÚN hay NaN, rellenar con media del overlap
                    if aligned_series.isnull().any():
                        mean_overlap = overlap_data.mean()
                        aligned_series = aligned_series.fillna(mean_overlap)

                    # VERIFICACIÓN FINAL
                    final_nan = aligned_series.isnull().sum()
                    if final_nan > 0:
                        if log_callback:
                            log_callback(f"RECHAZADA {var_code}: {final_nan} NaN finales")
                        continue

                    # ===== GUARDAR EN ESCALA ORIGINAL =====
                    exog_df[var_code] = aligned_series

                    exog_info[var_code] = {
                        "nombre": var_nombre,
                        "columna_clima": climate_col,
                        "correlacion": self._get_correlation_for_var(var_code, regional_code),
                        "scaled": False,  # CRÍTICO
                        "datos_reales_overlap": int(datos_reales_overlap),
                        "overlap_coverage_pct": float(overlap_pct),
                        "varianza_overlap": float(var_std),
                    }

                    if log_callback:
                        log_callback(f"✓ {var_code} -> ACEPTADA ({overlap_pct:.1f}% cobertura, escala original)")

                except Exception as e:
                    if log_callback:
                        log_callback(f"X ERROR {var_code}: {e}")
                    continue

            # VALIDACIÓN FINAL
            if exog_df.empty or exog_df.shape[1] == 0:
                if log_callback:
                    log_callback("ERROR: Ninguna variable aceptada")
                return None, None

            if log_callback:
                log_callback("=" * 60)
                log_callback(f"Variables preparadas: {len(exog_df.columns)}")
                log_callback("ESCALA: ORIGINAL (sin StandardScaler)")
                log_callback("Rangos:")
                for col in exog_df.columns:
                    log_callback(f"    - {col}: [{exog_df[col].min():.2f}, {exog_df[col].max():.2f}]")
                log_callback("=" * 60)

            return exog_df, exog_info if exog_info else None

        except Exception as e:
            if log_callback:
                log_callback(f"ERROR CRÍTICO: {e!s}")
            return None, None

    def _align_exog_to_saidi(self,
                            exog_series: pd.DataFrame,
                            df_saidi: pd.DataFrame,
                            var_code: str,
                            log_callback) -> pd.Series | None:
        """Alinear datos exogenos al indice de SAIDI."""
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
                log_callback(f"Error alineando variable {var_code}: {e!s}")
            return None

    def _apply_transformation(self, data: np.ndarray, transformation_type: str) -> tuple[np.ndarray, str]:
        """Aplicar transformacion a los datos."""
        if transformation_type == "original":
            return data, "Sin transformacion (datos originales)"

        if transformation_type == "standard":
            self.scaler = StandardScaler()
            transformed = self.scaler.fit_transform(data.reshape(-1, 1)).flatten()
            return transformed, f"StandardScaler (media={self.scaler.mean_[0]:.2f}, std={np.sqrt(self.scaler.var_[0]):.2f})"

        if transformation_type == "log":
            data_positive = np.maximum(data, 1e-10)
            transformed = np.log(data_positive)
            self.transformation_params["log_applied"] = True
            return transformed, "Transformacion logaritmica (log)"

        if transformation_type == "boxcox":
            data_positive = np.maximum(data, 1e-10)
            transformed, lambda_param = stats.boxcox(data_positive)
            self.transformation_params["boxcox_lambda"] = lambda_param
            return transformed, f"Box-Cox (lambda={lambda_param:.4f})"

        if transformation_type == "sqrt":
            data_positive = np.maximum(data, 0)
            transformed = np.sqrt(data_positive)
            self.transformation_params["sqrt_applied"] = True
            return transformed, "Sqrt"

        return data, "Sin transformacion (tipo desconocido)"

    def _inverse_transformation(self, data: np.ndarray, transformation_type: str) -> np.ndarray:
        """Revertir transformacion a escala original."""
        if transformation_type == "original":
            return data

        if transformation_type == "standard":
            if self.scaler is not None:
                return self.scaler.inverse_transform(data.reshape(-1, 1)).flatten()
            return data

        if transformation_type == "log":
            return np.exp(data)

        if transformation_type == "boxcox":
            lambda_param = self.transformation_params.get("boxcox_lambda", 0)
            if lambda_param == 0:
                return np.exp(data)
            return np.power(data * lambda_param + 1, 1 / lambda_param)

        if transformation_type == "sqrt":
            return np.power(data, 2)

        return data

    def _calcular_metricas_validacion_optimized(self,
                                               test_values: np.ndarray,
                                               pred_values: np.ndarray,
                                               order: tuple,
                                               seasonal_order: tuple,
                                               transformation: str,
                                               with_exogenous: bool,
                                               pct_validacion: float,
                                               n_test: int) -> dict[str, float]:
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
            "rmse": rmse,
            "mae": mae,
            "mape": mape,
            "r2_score": r2_score,
            "precision_final": precision_final,
            "n_test": n_test,
            "validation_pct": pct_validacion * 100,
        }

    def _calculate_stability_numpy(self,
                                   actual_values: np.ndarray,
                                   predicted_values: np.ndarray,
                                   precision: float,
                                   mape: float) -> float:
        """Calcular score de estabilidad."""
        try:
            errors = actual_values - predicted_values

            mean_abs_error = np.mean(np.abs(errors))
            std_error = np.std(errors)

            comparacion = 1e-8
            penalizacion_maxima = 50
            penalizacion_minima = 30

            if mean_abs_error > comparacion:
                cv_error = std_error / mean_abs_error
                stability_cv = max(0, 100 * (1 - min(cv_error, 1)))
            else:
                stability_cv = 50.0

            # Penalización por MAPE alto
            if mape > penalizacion_maxima:
                mape_penalty = 0.5
            elif mape > penalizacion_minima:
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
                               order: tuple,
                               seasonal_order: tuple,
                               metricas: dict,
                               pct_validacion: float,
                               transformation: str,
                               exog_info: dict | None = None,
                               simulation_config: dict | None = None) -> str | None:
        """Generar grafica de validacion con metricas alineadas y soporte COMPLETO para simulacion."""
        try:
            if datos_entrenamiento.empty or datos_validacion.empty or predicciones_validacion.empty:
                return None

            temp_dir = tempfile.gettempdir()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_path = os.path.join(temp_dir, f"saidi_validation_{timestamp}.png")

            plt.style.use("default")
            fig = plt.figure(figsize=(16, 10), dpi=100)

            # Detectar si hay simulacion
            simulation_applied = simulation_config and simulation_config.get("enabled", False)

            # Grafica principal - datos de entrenamiento
            plt.plot(datos_entrenamiento.index, datos_entrenamiento.values,
                    label=f"Datos de Entrenamiento ({100-int(pct_validacion*100)}% - {len(datos_entrenamiento)} obs.)",
                    color="blue", linewidth=3, marker="o", markersize=5)

            ultimo_punto_entrenamiento = datos_entrenamiento.iloc[-1]
            fecha_ultimo_entrenamiento = datos_entrenamiento.index[-1]

            # Conectar entrenamiento con validacion
            fechas_validacion_extendidas = [fecha_ultimo_entrenamiento] + list(datos_validacion.index)
            valores_validacion_extendidos = [ultimo_punto_entrenamiento] + list(datos_validacion.values)
            valores_prediccion_extendidos = [ultimo_punto_entrenamiento] + list(predicciones_validacion.values)

            # Datos reales de validacion
            plt.plot(fechas_validacion_extendidas, valores_validacion_extendidos,
                    label=f"Datos Reales de Validacion ({int(pct_validacion*100)}% - {len(datos_validacion)} obs.)",
                    color="navy", linewidth=3, linestyle=":", marker="s", markersize=7)

            # ========== PREDICCIONES CON ETIQUETA SEGÚN SIMULACIÓN ==========
            if simulation_applied:
                summary = simulation_config.get("summary", {})
                escenario_icon = "🌡️"
                escenario_name = summary.get("escenario", "N/A")
                exog_label = f" [{escenario_icon} SIMULADO: {escenario_name}]"
                pred_color = "red"
                pred_linestyle = "-"
                pred_linewidth = 3.5
            elif exog_info:
                exog_label = " [+EXOG]"
                pred_color = "orange"
                pred_linestyle = "-"
                pred_linewidth = 3
            else:
                exog_label = ""
                pred_color = "orange"
                pred_linestyle = "-"
                pred_linewidth = 3

            plt.plot(fechas_validacion_extendidas, valores_prediccion_extendidos,
                    label=f"Predicciones del Modelo ({transformation.upper()}){exog_label}",
                    color=pred_color, linewidth=pred_linewidth, linestyle=pred_linestyle,
                    marker="^", markersize=7, zorder=5)

            # Etiquetas de valores - datos de entrenamiento
            for x, y in zip(datos_entrenamiento.index, datos_entrenamiento.values):
                plt.text(x, y+0.3, f"{y:.1f}", color="blue", fontsize=8,
                        ha="center", va="bottom", rotation=0, alpha=0.9, weight="bold")

            # Etiquetas de valores - datos reales de validacion
            for x, y in zip(datos_validacion.index, datos_validacion.values):
                plt.text(x, y+0.4, f"{y:.1f}", color="navy", fontsize=9,
                        ha="center", va="bottom", rotation=0, weight="bold")

            # Etiquetas de valores - predicciones
            for x, y in zip(predicciones_validacion.index, predicciones_validacion.values):
                plt.text(x, y-0.5, f"{y:.1f}", color=pred_color, fontsize=9,
                        ha="center", va="top", rotation=0, weight="bold")

            # Area de error entre real y prediccion
            plt.fill_between(fechas_validacion_extendidas,
                            valores_validacion_extendidos,
                            valores_prediccion_extendidos,
                            alpha=0.2, color="red",
                            label="Area de Error")

            # Linea divisoria entre entrenamiento y validacion
            if not datos_entrenamiento.empty:
                separacion_x = datos_entrenamiento.index[-1]
                plt.axvline(x=separacion_x, color="gray", linestyle="--", alpha=0.8, linewidth=2)

                y_limits = plt.ylim()
                y_pos = y_limits[1] * 0.75
                plt.text(separacion_x, y_pos, "Division\nEntrenamiento/Validacion",
                        ha="center", va="center", color="gray", fontsize=10, weight="bold",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.9, edgecolor="gray"))

            # ========== CUADRO DE METRICAS ==========
            info_metricas = (f"METRICAS VALIDACION\n"
                            f"RMSE: {metricas['rmse']:.3f} | MAE: {metricas['mae']:.3f}\n"
                            f"MAPE: {metricas['mape']:.1f}% | R2: {metricas['r2_score']:.3f}\n"
                            f"Precision: {metricas['precision_final']:.1f}%")

            plt.text(0.01, 0.24, info_metricas, transform=plt.gca().transAxes,
                    fontsize=10, verticalalignment="top",
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.9, edgecolor="navy"))

            # ========== CUADRO DE ESTABILIDAD Y COMPLEJIDAD ==========
            stability = metricas.get("stability_score", 0)
            complexity = metricas.get("complexity", 0)
            composite = metricas.get("composite_score", 0)

            info_estabilidad = (f"ESTABILIDAD & COMPLEJIDAD\n"
                            f"Stability Score: {stability:.1f}/100\n"
                            f"Complejidad: {complexity} params\n"
                            f"Composite Score: {composite:.3f}")

            plt.text(0.01, 0.09, info_estabilidad, transform=plt.gca().transAxes,
                    fontsize=9, verticalalignment="top",
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.9, edgecolor="orange"))

            # ========== CUADRO DE PARAMETROS DEL MODELO ==========
            info_parametros = (f"PARAMETROS + {transformation.upper()}\n"
                            f"order = {order} | seasonal = {seasonal_order}\n"
                            f"Train: {len(datos_entrenamiento)} | Valid: {len(datos_validacion)}")

            if simulation_applied:
                summary = simulation_config.get("summary", {})
                info_parametros += f"\nSIMULACION: {summary.get('escenario', 'N/A')}"
                info_parametros += f"\nAlcance: {simulation_config.get('alcance_meses', 'N/A')} meses"
            elif exog_info:
                info_parametros += f"\nVariables exogenas: {len(exog_info)}"

            plt.text(0.985, 0.08, info_parametros, transform=plt.gca().transAxes,
                    fontsize=9, verticalalignment="top", horizontalalignment="right",
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="lightgreen", alpha=0.9, edgecolor="green"))

            # ========== CUADRO DE INFORMACIÓN DE SIMULACIÓN ==========
            if simulation_applied:
                summary = simulation_config.get("summary", {})
                escenario = summary.get("escenario", "N/A")
                dias_simulados = summary.get("dias_simulados", "N/A")
                alcance = simulation_config.get("alcance_meses", "N/A")

                info_simulacion = (f"SIMULACIÓN CLIMÁTICA\n"
                                f"Escenario: {escenario}\n"
                                f"Días simulados: {dias_simulados}\n"
                                f"Alcance: {alcance} mes(es)\n"
                                f"Condiciones HIPOTÉTICAS")

                plt.text(0.985, 0.30, info_simulacion, transform=plt.gca().transAxes,
                        fontsize=9, verticalalignment="top", horizontalalignment="right",
                        bbox=dict(boxstyle="round,pad=0.5", facecolor="#FFEBEE", alpha=0.95,
                                edgecolor="#F44336", linewidth=2),
                        color="darkred", weight="bold")

            # ========== INDICADOR DE CALIDAD ==========
            precision = metricas["precision_final"]

            precision_excelente = 60
            precision_buena = 40
            precision_aceptable = 20

            if precision >= precision_excelente:
                interpretacion = "EXCELENTE"
                color_interp = "green"
            elif precision >= precision_buena:
                interpretacion = "BUENO"
                color_interp = "limegreen"
            elif precision >= precision_aceptable:
                interpretacion = "ACEPTABLE"
                color_interp = "orange"
            else:
                interpretacion = "LIMITADO"
                color_interp = "red"

            # Si hay simulacion, agregar indicador
            if simulation_applied:
                interpretacion += "\n[SIMULADO]"

            plt.text(0.985, 0.97, f"{interpretacion}\n{precision:.1f}%",
                    transform=plt.gca().transAxes, fontsize=12, weight="bold",
                    verticalalignment="top", horizontalalignment="right",
                    bbox=dict(boxstyle="round,pad=0.5", facecolor=color_interp, alpha=0.8, edgecolor="black"),
                    color="black")

            # ========== CONFIGURAR EJES ==========
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
            meses_espanol = ["Ene", "Feb", "Mar", "Abr", "May", "Jun",
                            "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]

            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))

            fechas_mensuales = pd.date_range(start=x_min, end=x_max, freq="MS")
            labels_mensuales = []
            meses = 12

            for fecha in fechas_mensuales:
                mes_nombre = meses_espanol[fecha.month - 1]
                if fecha.month == 1 or len(fechas_mensuales) <= meses:
                    labels_mensuales.append(f"{mes_nombre}\n{fecha.year}")
                else:
                    labels_mensuales.append(mes_nombre)

            if len(fechas_mensuales) > 0:
                ax.set_xticks(fechas_mensuales)
                ax.set_xticklabels(labels_mensuales, rotation=45, ha="right", fontsize=9)

            # ========== TITULO CON INFO DE SIMULACION ==========
            title_text = f"Validacion Modelo: SARIMAX{order}x{seasonal_order} + {transformation.upper()}"

            if simulation_applied:
                summary = simulation_config.get("summary", {})
                escenario_name = summary.get("escenario", "N/A").upper()
                title_text += f" [🌡️ SIMULACIÓN: {escenario_name}]"
            elif exog_info:
                title_text += " [+EXOG]"

            plt.title(title_text, fontsize=18, fontweight="bold", pad=25)

            plt.xlabel("Fecha", fontsize=14, weight="bold")
            plt.ylabel("SAIDI (minutos)", fontsize=14, weight="bold")

            plt.legend(fontsize=11, loc="upper center", bbox_to_anchor=(0.25, -0.08),
                    ncol=2, frameon=True, shadow=True, fancybox=True)

            plt.grid(True, alpha=0.4, linestyle="-", linewidth=0.8)

            plt.tight_layout()
            plt.subplots_adjust(top=0.93, bottom=0.35, left=0.038, right=0.787)

            # ========== NOTA AL PIE CON INFO DE SIMULACION ==========
            footer_text = f"Transformacion: {transformation.upper()} - Precision calculada como OptimizationService"
            footer_color = "lightyellow"
            footer_edge_color = "darkblue"
            footer_text_color = "darkblue"
            footer_linewidth = 1

            if simulation_applied:
                summary = simulation_config.get("summary", {})
                escenario = summary.get("escenario", "N/A").upper()
                dias = summary.get("dias_simulados", "N/A")
                alcance = simulation_config.get("alcance_meses", "N/A")

                footer_text = f"🌡️ VALIDACIÓN CON SIMULACIÓN CLIMÁTICA - Escenario: {escenario} ({dias} días, {alcance} meses)"
                footer_text += " | Métricas bajo condiciones HIPOTÉTICAS simuladas"
                footer_color = "#FFEBEE"
                footer_edge_color = "#F44336"
                footer_text_color = "darkred"
                footer_linewidth = 2
            elif exog_info:
                footer_text += f" - Con {len(exog_info)} variables exogenas"

            footer_text += f" - Validacion: {metricas['validation_pct']:.0f}% test"

            plt.figtext(0.5, 0.02, footer_text,
                    ha="center", fontsize=12, style="italic", color=footer_text_color, weight="bold",
                    bbox=dict(boxstyle="round,pad=0.4", facecolor=footer_color, alpha=0.9,
                                edgecolor=footer_edge_color, linewidth=footer_linewidth))

            plt.savefig(plot_path, dpi=100, bbox_inches="tight", facecolor="white", edgecolor="none")
            plt.close(fig)

            self.plot_file_path = plot_path
            return plot_path

        except Exception as e:
            print(f"Error generando grafica de validacion: {e}")
            return None

    def cleanup_plot_file(self):
        """Limpiar archivo temporal de grafica."""
        if self.plot_file_path and os.path.exists(self.plot_file_path):
            try:
                os.remove(self.plot_file_path)
            except Exception as e:
                print(f"Error eliminando archivo temporal: {e}")
            finally:
                self.plot_file_path = None

    def _diagnose_exog_coverage(self,
                      serie_saidi: pd.Series,
                      exog_df: pd.DataFrame,
                      log_callback) -> bool:
        """
        Diagnosticar cobertura temporal de variables exógenas.

        Valida:
        1. Índices coinciden exactamente
        2. No hay NaN en ninguna columna
        3. No hay valores infinitos
        4. Variables tienen varianza > 0

        Args:
            serie_saidi: Serie temporal SAIDI
            exog_df: DataFrame con variables exógenas
            log_callback: Función para logging

        Returns:
            bool: True si pasa todas las validaciones

        """
        try:
            saidi_start = serie_saidi.index[0]
            saidi_end = serie_saidi.index[-1]
            exog_start = exog_df.index[0]
            exog_end = exog_df.index[-1]

            if log_callback:
                log_callback("=" * 60)
                log_callback("DIAGNOSTICO DE COBERTURA EXOGENA")
                log_callback("=" * 60)
                log_callback(f"SAIDI: {saidi_start.strftime('%Y-%m')} a {saidi_end.strftime('%Y-%m')} ({len(serie_saidi)} obs)")
                log_callback(f"EXOG:  {exog_start.strftime('%Y-%m')} a {exog_end.strftime('%Y-%m')} ({len(exog_df)} obs)")

            # 1. Verificar que los índices coinciden EXACTAMENTE
            porcentaje_faltante = 20
            if not exog_df.index.equals(serie_saidi.index):
                if log_callback:
                    log_callback("ADVERTENCIA: Indices no coinciden exactamente")

                # Verificar fechas faltantes
                missing_in_exog = [d for d in serie_saidi.index if d not in exog_df.index]

                if missing_in_exog:
                    pct_missing = len(missing_in_exog) / len(serie_saidi) * 100

                    if log_callback:
                        log_callback(f"Fechas SAIDI faltantes en EXOG: {len(missing_in_exog)} ({pct_missing:.1f}%)")

                    # CRÍTICO: Si falta >20% de fechas, rechazar
                    if pct_missing > porcentaje_faltante:
                        if log_callback:
                            log_callback("ERROR CRITICO: >20% de fechas faltantes")
                            log_callback("Las variables exogenas NO cubren suficiente periodo historico")
                        return False

            # 2. Verificar que NO hay NaN en ninguna columna
            if exog_df.isnull().any().any():
                nan_cols = exog_df.columns[exog_df.isnull().any()].tolist()

                if log_callback:
                    log_callback("ERROR: Columnas con NaN encontradas:")
                    for col in nan_cols:
                        nan_count = exog_df[col].isnull().sum()
                        pct_nan = (nan_count / len(exog_df)) * 100
                        log_callback(f"  - {col}: {nan_count} NaN ({pct_nan:.1f}%)")
                    log_callback("Variables exogenas deben estar completamente rellenas")

                return False

            # 3. Verificar valores infinitos
            if np.isinf(exog_df.values).any():
                if log_callback:
                    log_callback("ERROR: Variables exogenas contienen valores infinitos")
                return False

            # 4. Verificar que hay varianza en las variables
            zero_variance_vars = []
            for col in exog_df.columns:
                if exog_df[col].std() == 0:
                    zero_variance_vars.append(col)

            if zero_variance_vars:
                if log_callback:
                    log_callback("ADVERTENCIA: Variables con varianza cero:")
                    for var in zero_variance_vars:
                        log_callback(f"  - {var}")
                    log_callback("Estas variables no aportan informacion al modelo")
                # No rechazar por esto, solo advertir

            if log_callback:
                log_callback("✓ Cobertura temporal y calidad de datos OK")
                log_callback("=" * 60)

            return True

        except Exception as e:
            if log_callback:
                log_callback(f"ERROR durante diagnostico: {e}")
            return False

    def _get_correlation_for_var(self, var_code: str, regional_code: str) -> float:
        """
        Obtener correlación documentada de una variable específica.

        Args:
            var_code: Código de la variable (ej: 'realfeel_min')
            regional_code: Código de la regional (ej: 'SAIDI_O')

        Returns:
            float: Correlación documentada o 0.0 si no existe

        """
        # Correlaciones REALES documentadas por regional
        correlations = {
            "SAIDI_O": {  # Ocaña
                "realfeel_min": 0.689,              # *** FUERTE
                "windchill_avg": 0.520,             # ** MODERADA-FUERTE
                "dewpoint_avg": 0.470,              # ** MODERADA-FUERTE
                "windchill_max": 0.464,             # ** MODERADA-FUERTE
                "dewpoint_min": 0.456,              # ** MODERADA-FUERTE
                "precipitation_max_daily": 0.452,
                "precipitation_avg_daily": 0.438,
            },

            "SAIDI_C": {  # Cúcuta
                "realfeel_avg": 0.573,              # ** MODERADA-FUERTE
                "pressure_rel_avg": -0.358,         # Negativa
                "wind_speed_max": 0.356,
                "pressure_abs_avg": -0.356,         # Negativa
            },

            "SAIDI_T": {  # Tibú
                "realfeel_avg": 0.906,              # *** MUY FUERTE
                "wind_dir_avg": -0.400,             # Negativa
                "uv_index_avg": 0.385,
                "heat_index_avg": 0.363,
                "temperature_min": 0.352,
                "windchill_min": 0.340,
                "temperature_avg": 0.338,
                "pressure_rel_avg": -0.330,         # Negativa
            },

            "SAIDI_A": {  # Aguachica
                "uv_index_max": 0.664,              # *** FUERTE
                "days_with_rain": 0.535,            # ** MODERADA-FUERTE
            },

            "SAIDI_P": {  # Pamplona
                "precipitation_total": 0.577,       # ** MODERADA-FUERTE
                "precipitation_avg_daily": 0.552,
                "realfeel_min": 0.344,
            },
        }

        # Buscar correlación específica
        if regional_code in correlations and var_code in correlations[regional_code]:
            return correlations[regional_code][var_code]

        return 0.0
