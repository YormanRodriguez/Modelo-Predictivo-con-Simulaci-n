# services/climate_simulation_service.py - Servicio de simulación climática para SAIDI

# services/climate_simulation_service.py - Servicio de simulación climática para SAIDI

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import pandas as pd


@dataclass
class SimulationConfig:
    """Configuración para la simulación climática."""

    scenario_name: str
    intensity_adjustment: float
    alcance_meses: int
    percentiles: dict
    regional_code: str


class ClimateSimulationService:
    """Servicio para simular escenarios climáticos adaptativos por regional."""

    # ========== CATEGORÍAS DE VARIABLES CLIMÁTICAS ==========
    VARIABLE_CATEGORIES: ClassVar[dict] = {
        "temperature": {
            "variables": [
                "realfeel_min", "realfeel_max", "realfeel_avg",
                "temperature_min", "temperature_max", "temperature_avg",
                "windchill_min", "windchill_max", "windchill_avg",
                "dewpoint_min", "dewpoint_max", "dewpoint_avg",
                "heat_index_min", "heat_index_max", "heat_index_avg",
            ],
            "scenarios": {
                "calor_extremo": {"percentile": 90, "intensity": 1.2},
                "calor_moderado": {"percentile": 75, "intensity": 1.1},
                "normal": {"percentile": 50, "intensity": 1.0},
                "frio_moderado": {"percentile": 25, "intensity": 0.9},
                "frio_extremo": {"percentile": 10, "intensity": 0.8},
            },
        },
        "precipitation": {
            "variables": [
                "precipitation_total", "precipitation_max_daily",
                "precipitation_avg_daily", "days_with_rain",
            ],
            "scenarios": {
                "sequia": {"percentile": 10, "intensity": 0.3},
                "seco": {"percentile": 25, "intensity": 0.6},
                "normal": {"percentile": 50, "intensity": 1.0},
                "lluvioso": {"percentile": 75, "intensity": 1.5},
                "tormenta": {"percentile": 90, "intensity": 2.0},
            },
        },
        "pressure": {
            "variables": [
                "pressure_rel_avg", "pressure_rel_min", "pressure_rel_max",
                "pressure_abs_avg", "pressure_abs_min", "pressure_abs_max",
            ],
            "scenarios": {
                "baja_presion": {"percentile": 25, "intensity": 0.95},
                "normal": {"percentile": 50, "intensity": 1.0},
                "alta_presion": {"percentile": 75, "intensity": 1.05},
            },
        },
        "wind": {
            "variables": [
                "wind_speed_max", "wind_speed_avg", "wind_speed_min",
                "wind_gust_max", "wind_dir_avg",
            ],
            "scenarios": {
                "calma": {"percentile": 10, "intensity": 0.5},
                "brisa": {"percentile": 50, "intensity": 1.0},
                "ventoso": {"percentile": 75, "intensity": 1.3},
                "vendaval": {"percentile": 90, "intensity": 1.8},
            },
        },
        "uv": {
            "variables": [
                "uv_index_min", "uv_index_max", "uv_index_avg",
            ],
            "scenarios": {
                "bajo": {"percentile": 25, "intensity": 0.7},
                "moderado": {"percentile": 50, "intensity": 1.0},
                "alto": {"percentile": 75, "intensity": 1.3},
                "extremo": {"percentile": 90, "intensity": 1.6},
            },
        },
    }

    # ========== ESCENARIOS CLIMÁTICOS COMPUESTOS ==========
    COMPOSITE_SCENARIOS: ClassVar[dict] = {
        "calor_extremo": {
            "name": "Calor Extremo",
            "icon": "🌡️",
            "description": "Temperaturas muy altas, ambiente seco",
            "categories": {
                "temperature": "calor_extremo",
                "precipitation": "sequia",
                "wind": "calma",
                "uv": "extremo",
            },
        },
        "lluvias_intensas": {
            "name": "Lluvias Intensas",
            "icon": "⛈️",
            "description": "Precipitaciones abundantes con tormentas",
            "categories": {
                "temperature": "normal",
                "precipitation": "tormenta",
                "wind": "vendaval",
                "pressure": "baja_presion",
            },
        },
        "condiciones_normales": {
            "name": "Condiciones Normales",
            "icon": "🌤️",
            "description": "Clima típico sin extremos",
            "categories": {
                "temperature": "normal",
                "precipitation": "normal",
                "wind": "brisa",
                "pressure": "normal",
            },
        },
        "sequia": {
            "name": "Sequía",
            "icon": "🏜️",
            "description": "Ausencia prolongada de lluvias",
            "categories": {
                "temperature": "calor_moderado",
                "precipitation": "sequia",
                "wind": "calma",
                "uv": "alto",
            },
        },
        "vientos_fuertes": {
            "name": "Vientos Fuertes",
            "icon": "💨",
            "description": "Vientos intensos con ráfagas",
            "categories": {
                "temperature": "normal",
                "precipitation": "seco",
                "wind": "vendaval",
                "pressure": "baja_presion",
            },
        },
        "tiempo_humedo": {
            "name": "Tiempo Húmedo",
            "icon": "🌧️",
            "description": "Lluvias moderadas constantes",
            "categories": {
                "temperature": "normal",
                "precipitation": "lluvioso",
                "wind": "brisa",
                "pressure": "normal",
            },
        },
    }

    # ========== CONFIGURACIÓN DE REGIONALES ==========
    REGIONAL_CLIMATE_PROFILE: ClassVar[dict] = {
        "SAIDI_O": {  # Ocaña
            "tipo_clima": "calido_seco",
            "elevacion": "baja",  # ~1200 msnm
            "escenarios_excluidos": [],  # Todos aplicables
        },
        "SAIDI_C": {  # Cúcuta
            "tipo_clima": "calido_seco",
            "elevacion": "baja",  # ~320 msnm
            "escenarios_excluidos": [],  # Todos aplicables
        },
        "SAIDI_T": {  # Tibú
            "tipo_clima": "calido_humedo",
            "elevacion": "baja",  # ~75 msnm
            "escenarios_excluidos": [],  # Todos aplicables
        },
        "SAIDI_A": {  # Aguachica
            "tipo_clima": "calido_seco",
            "elevacion": "baja",  # ~150 msnm
            "escenarios_excluidos": [],  # Todos aplicables
        },
        "SAIDI_P": {  # Pamplona
            "tipo_clima": "frio_humedo",
            "elevacion": "alta",  # ~2200 msnm
            "escenarios_excluidos": ["calor_extremo", "sequia"],  # No aplican
        },
    }


    def __init__(self):
        """Inicializar servicio con configuración adaptativa."""
        self.variable_categories = self.VARIABLE_CATEGORIES
        self.composite_scenarios = self.COMPOSITE_SCENARIOS
        self.regional_climate_profile = self.REGIONAL_CLIMATE_PROFILE

        # Sincronizar con PredictionService (importación dinámica para evitar ciclos)
        try:
            from services.prediction_service import PredictionService  # noqa: PLC0415
            self.regional_exog_vars = PredictionService.REGIONAL_EXOG_VARS
        except ImportError:
            # Fallback si no se puede importar
            self.regional_exog_vars = {}

    def categorize_variable(self, var_code: str) -> str:
        """
        Determinar categoría de una variable climática.

        Args:
            var_code: Código de variable (ej: 'realfeel_min')

        Returns:
            Categoría (ej: 'temperature') o 'unknown'

        """
        # Búsqueda exacta en categorías definidas
        for category, config in self.variable_categories.items():
            if var_code in config["variables"]:
                return category

        # Fallback: inferir por nombre de la variable usando patrones
        var_lower = var_code.lower()

        category_patterns = {
            "temperature": ["temp", "feel", "chill", "heat", "dewpoint"],
            "precipitation": ["precip", "rain"],
            "pressure": ["pressure", "press"],
            "wind": ["wind", "gust"],
            "uv": ["uv"],
        }

        for category, keywords in category_patterns.items():
            if any(kw in var_lower for kw in keywords):
                return category

        return "unknown"

    def apply_simulation(
        self,
        exog_forecast: pd.DataFrame,
        config: SimulationConfig,
    ) -> pd.DataFrame:
        """
        Aplicar simulación climática adaptativa según variables de la regional.

        Args:
            exog_forecast: Variables exógenas SIN ESCALAR
            config: Configuración de la simulación

        Returns:
            DataFrame simulado (sin escalar)

        Raises:
            ValueError: Si el escenario no existe
            RuntimeError: Si hay errores en la aplicación de la simulación

        """
        # Validar escenario
        if config.scenario_name not in self.composite_scenarios:
            msg = f"Escenario '{config.scenario_name}' no existe"
            raise ValueError(msg)

        try:
            scenario = self.composite_scenarios[config.scenario_name]
            exog_simulated = exog_forecast.copy()

            regional_vars = self._get_regional_vars(config.regional_code, exog_forecast)
            meses_a_simular = min(config.alcance_meses, len(exog_simulated))

            self._apply_scenario_to_forecast(
                exog_simulated,
                scenario,
                regional_vars,
                meses_a_simular,
                config,
            )

        except ValueError:
            raise
        except (KeyError, AttributeError) as e:
            msg = f"Error de configuración en simulación: {e}"
            raise RuntimeError(msg) from e
        except (TypeError, IndexError) as e:
            msg = f"Error procesando datos de simulación: {e}"
            raise RuntimeError(msg) from e
        else:
            return exog_simulated

    def _get_regional_vars(
        self,
        regional_code: str,
        exog_forecast: pd.DataFrame,
    ) -> dict:
        """Obtener variables de la regional o todas las columnas disponibles."""
        regional_vars = self.regional_exog_vars.get(regional_code, {})

        if not regional_vars:
            regional_vars = {col: col for col in exog_forecast.columns}

        return regional_vars

    def _apply_scenario_to_forecast(
        self,
        exog_simulated: pd.DataFrame,
        scenario: dict,
        regional_vars: dict,  # noqa: ARG002
        meses_a_simular: int,
        config: SimulationConfig,
    ) -> None:
        """Aplicar el escenario a los meses del forecast (modifica in-place)."""
        for i in range(meses_a_simular):
            for var_code in exog_simulated.columns:
                self._simulate_variable(
                    exog_simulated,
                    i,
                    var_code,
                    scenario,
                    config,
                )

    def _simulate_variable(
        self,
        exog_simulated: pd.DataFrame,
        month_index: int,
        var_code: str,
        scenario: dict,
        config: SimulationConfig,
    ) -> None:
        """Simular una variable climática para un mes específico."""
        category = self.categorize_variable(var_code)

        if category == "unknown":
            return

        category_scenario = scenario["categories"].get(category)

        if not category_scenario:
            return

        # Obtener configuración del escenario
        scenario_config = self._get_scenario_config(category, category_scenario)

        if var_code not in config.percentiles:
            return

        # Calcular valor simulado
        simulated_value = self._calculate_simulated_value(
            exog_simulated.iloc[month_index][var_code],
            scenario_config,
            config.percentiles[var_code],
            config.intensity_adjustment,
        )

        # Aplicar límites seguros
        simulated_value = self._apply_safe_limits(
            simulated_value,
            var_code,
            category,
            config.percentiles[var_code],
        )

        # Actualizar valor
        exog_simulated.iloc[month_index, exog_simulated.columns.get_loc(var_code)] = simulated_value

    def _get_scenario_config(self, category: str, category_scenario: str) -> dict:
        """Obtener configuración del escenario para una categoría."""
        cat_config = self.variable_categories[category]
        return cat_config["scenarios"][category_scenario]

    def _calculate_simulated_value(
        self,
        current_value: float,
        scenario_config: dict,
        percentile_data: dict,
        intensity_adjustment: float,
    ) -> float:
        """Calcular el valor simulado basado en percentiles e intensidad."""
        base_percentile = scenario_config["percentile"]
        base_intensity = scenario_config["intensity"]
        final_intensity = base_intensity * intensity_adjustment

        # Mapeo de percentiles
        percentile_map = {
            10: "p10",
            25: "p25",
            50: "p50",
            75: "p75",
            90: "p90",
        }

        p_key = percentile_map.get(base_percentile, "p50")
        target = percentile_data[p_key]

        # Interpolar entre valor actual y objetivo
        return current_value + (target - current_value) * final_intensity

    def _apply_safe_limits(self,
                          value: float,
                          var_code: str,  # noqa: ARG002
                          category: str,
                          percentiles: dict) -> float:
        """
        Aplicar límites físicamente razonables según categoría.

        Args:
            value: Valor a limitar,
            var_code: Código de la variable,
            category: Categoría de la variable,
            percentiles: Percentiles históricos.

        Returns:
            Valor limitado dentro de rangos razonables.

        """
        if category == "temperature":
            # Temperatura: -10°C a 50°C
            return np.clip(value, -10, 50)

        if category == "precipitation":
            # Precipitación: 0 a 500mm (extremo)
            return np.clip(value, 0, 500)

        if category == "pressure":
            # Presión: 900 a 1100 hPa
            return np.clip(value, 900, 1100)

        if category == "wind":
            # Viento: 0 a 150 km/h (huracán categoría 1)
            return np.clip(value, 0, 150)

        if category == "uv":
            # Índice UV: 0 a 15 (extremo)
            return np.clip(value, 0, 15)

        # Límites por percentiles históricos (±50% del rango histórico)
        return np.clip(
            value,
            percentiles["p10"] * 0.5,
            percentiles["p90"] * 1.5,
        )

    def calculate_percentiles(self, climate_data: pd.DataFrame, regional_code: str) -> dict:
        """
        Calcular percentiles históricos para cada variable climática.

        Args: climate_data: DataFrame con datos climáticos históricos,
            regional_code: Código de la regional.

        Returns:
            Dict con percentiles por variable,

        Raises:
            ValueError: Si no hay datos climáticos disponibles o son insuficientes,

        RuntimeError:
            Si hay errores en el procesamiento de datos.

        """
        if climate_data is None or climate_data.empty:
            raise ValueError("No hay datos climáticos disponibles")

        try:
            # Obtener variables de esta regional
            regional_vars = self.regional_exog_vars.get(regional_code, {})

            if not regional_vars:
                # Si no hay configuración, usar todas las columnas numéricas
                regional_vars = {
                    col: col
                    for col in climate_data.select_dtypes(include=[np.number]).columns
                }

            percentiles_result = {}

            for var_code in regional_vars:  # Eliminado .keys()
                # Buscar columna en climate_data (puede tener nombre diferente)
                if var_code not in climate_data.columns:
                    continue

                data = climate_data[var_code].dropna()
                meses = 12
                if len(data) < meses:  # Necesitamos al menos 1 año de datos
                    continue

                percentiles_result[var_code] = {
                    "p10": float(np.percentile(data, 10)),
                    "p25": float(np.percentile(data, 25)),
                    "p50": float(np.percentile(data, 50)),
                    "p75": float(np.percentile(data, 75)),
                    "p90": float(np.percentile(data, 90)),
                    "mean": float(data.mean()),
                    "std": float(data.std()),
                    "min": float(data.min()),
                    "max": float(data.max()),
                }

        except (KeyError, AttributeError) as e:
            msg = f"Error de configuración calculando percentiles: {e}"
            raise RuntimeError(msg) from e
        except (TypeError, IndexError) as e:
            msg = f"Error procesando datos climáticos: {e}"
            raise RuntimeError(msg) from e
        else:
            return percentiles_result

    def get_simulation_summary(self, scenario_name: str, intensity_adjustment: float, alcance_meses: int, percentiles: dict,  # noqa: ARG002
        regional_code: str,
    )   -> dict:
        """
        Generar resumen de la configuración de simulación.

        Args:
            scenario_name: Nombre del escenario,
            intensity_adjustment: Factor de intensidad (0.5 - 2.0),
            alcance_meses: Número de meses afectados,
            percentiles: Percentiles históricos, regional_code: Código de la regional.
            regional_code: Código de la regional.

        Returns:
            Dict con información del resumen.

        """
        try:
            if scenario_name not in self.composite_scenarios:
                return {}

            scenario_info = self.composite_scenarios[scenario_name]

            # Obtener variables afectadas
            regional_vars = self.regional_exog_vars.get(regional_code, {})
            affected_vars = {}

            for var_code, var_nombre in regional_vars.items():
                category = self.categorize_variable(var_code)

                if category == "unknown":
                    continue

                category_scenario = scenario_info["categories"].get(category)

                if not category_scenario:
                    continue

                # Calcular cambio estimado
                cat_config = self.variable_categories[category]
                scenario_config = cat_config["scenarios"][category_scenario]

                base_intensity = scenario_config["intensity"]
                final_intensity = base_intensity * intensity_adjustment

                # Cambio porcentual
                change_pct = (final_intensity - 1.0) * 100

                affected_vars[var_code] = {
                    "nombre": var_nombre,
                    "categoria": category,
                    "escenario_categoria": category_scenario,
                    "cambio_porcentual": change_pct,
                    "intensidad_final": final_intensity,
                }

            return {
                "escenario": scenario_info["name"],
                "icono": scenario_info["icon"],
                "descripcion": scenario_info["description"],
                "intensity_adjustment": intensity_adjustment,
                "alcance_meses": alcance_meses,
                "variables_afectadas": affected_vars,
                "num_variables": len(affected_vars),
            }

        except (KeyError, AttributeError) as e:
            print(f"Error de configuración en resumen: {e}")
            return {}
        except (ValueError, TypeError) as e:
            print(f"Error de valor en cálculos: {e}")
            return {}

    def validate_simulation_params(
    self, scenario_name: str, intensity_adjustment: float, alcance_meses: int, regional_code: str | None = None) -> tuple[bool, str]:
        """
        Validar parámetros de simulación.

        Args: scenario_name: Nombre del escenario,
        intensity_adjustment: Factor de intensidad,
        alcance_meses: Número de meses,
        regional_code: Código de regional (opcional, para validar compatibilidad).

        Returns:tuple[bool, str]: (es_valido, mensaje_error).

        """
        if scenario_name not in self.composite_scenarios:
            return False, f"Escenario no válido: {scenario_name}"

        # Validar que el escenario sea aplicable a la regional
        if regional_code and regional_code in self.regional_climate_profile:
            exclusiones = self.regional_climate_profile[regional_code].get("escenarios_excluidos", [])

            if scenario_name in exclusiones:
                regional_info = self.regional_climate_profile[regional_code]
                return False, f"El escenario '{scenario_name}' no es aplicable a esta regional (clima: {regional_info['tipo_clima']})"

        if not isinstance(intensity_adjustment, (int, float)):
            return False, "Ajuste de intensidad debe ser un número"

        intensidad_menor = 0.5
        intensidad_mayor = 2.0
        if intensity_adjustment < intensidad_menor or intensity_adjustment > intensidad_mayor:
            return False, "Intensidad debe estar entre 0.5x y 2.0x"

        if alcance_meses not in [1, 3, 6]:
            return False, "Alcance debe ser 1, 3 o 6 meses"

        return True, ""

    def get_available_scenarios(self, regional_code: str | None = None) -> list:
        """
        Obtener lista de escenarios disponibles (filtrados por regional).

        Args:
            regional_code: Código de regional (opcional).
            Si se especifica, filtra escenarios no aplicables a esa región.

        Returns:Lista de dicts con información de cada escenario.

        """
        scenarios_list = []

        # Obtener exclusiones de la regional
        exclusiones = []
        if regional_code and regional_code in self.regional_climate_profile:
            exclusiones = self.regional_climate_profile[regional_code].get("escenarios_excluidos", [])

        for key, info in self.composite_scenarios.items():
            # Filtrar escenarios excluidos para esta regional
            if key in exclusiones:
                continue

            scenarios_list.append({
                "id": key,
                "name": info["name"],
                "icon": info["icon"],
                "description": info["description"],
                "categories": list(info["categories"].keys()),
            })

        return scenarios_list

    def get_regional_climate_info(self, regional_code: str) -> dict:
        """
        Obtener información climática de una regional.

        Args:
            regional_code: Código de la regional.

        Returns:
            Dict con información del perfil climático.

        """
        if regional_code not in self.regional_climate_profile:
            return {
                "tipo_clima": "desconocido",
                "elevacion": "desconocida",
                "escenarios_excluidos": [],
                "escenarios_aplicables": list(self.composite_scenarios.keys()),
            }

        profile = self.regional_climate_profile[regional_code]

        # Calcular escenarios aplicables
        todos_escenarios = set(self.composite_scenarios.keys())
        excluidos = set(profile.get("escenarios_excluidos", []))
        aplicables = list(todos_escenarios - excluidos)

        return {
            "tipo_clima": profile["tipo_clima"],
            "elevacion": profile["elevacion"],
            "escenarios_excluidos": list(excluidos),
            "escenarios_aplicables": aplicables,
        }

    def get_variable_category_info(self, var_code: str) -> dict:
        """
        Obtener información de categoría de una variable.

        Args:
          var_code: Código de la variable.

        Returns:Dict con información de la categoría.

        """
        category = self.categorize_variable(var_code)

        if category == "unknown":
            return {
                "category": "unknown",
                "scenarios": [],
                "unit": "N/A",
            }

        cat_config = self.variable_categories[category]

        # Determinar unidad según categoría
        unit_map = {
            "temperature": "°C",
            "precipitation": "mm",
            "pressure": "hPa",
            "wind": "km/h",
            "uv": "índice",
        }

        return {
            "category": category,
            "scenarios": list(cat_config["scenarios"].keys()),
            "unit": unit_map.get(category, ""),
        }
