# services/climate_simulation_service.py - Servicio de simulación climática para SAIDI
# VERSIÓN ACTUALIZADA: Simulación adaptativa por regional con categorización dinámica

import numpy as np
import pandas as pd


class ClimateSimulationService:
    """
    Servicio para simular escenarios climáticos adaptativos por regional
    
    NUEVA ARQUITECTURA:
    1. Categoriza variables automáticamente (temperatura, precipitación, presión, viento, UV)
    2. Aplica escenarios según categoría física de la variable
    3. Soporta cualquier combinación de variables específicas de cada regional
    4. Sincronizado con PredictionService para variables exógenas
    """

    # ========== CATEGORÍAS DE VARIABLES CLIMÁTICAS ==========
    VARIABLE_CATEGORIES = {
        'temperature': {
            'variables': [
                'realfeel_min', 'realfeel_max', 'realfeel_avg',
                'temperature_min', 'temperature_max', 'temperature_avg',
                'windchill_min', 'windchill_max', 'windchill_avg',
                'dewpoint_min', 'dewpoint_max', 'dewpoint_avg',
                'heat_index_min', 'heat_index_max', 'heat_index_avg'
            ],
            'scenarios': {
                'calor_extremo': {'percentile': 90, 'intensity': 1.2},
                'calor_moderado': {'percentile': 75, 'intensity': 1.1},
                'normal': {'percentile': 50, 'intensity': 1.0},
                'frio_moderado': {'percentile': 25, 'intensity': 0.9},
                'frio_extremo': {'percentile': 10, 'intensity': 0.8},
            }
        },
        
        'precipitation': {
            'variables': [
                'precipitation_total', 'precipitation_max_daily',
                'precipitation_avg_daily', 'days_with_rain'
            ],
            'scenarios': {
                'sequia': {'percentile': 10, 'intensity': 0.3},
                'seco': {'percentile': 25, 'intensity': 0.6},
                'normal': {'percentile': 50, 'intensity': 1.0},
                'lluvioso': {'percentile': 75, 'intensity': 1.5},
                'tormenta': {'percentile': 90, 'intensity': 2.0},
            }
        },
        
        'pressure': {
            'variables': [
                'pressure_rel_avg', 'pressure_rel_min', 'pressure_rel_max',
                'pressure_abs_avg', 'pressure_abs_min', 'pressure_abs_max'
            ],
            'scenarios': {
                'baja_presion': {'percentile': 25, 'intensity': 0.95},
                'normal': {'percentile': 50, 'intensity': 1.0},
                'alta_presion': {'percentile': 75, 'intensity': 1.05},
            }
        },
        
        'wind': {
            'variables': [
                'wind_speed_max', 'wind_speed_avg', 'wind_speed_min',
                'wind_gust_max', 'wind_dir_avg'
            ],
            'scenarios': {
                'calma': {'percentile': 10, 'intensity': 0.5},
                'brisa': {'percentile': 50, 'intensity': 1.0},
                'ventoso': {'percentile': 75, 'intensity': 1.3},
                'vendaval': {'percentile': 90, 'intensity': 1.8},
            }
        },
        
        'uv': {
            'variables': [
                'uv_index_min', 'uv_index_max', 'uv_index_avg'
            ],
            'scenarios': {
                'bajo': {'percentile': 25, 'intensity': 0.7},
                'moderado': {'percentile': 50, 'intensity': 1.0},
                'alto': {'percentile': 75, 'intensity': 1.3},
                'extremo': {'percentile': 90, 'intensity': 1.6},
            }
        },
    }

    # ========== ESCENARIOS CLIMÁTICOS COMPUESTOS ==========
    # Escenarios GENÉRICOS aplicables a cualquier región
    COMPOSITE_SCENARIOS = {
        'calor_extremo': {
            'name': 'Calor Extremo',
            'icon': '🌡️',
            'description': 'Temperaturas muy altas, ambiente seco',
            'categories': {
                'temperature': 'calor_extremo',
                'precipitation': 'sequia',
                'wind': 'calma',
                'uv': 'extremo',
            }
        },
        
        'lluvias_intensas': {
            'name': 'Lluvias Intensas',
            'icon': '⛈️',
            'description': 'Precipitaciones abundantes con tormentas',
            'categories': {
                'temperature': 'normal',
                'precipitation': 'tormenta',
                'wind': 'vendaval',
                'pressure': 'baja_presion',
            }
        },
        
        'condiciones_normales': {
            'name': 'Condiciones Normales',
            'icon': '🌤️',
            'description': 'Clima típico sin extremos',
            'categories': {
                'temperature': 'normal',
                'precipitation': 'normal',
                'wind': 'brisa',
                'pressure': 'normal',
            }
        },
        
        'sequia': {
            'name': 'Sequía',
            'icon': '🏜️',
            'description': 'Ausencia prolongada de lluvias',
            'categories': {
                'temperature': 'calor_moderado',
                'precipitation': 'sequia',
                'wind': 'calma',
                'uv': 'alto',
            }
        },
        
        'vientos_fuertes': {
            'name': 'Vientos Fuertes',
            'icon': '💨',
            'description': 'Vientos intensos con ráfagas',
            'categories': {
                'temperature': 'normal',
                'precipitation': 'seco',
                'wind': 'vendaval',
                'pressure': 'baja_presion',
            }
        },
        
        'tiempo_humedo': {
            'name': 'Tiempo Húmedo',
            'icon': '🌧️',
            'description': 'Lluvias moderadas constantes',
            'categories': {
                'temperature': 'normal',
                'precipitation': 'lluvioso',
                'wind': 'brisa',
                'pressure': 'normal',
            }
        },
    }

    # ========== CONFIGURACIÓN DE REGIONALES ==========
    # Metadata climática de cada regional para filtrar escenarios
    REGIONAL_CLIMATE_PROFILE = {
        'SAIDI_O': {  # Ocaña
            'tipo_clima': 'calido_seco',
            'elevacion': 'baja',  # ~1200 msnm
            'escenarios_excluidos': []  # Todos aplicables
        },
        'SAIDI_C': {  # Cúcuta
            'tipo_clima': 'calido_seco',
            'elevacion': 'baja',  # ~320 msnm
            'escenarios_excluidos': []  # Todos aplicables
        },
        'SAIDI_T': {  # Tibú
            'tipo_clima': 'calido_humedo',
            'elevacion': 'baja',  # ~75 msnm
            'escenarios_excluidos': []  # Todos aplicables
        },
        'SAIDI_A': {  # Aguachica
            'tipo_clima': 'calido_seco',
            'elevacion': 'baja',  # ~150 msnm
            'escenarios_excluidos': []  # Todos aplicables
        },
        'SAIDI_P': {  # Pamplona
            'tipo_clima': 'frio_humedo',
            'elevacion': 'alta',  # ~2200 msnm
            'escenarios_excluidos': ['calor_extremo', 'sequia']  # ❌ No aplican
        },
    }

    def __init__(self):
        """Inicializar servicio con configuración adaptativa"""
        self.variable_categories = self.VARIABLE_CATEGORIES
        self.composite_scenarios = self.COMPOSITE_SCENARIOS
        self.regional_climate_profile = self.REGIONAL_CLIMATE_PROFILE
        
        # Sincronizar con PredictionService (importación dinámica para evitar ciclos)
        try:
            from services.prediction_service import PredictionService
            self.regional_exog_vars = PredictionService.REGIONAL_EXOG_VARS
        except ImportError:
            # Fallback si no se puede importar
            self.regional_exog_vars = {}

    def categorize_variable(self, var_code: str) -> str:
        """
        Determinar categoría de una variable climática
        
        Args:
            var_code: Código de variable (ej: 'realfeel_min')
        
        Returns:
            Categoría (ej: 'temperature') o 'unknown'
        """
        # Búsqueda exacta en categorías definidas
        for category, config in self.variable_categories.items():
            if var_code in config['variables']:
                return category
        
        # Fallback: inferir por nombre de la variable
        var_lower = var_code.lower()
        
        if any(kw in var_lower for kw in ['temp', 'feel', 'chill', 'heat', 'dewpoint']):
            return 'temperature'
        elif any(kw in var_lower for kw in ['precip', 'rain']):
            return 'precipitation'
        elif any(kw in var_lower for kw in ['pressure', 'press']):
            return 'pressure'
        elif any(kw in var_lower for kw in ['wind', 'gust']):
            return 'wind'
        elif 'uv' in var_lower:
            return 'uv'
        
        return 'unknown'

    def apply_simulation(self,
                        exog_forecast: pd.DataFrame,
                        scenario_name: str,
                        intensity_adjustment: float,
                        alcance_meses: int,
                        percentiles: dict,
                        regional_code: str) -> pd.DataFrame:
        """
        Aplicar simulación climática adaptativa según variables de la regional
        
        MEJORAS:
        - Detecta automáticamente qué variables tiene la regional
        - Aplica transformación según categoría física
        - Maneja variables desconocidas con estrategia conservadora
        
        Args:
            exog_forecast: Variables exógenas SIN ESCALAR
            scenario_name: Nombre del escenario (ej: 'ola_calor')
            intensity_adjustment: Multiplicador de intensidad (0.5 - 2.0)
            alcance_meses: Meses a afectar (1, 3 o 6)
            percentiles: Percentiles históricos por variable
            regional_code: Código regional (ej: 'SAIDI_O')
        
        Returns:
            DataFrame simulado (sin escalar)
        """
        try:
            # Validar escenario
            if scenario_name not in self.composite_scenarios:
                raise ValueError(f"Escenario '{scenario_name}' no existe")
            
            scenario = self.composite_scenarios[scenario_name]
            exog_simulated = exog_forecast.copy()
            
            # Obtener variables de esta regional
            regional_vars = self.regional_exog_vars.get(regional_code, {})
            
            if not regional_vars:
                # Si no hay configuración específica, intentar con todas las columnas
                regional_vars = {col: col for col in exog_forecast.columns}
            
            # Aplicar simulación solo a primeros N meses
            meses_a_simular = min(alcance_meses, len(exog_simulated))
            
            for i in range(meses_a_simular):
                for var_code in exog_simulated.columns:
                    # Categorizar variable
                    category = self.categorize_variable(var_code)
                    
                    if category == 'unknown':
                        # Conservar valor original si no sabemos cómo simularlo
                        continue
                    
                    # Obtener configuración del escenario para esta categoría
                    category_scenario = scenario['categories'].get(category)
                    
                    if not category_scenario:
                        # Este escenario no afecta esta categoría
                        continue
                    
                    # Obtener configuración de la categoría
                    cat_config = self.variable_categories[category]
                    scenario_config = cat_config['scenarios'][category_scenario]
                    
                    # Calcular multiplicador
                    base_percentile = scenario_config['percentile']
                    base_intensity = scenario_config['intensity']
                    
                    # Aplicar ajuste de intensidad del usuario
                    final_intensity = base_intensity * intensity_adjustment
                    
                    # Obtener valor actual
                    current_value = exog_simulated.iloc[i][var_code]
                    
                    # Aplicar transformación según percentil
                    if var_code in percentiles:
                        p_data = percentiles[var_code]
                        
                        # Calcular valor objetivo del percentil
                        percentile_map = {
                            10: 'p10', 25: 'p25', 50: 'p50',
                            75: 'p75', 90: 'p90'
                        }
                        
                        p_key = percentile_map.get(base_percentile, 'p50')
                        target = p_data[p_key]
                        
                        # Interpolar entre valor actual y objetivo
                        simulated_value = current_value + (target - current_value) * final_intensity
                        
                        # Aplicar límites seguros
                        simulated_value = self._apply_safe_limits(
                            simulated_value, var_code, category, p_data
                        )
                        
                        # Actualizar
                        exog_simulated.iloc[i, exog_simulated.columns.get_loc(var_code)] = simulated_value
            
            return exog_simulated
        
        except Exception as e:
            raise Exception(f"Error aplicando simulación: {str(e)}")

    def _apply_safe_limits(self,
                          value: float,
                          var_code: str,
                          category: str,
                          percentiles: dict) -> float:
        """
        Aplicar límites físicamente razonables según categoría
        
        Args:
            value: Valor a limitar
            var_code: Código de la variable
            category: Categoría de la variable
            percentiles: Percentiles históricos
        
        Returns:
            Valor limitado dentro de rangos razonables
        """
        if category == 'temperature':
            # Temperatura: -10°C a 50°C
            return np.clip(value, -10, 50)
        
        elif category == 'precipitation':
            # Precipitación: 0 a 500mm (extremo)
            return np.clip(value, 0, 500)
        
        elif category == 'pressure':
            # Presión: 900 a 1100 hPa
            return np.clip(value, 900, 1100)
        
        elif category == 'wind':
            # Viento: 0 a 150 km/h (huracán categoría 1)
            return np.clip(value, 0, 150)
        
        elif category == 'uv':
            # Índice UV: 0 a 15 (extremo)
            return np.clip(value, 0, 15)
        
        else:
            # Límites por percentiles históricos (±50% del rango histórico)
            return np.clip(
                value,
                percentiles['p10'] * 0.5,
                percentiles['p90'] * 1.5
            )

    def calculate_percentiles(self, climate_data: pd.DataFrame, regional_code: str) -> dict:
        """
        Calcular percentiles históricos para cada variable climática
        
        Args:
            climate_data: DataFrame con datos climáticos históricos
            regional_code: Código de la regional
        
        Returns:
            Dict con percentiles por variable
        """
        try:
            if climate_data is None or climate_data.empty:
                raise ValueError("No hay datos climáticos disponibles")
            
            # Obtener variables de esta regional
            regional_vars = self.regional_exog_vars.get(regional_code, {})
            
            if not regional_vars:
                # Si no hay configuración, usar todas las columnas numéricas
                regional_vars = {col: col for col in climate_data.select_dtypes(include=[np.number]).columns}
            
            percentiles_result = {}
            
            for var_code in regional_vars.keys():
                # Buscar columna en climate_data (puede tener nombre diferente)
                if var_code not in climate_data.columns:
                    continue
                
                data = climate_data[var_code].dropna()
                
                if len(data) < 12:  # Necesitamos al menos 1 año de datos
                    continue
                
                percentiles_result[var_code] = {
                    'p10': float(np.percentile(data, 10)),
                    'p25': float(np.percentile(data, 25)),
                    'p50': float(np.percentile(data, 50)),
                    'p75': float(np.percentile(data, 75)),
                    'p90': float(np.percentile(data, 90)),
                    'mean': float(data.mean()),
                    'std': float(data.std()),
                    'min': float(data.min()),
                    'max': float(data.max()),
                }
            
            return percentiles_result
        
        except Exception as e:
            raise Exception(f"Error calculando percentiles: {str(e)}")

    def get_simulation_summary(self,
                               scenario_name: str,
                               intensity_adjustment: float,
                               alcance_meses: int,
                               percentiles: dict,
                               regional_code: str) -> dict:
        """
        Generar resumen de la configuración de simulación
        
        Args:
            scenario_name: Nombre del escenario
            intensity_adjustment: Factor de intensidad (0.5 - 2.0)
            alcance_meses: Número de meses afectados
            percentiles: Percentiles históricos
            regional_code: Código de la regional
        
        Returns:
            Dict con información del resumen
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
                
                if category == 'unknown':
                    continue
                
                category_scenario = scenario_info['categories'].get(category)
                
                if not category_scenario:
                    continue
                
                # Calcular cambio estimado
                cat_config = self.variable_categories[category]
                scenario_config = cat_config['scenarios'][category_scenario]
                
                base_intensity = scenario_config['intensity']
                final_intensity = base_intensity * intensity_adjustment
                
                # Cambio porcentual
                change_pct = (final_intensity - 1.0) * 100
                
                affected_vars[var_code] = {
                    'nombre': var_nombre,
                    'categoria': category,
                    'escenario_categoria': category_scenario,
                    'cambio_porcentual': change_pct,
                    'intensidad_final': final_intensity
                }
            
            return {
                'escenario': scenario_info['name'],
                'icono': scenario_info['icon'],
                'descripcion': scenario_info['description'],
                'intensity_adjustment': intensity_adjustment,
                'alcance_meses': alcance_meses,
                'variables_afectadas': affected_vars,
                'num_variables': len(affected_vars)
            }
        
        except Exception as e:
            print(f"Error generando resumen: {e}")
            return {}

    def validate_simulation_params(self,
                                   scenario_name: str,
                                   intensity_adjustment: float,
                                   alcance_meses: int,
                                   regional_code: str = None) -> tuple:
        """
        Validar parámetros de simulación
        
        Args:
            scenario_name: Nombre del escenario
            intensity_adjustment: Factor de intensidad
            alcance_meses: Número de meses
            regional_code: Código de regional (opcional, para validar compatibilidad)
        
        Returns:
            (es_valido, mensaje_error)
        """
        if scenario_name not in self.composite_scenarios:
            return False, f"Escenario no válido: {scenario_name}"
        
        # Validar que el escenario sea aplicable a la regional
        if regional_code and regional_code in self.regional_climate_profile:
            exclusiones = self.regional_climate_profile[regional_code].get('escenarios_excluidos', [])
            
            if scenario_name in exclusiones:
                regional_info = self.regional_climate_profile[regional_code]
                return False, f"El escenario '{scenario_name}' no es aplicable a esta regional (clima: {regional_info['tipo_clima']})"
        
        if not isinstance(intensity_adjustment, (int, float)):
            return False, "Ajuste de intensidad debe ser un número"
        
        if intensity_adjustment < 0.5 or intensity_adjustment > 2.0:
            return False, "Intensidad debe estar entre 0.5x y 2.0x"
        
        if alcance_meses not in [1, 3, 6]:
            return False, "Alcance debe ser 1, 3 o 6 meses"
        
        return True, ""

    def get_available_scenarios(self, regional_code: str = None) -> list:
        """
        Obtener lista de escenarios disponibles (filtrados por regional)
        
        Args:
            regional_code: Código de regional (opcional). Si se especifica,
                          filtra escenarios no aplicables a esa región.
        
        Returns:
            Lista de dicts con información de cada escenario
        """
        scenarios_list = []
        
        # Obtener exclusiones de la regional
        exclusiones = []
        if regional_code and regional_code in self.regional_climate_profile:
            exclusiones = self.regional_climate_profile[regional_code].get('escenarios_excluidos', [])
        
        for key, info in self.composite_scenarios.items():
            # Filtrar escenarios excluidos para esta regional
            if key in exclusiones:
                continue
            
            scenarios_list.append({
                'id': key,
                'name': info['name'],
                'icon': info['icon'],
                'description': info['description'],
                'categories': list(info['categories'].keys())
            })
        
        return scenarios_list

    def get_regional_climate_info(self, regional_code: str) -> dict:
        """
        Obtener información climática de una regional
        
        Args:
            regional_code: Código de la regional
        
        Returns:
            Dict con información del perfil climático
        """
        if regional_code not in self.regional_climate_profile:
            return {
                'tipo_clima': 'desconocido',
                'elevacion': 'desconocida',
                'escenarios_excluidos': [],
                'escenarios_aplicables': list(self.composite_scenarios.keys())
            }
        
        profile = self.regional_climate_profile[regional_code]
        
        # Calcular escenarios aplicables
        todos_escenarios = set(self.composite_scenarios.keys())
        excluidos = set(profile.get('escenarios_excluidos', []))
        aplicables = list(todos_escenarios - excluidos)
        
        return {
            'tipo_clima': profile['tipo_clima'],
            'elevacion': profile['elevacion'],
            'escenarios_excluidos': list(excluidos),
            'escenarios_aplicables': aplicables
        }

    def get_variable_category_info(self, var_code: str) -> dict:
        """
        Obtener información de categoría de una variable
        
        Args:
            var_code: Código de la variable
        
        Returns:
            Dict con información de la categoría
        """
        category = self.categorize_variable(var_code)
        
        if category == 'unknown':
            return {
                'category': 'unknown',
                'scenarios': [],
                'unit': 'N/A'
            }
        
        cat_config = self.variable_categories[category]
        
        # Determinar unidad según categoría
        unit_map = {
            'temperature': '°C',
            'precipitation': 'mm',
            'pressure': 'hPa',
            'wind': 'km/h',
            'uv': 'índice'
        }
        
        return {
            'category': category,
            'scenarios': list(cat_config['scenarios'].keys()),
            'unit': unit_map.get(category, '')
        }