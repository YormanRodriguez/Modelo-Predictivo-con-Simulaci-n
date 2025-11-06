# services/climate_simulation_service.py - Servicio de simulación climática para SAIDI
import pandas as pd
import numpy as np
from typing import Dict, Tuple

class ClimateSimulationService:
    """Servicio para simular escenarios climáticos y modificar variables exógenas"""
    
    # Definición de escenarios climáticos basados en percentiles
    SCENARIOS = {
        'soleado': {
            'name': 'Clima Seco/Soleado',
            'icon': '☀️',
            'description': 'Días soleados, baja humedad y poca lluvia',
            'percentiles': {
                'temp_max': 75,      # Temperatura alta
                'humedad_avg': 25,   # Humedad baja
                'precip_total': 10   # Precipitación muy baja
            }
        },
        'lluvioso': {
            'name': 'Clima Lluvioso',
            'icon': '🌧️',
            'description': 'Lluvias moderadas, humedad alta',
            'percentiles': {
                'temp_max': 50,      # Temperatura normal
                'humedad_avg': 75,   # Humedad alta
                'precip_total': 75   # Precipitación alta
            }
        },
        'tormentoso': {
            'name': 'Clima Tormentoso',
            'icon': '⛈️',
            'description': 'Tormentas intensas, lluvia extrema',
            'percentiles': {
                'temp_max': 25,      # Temperatura baja
                'humedad_avg': 90,   # Humedad muy alta
                'precip_total': 90   # Precipitación extrema
            }
        },
        'ola_calor': {
            'name': 'Ola de Calor',
            'icon': '🌡️',
            'description': 'Calor extremo, sequía',
            'percentiles': {
                'temp_max': 90,      # Temperatura muy alta
                'humedad_avg': 10,   # Humedad muy baja
                'precip_total': 10   # Precipitación muy baja
            }
        }
    }
    
    # Variables exógenas por regional (debe coincidir con PredictionService)
    REGIONAL_EXOG_VARS = {
        'SAIDI_O': ['temp_max', 'humedad_avg', 'precip_total'],
        'SAIDI_C': ['temp_max', 'humedad_avg', 'precip_total'],
        'SAIDI_A': ['temp_max', 'humedad_avg', 'precip_total'],
        'SAIDI_P': ['temp_max', 'humedad_avg', 'precip_total'],
        'SAIDI_T': ['temp_max', 'humedad_avg', 'precip_total']
    }
    
    def __init__(self):
        self.percentiles_cache = {}
        self.base_days_cache = {}
    
    def calculate_percentiles(self, climate_data: pd.DataFrame, regional_code: str) -> Dict[str, Dict[str, float]]:
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
            
            cache_key = f"{regional_code}_{len(climate_data)}"
            if cache_key in self.percentiles_cache:
                return self.percentiles_cache[cache_key]
            
            variables = self.REGIONAL_EXOG_VARS.get(regional_code, [])
            percentiles_result = {}
            
            for var in variables:
                if var not in climate_data.columns:
                    continue
                
                data = climate_data[var].dropna()
                
                if len(data) < 12:  # Necesitamos al menos 1 año de datos
                    continue
                
                percentiles_result[var] = {
                    'p10': float(np.percentile(data, 10)),
                    'p25': float(np.percentile(data, 25)),
                    'p50': float(np.percentile(data, 50)),
                    'p75': float(np.percentile(data, 75)),
                    'p90': float(np.percentile(data, 90)),
                    'mean': float(data.mean()),
                    'std': float(data.std()),
                    'min': float(data.min()),
                    'max': float(data.max())
                }
            
            self.percentiles_cache[cache_key] = percentiles_result
            return percentiles_result
            
        except Exception as e:
            raise Exception(f"Error calculando percentiles: {str(e)}")
    
    def calculate_base_days(self, climate_data: pd.DataFrame, mes_prediccion: int, 
                           regional_code: str) -> Dict[str, int]:
        """
        Calcular días base para cada tipo de clima según histórico del mes
        
        Args:
            climate_data: DataFrame con datos climáticos históricos
            mes_prediccion: Mes a predecir (1-12)
            regional_code: Código de la regional
            
        Returns:
            Dict con días base por escenario
        """
        try:
            if climate_data is None or climate_data.empty:
                return {'soleado': 10, 'lluvioso': 12, 'tormentoso': 8}
            
            cache_key = f"{regional_code}_{mes_prediccion}"
            if cache_key in self.base_days_cache:
                return self.base_days_cache[cache_key]
            
            # Filtrar datos del mes específico
            mes_data = climate_data[climate_data['month'] == mes_prediccion].copy()
            
            if len(mes_data) < 3:  # Necesitamos al menos 3 años de ese mes
                return {'soleado': 10, 'lluvioso': 12, 'tormentoso': 8}
            
            # Calcular percentiles de precipitación para el mes
            precip_col = 'precip_total'
            if precip_col not in mes_data.columns:
                return {'soleado': 10, 'lluvioso': 12, 'tormentoso': 8}
            
            precip_data = mes_data[precip_col].dropna()
            
            if len(precip_data) == 0:
                return {'soleado': 10, 'lluvioso': 12, 'tormentoso': 8}
            
            p25_precip = np.percentile(precip_data, 25)
            p75_precip = np.percentile(precip_data, 75)
            p90_precip = np.percentile(precip_data, 90)
            
            # Contar meses según rangos de precipitación
            soleados = len(precip_data[precip_data < p25_precip])
            lluviosos = len(precip_data[(precip_data >= p25_precip) & (precip_data < p75_precip)])
            tormentosos = len(precip_data[precip_data >= p90_precip])
            
            total_meses = len(precip_data)
            
            # Normalizar a 30 días
            dias_base = {
                'soleado': int(round((soleados / total_meses) * 30)),
                'lluvioso': int(round((lluviosos / total_meses) * 30)),
                'tormentoso': int(round((tormentosos / total_meses) * 30))
            }
            
            # Asegurar al menos 6 días para cada tipo
            dias_base = {k: max(6, v) for k, v in dias_base.items()}
            
            # Asegurar que sumen aproximadamente 30
            total = sum(dias_base.values())
            if total < 28 or total > 32:
                # Reajustar proporcionalmente
                factor = 30 / total
                dias_base = {k: int(round(v * factor)) for k, v in dias_base.items()}
            
            self.base_days_cache[cache_key] = dias_base
            return dias_base
            
        except Exception as e:
            print(f"Error calculando días base: {e}")
            return {'soleado': 10, 'lluvioso': 12, 'tormentoso': 8}
    
    def calculate_multipliers(self, percentiles: Dict[str, Dict[str, float]], 
                             escenario: str, regional_code: str) -> Dict[str, float]:
        """
        Calcular multiplicadores para cada variable según escenario
        
        Args:
            percentiles: Percentiles calculados de las variables
            escenario: Nombre del escenario ('soleado', 'lluvioso', etc.)
            regional_code: Código de la regional
            
        Returns:
            Dict con multiplicadores por variable
        """
        try:
            if escenario not in self.SCENARIOS:
                raise ValueError(f"Escenario no válido: {escenario}")
            
            scenario_config = self.SCENARIOS[escenario]
            multipliers = {}
            
            variables = self.REGIONAL_EXOG_VARS.get(regional_code, [])
            
            for var in variables:
                if var not in percentiles:
                    multipliers[var] = 1.0
                    continue
                
                p_data = percentiles[var]
                p50 = p_data['p50']  # Mediana
                
                # Obtener el percentil del escenario
                target_percentile = scenario_config['percentiles'].get(var, 50)
                
                # Calcular valor del percentil objetivo
                if target_percentile == 10:
                    target_value = p_data['p10']
                elif target_percentile == 25:
                    target_value = p_data['p25']
                elif target_percentile == 50:
                    target_value = p_data['p50']
                elif target_percentile == 75:
                    target_value = p_data['p75']
                elif target_percentile == 90:
                    target_value = p_data['p90']
                else:
                    target_value = p50
                
                # Calcular multiplicador como ratio
                if p50 > 0:
                    multipliers[var] = target_value / p50
                else:
                    multipliers[var] = 1.0
            
            return multipliers
            
        except Exception as e:
            print(f"Error calculando multiplicadores: {e}")
            return {var: 1.0 for var in self.REGIONAL_EXOG_VARS.get(regional_code, [])}
    
    def calculate_slider_ranges(self, climate_data: pd.DataFrame, mes_prediccion: int,
                                escenario: str, regional_code: str) -> Dict[str, Tuple[int, int, int]]:
        """
        Calcular rangos del slider basados en desviación estándar
        
        Args:
            climate_data: DataFrame con datos climáticos
            mes_prediccion: Mes a predecir
            escenario: Nombre del escenario
            regional_code: Código de la regional
            
        Returns:
            Dict con (min, max, base) por tipo de clima
        """
        try:
            dias_base = self.calculate_base_days(climate_data, mes_prediccion, regional_code)
            
            # Calcular desviación estándar de ocurrencias por tipo de clima
            # Usamos 2 desviaciones estándar como rango
            
            ranges = {}
            
            for tipo_clima in ['soleado', 'lluvioso', 'tormentoso']:
                base = dias_base[tipo_clima]
                
                # Estimar desviación estándar (aproximadamente 30% del valor base)
                std_dev = max(3, int(round(base * 0.3)))
                
                # Rango = ±2 * desviación estándar
                min_dias = max(2, base - 2 * std_dev)
                max_dias = min(28, base + 2 * std_dev)
                
                ranges[tipo_clima] = (min_dias, max_dias, base)
            
            return ranges
            
        except Exception as e:
            print(f"Error calculando rangos de slider: {e}")
            return {
                'soleado': (4, 16, 10),
                'lluvioso': (6, 18, 12),
                'tormentoso': (2, 14, 8)
            }
    
    def apply_simulation(self, exog_forecast: pd.DataFrame, escenario: str, 
                        slider_adjustment: int, dias_base: int, alcance_meses: int,
                        percentiles: Dict[str, Dict[str, float]], 
                        regional_code: str) -> pd.DataFrame:
        """
        Aplicar simulación climática a las variables exógenas de predicción
        
        Args:
            exog_forecast: DataFrame con variables exógenas proyectadas
            escenario: Nombre del escenario
            slider_adjustment: Ajuste de días del slider (ej: +3, -2)
            dias_base: Días base del escenario
            alcance_meses: Número de meses a afectar (1, 3, o 6)
            percentiles: Percentiles calculados de las variables
            regional_code: Código de la regional
            
        Returns:
            DataFrame con variables exógenas simuladas
        """
        try:
            if exog_forecast is None or exog_forecast.empty:
                raise ValueError("No hay variables exógenas para simular")
            
            exog_simulated = exog_forecast.copy()
            
            # Calcular multiplicadores base del escenario
            base_multipliers = self.calculate_multipliers(percentiles, escenario, regional_code)
            
            # Calcular factor de intensidad por días adicionales
            dias_simulados = dias_base + slider_adjustment
            intensity_factor = dias_simulados / dias_base if dias_base > 0 else 1.0
            
            # Aplicar simulación solo a los primeros N meses
            meses_a_simular = min(alcance_meses, len(exog_simulated))
            
            for i in range(meses_a_simular):
                for var, base_mult in base_multipliers.items():
                    if var not in exog_simulated.columns:
                        continue
                    
                    # Multiplicador final = base * intensidad
                    final_mult = base_mult * intensity_factor
                    
                    # Aplicar multiplicador
                    current_value = exog_simulated.iloc[i][var]
                    simulated_value = current_value * final_mult
                    
                    # Aplicar límites de seguridad (clip)
                    if var in percentiles:
                        p_data = percentiles[var]
                        
                        if var == 'temp_max':
                            min_limit = p_data['p10']
                            max_limit = p_data['p90'] * 1.1
                        elif var == 'humedad_avg':
                            min_limit = 30
                            max_limit = 100
                        elif var == 'precip_total':
                            min_limit = 0
                            max_limit = p_data['p90'] * 1.5
                        else:
                            min_limit = p_data['min']
                            max_limit = p_data['max']
                        
                        simulated_value = np.clip(simulated_value, min_limit, max_limit)
                    
                    exog_simulated.iloc[i, exog_simulated.columns.get_loc(var)] = simulated_value
            
            return exog_simulated
            
        except Exception as e:
            raise Exception(f"Error aplicando simulación: {str(e)}")
    
    def get_simulation_summary(self, escenario: str, slider_adjustment: int, 
                               dias_base: int, alcance_meses: int, 
                               percentiles: Dict[str, Dict[str, float]],
                               regional_code: str) -> Dict:
        """
        Generar resumen de la configuración de simulación
        
        Returns:
            Dict con información del resumen
        """
        try:
            scenario_info = self.SCENARIOS.get(escenario, {})
            base_multipliers = self.calculate_multipliers(percentiles, escenario, regional_code)
            
            dias_simulados = dias_base + slider_adjustment
            intensity_factor = dias_simulados / dias_base if dias_base > 0 else 1.0
            
            # Calcular multiplicadores finales
            final_multipliers = {
                var: base_mult * intensity_factor 
                for var, base_mult in base_multipliers.items()
            }
            
            # Calcular cambios porcentuales
            percentage_changes = {
                var: (final_mult - 1.0) * 100
                for var, final_mult in final_multipliers.items()
            }
            
            return {
                'escenario': scenario_info.get('name', escenario),
                'icono': scenario_info.get('icon', ''),
                'descripcion': scenario_info.get('description', ''),
                'dias_base': dias_base,
                'slider_adjustment': slider_adjustment,
                'dias_simulados': dias_simulados,
                'alcance_meses': alcance_meses,
                'intensity_factor': intensity_factor,
                'base_multipliers': base_multipliers,
                'final_multipliers': final_multipliers,
                'percentage_changes': percentage_changes
            }
            
        except Exception as e:
            print(f"Error generando resumen: {e}")
            return {}
    
    def validate_simulation_params(self, escenario: str, slider_adjustment: int,
                                   dias_base: int, alcance_meses: int) -> Tuple[bool, str]:
        """
        Validar parámetros de simulación
        
        Returns:
            (es_valido, mensaje_error)
        """
        if escenario not in self.SCENARIOS:
            return False, f"Escenario no válido: {escenario}"
        
        if not isinstance(slider_adjustment, int):
            return False, "Ajuste de días debe ser un número entero"
        
        if not isinstance(dias_base, int) or dias_base < 1:
            return False, "Días base debe ser un número positivo"
        
        dias_simulados = dias_base + slider_adjustment
        if dias_simulados < 2 or dias_simulados > 28:
            return False, f"Días simulados ({dias_simulados}) fuera de rango válido (2-28)"
        
        if alcance_meses not in [1, 3, 6]:
            return False, "Alcance debe ser 1, 3 o 6 meses"
        
        return True, ""