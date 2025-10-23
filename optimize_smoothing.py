"""
optimize_smoothing.py - Búsqueda exhaustiva del suavizado óptimo por regional

PROPÓSITO:
Encontrar la mejor configuración de suavizado para cada regional, dados los 
parámetros SARIMAX óptimos ya conocidos.

USO:
1. Ajusta OPTIMAL_SARIMAX_PARAMS con tus parámetros óptimos encontrados
2. Ejecuta: python optimize_smoothing.py ruta_archivo.xlsx
3. Revisa resultados en: smoothing_optimization_results.json

BÚSQUEDA:
- Prueba 15+ configuraciones de suavizado por regional
- Calcula métricas completas en validación cruzada
- Selecciona la mejor configuración según precisión compuesta
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.nonparametric.smoothers_lowess import lowess
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from scipy import stats
import json
from datetime import datetime
import os
import sys

# Importar ExcelModel para cargar datos correctamente
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model.excel_model import ExcelModel

# ============================================================================
# CONFIGURACIÓN: Ajusta estos parámetros SARIMAX óptimos por regional
# ============================================================================
OPTIMAL_SARIMAX_PARAMS = {
    'SAIDI_O': {
        'order': (3, 0, 3),
        'seasonal_order': (3, 1, 3, 12),
        'transformation': 'boxcox'
    },
    'SAIDI_C': {
        'order': (2, 0, 1),
        'seasonal_order': (1, 0, 2, 12),
        'transformation': 'original'
    },
    'SAIDI_A': {
        'order': (0, 1, 0),
        'seasonal_order': (3, 0, 1, 12),
        'transformation': 'original'
    },
    'SAIDI_P': {
        'order': (2, 0, 0),
        'seasonal_order': (3, 0, 3, 12),
        'transformation': 'boxcox'
    },
    'SAIDI_T': {
        'order': (0, 0, 3),
        'seasonal_order': (1, 2, 3, 12),
        'transformation': 'sqrt'
    },
    'SAIDI_Cens': {
        'order': (2, 1, 2),
        'seasonal_order': (0, 1, 1, 12),
        'transformation': 'original'
    }
}

# ============================================================================
# CONFIGURACIONES DE SUAVIZADO A PROBAR (EXPANDIDAS)
# ============================================================================
SMOOTHING_CONFIGS = [
    # Sin suavizado (baseline)
    {'method': 'none', 'window': None, 'name': 'Sin suavizado'},
    
    # Media móvil - ventanas pequeñas (2-10)
    {'method': 'moving_average', 'window': 2, 'name': 'MA window=2'},
    {'method': 'moving_average', 'window': 3, 'name': 'MA window=3'},
    {'method': 'moving_average', 'window': 4, 'name': 'MA window=4'},
    {'method': 'moving_average', 'window': 5, 'name': 'MA window=5'},
    {'method': 'moving_average', 'window': 6, 'name': 'MA window=6'},
    {'method': 'moving_average', 'window': 7, 'name': 'MA window=7'},
    {'method': 'moving_average', 'window': 8, 'name': 'MA window=8'},
    {'method': 'moving_average', 'window': 9, 'name': 'MA window=9'},
    {'method': 'moving_average', 'window': 10, 'name': 'MA window=10'},
    
    # Media móvil - ventanas medianas (12-24, para capturas estacionalidad)
    {'method': 'moving_average', 'window': 12, 'name': 'MA window=12'},
    {'method': 'moving_average', 'window': 15, 'name': 'MA window=15'},
    {'method': 'moving_average', 'window': 18, 'name': 'MA window=18'},
    {'method': 'moving_average', 'window': 24, 'name': 'MA window=24'},
    
    # Suavizado exponencial - spans pequeños (2-10)
    {'method': 'exponential', 'window': 2, 'name': 'EWM span=2'},
    {'method': 'exponential', 'window': 3, 'name': 'EWM span=3'},
    {'method': 'exponential', 'window': 4, 'name': 'EWM span=4'},
    {'method': 'exponential', 'window': 5, 'name': 'EWM span=5'},
    {'method': 'exponential', 'window': 6, 'name': 'EWM span=6'},
    {'method': 'exponential', 'window': 7, 'name': 'EWM span=7'},
    {'method': 'exponential', 'window': 8, 'name': 'EWM span=8'},
    {'method': 'exponential', 'window': 9, 'name': 'EWM span=9'},
    {'method': 'exponential', 'window': 10, 'name': 'EWM span=10'},
    
    # Suavizado exponencial - spans medianos (12-24)
    {'method': 'exponential', 'window': 12, 'name': 'EWM span=12'},
    {'method': 'exponential', 'window': 15, 'name': 'EWM span=15'},
    {'method': 'exponential', 'window': 18, 'name': 'EWM span=18'},
    {'method': 'exponential', 'window': 24, 'name': 'EWM span=24'},
    
    # LOWESS - fracs muy pequeñas (suavizado mínimo)
    {'method': 'lowess', 'frac': 0.02, 'name': 'LOWESS frac=0.02'},
    {'method': 'lowess', 'frac': 0.03, 'name': 'LOWESS frac=0.03'},
    {'method': 'lowess', 'frac': 0.05, 'name': 'LOWESS frac=0.05'},
    {'method': 'lowess', 'frac': 0.07, 'name': 'LOWESS frac=0.07'},
    
    # LOWESS - fracs pequeñas a medianas
    {'method': 'lowess', 'frac': 0.10, 'name': 'LOWESS frac=0.10'},
    {'method': 'lowess', 'frac': 0.12, 'name': 'LOWESS frac=0.12'},
    {'method': 'lowess', 'frac': 0.15, 'name': 'LOWESS frac=0.15'},
    {'method': 'lowess', 'frac': 0.18, 'name': 'LOWESS frac=0.18'},
    {'method': 'lowess', 'frac': 0.20, 'name': 'LOWESS frac=0.20'},
    {'method': 'lowess', 'frac': 0.25, 'name': 'LOWESS frac=0.25'},
    {'method': 'lowess', 'frac': 0.30, 'name': 'LOWESS frac=0.30'},
    
    # Media móvil ponderada - ventanas pequeñas
    {'method': 'moving_average_weighted', 'window': 2, 'name': 'WMA window=2'},
    {'method': 'moving_average_weighted', 'window': 3, 'name': 'WMA window=3'},
    {'method': 'moving_average_weighted', 'window': 4, 'name': 'WMA window=4'},
    {'method': 'moving_average_weighted', 'window': 5, 'name': 'WMA window=5'},
    {'method': 'moving_average_weighted', 'window': 6, 'name': 'WMA window=6'},
    {'method': 'moving_average_weighted', 'window': 7, 'name': 'WMA window=7'},
    {'method': 'moving_average_weighted', 'window': 8, 'name': 'WMA window=8'},
]


class SmoothingOptimizer:
    """Optimizador de suavizado para series temporales SAIDI"""
    
    def __init__(self):
        self.results = {}
        self.scaler = None
        self.transformation_params = {}
    
    def apply_smoothing(self, data, config):
        """Aplicar suavizado según configuración"""
        method = config['method']
        
        if method == 'none':
            return data, "Sin suavizado"
        
        elif method == 'moving_average':
            window = config['window']
            series = pd.Series(data)
            smoothed = series.rolling(window=window, center=True, min_periods=1).mean()
            smoothed = smoothed.fillna(series).values
            return smoothed, f"MA(window={window})"
        
        elif method == 'exponential':
            window = config['window']
            series = pd.Series(data)
            smoothed = series.ewm(span=window, adjust=False, min_periods=1).mean().values
            return smoothed, f"EWM(span={window})"
        
        elif method == 'lowess':
            frac = config.get('frac', 0.10)
            smoothed = lowess(data, np.arange(len(data)), frac=frac, return_sorted=False)
            return smoothed, f"LOWESS(frac={frac})"
        
        elif method == 'moving_average_weighted':
            window = config['window']
            series = pd.Series(data)
            weights = np.arange(1, window + 1)
            smoothed = series.rolling(window=window, center=True, min_periods=1).apply(
                lambda x: np.sum(weights[:len(x)] * x) / np.sum(weights[:len(x)]), raw=True
            )
            smoothed = smoothed.fillna(series).values
            return smoothed, f"WMA(window={window})"
        
        else:
            return data, "Desconocido"
    
    def apply_transformation(self, data, transformation_type):
        """Aplicar transformación a los datos"""
        if transformation_type == 'original':
            return data
        
        elif transformation_type == 'standard':
            self.scaler = StandardScaler()
            return self.scaler.fit_transform(data.reshape(-1, 1)).flatten()
        
        elif transformation_type == 'log':
            data_positive = np.maximum(data, 1e-10)
            return np.log(data_positive)
        
        elif transformation_type == 'boxcox':
            data_positive = np.maximum(data, 1e-10)
            transformed, lambda_param = stats.boxcox(data_positive)
            self.transformation_params['boxcox_lambda'] = lambda_param
            return transformed
        
        else:
            return data
    
    def inverse_transformation(self, data, transformation_type):
        """Revertir transformación"""
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
        
        else:
            return data
    
    def calculate_metrics(self, actual, predicted):
        """Calcular métricas completas"""
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        mae = np.mean(np.abs(actual - predicted))
        
        epsilon = 1e-8
        mape = np.mean(np.abs((actual - predicted) / (actual + epsilon))) * 100
        
        ss_res = np.sum((actual - predicted) ** 2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + epsilon))
        
        # Precisión compuesta
        precision_mape = max(0, 100 - mape)
        precision_r2 = max(0, r2 * 100)
        mean_actual = np.mean(actual)
        precision_rmse = max(0, (1 - rmse/mean_actual) * 100) if mean_actual > 0 else 0
        
        precision_final = (precision_mape * 0.4 + precision_r2 * 0.4 + precision_rmse * 0.2)
        precision_final = max(0, min(100, precision_final))
        
        return {
            'rmse': float(rmse),
            'mae': float(mae),
            'mape': float(mape),
            'r2': float(r2),
            'precision_mape': float(precision_mape),
            'precision_r2': float(precision_r2),
            'precision_rmse': float(precision_rmse),
            'precision_final': float(precision_final)
        }
    
    def evaluate_smoothing_config(self, serie_original, smoothing_config, 
                                  order, seasonal_order, transformation):
        """Evaluar una configuración de suavizado"""
        try:
            # Determinar tamaño de validación
            if len(serie_original) >= 60:
                pct_validacion = 0.30
            elif len(serie_original) >= 36:
                pct_validacion = 0.25
            else:
                pct_validacion = 0.20
            
            n_test = max(6, int(len(serie_original) * pct_validacion))
            
            # Separar train/test
            train_original = serie_original[:-n_test]
            test_original = serie_original[-n_test:]
            
            # PASO 1: SUAVIZADO
            train_smoothed, smooth_info = self.apply_smoothing(
                train_original.values, smoothing_config
            )
            
            # PASO 2: TRANSFORMACIÓN
            train_transformed = self.apply_transformation(train_smoothed, transformation)
            train_series = pd.Series(train_transformed, index=train_original.index)
            
            # PASO 3: MODELO SARIMAX
            model = SARIMAX(
                train_series,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=True,
                enforce_invertibility=True
            )
            results = model.fit(disp=False, maxiter=50)
            
            # PASO 4: PREDICCIÓN
            pred = results.get_forecast(steps=n_test)
            pred_transformed = pred.predicted_mean
            
            # PASO 5: INVERSIÓN
            pred_original = self.inverse_transformation(
                pred_transformed.values, transformation
            )
            
            # PASO 6: MÉTRICAS
            metrics = self.calculate_metrics(test_original.values, pred_original)
            metrics['smooth_info'] = smooth_info
            metrics['n_test'] = n_test
            metrics['aic'] = float(results.aic)
            metrics['bic'] = float(results.bic)
            
            return metrics
            
        except Exception as e:
            return {
                'rmse': float('inf'),
                'mae': float('inf'),
                'mape': 100.0,
                'r2': -1.0,
                'precision_final': 0.0,
                'error': str(e)
            }
    
    def optimize_regional(self, file_path, regional_code):
        """Optimizar suavizado para una regional específica"""
        print(f"\n{'='*80}")
        print(f"OPTIMIZANDO SUAVIZADO PARA: {regional_code}")
        print(f"{'='*80}")
        
        # CORRECCIÓN: Usar ExcelModel para cargar datos correctamente
        excel_model = ExcelModel()
        
        if not excel_model.load_excel_file(file_path):
            raise Exception(f"Error cargando archivo: {file_path}")
        
        # Si es formato regional, seleccionar la regional
        if excel_model.is_regional_format():
            if not excel_model.set_selected_regional(regional_code):
                raise Exception(f"No se pudo seleccionar regional: {regional_code}")
            print(f"[INFO] Formato regional detectado - Regional seleccionada: {regional_code}")
        else:
            print(f"[INFO] Formato tradicional detectado")
        
        # Obtener datos preparados para análisis (ya en formato unificado)
        df = excel_model.get_excel_data_for_analysis()
        
        if df is None:
            raise Exception("No se pudieron obtener datos para análisis")
        
        # Preparar datos (ahora siempre es formato unificado: 'Fecha' y 'SAIDI')
        if not isinstance(df.index, pd.DatetimeIndex):
            df["Fecha"] = pd.to_datetime(df["Fecha"])
            df.set_index("Fecha", inplace=True)
        
        col_saidi = "SAIDI"  # Siempre es 'SAIDI' después de get_excel_data_for_analysis()
        serie_original = df[df[col_saidi].notna()][col_saidi]
        
        print(f"Dataset: {len(serie_original)} observaciones")
        print(f"Período: {serie_original.index[0].strftime('%Y-%m')} a {serie_original.index[-1].strftime('%Y-%m')}")
        
        # Obtener parámetros óptimos
        params = OPTIMAL_SARIMAX_PARAMS.get(regional_code, {
            'order': (1, 1, 2),
            'seasonal_order': (1, 0, 1, 12),
            'transformation': 'original'
        })
        
        order = params['order']
        seasonal_order = params['seasonal_order']
        transformation = params['transformation']
        
        print(f"Parámetros SARIMAX: order={order}, seasonal={seasonal_order}")
        print(f"Transformación: {transformation}")
        print(f"\nProbando {len(SMOOTHING_CONFIGS)} configuraciones de suavizado...")
        
        # Evaluar cada configuración
        results_list = []
        for i, config in enumerate(SMOOTHING_CONFIGS, 1):
            print(f"\n[{i}/{len(SMOOTHING_CONFIGS)}] Evaluando: {config['name']}")
            
            metrics = self.evaluate_smoothing_config(
                serie_original, config, order, seasonal_order, transformation
            )
            
            result = {
                'config': config,
                'metrics': metrics
            }
            results_list.append(result)
            
            # Mostrar resultado
            if 'error' in metrics:
                print(f"    ❌ ERROR: {metrics['error']}")
            else:
                print(f"    Precisión: {metrics['precision_final']:.1f}%")
                print(f"    RMSE: {metrics['rmse']:.4f} | MAE: {metrics['mae']:.4f}")
                print(f"    MAPE: {metrics['mape']:.1f}% | R²: {metrics['r2']:.3f}")
        
        # Ordenar por precisión
        results_list.sort(key=lambda x: x['metrics'].get('precision_final', 0), reverse=True)
        
        # Mostrar TOP 5
        print(f"\n{'='*80}")
        print(f"TOP 5 CONFIGURACIONES PARA {regional_code}")
        print(f"{'='*80}")
        
        for i, result in enumerate(results_list[:5], 1):
            config = result['config']
            metrics = result['metrics']
            medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"#{i}"
            
            print(f"\n{medal} {config['name']}")
            print(f"   Precisión: {metrics['precision_final']:.1f}% | R²: {metrics['r2']:.3f}")
            print(f"   RMSE: {metrics['rmse']:.4f} | MAPE: {metrics['mape']:.1f}%")
            print(f"   AIC: {metrics.get('aic', 'N/A'):.1f} | BIC: {metrics.get('bic', 'N/A'):.1f}")
        
        # Guardar resultados
        self.results[regional_code] = {
            'optimal_sarimax': params,
            'all_smoothing_results': results_list,
            'best_smoothing': results_list[0]['config'],
            'best_metrics': results_list[0]['metrics'],
            'baseline_metrics': next((r['metrics'] for r in results_list if r['config']['method'] == 'none'), None)
        }
        
        return results_list[0]
    
    def optimize_all_regionals(self, file_path):
        """Optimizar todas las regionales desde un archivo"""
        print(f"\n{'#'*80}")
        print(f"INICIANDO OPTIMIZACIÓN DE SUAVIZADO - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'#'*80}")
        
        if not os.path.exists(file_path):
            print(f"❌ ERROR: Archivo no encontrado: {file_path}")
            return
        
        print(f"\n[INFO] Archivo: {file_path}")
        
        excel_model = ExcelModel()
        if not excel_model.load_excel_file(file_path):
            print(f"❌ ERROR: No se pudo cargar {file_path}")
            return
        
        if excel_model.is_regional_format():
            # Formato regional: optimizar cada regional
            regionales = excel_model.get_available_regionales()
            print(f"[INFO] Formato REGIONAL detectado")
            print(f"[INFO] Regionales detectadas: {[r['codigo'] for r in regionales]}")
            
            for regional_info in regionales:
                regional_code = regional_info['codigo']
                try:
                    self.optimize_regional(file_path, regional_code)
                except Exception as e:
                    print(f"\n❌ ERROR en {regional_code}: {str(e)}")
                    continue
        else:
            # Formato tradicional: necesitas especificar el código de regional
            print(f"❌ ERROR: El archivo es formato TRADICIONAL")
            print(f"    Para archivos tradicionales, especifica el código de regional:")
            print(f"    python optimize_smoothing.py archivo.xlsx SAIDI_C")
            return
        
        # Generar resumen final
        self.generate_summary()
        
        # Guardar resultados
        self.save_results()
    
    def generate_summary(self):
        """Generar resumen comparativo"""
        print(f"\n{'#'*80}")
        print(f"RESUMEN FINAL - MEJORES CONFIGURACIONES POR REGIONAL")
        print(f"{'#'*80}\n")
        
        summary_data = []
        
        for regional_code, data in self.results.items():
            best_config = data['best_smoothing']
            best_metrics = data['best_metrics']
            baseline = data.get('baseline_metrics', {})
            
            improvement = best_metrics['precision_final'] - baseline.get('precision_final', 0)
            
            summary_data.append({
                'Regional': regional_code,
                'Mejor Config': best_config['name'],
                'Precisión': f"{best_metrics['precision_final']:.1f}%",
                'R²': f"{best_metrics['r2']:.3f}",
                'MAPE': f"{best_metrics['mape']:.1f}%",
                'RMSE': f"{best_metrics['rmse']:.4f}",
                'Mejora vs Baseline': f"{improvement:+.1f}%"
            })
        
        # Crear DataFrame para tabla bonita
        df_summary = pd.DataFrame(summary_data)
        print(df_summary.to_string(index=False))
        
        print(f"\n{'='*80}")
        print("CÓDIGO PYTHON PARA COPIAR EN TUS SERVICIOS:")
        print(f"{'='*80}\n")
        
        print("REGIONAL_SMOOTHING = {")
        for regional_code, data in self.results.items():
            config = data['best_smoothing']
            method = config['method']
            
            if method == 'none':
                print(f"    '{regional_code}': {{'method': 'none', 'window': None}},")
            elif method in ['moving_average', 'exponential', 'moving_average_weighted']:
                window = config['window']
                print(f"    '{regional_code}': {{'method': '{method}', 'window': {window}}},")
            elif method == 'lowess':
                frac = config['frac']
                print(f"    '{regional_code}': {{'method': 'lowess', 'frac': {frac}}},")
        
        print("}")
    
    def save_results(self, output_file='smoothing_optimization_results.json'):
        """Guardar resultados en JSON"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Resultados guardados en: {output_file}")


# ============================================================================
# EJECUCIÓN PRINCIPAL
# ============================================================================
if __name__ == "__main__":
    optimizer = SmoothingOptimizer()
    
    print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                    OPTIMIZADOR DE SUAVIZADO SAIDI                          ║
║                                                                            ║
║  USO:                                                                      ║
║  python optimize_smoothing.py [archivo.xlsx]                              ║
║                                                                            ║
║  Sin argumentos: usa 'SAIDI_regionales.xlsx' por defecto                  ║
╚════════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Determinar archivo a usar
    if len(sys.argv) >= 2:
        file_path = sys.argv[1]
    else:
        # Buscar SAIDI_regionales.xlsx en varios lugares
        possible_paths = [
            'SAIDI_regionales.xlsx',
            'data/SAIDI_regionales.xlsx',
            '../SAIDI_regionales.xlsx',
            os.path.join(os.path.dirname(__file__), 'SAIDI_regionales.xlsx'),
            os.path.join(os.path.dirname(__file__), 'data', 'SAIDI_regionales.xlsx')
        ]
        
        file_path = None
        for path in possible_paths:
            if os.path.exists(path):
                file_path = path
                break
        
        if file_path is None:
            print("❌ ERROR: No se encontró 'SAIDI_regionales.xlsx'")
            print("\nBuscado en:")
            for path in possible_paths:
                print(f"  - {path}")
            print("\nPor favor, especifique la ruta manualmente:")
            print("  python optimize_smoothing.py ruta/a/SAIDI_regionales.xlsx")
            sys.exit(1)
    
    if not os.path.exists(file_path):
        print(f"❌ ERROR: Archivo no encontrado: {file_path}")
        sys.exit(1)
    
    print(f"[INFO] Usando archivo: {file_path}\n")
    
    optimizer.optimize_all_regionals(file_path)
    
    print("\n" + "="*80)
    print("OPTIMIZACIÓN COMPLETADA")
    print("="*80)