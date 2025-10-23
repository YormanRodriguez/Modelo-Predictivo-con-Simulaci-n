# services/export_service.py - Servicio para exportar predicciones a Excel
import pandas as pd
import os
from datetime import datetime
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side

class ExportService:
    """Servicio para exportar predicciones SAIDI a Excel con formato profesional"""
    
    def __init__(self):
        self.last_export_path = None
    
    def export_predictions_to_excel(self, predictions_dict, regional_code, 
                                   regional_nombre, output_dir=None, 
                                   include_confidence_intervals=True,
                                   model_info=None):
        """
        Exportar predicciones a archivo Excel con formato
        
        Args:
            predictions_dict: Diccionario con predicciones {fecha_str: valor o dict}
            regional_code: Codigo de la regional (ej: 'SAIDI_O')
            regional_nombre: Nombre completo de la regional (ej: 'Ocana')
            output_dir: Directorio donde guardar el archivo (opcional)
            include_confidence_intervals: Incluir intervalos de confianza si existen
            model_info: Informacion del modelo (opcional)
        
        Returns:
            str: Ruta del archivo exportado o None si hay error
        """
        try:
            if not predictions_dict:
                raise ValueError("No hay predicciones para exportar")
            
            # Determinar directorio de salida
            if output_dir is None:
                output_dir = os.path.expanduser("~/Desktop")
            
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            
            # Generar nombre de archivo
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            regional_clean = regional_nombre.replace(" ", "_")
            filename = f"Predicciones_SAIDI_{regional_clean}_{timestamp}.xlsx"
            filepath = os.path.join(output_dir, filename)
            
            # Preparar datos para DataFrame
            data_rows = []
            
            # Determinar si hay intervalos de confianza
            first_entry = next(iter(predictions_dict.values()))
            has_intervals = isinstance(first_entry, dict) and 'limite_inferior' in first_entry
            
            for fecha_str, entry in predictions_dict.items():
                if isinstance(entry, dict):
                    # Formato con intervalos de confianza
                    valor = entry.get('valor_predicho', entry.get('valor', 0))
                    
                    if include_confidence_intervals and has_intervals:
                        inferior = entry.get('limite_inferior', None)
                        superior = entry.get('limite_superior', None)
                        margen = entry.get('margen_error', None)
                        
                        data_rows.append({
                            'Fecha': fecha_str,
                            f'SAIDI {regional_nombre} Predicho': round(valor, 2),
                            'Limite Inferior (95%)': round(inferior, 2) if inferior is not None else None,
                            'Limite Superior (95%)': round(superior, 2) if superior is not None else None,
                            'Margen de Error': round(margen, 2) if margen is not None else None
                        })
                    else:
                        data_rows.append({
                            'Fecha': fecha_str,
                            f'SAIDI {regional_nombre} Predicho': round(valor, 2)
                        })
                else:
                    # Formato simple (solo valor numerico)
                    data_rows.append({
                        'Fecha': fecha_str,
                        f'SAIDI {regional_nombre} Predicho': round(float(entry), 2)
                    })
            
            # Crear DataFrame
            df = pd.DataFrame(data_rows)
            
            # Crear workbook con openpyxl para formato avanzado
            wb = Workbook()
            ws = wb.active
            ws.title = "Predicciones SAIDI"
            
            # Estilos
            header_fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")
            header_font = Font(bold=True, color="FFFFFF", size=12)
            header_alignment = Alignment(horizontal="center", vertical="center")
            
            cell_border = Border(
                left=Side(style='thin'),
                right=Side(style='thin'),
                top=Side(style='thin'),
                bottom=Side(style='thin')
            )
            
            data_alignment = Alignment(horizontal="center", vertical="center")
            
            # Escribir encabezados
            for col_idx, column in enumerate(df.columns, start=1):
                cell = ws.cell(row=1, column=col_idx, value=column)
                cell.fill = header_fill
                cell.font = header_font
                cell.alignment = header_alignment
                cell.border = cell_border
            
            # Escribir datos
            for row_idx, row_data in enumerate(df.values, start=2):
                for col_idx, value in enumerate(row_data, start=1):
                    cell = ws.cell(row=row_idx, column=col_idx, value=value)
                    cell.alignment = data_alignment
                    cell.border = cell_border
                    
                    # Formato numerico para columnas de valores
                    if col_idx > 1 and value is not None:
                        cell.number_format = '0.00'
            
            # Ajustar ancho de columnas
            for column in ws.columns:
                max_length = 0
                column_letter = column[0].column_letter
                
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                
                adjusted_width = min(max_length + 2, 30)
                ws.column_dimensions[column_letter].width = adjusted_width
            
            # Agregar informacion del modelo en una hoja separada (opcional)
            if model_info:
                ws_info = wb.create_sheet("Informacion del Modelo")
                
                info_rows = [
                    ["Informacion del Modelo de Prediccion", ""],
                    ["", ""],
                    ["Regional", regional_nombre],
                    ["Codigo Regional", regional_code],
                    ["Fecha de Exportacion", datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
                    ["", ""],
                    ["Parametros del Modelo", ""],
                ]
                
                if 'order' in model_info:
                    info_rows.append(["Order (p,d,q)", str(model_info['order'])])
                if 'seasonal_order' in model_info:
                    info_rows.append(["Seasonal Order (P,D,Q,s)", str(model_info['seasonal_order'])])
                if 'transformation' in model_info:
                    info_rows.append(["Transformacion", model_info['transformation'].upper()])
                if 'with_exogenous' in model_info:
                    info_rows.append(["Variables Exogenas", "SI" if model_info['with_exogenous'] else "NO"])
                if 'with_simulation' in model_info:
                    info_rows.append(["Simulacion Climatica", "SI" if model_info['with_simulation'] else "NO"])
                if 'confidence_level' in model_info:
                    info_rows.append(["Nivel de Confianza", f"{model_info['confidence_level']*100:.0f}%"])
                
                info_rows.append(["", ""])
                info_rows.append(["Metricas del Modelo", ""])
                
                if 'metrics' in model_info and model_info['metrics']:
                    metrics = model_info['metrics']
                    if 'rmse' in metrics:
                        info_rows.append(["RMSE", f"{metrics['rmse']:.4f}"])
                    if 'mae' in metrics:
                        info_rows.append(["MAE", f"{metrics['mae']:.4f}"])
                    if 'mape' in metrics:
                        info_rows.append(["MAPE", f"{metrics['mape']:.2f}%"])
                    if 'r2_score' in metrics:
                        info_rows.append(["R2 Score", f"{metrics['r2_score']:.4f}"])
                    if 'precision_final' in metrics:
                        info_rows.append(["Precision Final", f"{metrics['precision_final']:.2f}%"])
                
                # Escribir informacion
                for row_idx, (label, value) in enumerate(info_rows, start=1):
                    ws_info.cell(row=row_idx, column=1, value=label)
                    ws_info.cell(row=row_idx, column=2, value=value)
                    
                    if label and not value:  # Titulos de seccion
                        cell = ws_info.cell(row=row_idx, column=1)
                        cell.font = Font(bold=True, size=11)
                
                ws_info.column_dimensions['A'].width = 30
                ws_info.column_dimensions['B'].width = 25
            
            # Guardar archivo
            wb.save(filepath)
            
            self.last_export_path = filepath
            return filepath
            
        except Exception as e:
            print(f"Error exportando predicciones: {str(e)}")
            return None
    
    def get_last_export_path(self):
        """Obtener la ruta del ultimo archivo exportado"""
        return self.last_export_path
    
    def export_to_custom_location(self, predictions_dict, regional_code, 
                                  regional_nombre, custom_path,
                                  include_confidence_intervals=True,
                                  model_info=None):
        """
        Exportar predicciones a una ubicacion especifica
        
        Args:
            predictions_dict: Diccionario con predicciones
            regional_code: Codigo de regional
            regional_nombre: Nombre de regional
            custom_path: Ruta completa del archivo (incluyendo nombre)
            include_confidence_intervals: Incluir intervalos
            model_info: Informacion del modelo
        
        Returns:
            str: Ruta del archivo o None
        """
        try:
            directory = os.path.dirname(custom_path)
            
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
            
            # Temporalmente cambiar output_dir
            temp_dir = directory if directory else os.path.dirname(custom_path)
            
            result_path = self.export_predictions_to_excel(
                predictions_dict=predictions_dict,
                regional_code=regional_code,
                regional_nombre=regional_nombre,
                output_dir=temp_dir,
                include_confidence_intervals=include_confidence_intervals,
                model_info=model_info
            )
            
            if result_path and custom_path != result_path:
                # Renombrar al nombre deseado
                os.rename(result_path, custom_path)
                self.last_export_path = custom_path
                return custom_path
            
            return result_path
            
        except Exception as e:
            print(f"Error exportando a ubicacion personalizada: {str(e)}")
            return None