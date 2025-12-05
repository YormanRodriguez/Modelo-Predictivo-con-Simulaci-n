# services/report_generation_service.py
"""
Servicio de Generación de Informes PDF para Validación Temporal SAIDI.

Genera reportes profesionales con análisis explicativo + gráficas
"""
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


class ReportGenerationError(Exception):
    """Excepción personalizada para errores en la generación de informes PDF."""

class ValidationReportService:
    """Servicio para generar informes PDF de validación temporal."""

    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()

    def _setup_custom_styles(self):
        """Configurar estilos personalizados para el PDF."""
        # Título principal
        self.styles.add(ParagraphStyle(
            name="CustomTitle",
            parent=self.styles["Heading1"],
            fontSize=24,
            textColor=colors.HexColor("#1976D2"),
            spaceAfter=30,
            alignment=TA_CENTER,
            fontName="Helvetica-Bold",
        ))

        # Subtítulo
        self.styles.add(ParagraphStyle(
            name="CustomSubtitle",
            parent=self.styles["Heading2"],
            fontSize=16,
            textColor=colors.HexColor("#2196F3"),
            spaceAfter=12,
            spaceBefore=20,
            fontName="Helvetica-Bold",
        ))

        # Sección
        self.styles.add(ParagraphStyle(
            name="SectionHeader",
            parent=self.styles["Heading2"],
            fontSize=14,
            textColor=colors.HexColor("#FF6B6B"),
            spaceAfter=10,
            spaceBefore=15,
            fontName="Helvetica-Bold",
            borderPadding=5,
            leftIndent=0,
        ))

        # Texto justificado
        self.styles.add(ParagraphStyle(
            name="JustifiedBody",
            parent=self.styles["BodyText"],
            fontSize=10,
            alignment=TA_JUSTIFY,
            spaceAfter=8,
            leading=14,
        ))

        # Texto de métrica
        self.styles.add(ParagraphStyle(
            name="MetricText",
            parent=self.styles["BodyText"],
            fontSize=10,
            textColor=colors.HexColor("#424242"),
            leftIndent=20,
            spaceAfter=5,
        ))

    def generate_validation_report(
    self,
    result: dict[str, Any],
    output_path: str | None = None,
    log_callback=None,
    ) -> str:
        """
        Generar informe PDF completo de validación temporal.

        Args:
            result: Diccionario con resultados de rolling_validation_service
            output_path: Ruta opcional para guardar el PDF
            log_callback: Función para logging

        Returns:
            Ruta del archivo PDF generado

        Raises:
            ReportGenerationError: Si ocurre un error durante la generación del PDF

        """
        try:
            # Preparar ruta de salida
            if not output_path:
                temp_dir = Path(tempfile.gettempdir())
                timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
                output_path = str(temp_dir / f"Informe_Validacion_SAIDI_{timestamp}.pdf")

            # Crear documento
            doc = SimpleDocTemplate(
                output_path,
                pagesize=A4,
                rightMargin=50,
                leftMargin=50,
                topMargin=50,
                bottomMargin=50,
            )

            # Contenedor de elementos
            story = []

            # 1. Portada
            story.extend(self._create_cover_page(result))
            story.append(PageBreak())

            # 2. Resumen Ejecutivo
            story.extend(self._create_executive_summary(result))
            story.append(PageBreak())

            # 3. Sección 1: Rolling Forecast
            story.extend(self._create_rolling_forecast_section(result))
            story.append(PageBreak())

            # 4. Sección 2: Cross-Validation
            story.extend(self._create_cv_section(result))
            story.append(PageBreak())

            # 5. Sección 3: Estabilidad de Parámetros
            story.extend(self._create_parameter_stability_section(result))
            story.append(PageBreak())

            # 6. Sección 4: Backtesting
            story.extend(self._create_backtesting_section(result))
            story.append(PageBreak())

            # 7. Sección 5: Diagnóstico Final
            story.extend(self._create_final_diagnosis_section(result))
            story.append(PageBreak())

            # 8. Sección 6: Gráficas Integradas
            story.extend(self._create_integrated_plots_section(result))

            # Generar PDF
            doc.build(story)

        except (OSError, ValueError, KeyError, AttributeError) as e:
            error_msg = f"Error generando informe PDF: {e!s}"
            if log_callback:
                log_callback(error_msg)
            raise ReportGenerationError(error_msg) from e
        else:
            # Solo se ejecuta si no hubo excepciones
            if log_callback:
                log_callback("Generando informe PDF...")
                log_callback(f"Informe PDF generado: {Path(output_path).name}")

            return output_path

    def _create_cover_page(self, result: dict) -> list:
        """Crear portada del informe."""
        elements = []

        # Espaciado superior
        elements.append(Spacer(1, 1.5*inch))

        # Título principal
        title = Paragraph(
            "INFORME DE VALIDACIÓN TEMPORAL<br/>MODELO SAIDI",
            self.styles["CustomTitle"],
        )
        elements.append(title)
        elements.append(Spacer(1, 0.3*inch))

        # Subtítulo
        model_params = result.get("model_params", {})
        regional_code = model_params.get("regional_code", "N/A")

        subtitle = Paragraph(
            f"<b>Regional:</b> {regional_code}<br/>"
            f"<b>Fecha de generación:</b> {datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")}",
            self.styles["Normal"],
        )
        elements.append(subtitle)
        elements.append(Spacer(1, 0.5*inch))

        # Información del modelo
        order = model_params.get("order", "N/A")
        seasonal_order = model_params.get("seasonal_order", "N/A")
        transformation = model_params.get("transformation", "N/A")

        model_info = [
            ["Configuración del Modelo", "Parámetros"],
            ["Orden ARIMA:", f"{order}"],
            ["Orden Estacional:", f"{seasonal_order}"],
            ["Transformación:", f"{transformation.upper()}"],
            ["Variables Exógenas:", "Sí" if model_params.get("with_exogenous") else "No"],
        ]

        table = Table(model_info, colWidths=[3*inch, 2*inch])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2196F3")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 12),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
            ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
            ("GRID", (0, 0), (-1, -1), 1, colors.black),
            ("FONTSIZE", (0, 1), (-1, -1), 10),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ]))

        elements.append(table)

        return elements

    def _create_executive_summary(self, result: dict) -> list:
        """Crear resumen ejecutivo."""
        elements = []

        elements.append(Paragraph("RESUMEN EJECUTIVO", self.styles["CustomSubtitle"]))

        final_diagnosis = result.get("validation_analysis", {}).get("final_diagnosis", {})

        # Calidad del modelo
        quality = final_diagnosis.get("model_quality", "N/A")
        confidence = final_diagnosis.get("confidence_level", 0)

        quality_color = self._get_quality_color(quality)

        summary_text = f"""
        Este informe presenta los resultados de una <b>validación temporal exhaustiva</b>
        del modelo SAIDI utilizando cuatro metodologías complementarias: Rolling Forecast,
        Cross-Validation, Análisis de Estabilidad de Parámetros y Backtesting Multi-Horizonte.
        <br/><br/>
        <b><font color="{quality_color}">Calidad del Modelo: {quality}</font></b><br/>
        <b>Nivel de Confianza: {confidence:.1f}%</b>
        <br/><br/>
        <b>Recomendación:</b> {final_diagnosis.get('recommendation', 'N/A')}
        """

        elements.append(Paragraph(summary_text, self.styles["JustifiedBody"]))
        elements.append(Spacer(1, 0.2*inch))

        # Tabla de métricas clave
        rolling_results = result.get("validation_analysis", {}).get("rolling_forecast", {})
        cv_results = result.get("validation_analysis", {}).get("cross_validation", {})

        metrics_data = [
            ["Métrica", "Valor"],
            ["Rolling Forecast RMSE", f"{rolling_results.get('rmse', 0):.2f} min"],
            ["Precisión Rolling", f"{rolling_results.get('precision', 0):.1f}%"],
            ["CV Stability Score", f"{cv_results.get('cv_stability_score', 0):.1f}/100"],
        ]

        metrics_table = Table(metrics_data, colWidths=[3*inch, 2*inch])
        metrics_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4CAF50")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 1, colors.black),
            ("BACKGROUND", (0, 1), (-1, -1), colors.lightgrey),
        ]))

        elements.append(metrics_table)

        return elements

    def _create_rolling_forecast_section(self, result: dict) -> list:
        """Sección 1: Rolling Forecast - Walk-Forward Validation."""
        elements = []

        elements.append(Paragraph(
            "1. ROLLING FORECAST - WALK-FORWARD VALIDATION",
            self.styles["SectionHeader"],
        ))

        # Explicación teórica
        theory_text = """
        <b>¿Qué es Rolling Forecast?</b><br/>
        El Rolling Forecast (también llamado Walk-Forward Validation) es una técnica de
        validación temporal que simula el proceso real de predicción. En lugar de dividir
        los datos en un único conjunto de entrenamiento y prueba, este método:
        <br/><br/>
        • <b>Entrena el modelo</b> con todos los datos disponibles hasta el momento t<br/>
        • <b>Predice</b> el siguiente período (t+1)<br/>
        • <b>Compara</b> la predicción con el valor real observado<br/>
        • <b>Avanza</b> un período e incorpora el nuevo dato al entrenamiento<br/>
        • <b>Repite</b> el proceso para cada período de validación
        <br/><br/>
        <b>¿Por qué es importante?</b><br/>
        Esta metodología es especialmente valiosa para series temporales porque:
        <br/>
        1. Respeta el orden temporal de los datos (no hay "data leakage")<br/>
        2. Simula exactamente cómo funcionará el modelo en producción<br/>
        3. Evalúa la capacidad del modelo para adaptarse a nueva información<br/>
        4. Identifica si el modelo mantiene su precisión a lo largo del tiempo
        """

        elements.append(Paragraph(theory_text, self.styles["JustifiedBody"]))
        elements.append(Spacer(1, 0.15*inch))

        # Resultados
        rolling_results = result.get("validation_analysis", {}).get("rolling_forecast", {})

        results_text = f"""
        <b>Resultados del Rolling Forecast:</b><br/><br/>
        En este análisis se realizaron <b>{rolling_results.get('n_predictions', 0)} predicciones</b>
        secuenciales, donde cada predicción se generó con toda la información disponible hasta
        ese momento.
        <br/><br/>
        <b>Métricas de Desempeño:</b><br/>
        • <b>RMSE (Error Cuadrático Medio):</b> {rolling_results.get('rmse', 0):.2f} minutos<br/>
        • <b>MAE (Error Absoluto Medio):</b> {rolling_results.get('mae', 0):.2f} minutos<br/>
        • <b>Precisión Final:</b> {rolling_results.get('precision', 0):.1f}%<br/>
        • <b>Calidad de Predicción:</b> {rolling_results.get('prediction_quality', 'N/A')}
        <br/><br/>
        <b>Interpretación:</b><br/>
        """

        quality = rolling_results.get("prediction_quality", "")
        if quality == "EXCELENTE":
            interpretation = """
            El modelo demostró un <b>desempeño excepcional</b> en las predicciones rolling.
            Con RMSE bajo y precisión superior al 85%, el modelo es altamente confiable
            para pronósticos futuros. Los errores son consistentemente pequeños y el modelo
            se adapta bien a nuevos datos. <b>Apto para uso en producción.</b>
            """
        elif quality == "BUENA":
            interpretation = """
            El modelo mostró un <b>desempeño sólido</b> en las predicciones secuenciales.
            La precisión es buena (78-85%) y los errores están dentro de rangos aceptables
            para series temporales con volatilidad natural. El modelo es <b>confiable para
            uso operativo</b> con monitoreo regular. La variabilidad observada es típica
            de series SAIDI.
            """
        elif quality == "REGULAR":
            interpretation = """
            El modelo presentó un <b>desempeño aceptable</b> para una serie temporal compleja.
            Aunque muestra variabilidad, la precisión (70-78%) es suficiente para pronósticos
            informativos. <b>Útil para análisis de tendencias</b> y planificación con intervalos
            de confianza amplios. Se recomienda monitoreo continuo.
            """
        else:
            interpretation = """
            El modelo mostró <b>limitaciones significativas</b> en su capacidad predictiva.
            Los errores son considerables y la precisión está por debajo de umbrales
            recomendables. Se sugiere reformular el modelo con variables adicionales o
            especificaciones alternativas.
            """

        elements.append(Paragraph(results_text + interpretation, self.styles["JustifiedBody"]))
        elements.append(Spacer(1, 0.2*inch))

        # Insertar gráfica de rolling forecast
        plot_file = result.get("plot_file")
        if plot_file and Path(plot_file).exists():
            try:
                rolling_plot = self._extract_plot_section(plot_file, "rolling_forecast")
                if rolling_plot:
                    img = Image(rolling_plot, width=5*inch, height=3.5*inch)
                    elements.append(img)
                    elements.append(Spacer(1, 0.1*inch))
                    caption = Paragraph(
                        "<i>Figura 1.1: Predicciones vs. valores reales en validación walk-forward</i>",
                        self.styles["Normal"],
                    )
                    elements.append(caption)
            except (OSError, ValueError, KeyError):
                # Silenciosamente ignorar errores al cargar imágenes
                # El informe continuará sin la gráfica
                pass

        return elements

    def _create_cv_section(self, result: dict) -> list:
        """Sección 2: Time Series Cross-Validation."""
        elements = []

        elements.append(Paragraph(
            "2. TIME SERIES CROSS-VALIDATION",
            self.styles["SectionHeader"],
        ))

        theory_text = """
        <b>¿Qué es Time Series Cross-Validation?</b><br/>
        La validación cruzada temporal es una extensión del concepto de cross-validation
        tradicional adaptada específicamente para series temporales. A diferencia del CV
        estándar que usa particiones aleatorias, el Time Series CV:
        <br/><br/>
        • <b>Mantiene el orden temporal</b> de las observaciones<br/>
        • <b>Crea múltiples ventanas</b> de entrenamiento/validación crecientes<br/>
        • <b>Evalúa la estabilidad</b> del modelo con diferentes tamaños de muestra<br/>
        • <b>Detecta sobreajuste</b> comparando desempeño entre ventanas
        <br/><br/>
        <b>Metodología aplicada:</b><br/>
        Se crearon múltiples "splits" temporales donde cada uno tiene:<br/>
        • Un conjunto de entrenamiento creciente (desde el inicio hasta t)<br/>
        • Un conjunto de validación de tamaño fijo (t+1 hasta t+k)<br/>
        • El modelo se reentrena completamente en cada split
        """

        elements.append(Paragraph(theory_text, self.styles["JustifiedBody"]))
        elements.append(Spacer(1, 0.15*inch))

        # Resultados
        cv_results = result.get("validation_analysis", {}).get("cross_validation", {})

        results_text = f"""
        <b>Resultados del Cross-Validation:</b><br/><br/>
        Se completaron <b>{cv_results.get('n_splits', 0)} splits temporales</b> exitosos.
        <br/><br/>
        <b>Métricas de Estabilidad:</b><br/>
        • <b>RMSE Promedio:</b> {cv_results.get('mean_rmse', 0):.4f} ± {cv_results.get('std_rmse', 0):.4f} minutos<br/>
        • <b>Precisión Promedio:</b> {cv_results.get('mean_precision', 0):.1f}%<br/>
        • <b>Rango de Precisión:</b> {cv_results.get('precision_range', [0,0])[0]:.1f}% - {cv_results.get('precision_range', [0,0])[1]:.1f}%<br/>
        • <b>Stability Score:</b> {cv_results.get('cv_stability_score', 0):.1f}/100
        <br/><br/>
        <b>Interpretación de Estabilidad:</b><br/>
        """
        estabilidad_mayor = 80
        estabilidad_media = 68
        estabilidad_menor = 58

        stability = cv_results.get("cv_stability_score", 0)
        if stability >= estabilidad_mayor:
            interpretation = """
            El modelo demostró una <b>estabilidad muy buena</b> a través de diferentes
            ventanas temporales. La variabilidad en las métricas está dentro de límites
            razonables para series con patrones estacionales y tendencias cambiantes.
            La desviación estándar controlada del RMSE confirma predicciones consistentes.
            <b>Modelo confiable para generalización.</b>
            """
        elif stability >= estabilidad_media:
            interpretation = """
            El modelo mostró <b>buena estabilidad</b> en los diferentes splits. La variación
            observada refleja la naturaleza dinámica de la serie temporal, no sobreajuste.
            El modelo es <b>apto para uso operativo</b> considerando la complejidad inherente
            de SAIDI. Monitoreo regular es suficiente.
            """
        elif stability >= estabilidad_menor:
            interpretation = """
            El modelo presentó <b>estabilidad moderada pero funcional</b>. La variación en
            las métricas sugiere sensibilidad a eventos atípicos, lo cual es esperado en
            series de infraestructura. <b>Útil para análisis con intervalos de confianza</b>
            que capturen esta variabilidad natural.
            """
        else:
            interpretation = """
            El modelo mostró <b>baja estabilidad</b> entre diferentes ventanas temporales.
            La alta variabilidad sugiere posible sobreajuste o problemas estructurales.
            Es necesario revisar la especificación del modelo o considerar técnicas de
            regularización.
            """

        elements.append(Paragraph(results_text + interpretation, self.styles["JustifiedBody"]))
        elements.append(Spacer(1, 0.2*inch))

        # Insertar gráfica de CV
        plot_file = result.get("plot_file")
        if plot_file and Path(plot_file).exists():
            try:
                cv_plot = self._extract_plot_section(plot_file, "cv_stability")
                if cv_plot:
                    img = Image(cv_plot, width=5*inch, height=3.5*inch)
                    elements.append(img)
                    elements.append(Spacer(1, 0.1*inch))
                    caption = Paragraph(
                        "<i>Figura 2.1: Distribución de métricas en splits temporales</i>",
                        self.styles["Normal"],
                    )
                    elements.append(caption)
            except (OSError, ValueError, KeyError):
                # Silenciosamente ignorar errores al cargar imágenes
                # El informe continuará sin la gráfica
                pass

        return elements

    def _create_parameter_stability_section(self, result: dict) -> list:
        """Sección 3: Estabilidad de Parámetros."""
        elements = []

        elements.append(Paragraph(
            "3. ANÁLISIS DE ESTABILIDAD DE PARÁMETROS",
            self.styles["SectionHeader"],
        ))

        theory_text = """
        <b>¿Por qué analizar la estabilidad de parámetros?</b><br/>
        Los coeficientes de un modelo SARIMAX deben ser relativamente estables cuando se
        ajusta el modelo con diferentes tamaños de muestra. Si los parámetros varían
        drásticamente:
        <br/><br/>
        • Indica que el modelo está <b>sobreajustado</b> a patrones específicos del conjunto
        de entrenamiento<br/>
        • Sugiere que el modelo es <b>sensible a valores atípicos</b><br/>
        • Implica que las predicciones futuras pueden ser <b>poco confiables</b><br/>
        • Revela posibles <b>problemas de especificación</b> del modelo
        <br/><br/>
        <b>Metodología:</b><br/>
        Se entrenó el modelo con ventanas de datos crecientes (24, 30, 36... hasta todos
        los datos) y se analizó cómo evolucionan los coeficientes AR, MA, SAR y SMA.
        Parámetros estables indican un modelo robusto.
        """

        elements.append(Paragraph(theory_text, self.styles["JustifiedBody"]))
        elements.append(Spacer(1, 0.15*inch))

        # Resultados
        param_stability = result.get("validation_analysis", {}).get("parameter_stability", {})

        results_text = f"""
        <b>Resultados del Análisis:</b><br/><br/>
        <b>Overall Stability Score:</b> {param_stability.get('overall_stability_score', 0):.1f}/100<br/>
        <b>Interpretación:</b> {param_stability.get('interpretation', 'N/A')}
        <br/><br/>
        """

        unstable_params = param_stability.get("unstable_params", [])
        if unstable_params:
            results_text += """
            <b>Parámetros inestables detectados:</b><br/>
            """
            for param in unstable_params:
                results_text += f"• {param}<br/>"
        else:
            results_text += """
            <b>✓ Todos los parámetros mostraron estabilidad adecuada</b><br/>
            """

        results_text += """
        <br/>
        <b>Implicaciones:</b><br/>
        """

        stability_score = param_stability.get("overall_stability_score", 0)
        estabilidad_mayor = 85
        estabilidad_media = 75
        estabilidad_menor = 65

        if stability_score >= estabilidad_mayor:
            interpretation = """
            Los parámetros del modelo son <b>altamente estables</b>. Esto es un fuerte
            indicador de que el modelo ha capturado correctamente la estructura subyacente
            de los datos sin sobreajuste. Los coeficientes convergen a valores consistentes
            independientemente del tamaño de muestra, lo cual es ideal para predicciones
            futuras confiables.
            """
        elif stability_score >= estabilidad_media:
            interpretation = """
            Los parámetros muestran <b>buena estabilidad</b>. Hay variación moderada pero
            dentro de límites aceptables. El modelo es confiable aunque puede haber
            sensibilidad menor a datos extremos.
            """
        elif stability_score >= estabilidad_menor:
            interpretation = """
            Los parámetros presentan <b>estabilidad moderada</b>. Algunos coeficientes
            varían significativamente con el tamaño de muestra, sugiriendo posible
            sobreajuste parcial. Se recomienda considerar simplificar el modelo.
            """
        else:
            interpretation = """
            Los parámetros muestran <b>baja estabilidad</b>. La variación significativa
            es una señal de alerta de sobreajuste o mala especificación. El modelo
            probablemente no generalizará bien a datos futuros.
            """

        elements.append(Paragraph(results_text + interpretation, self.styles["JustifiedBody"]))
        elements.append(Spacer(1, 0.2*inch))

        # Insertar gráfica de estabilidad de parámetros
        plot_file = result.get("plot_file")
        if plot_file and Path(plot_file).exists():
            try:
                param_plot = self._extract_plot_section(plot_file, "parameter_stability")
                if param_plot:
                    img = Image(param_plot, width=5*inch, height=3.5*inch)
                    elements.append(img)
                    elements.append(Spacer(1, 0.1*inch))
                    caption = Paragraph(
                        "<i>Figura 3.1: Evolución de coeficientes SARIMAX según tamaño de muestra</i>",
                        self.styles["Normal"],
                    )
                    elements.append(caption)
            except (OSError, ValueError, KeyError):
                # Silenciosamente ignorar errores al cargar imágenes
                # El informe continuará sin la gráfica
                pass

        return elements

    def _create_backtesting_section(self, result: dict) -> list:
        """Sección 4: Backtesting Multi-Horizonte."""
        elements = []

        elements.append(Paragraph(
            "4. BACKTESTING MULTI-HORIZONTE",
            self.styles["SectionHeader"],
        ))

        theory_text = """
        <b>¿Qué es el Backtesting Multi-Horizonte?</b><br/>
        El backtesting evalúa cómo se degrada el desempeño del modelo al hacer predicciones
        cada vez más lejanas en el futuro. Es crítico porque:
        <br/><br/>
        • Las predicciones a <b>1 mes</b> son típicamente más precisas que a 12 meses<br/>
        • Permite identificar el <b>"horizonte óptimo"</b> del modelo<br/>
        • Revela la <b>tasa de degradación</b> de la precisión con el tiempo<br/>
        • Ayuda a establecer <b>límites de confianza</b> según el horizonte
        <br/><br/>
        <b>Horizontes evaluados:</b><br/>
        Se probaron predicciones a 1, 3, 6 y 12 meses desde múltiples puntos de inicio
        en el histórico, calculando el desempeño promedio para cada horizonte.
        """

        elements.append(Paragraph(theory_text, self.styles["JustifiedBody"]))
        elements.append(Spacer(1, 0.15*inch))

        # Resultados
        backtesting_results = result.get("validation_analysis", {}).get("backtesting", {})
        metrics_by_horizon = backtesting_results.get("metrics_by_horizon", {})

        results_text = f"""
        <b>Resultados del Backtesting:</b><br/><br/>
        <b>Horizonte Óptimo Identificado:</b> {backtesting_results.get('optimal_horizon', 0)} meses<br/>
        <b>Tasa de Degradación:</b> {backtesting_results.get('degradation_rate', 0):.2f}% por mes
        <br/><br/>
        <b>Desempeño por Horizonte:</b><br/>
        """

        for horizon in sorted(metrics_by_horizon.keys()):
            metrics = metrics_by_horizon[horizon]
            results_text += f"""
            • <b>H={horizon} meses:</b> Precisión {metrics['precision']:.1f}%,
            RMSE {metrics['rmse']:.3f} min ({metrics['n_tests']} pruebas)<br/>
            """

        results_text += """
        <br/>
        <b>Interpretación de Degradación:</b><br/>
        """

        degradation = backtesting_results.get("degradation_rate", 0)
        optimal = backtesting_results.get("optimal_horizon", 0)

        degradacion_menor = 2.0
        degradacion_mayor = 3.5

        if abs(degradation) < degradacion_menor:
            interpretation = f"""
            El modelo muestra una <b>degradación mínima</b> en horizontes de hasta
            {optimal} meses. La pérdida de precisión es gradual y predecible, lo cual
            es excelente. El modelo es confiable para pronósticos de mediano plazo.
            """
        elif abs(degradation) < degradacion_mayor:
            interpretation = f"""
            El modelo presenta una <b>degradación moderada</b> pero manejable. La precisión
            se mantiene aceptable hasta {optimal} meses. Para horizontes más largos,
            considere intervalos de confianza más amplios.
            """
        else:
            interpretation = f"""
            El modelo muestra <b>degradación significativa</b> después de {optimal} meses.
            Se recomienda limitar las predicciones a horizontes cortos o revisar la
            especificación del modelo para mejorar pronósticos a largo plazo.
            """

        elements.append(Paragraph(results_text + interpretation, self.styles["JustifiedBody"]))
        elements.append(Spacer(1, 0.2*inch))

        # Insertar gráfica de backtesting
        plot_file = result.get("plot_file")
        if plot_file and Path(plot_file).exists():
            try:
                backtesting_plot = self._extract_plot_section(plot_file, "backtesting")
                if backtesting_plot:
                    img = Image(backtesting_plot, width=5*inch, height=3.5*inch)
                    elements.append(img)
                    elements.append(Spacer(1, 0.1*inch))
                    caption = Paragraph(
                        "<i>Figura 4.1: Degradación de precisión según horizonte de predicción</i>",
                        self.styles["Normal"],
                    )
                    elements.append(caption)
            except (OSError, ValueError, KeyError):
                # Silenciosamente ignorar errores al cargar imágenes
                # El informe continuará sin la gráfica
                pass

        return elements

    def _extract_plot_section(self, plot_file: str, section: str) -> str | None:
        """
        Extraer una sección específica de la imagen completa.

        Args:
            plot_file: Ruta de la imagen completa
            section: Nombre de la sección ('rolling_forecast', 'cv_stability',
                    'parameter_stability', 'backtesting', 'diagnosis', 'scores')

        Returns:
            Ruta del archivo temporal con la sección extraída

        """
        try:
            # Abrir imagen completa
            img = PILImage.open(plot_file)
            width, height = img.size

            # Definir coordenadas de recorte según sección (basado en layout 2x3)
            col_width = width // 3
            row_height = height // 2

            crop_coords = {
                "rolling_forecast": (0, 0, col_width, row_height),  # Superior izquierda
                "cv_stability": (col_width, 0, 2*col_width, row_height),  # Superior centro
                "parameter_stability": (2*col_width, 0, width, row_height),  # Superior derecha
                "backtesting": (0, row_height, col_width, height),  # Inferior izquierda
                "diagnosis": (col_width, row_height, 2*col_width, height),  # Inferior centro
                "scores": (2*col_width, row_height, width, height),  # Inferior derecha
            }

            if section not in crop_coords:
                return None

            # Recortar sección
            cropped = img.crop(crop_coords[section])

            # Guardar en archivo temporal
            temp_dir = Path(tempfile.gettempdir())
            temp_file = temp_dir / f'section_{section}_{datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")}.png'
            cropped.save(temp_file, "PNG")

        except (OSError, ValueError, KeyError) as e:
            print(f"Error extrayendo sección {section}: {e}")
            return None
        else:
            return str(temp_file)

    def _create_final_diagnosis_section(self, result: dict) -> list:
        """Sección 5: Diagnóstico Final Integrado."""
        elements = []

        elements.append(Paragraph(
            "5. DIAGNÓSTICO FINAL INTEGRADO",
            self.styles["SectionHeader"],
        ))

        intro_text = """
        <b>Síntesis de Todas las Validaciones</b><br/>
        Este diagnóstico integra los resultados de las cuatro metodologías anteriores para
        proporcionar una evaluación holística del modelo. Cada componente aporta una
        perspectiva única:
        <br/><br/>
        • <b>Rolling Forecast:</b> Capacidad predictiva en escenarios reales<br/>
        • <b>Cross-Validation:</b> Estabilidad y generalización<br/>
        • <b>Estabilidad de Parámetros:</b> Robustez estructural<br/>
        • <b>Backtesting:</b> Desempeño en diferentes horizontes temporales
        """

        elements.append(Paragraph(intro_text, self.styles["JustifiedBody"]))
        elements.append(Spacer(1, 0.15*inch))

        # Diagnóstico final
        final_diagnosis = result.get("validation_analysis", {}).get("final_diagnosis", {})
        component_scores = final_diagnosis.get("component_scores", {})

        # Tabla de scores
        scores_data = [["Componente", "Score", "Estado"]]

        puntaje_mayor = 80
        puntaje_baja = 65

        for component, score in component_scores.items():
            component_name = self._get_component_display_name(component)
            status = "Excelente" if score >= puntaje_mayor else "○ Aceptable" if score >= puntaje_baja else "✗ Mejorable"
            scores_data.append([component_name, f"{score:.1f}/100", status])

        scores_table = Table(scores_data, colWidths=[2.5*inch, 1.5*inch, 1.5*inch])
        scores_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#FF6B6B")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 1, colors.black),
            ("BACKGROUND", (0, 1), (-1, -1), colors.lightgrey),
        ]))

        elements.append(scores_table)
        elements.append(Spacer(1, 0.15*inch))

        # Diagnóstico textual
        quality = final_diagnosis.get("model_quality", "N/A")
        confidence = final_diagnosis.get("confidence_level", 0)
        recommendation = final_diagnosis.get("recommendation", "N/A")
        limitations = final_diagnosis.get("limitations", [])

        diagnosis_text = f"""
        <b>Calidad del Modelo:</b> <font color="{self._get_quality_color(quality)}">{quality}</font><br/>
        <b>Nivel de Confianza Global:</b> {confidence:.1f}%<br/><br/>
        <b>Recomendación Principal:</b><br/>
        {recommendation}
        <br/><br/>
        """

        if limitations:
            diagnosis_text += "<b>Limitaciones Identificadas:</b><br/>"
            for i, limitation in enumerate(limitations, 1):
                diagnosis_text += f"{i}. {limitation}<br/>"

        elements.append(Paragraph(diagnosis_text, self.styles["JustifiedBody"]))
        elements.append(Spacer(1, 0.15*inch))

        # Conclusión final
        conclusion = self._generate_final_conclusion(quality, confidence, limitations)
        elements.append(Paragraph(f"<b>Conclusión:</b><br/>{conclusion}", self.styles["JustifiedBody"]))

        return elements

    def _create_integrated_plots_section(self, result: dict) -> list:
        """Sección 6: Gráficas Integradas del Análisis."""
        elements = []

        elements.append(Paragraph(
            "6. VISUALIZACIÓN INTEGRADA DE RESULTADOS",
            self.styles["SectionHeader"],
        ))

        intro_text = """
        <b>Panel de Gráficas Comprehensivo</b><br/>
        A continuación se presenta el panel visual completo generado por el análisis de
        validación temporal. Esta visualización integrada combina los seis componentes
        principales del análisis en un solo dashboard interconectado.
        <br/><br/>
        <b>Componentes del Panel:</b><br/>
        1. <b>Rolling Forecast Performance:</b> Predicciones vs. valores reales en validación secuencial<br/>
        2. <b>Cross-Validation Stability:</b> Distribución de métricas entre splits temporales<br/>
        3. <b>Parameter Evolution:</b> Evolución de coeficientes SARIMAX con diferentes tamaños de muestra<br/>
        4. <b>Backtesting Degradation:</b> Precisión y RMSE según horizonte de predicción<br/>
        5. <b>Diagnóstico Completo:</b> Resumen textual con todas las métricas clave<br/>
        6. <b>Component Scores:</b> Radar/barras de scores individuales por componente
        """

        elements.append(Paragraph(intro_text, self.styles["JustifiedBody"]))
        elements.append(Spacer(1, 0.2*inch))

        # Insertar la gráfica completa
        plot_file = result.get("plot_file")

        if plot_file and Path(plot_file).exists():
            try:
                # Calcular dimensiones para ajustar a página A4
                img_width = 7*inch
                img_height = 5*inch

                img = Image(plot_file, width=img_width, height=img_height)
                elements.append(img)
                elements.append(Spacer(1, 0.1*inch))

                caption = Paragraph(
                    "<i>Figura 6.1: Panel de validación temporal completa con los seis componentes del análisis integrado.</i>",
                    self.styles["Normal"],
                )
                elements.append(caption)

            except (OSError, ValueError, KeyError):
                error_text = "<i>Error al cargar la gráfica integrada</i>"
                elements.append(Paragraph(error_text, self.styles["Normal"]))
        else:
            no_plot_text = "<i>Gráfica no disponible en el resultado del análisis.</i>"
            elements.append(Paragraph(no_plot_text, self.styles["Normal"]))

        # Nota interpretativa
        elements.append(Spacer(1, 0.2*inch))
        note_text = """
        <b>Nota sobre Interpretación Visual:</b><br/>
        Al analizar el panel gráfico, preste especial atención a:<br/>
        • <b>Convergencia visual:</b> Las predicciones rolling deben seguir de cerca los valores reales<br/>
        • <b>Variabilidad de boxes:</b> Cajas estrechas en CV indican alta estabilidad<br/>
        • <b>Parámetros constantes:</b> Líneas horizontales sugieren coeficientes estables<br/>
        • <b>Degradación gradual:</b> La caída de precisión debe ser suave, no abrupta<br/>
        • <b>Scores balanceados:</b> Todos los componentes deben estar en rangos similares
        """

        elements.append(Paragraph(note_text, self.styles["JustifiedBody"]))

        return elements

    def _get_component_display_name(self, component: str) -> str:
        """Obtener nombre amigable de componente."""
        names = {
            "rolling_forecast": "Rolling Forecast",
            "precision": "Precisión Global",
            "cv_stability": "CV Stability",
            "parameter_stability": "Estabilidad de Parámetros",
            "degradation": "Control de Degradación",
        }
        return names.get(component, component)

    def _get_quality_color(self, quality: str) -> str:
        """Obtener color según calidad."""
        colors_map = {
            "EXCELENTE": "#4CAF50",
            "CONFIABLE": "#2196F3",
            "CUESTIONABLE": "#FF9800",
            "NO CONFIABLE": "#F44336",
        }
        return colors_map.get(quality, "#757575")

    def _generate_final_conclusion(self, quality: str, confidence: float, limitations: list) -> str:
        """Generar conclusión final personalizada."""
        calidad_alta = 80
        calidad_media = 70

        if quality == "EXCELENTE" and confidence >= calidad_alta:
            conclusion = """
            El modelo ha superado las pruebas de validación temporal con resultados
            sobresalientes. La combinación de buena precisión en rolling forecast,
            estabilidad en cross-validation, parámetros robustos y degradación controlada
            confirma que este modelo es <b>confiable para uso en producción</b>.
            Se recomienda su implementación con monitoreo periódico estándar.
            """
        elif quality in ["EXCELENTE", "CONFIABLE"] and confidence >= calidad_media:
            conclusion = """
            El modelo demostró un desempeño sólido considerando la complejidad de la serie
            temporal SAIDI. Aunque existen variaciones naturales identificadas, el modelo
            es <b>confiable para uso operativo</b> en el contexto de planificación y
            gestión de infraestructura. Se recomienda implementación con intervalos de
            confianza y revisión trimestral de métricas.
            """
        elif quality == "CUESTIONABLE" or confidence < calidad_media:
            conclusion = """
            El modelo presenta resultados que requieren consideración del contexto.
            Algunas métricas son funcionales para análisis de tendencias mientras que
            otras sugieren limitaciones para predicciones precisas. Se recomienda
            <b>uso informativo con intervalos amplios</b> y considerar mejoras si se
            requiere mayor precisión. Útil para escenarios "what-if" y planificación
            de alto nivel.
            """
        else:
            conclusion = """
            Los resultados de validación indican que el modelo actual <b>no es adecuado
            para uso en producción</b> sin modificaciones sustanciales. Se recomienda
            revisar la especificación del modelo, considerar variables adicionales,
            o explorar metodologías alternativas antes de su implementación.
            """

        if limitations:
            conclusion += f"""
            <br/><br/>
            Es crítico abordar las {len(limitations)} limitaciones identificadas antes
            de tomar decisiones basadas en las predicciones del modelo.
            """

        return conclusion
