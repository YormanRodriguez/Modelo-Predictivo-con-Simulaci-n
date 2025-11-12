# view/climate_simulation_dialog.py - Diálogo de simulación climática ACTUALIZADO
from dataclasses import dataclass

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QButtonGroup,
    QDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from services.climate_simulation_service import ClimateSimulationService


@dataclass
class SimulationConfig:
    """Configuración para el diálogo de simulación."""

    climate_data: any
    mes_prediccion: int
    regional_code: str
    regional_nombre: str
    mode: str = "prediction"


class ClimateSimulationDialog(QDialog):
    """
    Diálogo para configurar simulación climática adaptativa.
    
    ACTUALIZADO para usar:
    - Escenarios genéricos universales
    - Filtrado automático por regional
    - Slider de intensidad (0.5x - 2.0x)
    - Nueva estructura de configuración
    """

    # Señales
    simulation_accepted = pyqtSignal(dict)
    simulation_cancelled = pyqtSignal()

    def __init__(self, config: SimulationConfig, parent=None):
        super().__init__(parent)

        self.climate_data = config.climate_data
        self.mes_prediccion = config.mes_prediccion
        self.regional_code = config.regional_code
        self.regional_nombre = config.regional_nombre
        self.mode = config.mode

        self.simulation_service = ClimateSimulationService()

        # Datos calculados
        self.percentiles = None

        # Estado actual
        self.escenario_seleccionado = None
        self.intensity_adjustment = 1.0  # ✅ NUEVO: Factor de intensidad

        self.setup_ui()
        self.calculate_simulation_data()

    def setup_ui(self):
        """Configurar interfaz de usuario."""
        self.setWindowTitle(f"Simulador Climático - {self.regional_nombre}")
        self.setModal(True)
        self.setMinimumSize(500, 550)
        self.resize(550, 600)

        # Layout principal del diálogo
        dialog_layout = QVBoxLayout(self)
        dialog_layout.setContentsMargins(0, 0, 0, 0)
        dialog_layout.setSpacing(0)

        # Crear área scrolleable con contenido
        scroll = self._create_scroll_area()
        dialog_layout.addWidget(scroll)

        # Separador
        separator = self._create_separator()
        dialog_layout.addWidget(separator)

        # Botones de acción
        buttons_layout = self._create_action_buttons()
        dialog_layout.addLayout(buttons_layout)

        self.apply_dialog_styles()

    def _create_scroll_area(self):
        """Crear área scrolleable con todo el contenido."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        # Widget contenedor
        scroll_content = QWidget()
        main_layout = QVBoxLayout(scroll_content)
        main_layout.setSpacing(12)
        main_layout.setContentsMargins(16, 16, 16, 16)

        # Agregar todos los componentes
        header = self._create_header()
        main_layout.addWidget(header)

        desc = self._create_description()
        main_layout.addWidget(desc)

        self.create_scenario_group(main_layout)

        # ❌ ELIMINADO: historical_group (ya no se usan "días base")
        
        adjustment_group = self._create_adjustment_group()
        main_layout.addWidget(adjustment_group)

        alcance_group = self._create_alcance_group()
        main_layout.addWidget(alcance_group)

        scroll.setWidget(scroll_content)
        return scroll

    def _create_header(self):
        """Crear encabezado del diálogo."""
        mode_text = "VALIDACIÓN" if self.mode == "validation" else "PREDICCIÓN"
        mode_icon = "📊" if self.mode == "validation" else "🔮"

        header = QLabel(f"{mode_icon} Simulador Climático\n{mode_text} - {self.regional_nombre}")
        header.setFont(QFont("Segoe UI", 13, QFont.Weight.Bold))
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Color según modo
        header_color = "#1976D2" if self.mode == "prediction" else "#FF6F00"
        gradient_start = "#E3F2FD" if self.mode == "prediction" else "#FFF3E0"
        gradient_end = "#BBDEFB" if self.mode == "prediction" else "#FFE0B2"

        header.setStyleSheet(f"""
            QLabel {{
                color: {header_color};
                padding: 10px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {gradient_start}, stop:1 {gradient_end});
                border-radius: 6px;
            }}
        """)
        return header

    def _create_description(self):
        """Crear descripción según el modo."""
        if self.mode == "validation":
            desc_text = (
                "Configure un escenario climático para evaluar la SENSIBILIDAD "
                "del modelo bajo diferentes condiciones meteorológicas."
            )
        else:
            desc_text = (
                "Configure un escenario climático para ajustar la predicción SAIDI "
                "según condiciones meteorológicas esperadas."
            )

        desc = QLabel(desc_text)
        desc.setWordWrap(True)
        desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        desc.setStyleSheet("color: #666; font-size: 10px; padding: 6px;")
        return desc

    def _create_adjustment_group(self):
        """Crear grupo de ajuste de intensidad - ACTUALIZADO."""
        self.adjustment_group = QGroupBox("⚙️ Ajustar Intensidad del Escenario")
        self.adjustment_group.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        self.adjustment_group.setEnabled(False)

        adjustment_layout = QVBoxLayout(self.adjustment_group)
        adjustment_layout.setContentsMargins(8, 12, 8, 8)
        adjustment_layout.setSpacing(6)

        # Label de ajuste
        self.adjustment_label = QLabel("Seleccione un escenario primero")
        self.adjustment_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.adjustment_label.setStyleSheet("color: #666; font-size: 10px; padding: 4px;")
        adjustment_layout.addWidget(self.adjustment_label)

        # ✅ NUEVO: Slider de intensidad (0.5x - 2.0x)
        self._add_intensity_slider(adjustment_layout)

        # Vista previa
        self.preview_label = QLabel("")
        self.preview_label.setWordWrap(True)
        self.preview_label.setStyleSheet("""
            QLabel {
                background-color: #E3F2FD;
                padding: 8px;
                border-radius: 4px;
                border-left: 3px solid #2196F3;
                font-size: 10px;
                color: #1565C0;
            }
        """)
        adjustment_layout.addWidget(self.preview_label)

        return self.adjustment_group

    def _add_intensity_slider(self, layout):
        """✅ NUEVO: Agregar slider de intensidad (0.5x - 2.0x)."""
        # Label de rango
        range_label = QLabel("Rango: 0.5x (débil) ← 1.0x (normal) → 2.0x (extremo)")
        range_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        range_label.setStyleSheet("font-size: 9px; color: #888;")
        layout.addWidget(range_label)

        # Slider horizontal
        self.intensity_slider = QSlider(Qt.Orientation.Horizontal)
        self.intensity_slider.setMinimum(50)   # 0.5x
        self.intensity_slider.setMaximum(200)  # 2.0x
        self.intensity_slider.setValue(100)    # 1.0x (normal)
        self.intensity_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.intensity_slider.setTickInterval(10)
        self.intensity_slider.valueChanged.connect(self.on_intensity_changed)
        
        self.intensity_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #bbb;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #2196F3, stop:0.5 #FFC107, stop:1 #F44336);
                height: 8px;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: white;
                border: 3px solid #1976D2;
                width: 20px;
                margin: -6px 0;
                border-radius: 10px;
            }
            QSlider::handle:horizontal:hover {
                border-color: #0D47A1;
                width: 22px;
                margin: -7px 0;
            }
        """)
        layout.addWidget(self.intensity_slider)

        # Label de valor actual
        self.intensity_value_label = QLabel("1.0x (Normal)")
        self.intensity_value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.intensity_value_label.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        self.intensity_value_label.setStyleSheet("color: #FFC107; padding: 3px;")
        layout.addWidget(self.intensity_value_label)

    def _create_alcance_group(self):
        """Crear grupo de alcance temporal."""
        alcance_group = QGroupBox("📅 Alcance Temporal")
        alcance_group.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))

        alcance_layout = QVBoxLayout(alcance_group)
        alcance_layout.setContentsMargins(8, 12, 8, 8)
        alcance_layout.setSpacing(6)

        # Descripción
        alcance_desc = QLabel("¿A cuántos meses aplicar la simulación?")
        alcance_desc.setStyleSheet("color: #666; font-size: 9px; padding: 2px;")
        alcance_layout.addWidget(alcance_desc)

        # Botones de radio
        self.alcance_group = QButtonGroup(self)
        alcance_buttons_layout = QHBoxLayout()
        alcance_buttons_layout.setSpacing(8)

        self.alcance_1 = QRadioButton("1 mes")
        self.alcance_3 = QRadioButton("3 meses")
        self.alcance_6 = QRadioButton("6 meses")

        self.alcance_3.setChecked(True)

        self.alcance_group.addButton(self.alcance_1, 1)
        self.alcance_group.addButton(self.alcance_3, 3)
        self.alcance_group.addButton(self.alcance_6, 6)

        radio_style = """
            QRadioButton {
                font-size: 10px;
                spacing: 6px;
            }
            QRadioButton::indicator {
                width: 14px;
                height: 14px;
            }
        """

        for btn in [self.alcance_1, self.alcance_3, self.alcance_6]:
            btn.setStyleSheet(radio_style)
            alcance_buttons_layout.addWidget(btn)

        alcance_layout.addLayout(alcance_buttons_layout)
        return alcance_group

    def _create_separator(self):
        """Crear separador horizontal."""
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setStyleSheet("background-color: #ddd;")
        return separator

    def _create_action_buttons(self):
        """Crear botones de acción del diálogo."""
        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(10)
        buttons_layout.setContentsMargins(16, 12, 16, 12)

        # Botón "Sin Simulación"
        self.no_simulation_button = QPushButton("❌ Sin Simulación")
        self.no_simulation_button.setMinimumHeight(38)
        self.no_simulation_button.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        self.no_simulation_button.clicked.connect(self.on_no_simulation)
        self.no_simulation_button.setStyleSheet("""
            QPushButton {
                background-color: #757575;
                color: white;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #616161;
            }
        """)
        buttons_layout.addWidget(self.no_simulation_button)

        # Botón de simulación
        button_text = "✅ Validar con Simulación" if self.mode == "validation" else "🌦️ Aplicar Simulación"
        self.simulate_button = QPushButton(button_text)
        self.simulate_button.setMinimumHeight(38)
        self.simulate_button.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        self.simulate_button.setEnabled(False)
        self.simulate_button.clicked.connect(self.on_simulate)

        # Color según modo
        button_color = "#FF6F00" if self.mode == "validation" else "#4CAF50"
        button_hover = "#F57C00" if self.mode == "validation" else "#45a049"

        self.simulate_button.setStyleSheet(f"""
            QPushButton {{
                background-color: {button_color};
                color: white;
                border-radius: 5px;
                padding: 0 20px;
            }}
            QPushButton:hover {{
                background-color: {button_hover};
            }}
            QPushButton:disabled {{
                background-color: #e0e0e0;
                color: #9e9e9e;
            }}
        """)
        buttons_layout.addWidget(self.simulate_button)

        return buttons_layout

    def create_scenario_group(self, parent_layout):
        """✅ ACTUALIZADO: Crear grupo con escenarios FILTRADOS por regional."""
        scenario_group = QGroupBox("🌤️ Escenario Climático")
        scenario_group.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        scenario_layout = QVBoxLayout(scenario_group)
        scenario_layout.setContentsMargins(8, 12, 8, 8)
        scenario_layout.setSpacing(6)

        self.scenario_buttons = {}
        self.scenario_group = QButtonGroup(self)

        # ✅ NUEVO: Obtener escenarios filtrados por regional
        available_scenarios = self.simulation_service.get_available_scenarios(self.regional_code)

        # Mapeo de colores por escenario
        colors = {
            "calor_extremo": "#F44336",
            "lluvias_intensas": "#2196F3",
            "condiciones_normales": "#4CAF50",
            "sequia": "#FF9800",
            "vientos_fuertes": "#9C27B0",
            "tiempo_humedo": "#00BCD4",
        }

        for idx, scenario_info in enumerate(available_scenarios):
            scenario_id = scenario_info['id']
            scenario_name = scenario_info['name']
            scenario_icon = scenario_info['icon']
            scenario_desc = self.simulation_service.composite_scenarios[scenario_id]['description']

            btn = QPushButton(f"{scenario_icon} {scenario_name}")
            btn.setMinimumHeight(40)
            btn.setCheckable(True)
            btn.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
            btn.clicked.connect(lambda _checked, k=scenario_id: self.on_scenario_selected(k))

            color = colors.get(scenario_id, "#757575")

            btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: white;
                    border: 2px solid {color};
                    border-radius: 6px;
                    text-align: left;
                    padding-left: 10px;
                    color: #333;
                }}
                QPushButton:hover {{
                    background-color: #f5f5f5;
                    border-width: 2px;
                }}
                QPushButton:checked {{
                    background-color: {color};
                    color: white;
                    border-color: {color};
                }}
            """)

            scenario_layout.addWidget(btn)
            self.scenario_buttons[scenario_id] = btn
            self.scenario_group.addButton(btn, idx)

            # Descripción compacta
            desc = QLabel(f"   {scenario_desc}")
            desc.setStyleSheet("color: #888; font-size: 9px; font-style: italic; padding-left: 16px;")
            scenario_layout.addWidget(desc)

        parent_layout.addWidget(scenario_group)

    def calculate_simulation_data(self):
        """✅ ACTUALIZADO: Calcular solo percentiles (sin días base)."""
        try:
            self.percentiles = self.simulation_service.calculate_percentiles(
                self.climate_data, self.regional_code,
            )

        except Exception as e:
            error_msg = f"Error calculando percentiles: {str(e)}"
            QMessageBox.warning(self, "Error", error_msg)
            print(f"Error en calculate_simulation_data: {e}")

    def on_scenario_selected(self, escenario_key):
        """✅ ACTUALIZADO: Callback cuando se selecciona un escenario."""
        try:
            self.escenario_seleccionado = escenario_key

            self.adjustment_group.setEnabled(True)
            self.simulate_button.setEnabled(True)

            # Resetear slider a valor normal
            self.intensity_slider.setValue(100)  # 1.0x

            scenario_info = self.simulation_service.composite_scenarios[escenario_key]
            self.adjustment_label.setText(f"Ajustar intensidad: {scenario_info['name']}")

            self.update_preview()

        except KeyError as e:
            QMessageBox.warning(self, "Error", f"Error: Escenario no encontrado - {str(e)}")

    def on_intensity_changed(self, value):
        """Callback cuando cambia el slider de intensidad."""
        self.intensity_adjustment = value / 100.0  # Convertir 50-200 a 0.5-2.0

        # Actualizar label con colores según intensidad
        if self.intensity_adjustment < 0.8:
            color = "#2196F3"  # Azul (débil)
            text = f"{self.intensity_adjustment:.1f}x (Débil)"
        elif self.intensity_adjustment < 1.2:
            color = "#FFC107"  # Amarillo (normal)
            text = f"{self.intensity_adjustment:.1f}x (Normal)"
        else:
            color = "#F44336"  # Rojo (extremo)
            text = f"{self.intensity_adjustment:.1f}x (Extremo)"

        self.intensity_value_label.setText(text)
        self.intensity_value_label.setStyleSheet(f"color: {color}; padding: 3px; font-weight: bold;")

        self.update_preview()

    def update_preview(self):
        """✅ ACTUALIZADO: Vista previa con nueva estructura."""
        try:
            if not self.escenario_seleccionado:
                return

            alcance = self.alcance_group.checkedId()

            # Obtener resumen con NUEVA FIRMA
            summary = self.simulation_service.get_simulation_summary(
                scenario_name=self.escenario_seleccionado,  # ✅ NUEVO
                intensity_adjustment=self.intensity_adjustment,  # ✅ NUEVO
                alcance_meses=alcance,
                percentiles=self.percentiles,
                regional_code=self.regional_code,
            )

            scenario_info = self.simulation_service.composite_scenarios[self.escenario_seleccionado]

            # Construir preview
            preview = f"<b>{scenario_info['icon']} {scenario_info['name']}</b> | "
            preview += f"Intensidad: {self.intensity_adjustment:.1f}x | {alcance} meses<br>"

            # Mostrar variables afectadas
            affected_vars = summary.get('variables_afectadas', {})
            
            if affected_vars:
                changes = []
                for var_code, var_info in list(affected_vars.items())[:5]:  # Mostrar máximo 5
                    var_name = var_info['nombre'][:20]  # Truncar nombre  # noqa: F841
                    change_pct = var_info['cambio_porcentual']
                    
                    if abs(change_pct) < 1:
                        arrow = "→"
                        color = ""
                    elif change_pct > 0:
                        arrow = "↑"
                        color = " style='color: #F44336;'"
                    else:
                        arrow = "↓"
                        color = " style='color: #2196F3;'"
                    
                    changes.append(f"<b{color}>{arrow}{change_pct:+.1f}%</b>")
                
                preview += f"{len(affected_vars)} variables: " + " | ".join(changes)
            else:
                preview += "<i>Sin variables afectadas</i>"

            # Advertencia según modo
            if self.mode == "validation":
                preview += "<br><b style='color: #F57C00;'>⚠ Validación: Evalúa sensibilidad del modelo</b>"

            self.preview_label.setText(preview)

        except Exception as e:
            self.preview_label.setText(f"Error: {str(e)}")

    def on_simulate(self):
        """✅ ACTUALIZADO: Emitir configuración en NUEVO formato."""
        try:
            if not self.escenario_seleccionado:
                QMessageBox.warning(self, "Advertencia", "Seleccione un escenario")
                return

            alcance = self.alcance_group.checkedId()

            # Validar parámetros con NUEVA FIRMA
            is_valid, error_msg = self.simulation_service.validate_simulation_params(
                scenario_name=self.escenario_seleccionado,  # ✅ NUEVO
                intensity_adjustment=self.intensity_adjustment,  # ✅ NUEVO
                alcance_meses=alcance,
                regional_code=self.regional_code,  # ✅ NUEVO
            )

            if not is_valid:
                QMessageBox.critical(self, "Error", error_msg)
                return

            # ✅ NUEVA ESTRUCTURA DE CONFIGURACIÓN
            config = {
                "enabled": True,
                "scenario_name": self.escenario_seleccionado,  # ✅ NUEVO
                "intensity_adjustment": self.intensity_adjustment,  # ✅ NUEVO
                "alcance_meses": alcance,
                "percentiles": self.percentiles,
                "regional_code": self.regional_code,
                "summary": self.simulation_service.get_simulation_summary(
                    scenario_name=self.escenario_seleccionado,  # ✅ NUEVO
                    intensity_adjustment=self.intensity_adjustment,  # ✅ NUEVO
                    alcance_meses=alcance,
                    percentiles=self.percentiles,
                    regional_code=self.regional_code,
                ),
            }

            self.simulation_accepted.emit(config)
            self.accept()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error en la configuración: {str(e)}")

    def on_no_simulation(self):
        """Callback cuando se cancela la simulación."""
        config = {"enabled": False}
        self.simulation_accepted.emit(config)
        self.accept()

    def apply_dialog_styles(self):
        """Aplicar estilos globales al diálogo."""
        self.setStyleSheet("""
            QDialog {
                background-color: #f5f5f5;
            }
            QGroupBox {
                border: 2px solid #2196F3;
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 10px;
                background-color: white;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 6px;
                color: #2196F3;
                background-color: white;
            }
            QRadioButton::indicator {
                width: 14px;
                height: 14px;
            }
            QRadioButton::indicator:unchecked {
                border: 2px solid #999;
                border-radius: 7px;
                background-color: white;
            }
            QRadioButton::indicator:checked {
                border: 2px solid #2196F3;
                border-radius: 7px;
                background-color: #2196F3;
            }
        """)