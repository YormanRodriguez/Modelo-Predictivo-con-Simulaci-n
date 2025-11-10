# view/climate_simulation_dialog.py - Diálogo de simulación climática
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
    """Diálogo para configurar simulación climática antes de predicción."""

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
        self.dias_base = None
        self.slider_ranges = None

        # Estado actual
        self.escenario_seleccionado = None
        self.slider_adjustment = 0

        self.setup_ui()
        self.calculate_simulation_data()

    def setup_ui(self):
        """Configurar interfaz de usuario."""
        self.setWindowTitle(f"Simulador - {self.regional_nombre}")
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

        historical_group = self._create_historical_group()
        main_layout.addWidget(historical_group)

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

    def _create_historical_group(self):
        """Crear grupo de datos históricos."""
        self.historical_group = QGroupBox("Datos Históricos")
        self.historical_group.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))

        historical_layout = QVBoxLayout(self.historical_group)
        historical_layout.setContentsMargins(8, 12, 8, 8)

        self.historical_label = QLabel("Calculando...")
        self.historical_label.setWordWrap(True)
        self.historical_label.setStyleSheet("""
            QLabel {
                background-color: #F5F5F5;
                padding: 8px;
                border-radius: 4px;
                font-size: 10px;
                color: #333;
            }
        """)
        historical_layout.addWidget(self.historical_label)

        return self.historical_group

    def _create_adjustment_group(self):
        """Crear grupo de ajuste de intensidad."""
        self.adjustment_group = QGroupBox("Ajustar Intensidad")
        self.adjustment_group.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        self.adjustment_group.setEnabled(False)

        adjustment_layout = QVBoxLayout(self.adjustment_group)
        adjustment_layout.setContentsMargins(8, 12, 8, 8)
        adjustment_layout.setSpacing(6)

        # Label de ajuste
        self.adjustment_label = QLabel("Seleccione un escenario")
        self.adjustment_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.adjustment_label.setStyleSheet("color: #666; font-size: 10px; padding: 4px;")
        adjustment_layout.addWidget(self.adjustment_label)

        # Slider y controles
        self._add_slider_controls(adjustment_layout)

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

    def _add_slider_controls(self, layout):
        """Agregar controles del slider al layout."""
        # Label de rango
        self.slider_range_label = QLabel("")
        self.slider_range_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.slider_range_label.setStyleSheet("font-size: 9px; color: #888;")
        layout.addWidget(self.slider_range_label)

        # Slider
        self.intensity_slider = QSlider(Qt.Orientation.Horizontal)
        self.intensity_slider.setMinimum(-10)
        self.intensity_slider.setMaximum(10)
        self.intensity_slider.setValue(0)
        self.intensity_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.intensity_slider.setTickInterval(2)
        self.intensity_slider.valueChanged.connect(self.on_slider_changed)
        self.intensity_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #bbb;
                background: white;
                height: 8px;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #2196F3;
                border: 2px solid #1976D2;
                width: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
        """)
        layout.addWidget(self.intensity_slider)

        # Label de valor
        self.slider_value_label = QLabel("0 días")
        self.slider_value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.slider_value_label.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        self.slider_value_label.setStyleSheet("color: #2196F3; padding: 3px;")
        layout.addWidget(self.slider_value_label)

    def _create_alcance_group(self):
        """Crear grupo de alcance temporal."""
        alcance_group = QGroupBox("Alcance Temporal")
        alcance_group.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))

        alcance_layout = QVBoxLayout(alcance_group)
        alcance_layout.setContentsMargins(8, 12, 8, 8)
        alcance_layout.setSpacing(6)

        # Descripción
        alcance_desc = QLabel("¿A cuántos meses?")
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
        self.no_simulation_button = QPushButton("Sin Simulación")
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
        button_text = "Validar con Simulación" if self.mode == "validation" else "Simular Predicción"
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
        """Crear grupo de selección de escenarios."""
        scenario_group = QGroupBox("Escenario Climático")
        scenario_group.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        scenario_layout = QVBoxLayout(scenario_group)
        scenario_layout.setContentsMargins(8, 12, 8, 8)
        scenario_layout.setSpacing(6)

        self.scenario_buttons = {}
        self.scenario_group = QButtonGroup(self)

        scenarios = ClimateSimulationService.SCENARIOS

        for idx, (key, scenario) in enumerate(scenarios.items()):
            btn = QPushButton(f"{scenario['icon']} {scenario['name']}")
            btn.setMinimumHeight(40)
            btn.setCheckable(True)
            btn.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
            btn.clicked.connect(lambda _checked, k=key: self.on_scenario_selected(k))

            colors = {
                "soleado": "#FF9800",
                "lluvioso": "#2196F3",
                "tormentoso": "#9C27B0",
                "ola_calor": "#F44336",
            }

            color = colors.get(key, "#757575")

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
            self.scenario_buttons[key] = btn
            self.scenario_group.addButton(btn, idx)

            # Descripción compacta
            desc = QLabel(f"   {scenario['description']}")
            desc.setStyleSheet("color: #888; font-size: 9px; font-style: italic; padding-left: 16px;")
            scenario_layout.addWidget(desc)

        parent_layout.addWidget(scenario_group)

    def calculate_simulation_data(self):
        """Calcular datos necesarios para la simulación."""
        try:
            self.percentiles = self.simulation_service.calculate_percentiles(
                self.climate_data, self.regional_code,
            )

            self.dias_base = self.simulation_service.calculate_base_days(
                self.climate_data, self.mes_prediccion, self.regional_code,
            )

            self.slider_ranges = self.simulation_service.calculate_slider_ranges(
                self.climate_data, self.mes_prediccion, "lluvioso", self.regional_code,
            )

            self.update_historical_info()

        except AttributeError as e:
            error_msg = f"Error: Servicio no inicializado - {e!s}"
            self.historical_label.setText(error_msg)
            print(f"Error en calculate_simulation_data: {e}")
        except (KeyError, ValueError) as e:
            error_msg = f"Error: Datos climáticos inválidos - {e!s}"
            self.historical_label.setText(error_msg)
            print(f"Error en calculate_simulation_data: {e}")
        except TypeError as e:
            error_msg = f"Error: Parámetros incorrectos - {e!s}"
            self.historical_label.setText(error_msg)
            print(f"Error en calculate_simulation_data: {e}")

    def update_historical_info(self):
        """Actualizar información histórica del mes."""
        try:
            meses_nombres = ["Ene", "Feb", "Mar", "Abr", "May", "Jun",
                        "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]

            mes_nombre = meses_nombres[self.mes_prediccion - 1]

            info = f"<b>Mes:</b> {mes_nombre} | "
            info += f"<b>☀</b> {self.dias_base['soleado']}d | "
            info += f"<b>🌧</b> {self.dias_base['lluvioso']}d | "
            info += f"<b>⛈</b> {self.dias_base['tormentoso']}d"

            self.historical_label.setText(info)

        except IndexError as e:
            self.historical_label.setText(f"Error: Mes inválido - {e!s}")
        except KeyError as e:
            self.historical_label.setText(f"Error: Tipo de clima no encontrado - {e!s}")
        except (AttributeError, TypeError) as e:
            self.historical_label.setText(f"Error: Datos no disponibles - {e!s}")

    def on_scenario_selected(self, escenario_key):
        """Callback cuando se selecciona un escenario."""
        try:
            self.escenario_seleccionado = escenario_key

            self.adjustment_group.setEnabled(True)
            self.simulate_button.setEnabled(True)

            scenario_to_climate = {
                "soleado": "soleado",
                "lluvioso": "lluvioso",
                "tormentoso": "tormentoso",
                "ola_calor": "soleado",
            }

            climate_type = scenario_to_climate.get(escenario_key, "lluvioso")

            if climate_type in self.slider_ranges:
                min_dias, max_dias, base_dias = self.slider_ranges[climate_type]

                min_adjustment = min_dias - base_dias
                max_adjustment = max_dias - base_dias

                self.intensity_slider.setMinimum(min_adjustment)
                self.intensity_slider.setMaximum(max_adjustment)
                self.intensity_slider.setValue(0)

                self.slider_range_label.setText(
                    f"Rango: {min_dias}-{max_dias}d (base: {base_dias}d)",
                )

            scenario_info = ClimateSimulationService.SCENARIOS[escenario_key]
            self.adjustment_label.setText(f"Ajustar: {scenario_info['name']}")

            self.update_preview()

        except KeyError as e:
            QMessageBox.warning(self, "Error", f"Error: Escenario no encontrado - {e!s}")
        except (ValueError, TypeError) as e:
            QMessageBox.warning(self, "Error", f"Error en los valores del rango - {e!s}")
        except AttributeError as e:
            QMessageBox.warning(self, "Error", f"Error: Componente no inicializado - {e!s}")

    def on_slider_changed(self, value):
        """Callback cuando cambia el slider."""
        self.slider_adjustment = value

        if value > 0:
            self.slider_value_label.setText(f"+{value} días")
            self.slider_value_label.setStyleSheet("color: #FF5722; padding: 3px; font-weight: bold;")
        elif value < 0:
            self.slider_value_label.setText(f"{value} días")
            self.slider_value_label.setStyleSheet("color: #2196F3; padding: 3px; font-weight: bold;")
        else:
            self.slider_value_label.setText("0 días")
            self.slider_value_label.setStyleSheet("color: #757575; padding: 3px; font-weight: bold;")

        self.update_preview()

    def update_preview(self):
        """Actualizar vista previa de la simulación."""
        try:
            if not self.escenario_seleccionado:
                return

            scenario_to_climate = {
                "soleado": "soleado",
                "lluvioso": "lluvioso",
                "tormentoso": "tormentoso",
                "ola_calor": "soleado",
            }

            climate_type = scenario_to_climate.get(self.escenario_seleccionado, "lluvioso")
            dias_base = self.dias_base[climate_type]
            dias_simulados = dias_base + self.slider_adjustment

            alcance = self.alcance_group.checkedId()

            summary = self.simulation_service.get_simulation_summary(
                self.escenario_seleccionado,
                self.slider_adjustment,
                dias_base,
                alcance,
                self.percentiles,
                self.regional_code,
            )

            scenario_info = ClimateSimulationService.SCENARIOS[self.escenario_seleccionado]

            preview = f"<b>{scenario_info['icon']} {scenario_info['name']}</b> | "
            preview += f"{dias_simulados}d ({climate_type}) | {alcance} meses<br>"

            var_names = {
                "temp_max": "Temp máx",
                "humedad_avg": "Humedad",
                "precip_total": "Precip",
            }

            changes = []
            for var, change_pct in summary["percentage_changes"].items():
                var_name = var_names.get(var, var)

                if abs(change_pct) < 1:
                    arrow = "→"
                    color = ""
                elif change_pct > 0:
                    arrow = "↑"
                    color = " style='color: #F44336;'"
                else:
                    arrow = "↓"
                    color = " style='color: #2196F3;'"

                changes.append(f"{var_name}: <b{color}>{arrow}{change_pct:+.1f}%</b>")

            preview += " | ".join(changes)

            # Advertencia según modo
            if self.mode == "validation":
                preview += "<br><b style='color: #F57C00;'>⚠ Validación: Evalúa sensibilidad del modelo</b>"

            self.preview_label.setText(preview)

        except KeyError as e:
            self.preview_label.setText(f"Error: Clave no encontrada - {e!s}")
        except (AttributeError, TypeError) as e:
            self.preview_label.setText(f"Error: Datos inválidos - {e!s}")
        except ValueError as e:
            self.preview_label.setText(f"Error: Valor incorrecto - {e!s}")

    def on_simulate(self):
        """Callback cuando se acepta la simulación."""
        try:
            if not self.escenario_seleccionado:
                QMessageBox.warning(self, "Advertencia", "Seleccione un escenario")
                return

            scenario_to_climate = {
                "soleado": "soleado",
                "lluvioso": "lluvioso",
                "tormentoso": "tormentoso",
                "ola_calor": "soleado",
            }

            climate_type = scenario_to_climate.get(self.escenario_seleccionado, "lluvioso")
            dias_base = self.dias_base[climate_type]

            alcance = self.alcance_group.checkedId()

            is_valid, error_msg = self.simulation_service.validate_simulation_params(
                self.escenario_seleccionado,
                self.slider_adjustment,
                dias_base,
                alcance,
            )

            if not is_valid:
                QMessageBox.critical(self, "Error", error_msg)
                return

            config = {
                "enabled": True,
                "escenario": self.escenario_seleccionado,
                "slider_adjustment": self.slider_adjustment,
                "dias_base": dias_base,
                "alcance_meses": alcance,
                "percentiles": self.percentiles,
                "regional_code": self.regional_code,
                "summary": self.simulation_service.get_simulation_summary(
                    self.escenario_seleccionado,
                    self.slider_adjustment,
                    dias_base,
                    alcance,
                    self.percentiles,
                    self.regional_code,
                ),
            }

            self.simulation_accepted.emit(config)
            self.accept()

        except (KeyError, AttributeError, ValueError, TypeError) as e:
            QMessageBox.critical(self, "Error", f"Error en la configuración: {e!s}")
        except RuntimeError as e:
            QMessageBox.critical(self, "Error", f"Error de ejecución: {e!s}")

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
