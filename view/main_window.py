# view/main_window.py - Vista principal mejorada con diseño responsive
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QPushButton, QLabel, QTextEdit, QGroupBox, 
                            QProgressBar, QStatusBar, QSplitter, QScrollArea, QDialog,
                            QDialogButtonBox, QComboBox, QFrame, QSizePolicy, QCheckBox)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QIcon, QPixmap
import os

class PlotViewerDialog(QDialog):
    """Diálogo para mostrar gráficas generadas"""
    
    def __init__(self, plot_file_path, title="Gráfica SAIDI", parent=None):
        super().__init__(parent)
        self.plot_file_path = plot_file_path
        self.setup_ui(title)
        self.load_plot()
    
    def setup_ui(self, title):
        """Configurar interfaz del visor de gráficas"""
        self.setWindowTitle(title)
        self.setModal(False)
        self.resize(1000, 700)
        
        layout = QVBoxLayout(self)
        
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("border: 1px solid #ccc; background-color: white;")
        scroll_area.setWidget(self.image_label)
        
        layout.addWidget(scroll_area)
        
        button_layout = QHBoxLayout()
        
        self.save_button = QPushButton(" Guardar Como...")
        self.save_button.clicked.connect(self.save_plot)
        button_layout.addWidget(self.save_button)
        
        button_layout.addStretch()
        
        self.close_button = QPushButton("Cerrar")
        self.close_button.clicked.connect(self.accept)
        button_layout.addWidget(self.close_button)
        
        layout.addLayout(button_layout)
        
        self.setStyleSheet("""
            QDialog {
                background-color: #f5f5f5;
            }
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: bold;
                font-size: 12px;
                min-height: 30px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
        """)
    
    def load_plot(self):
        """Cargar y mostrar la gráfica"""
        if not self.plot_file_path or not os.path.exists(self.plot_file_path):
            self.image_label.setText("Error: No se pudo cargar la gráfica")
            return
        
        try:
            pixmap = QPixmap(self.plot_file_path)
            if pixmap.isNull():
                self.image_label.setText("Error: Formato de imagen no válido")
                return
                
            scaled_pixmap = pixmap.scaled(
                900, 600, 
                Qt.AspectRatioMode.KeepAspectRatio, 
                Qt.TransformationMode.SmoothTransformation
            )
            
            self.image_label.setPixmap(scaled_pixmap)
            
        except Exception as e:
            self.image_label.setText(f"Error cargando imagen: {str(e)}")
    
    def save_plot(self):
        """Guardar la gráfica en una ubicación personalizada"""
        from PyQt6.QtWidgets import QFileDialog
        
        if not self.plot_file_path or not os.path.exists(self.plot_file_path):
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Guardar Gráfica SAIDI",
            "saidi_grafica.png",
            "Imágenes PNG (*.png);;Imágenes JPG (*.jpg);;Todos los archivos (*.*)"
        )
        
        if file_path:
            try:
                import shutil
                shutil.copy2(self.plot_file_path, file_path)
                
                from PyQt6.QtWidgets import QMessageBox
                QMessageBox.information(
                    self, 
                    "Guardado Exitoso", 
                    f"Gráfica guardada exitosamente en:\n{file_path}"
                )
                
            except Exception as e:
                from PyQt6.QtWidgets import QMessageBox
                QMessageBox.critical(
                    self,
                    "Error",
                    f"Error guardando la gráfica:\n{str(e)}"
                )

class MainWindow(QMainWindow):
    """Ventana principal de la aplicación SAIDI"""
    
    # Señales personalizadas
    excel_load_requested = pyqtSignal()
    prediction_requested = pyqtSignal()
    optimization_requested = pyqtSignal()
    validation_requested = pyqtSignal()
    overfitting_requested = pyqtSignal()
    regional_selected = pyqtSignal(str)
    climate_load_requested = pyqtSignal(str)
    export_requested = pyqtSignal()   # Nueva señal para exportar a Excel
    
    def __init__(self):
        super().__init__()
        self.climate_load_buttons = {}
        self.climate_status_labels = {}
        self.setup_ui()
        
    def setup_ui(self):
        """Configurar interfaz de usuario"""
        self.setWindowTitle("SAIDI Analysis Tool - Sistema de Análisis y Predicción")
        self.setGeometry(100, 100, 1600, 900)
        self.setMinimumSize(1200, 700)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(12)
        
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)
        
        # Panel izquierdo - Controles (con scroll)
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        left_scroll.setFrameShape(QFrame.Shape.NoFrame)
        
        self.create_control_panel()
        left_scroll.setWidget(self.control_panel)
        left_scroll.setMinimumWidth(420)
        left_scroll.setMaximumWidth(550)
        
        splitter.addWidget(left_scroll)
        
        # Panel derecho - Información y Log
        self.create_info_panel()
        splitter.addWidget(self.info_panel)
        
        splitter.setSizes([450, 1150])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("✓ Sistema listo - Cargue archivos Excel para comenzar")
        
        self.apply_styles()
        
    def create_control_panel(self):
        """Crear panel de controles"""
        self.control_panel = QWidget()
        layout = QVBoxLayout(self.control_panel)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(12)
        
        # Encabezado mejorado
        header_widget = QWidget()
        header_layout = QVBoxLayout(header_widget)
        header_layout.setContentsMargins(12, 16, 12, 16)
        header_layout.setSpacing(4)
        
        header_widget.setStyleSheet("""
            QWidget {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #2196F3, stop:1 #1976D2);
                border-radius: 10px;
            }
        """)
        
        title = QLabel("SAIDI Analysis")
        title.setFont(QFont("Segoe UI", 20, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("color: white; background: transparent;")
        header_layout.addWidget(title)
        
        subtitle = QLabel("Sistema de Análisis y Predicción")
        subtitle.setFont(QFont("Segoe UI", 10))
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet("color: rgba(255, 255, 255, 0.9); background: transparent;")
        header_layout.addWidget(subtitle)
        
        layout.addWidget(header_widget)
        
        # Grupo 1: Carga de datos SAIDI
        data_group = self.create_styled_group("1. Cargar Datos SAIDI", "#4CAF50")
        data_layout = QVBoxLayout(data_group)
        data_layout.setSpacing(10)
        
        self.load_excel_button = QPushButton(" Cargar Archivo Excel SAIDI")
        self.load_excel_button.setMinimumHeight(50)
        self.load_excel_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.load_excel_button.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        data_layout.addWidget(self.load_excel_button)
        
        self.file_info_label = QLabel("No hay archivo cargado")
        self.file_info_label.setWordWrap(True)
        self.file_info_label.setStyleSheet("""
            QLabel {
                background-color: #f9f9f9;
                padding: 10px;
                border-radius: 6px;
                border: 1px dashed #ccc;
                color: #666;
                font-size: 11px;
            }
        """)
        data_layout.addWidget(self.file_info_label)
        
        # Selector de regionales
        self.regional_selector_widget = QWidget()
        regional_selector_layout = QVBoxLayout(self.regional_selector_widget)
        regional_selector_layout.setContentsMargins(0, 10, 0, 0)
        regional_selector_layout.setSpacing(8)
        
        regional_label = QLabel(" Seleccione Regional:")
        regional_label.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        regional_label.setStyleSheet("color: #2196F3;")
        regional_selector_layout.addWidget(regional_label)
        
        self.regional_combo = QComboBox()
        self.regional_combo.setMinimumHeight(40)
        self.regional_combo.setFont(QFont("Segoe UI", 10))
        self.regional_combo.setCursor(Qt.CursorShape.PointingHandCursor)
        self.regional_combo.setStyleSheet("""
            QComboBox {
                background-color: white;
                border: 2px solid #2196F3;
                border-radius: 8px;
                padding: 8px 12px;
                font-weight: bold;
                color: #333;
            }
            QComboBox:hover {
                border-color: #1976D2;
                background-color: #f5f5f5;
            }
            QComboBox::drop-down {
                border: none;
                width: 30px;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 6px solid #2196F3;
                margin-right: 8px;
            }
        """)
        self.regional_combo.currentTextChanged.connect(self.on_regional_changed)
        regional_selector_layout.addWidget(self.regional_combo)
        
        self.regional_info_label = QLabel("")
        self.regional_info_label.setWordWrap(True)
        self.regional_info_label.setStyleSheet("""
            QLabel {
                background-color: #E3F2FD;
                padding: 8px;
                border-radius: 6px;
                border-left: 4px solid #2196F3;
                color: #1565C0;
                font-size: 10px;
            }
        """)
        regional_selector_layout.addWidget(self.regional_info_label)
        
        self.regional_selector_widget.setVisible(False)
        data_layout.addWidget(self.regional_selector_widget)
        
        layout.addWidget(data_group)
        
        # Grupo 2: Carga de datos climáticos
        climate_group = self.create_styled_group("2. Cargar Datos Climáticos", "#FF9800")
        climate_layout = QVBoxLayout(climate_group)
        climate_layout.setSpacing(8)
        
        climate_info = QLabel("Cargar datos climáticos para cada regional\n(Excepto CENS - Empresa General)")
        climate_info.setWordWrap(True)
        climate_info.setStyleSheet("""
            QLabel {
                background-color: #FFF3E0;
                padding: 10px;
                border-radius: 6px;
                border-left: 4px solid #FF9800;
                color: #E65100;
                font-size: 10px;
                font-style: italic;
            }
        """)
        climate_layout.addWidget(climate_info)
        
        # Scroll para botones climáticos
        climate_scroll = QScrollArea()
        climate_scroll.setWidgetResizable(True)
        climate_scroll.setMaximumHeight(280)
        climate_scroll.setFrameShape(QFrame.Shape.NoFrame)
        climate_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        climate_buttons_widget = QWidget()
        climate_buttons_layout = QVBoxLayout(climate_buttons_widget)
        climate_buttons_layout.setContentsMargins(0, 0, 0, 0)
        climate_buttons_layout.setSpacing(6)
        
        regionales_clima = {
            'SAIDI_C': 'Cúcuta',
            'SAIDI_O': 'Ocaña',
            'SAIDI_A': 'Aguachica',
            'SAIDI_P': 'Pamplona',
            'SAIDI_T': 'Tibú'
        }
        
        for code, nombre in regionales_clima.items():
            regional_widget = QWidget()
            regional_layout = QHBoxLayout(regional_widget)
            regional_layout.setContentsMargins(0, 0, 0, 0)
            regional_layout.setSpacing(8)
            
            btn = QPushButton(f" {nombre}")
            btn.setMinimumHeight(42)
            btn.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
            btn.setProperty('regional_code', code)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.clicked.connect(lambda checked, c=code: self.on_climate_load_clicked(c))
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #FF9800;
                    text-align: left;
                    padding-left: 12px;
                    border-radius: 6px;
                }
                QPushButton:hover {
                    background-color: #F57C00;
                }
                QPushButton:pressed {
                    background-color: #E65100;
                }
            """)
            
            status_label = QLabel(" 0 ")
            status_label.setFixedWidth(30)
            status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            status_label.setStyleSheet("font-size: 18px;")
            
            regional_layout.addWidget(btn, 4)
            regional_layout.addWidget(status_label, 1)
            
            climate_buttons_layout.addWidget(regional_widget)
            
            self.climate_load_buttons[code] = btn
            self.climate_status_labels[code] = status_label
        
        climate_scroll.setWidget(climate_buttons_widget)
        climate_layout.addWidget(climate_scroll)
        
        self.climate_progress_label = QLabel("")
        self.climate_progress_label.setWordWrap(True)
        self.climate_progress_label.setStyleSheet("""
            QLabel {
                background-color: #E8F5E9;
                padding: 8px;
                border-radius: 6px;
                border-left: 4px solid #4CAF50;
                color: #2E7D32;
                font-size: 10px;
                font-weight: bold;
            }
        """)
        climate_layout.addWidget(self.climate_progress_label)
        
        layout.addWidget(climate_group)
        
        # Grupo 2.5: Simulación Climática (NUEVO)
        simulation_group = self.create_styled_group("🌦️ Simulación Climática", "#9C27B0")
        simulation_layout = QVBoxLayout(simulation_group)
        simulation_layout.setSpacing(10)
        
        # Checkbox para habilitar simulación
        self.enable_simulation_checkbox = QCheckBox("Habilitar simulación climática antes de predicción")
        self.enable_simulation_checkbox.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        self.enable_simulation_checkbox.setStyleSheet("""
            QCheckBox {
                spacing: 8px;
                color: #333;
            }
            QCheckBox::indicator {
                width: 20px;
                height: 20px;
                border: 2px solid #9C27B0;
                border-radius: 4px;
                background-color: white;
            }
            QCheckBox::indicator:checked {
                background-color: #9C27B0;
                image: url(none);
            }
            QCheckBox::indicator:hover {
                border-color: #7B1FA2;
            }
        """)
        self.enable_simulation_checkbox.stateChanged.connect(self.on_simulation_checkbox_changed)
        simulation_layout.addWidget(self.enable_simulation_checkbox)
        
        # Descripción de la funcionalidad
        simulation_desc = QLabel(
            "Cuando esté habilitado, podrá configurar escenarios climáticos "
            "hipotéticos (☀️ soleado, 🌧️ lluvioso, ⛈️ tormentoso, 🌡️ ola de calor) "
            "antes de generar la predicción SAIDI."
        )
        simulation_desc.setWordWrap(True)
        simulation_desc.setStyleSheet("""
            QLabel {
                background-color: #F3E5F5;
                padding: 10px;
                border-radius: 6px;
                border-left: 4px solid #9C27B0;
                color: #4A148C;
                font-size: 10px;
                font-style: italic;
            }
        """)
        simulation_layout.addWidget(simulation_desc)
        
        # Indicador de estado
        self.simulation_status_label = QLabel("")
        self.simulation_status_label.setWordWrap(True)
        self.simulation_status_label.setVisible(False)
        self.simulation_status_label.setStyleSheet("""
            QLabel {
                background-color: #E1BEE7;
                padding: 8px;
                border-radius: 6px;
                color: #6A1B9A;
                font-size: 10px;
                font-weight: bold;
            }
        """)
        simulation_layout.addWidget(self.simulation_status_label)
        
        layout.addWidget(simulation_group)
        
        # Grupo 3: Análisis SAIDI
        analysis_group = self.create_styled_group("3. Análisis SAIDI", "#2196F3")
        analysis_layout = QVBoxLayout(analysis_group)
        analysis_layout.setSpacing(8)
        
        analysis_buttons = [
            (" Generar Predicción", "predict_button", "#2196F3"),
            (" Optimizar Parámetros", "optimize_button", "#9C27B0"),
            (" Validar Modelo", "validate_button", "#00BCD4"),
            (" Detectar Overfitting", "overfitting_button", "#FF6B6B")
        ]
        
        for text, attr_name, color in analysis_buttons:
            btn = QPushButton(text)
            btn.setMinimumHeight(44)
            btn.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
            btn.setEnabled(False)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {color};
                    border-radius: 6px;
                }}
                QPushButton:hover {{
                    background-color: {color};
                    filter: brightness(110%);
                }}
                QPushButton:disabled {{
                    background-color: #e0e0e0;
                    color: #9e9e9e;
                }}
            """)
            analysis_layout.addWidget(btn)
            setattr(self, attr_name, btn)
        
        layout.addWidget(analysis_group)
        
        # Botón para exportar predicciones a Excel (será conectado al controlador si está disponible)
        self.export_excel_button = QPushButton("Exportar a Excel")
        self.export_excel_button.setMinimumHeight(45)
        self.export_excel_button.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        self.export_excel_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.export_excel_button.setEnabled(False)  # se habilita cuando haya predicciones/generación disponible
        # Estilo parecido a otros botones
        self.export_excel_button.setStyleSheet("""
            QPushButton {
                background-color: #607D8B;
                color: white;
                border-radius: 6px;
                padding: 8px 12px;
            }
            QPushButton:hover {
                background-color: #546E7A;
            }
        """)
        # Conectar a la señal interna; el controlador debe conectarse a export_requested desde main.py
        self.export_excel_button.clicked.connect(lambda: self.export_requested.emit())

        # Añadir el botón al panel de controles (debajo de los grupos principales)
        layout.addWidget(self.export_excel_button)
        
        # Grupo 4: Mejoras anti-overfitting
        improvements_group = self.create_styled_group("4. Mejoras Anti-Overfitting", "#FF5722")
        improvements_layout = QVBoxLayout(improvements_group)
        improvements_layout.setSpacing(8)
        
        improvement_buttons = [
            (" Cross-Validation Temporal", "cv_button", "#00BCD4"),
            (" Buscar Modelo Simple", "simple_model_button", "#8BC34A"),
            (" Comparar Transformaciones", "transformation_button", "#FF9800")
        ]
        
        for text, attr_name, color in improvement_buttons:
            btn = QPushButton(text)
            btn.setMinimumHeight(44)
            btn.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
            btn.setEnabled(False)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {color};
                    border-radius: 6px;
                }}
                QPushButton:hover {{
                    background-color: {color};
                    filter: brightness(110%);
                }}
                QPushButton:disabled {{
                    background-color: #e0e0e0;
                    color: #9e9e9e;
                }}
            """)
            improvements_layout.addWidget(btn)
            setattr(self, attr_name, btn)
        
        info_label = QLabel("💡 Usar estas herramientas para reducir overfitting")
        info_label.setWordWrap(True)
        info_label.setStyleSheet("""
            QLabel {
                background-color: #FFF3E0;
                padding: 10px;
                border-radius: 6px;
                border-left: 4px solid #FF9800;
                color: #E65100;
                font-size: 10px;
                font-style: italic;
            }
        """)
        improvements_layout.addWidget(info_label)
        
        layout.addWidget(improvements_group)
        
        # Progreso
        progress_group = self.create_styled_group("Progreso", "#757575")
        progress_layout = QVBoxLayout(progress_group)
        progress_layout.setSpacing(8)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setMinimumHeight(24)
        self.progress_bar.setTextVisible(True)
        progress_layout.addWidget(self.progress_bar)
        
        self.progress_label = QLabel("")
        self.progress_label.setVisible(False)
        self.progress_label.setStyleSheet("color: #555; font-size: 10px;")
        progress_layout.addWidget(self.progress_label)
        
        layout.addWidget(progress_group)
        
        layout.addStretch()
        
    def create_styled_group(self, title, color):
        """Crear grupo estilizado con color personalizado"""
        group = QGroupBox(title)
        group.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        group.setStyleSheet(f"""
            QGroupBox {{
                border: 2px solid {color};
                border-radius: 10px;
                margin-top: 12px;
                padding-top: 12px;
                background-color: white;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 8px;
                color: {color};
                background-color: white;
            }}
        """)
        return group
        
    def create_info_panel(self):
        """Crear panel de información responsive"""
        self.info_panel = QWidget()
        layout = QVBoxLayout(self.info_panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        
        # Información del archivo SAIDI
        self.excel_info_group = self.create_styled_group("Información del Archivo SAIDI", "#4CAF50")
        excel_info_layout = QVBoxLayout(self.excel_info_group)
        
        self.excel_details_label = QLabel("Cargue un archivo Excel para ver detalles")
        self.excel_details_label.setWordWrap(True)
        self.excel_details_label.setFont(QFont("Segoe UI", 10))
        self.excel_details_label.setStyleSheet("""
            QLabel {
                padding: 12px;
                background-color: #fafafa;
                border-radius: 6px;
                color: #666;
            }
        """)
        self.excel_details_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        excel_info_layout.addWidget(self.excel_details_label)
        
        layout.addWidget(self.excel_info_group)
        
        # Información de datos climáticos
        self.climate_info_group = self.create_styled_group("Información de Datos Climáticos", "#FF9800")
        climate_info_layout = QVBoxLayout(self.climate_info_group)
        
        self.climate_details_label = QLabel("Cargue archivos climáticos para ver detalles")
        self.climate_details_label.setWordWrap(True)
        self.climate_details_label.setFont(QFont("Segoe UI", 10))
        self.climate_details_label.setStyleSheet("""
            QLabel {
                padding: 12px;
                background-color: #fafafa;
                border-radius: 6px;
                color: #666;
            }
        """)
        self.climate_details_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        climate_info_layout.addWidget(self.climate_details_label)
        
        layout.addWidget(self.climate_info_group)
        
        # Log de consola (expandible)
        log_group = self.create_styled_group("Log de Consola", "#2196F3")
        log_layout = QVBoxLayout(log_group)
        
        self.console_log = QTextEdit()
        self.console_log.setReadOnly(True)
        self.console_log.setFont(QFont("Consolas", 9))
        self.console_log.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        log_layout.addWidget(self.console_log)
        
        log_buttons_layout = QHBoxLayout()
        
        self.clear_log_button = QPushButton(" Limpiar Log")
        self.clear_log_button.setMinimumHeight(36)
        self.clear_log_button.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        self.clear_log_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.clear_log_button.clicked.connect(self.clear_console_log)
        log_buttons_layout.addWidget(self.clear_log_button)
        
        log_buttons_layout.addStretch()
        log_layout.addLayout(log_buttons_layout)
        
        layout.addWidget(log_group, 1)
        
    def apply_styles(self):
        """Aplicar estilos CSS modernos a la interfaz"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f2f5;
            }
            
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 10px 16px;
                font-weight: bold;
                font-size: 11px;
            }
            
            QPushButton:hover {
                background-color: #45a049;
            }
            
            QPushButton:pressed {
                background-color: #3d8b40;
            }
            
            QPushButton:disabled {
                background-color: #e0e0e0;
                color: #9e9e9e;
            }
            
            QTextEdit {
                border: 1px solid #ddd;
                border-radius: 6px;
                background-color: #ffffff;
                padding: 8px;
                selection-background-color: #2196F3;
            }
            
            QProgressBar {
                border: 1px solid #ddd;
                border-radius: 6px;
                text-align: center;
                background-color: #f5f5f5;
                font-weight: bold;
                color: #333;
            }
            
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #4CAF50, stop:1 #66BB6A);
                border-radius: 5px;
            }
            
            QStatusBar {
                background-color: white;
                border-top: 1px solid #ddd;
                color: #555;
                font-size: 11px;
                padding: 4px;
            }
            
            QScrollArea {
                border: none;
                background-color: transparent;
            }
            
            QScrollBar:vertical {
                border: none;
                background-color: #f5f5f5;
                width: 10px;
                border-radius: 5px;
            }
            
            QScrollBar::handle:vertical {
                background-color: #bbb;
                border-radius: 5px;
                min-height: 30px;
            }
            
            QScrollBar::handle:vertical:hover {
                background-color: #999;
            }
            
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)
    
    def on_simulation_checkbox_changed(self, state):
        """Callback cuando cambia el estado del checkbox de simulación"""
        if state == Qt.CheckState.Checked.value:
            self.simulation_status_label.setText(
                "✓ Simulación habilitada - Se abrirá configurador antes de predecir"
            )
            self.simulation_status_label.setVisible(True)
            self.log_message("🌦️ Simulación climática HABILITADA")
            self.log_message("   Al generar predicción, se abrirá el configurador de escenarios")
        else:
            self.simulation_status_label.setVisible(False)
            self.log_message("🌦️ Simulación climática DESHABILITADA")
            self.log_message("   Las predicciones usarán proyección normal de variables exógenas")
    
    def on_climate_load_clicked(self, regional_code):
        """Callback cuando se hace clic en un botón de carga climática"""
        self.climate_load_requested.emit(regional_code)
    
    def update_climate_status(self, regional_code, status):
        """Actualizar estado visual de carga climática"""
        if regional_code not in self.climate_status_labels:
            return
        
        label = self.climate_status_labels[regional_code]
        
        status_icons = {
            'loading': '...',
            'success': 'ON',
            'error': 'OFF',
            'empty': '0'
        }
        
        label.setText(status_icons.get(status, '0'))
        
    def update_climate_progress_summary(self, loaded_count, total_count):
        """Actualizar resumen de progreso de carga climática"""
        if loaded_count == 0:
            self.climate_progress_label.setText("")
        elif loaded_count == total_count:
            self.climate_progress_label.setText(
                f" Completo: {loaded_count}/{total_count} regionales cargadas"
            )
            self.climate_progress_label.setStyleSheet("""
                QLabel {
                    background-color: #C8E6C9;
                    padding: 8px;
                    border-radius: 6px;
                    border-left: 4px solid #4CAF50;
                    color: #1B5E20;
                    font-size: 10px;
                    font-weight: bold;
                }
            """)
        else:
            self.climate_progress_label.setText(
                f" Progreso: {loaded_count}/{total_count} regionales cargadas"
            )
            self.climate_progress_label.setStyleSheet("""
                QLabel {
                    background-color: #FFF9C4;
                    padding: 8px;
                    border-radius: 6px;
                    border-left: 4px solid #FDD835;
                    color: #F57F17;
                    font-size: 10px;
                    font-weight: bold;
                }
            """)
    
    def update_climate_details(self, climate_summary):
        """Actualizar panel de detalles climáticos"""
        if not climate_summary:
            self.climate_details_label.setText("No hay datos climáticos cargados")
            return
        
        details = "<div style='font-size: 11px;'><b style='color: #FF6F00; font-size: 12px;'>DATOS CLIMÁTICOS CARGADOS:</b><br><br>"
        
        for regional_code, info in climate_summary.items():
            if info:
                details += f"<div style='margin-bottom: 12px; padding: 8px; background-color: #fff; border-left: 3px solid #FF9800; border-radius: 4px;'>"
                details += f"<b style='color: #FF6F00;'> {info['regional_name']}:</b><br>"
                details += f"<span style='color: #666;'> Archivo: <b>{info['file_name']}</b></span><br>"
                details += f"<span style='color: #666;'> Registros: <b>{info['total_records']:,}</b></span><br>"
                details += f"<span style='color: #666;'> Período: <b>{info['date_range']['start'].strftime('%Y-%m-%d')}</b> a <b>{info['date_range']['end'].strftime('%Y-%m-%d')}</b></span><br>"
                details += f"<span style='color: #666;'>✓ Completitud: <b style='color: #4CAF50;'>{info['avg_completeness']:.1f}%</b></span>"
                details += f"</div>"
        
        details += "</div>"
        
        self.climate_details_label.setText(details)
        
        # Habilitar simulación si hay datos climáticos cargados
        if hasattr(self, 'enable_simulation_checkbox') and climate_summary:
            self.enable_simulation_checkbox.setEnabled(True)
            self.enable_simulation_checkbox.setToolTip(
                "Habilite para configurar escenarios climáticos antes de predecir"
            )
            self.log_message("✓ Simulación climática disponible")
    
    def on_regional_changed(self, regional_text):
        """Callback cuando cambia la selección de regional"""
        if regional_text:
            if '(' in regional_text and ')' in regional_text:
                codigo = regional_text.split('(')[1].split(')')[0]
                self.log_message(f"Regional seleccionada: {regional_text}")
                self.regional_selected.emit(codigo)
                self.regional_info_label.setText(f"✓ Trabajando con: {regional_text}")
            else:
                self.log_message(f"Regional seleccionada: {regional_text}")
    
    def show_regional_selector(self, regionales_list):
        """Mostrar selector de regionales"""
        self.log_message(f"Mostrando selector con {len(regionales_list)} regionales")
        
        self.regional_combo.clear()
        
        for regional in regionales_list:
            codigo = regional.get('codigo', '')
            nombre = regional.get('nombre', codigo)
            display_text = f"{nombre} ({codigo})"
            self.regional_combo.addItem(display_text)
        
        self.regional_selector_widget.setVisible(True)
        
        if len(regionales_list) > 0:
            self.regional_combo.setCurrentIndex(0)
            first_regional = regionales_list[0]
            self.regional_info_label.setText(
                f"✓ Trabajando con: {first_regional['nombre']} ({first_regional['codigo']})"
            )
    
    def hide_regional_selector(self):
        """Ocultar selector de regionales"""
        self.regional_selector_widget.setVisible(False)
        self.regional_combo.clear()
        self.regional_info_label.setText("")
        self.log_message("Selector de regionales ocultado")
    
    def log_message(self, message):
        """Agregar mensaje al log de consola"""
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.console_log.append(f'<span style="color: #2196F3;">[{timestamp}]</span> {message}')
        self.console_log.ensureCursorVisible()
        
    def log_error(self, message):
        """Agregar mensaje de error al log"""
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.console_log.append(f'<span style="color: #888;">[{timestamp}]</span> <span style="color: #f44336; font-weight: bold;">❌ ERROR:</span> {message}')
        self.console_log.ensureCursorVisible()
        
    def log_success(self, message):
        """Agregar mensaje de éxito al log"""
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.console_log.append(f'<span style="color: #888;">[{timestamp}]</span> <span style="color: #4CAF50; font-weight: bold;">✓ ÉXITO:</span> {message}')
        self.console_log.ensureCursorVisible()
        
    def clear_console_log(self):
        """Limpiar el log de consola"""
        self.console_log.clear()
        self.log_message("Log limpiado")
        
    def on_excel_loaded(self, file_info):
        """Callback cuando se carga un archivo Excel"""
        self.file_info_label.setText(f" {file_info.get('file_name', 'Archivo cargado')}")
        self.file_info_label.setStyleSheet("""
            QLabel {
                background-color: #E8F5E9;
                padding: 10px;
                border-radius: 6px;
                border-left: 4px solid #4CAF50;
                color: #2E7D32;
                font-size: 11px;
                font-weight: bold;
            }
        """)
        
        is_regional = file_info.get('is_regional_format', False)
        
        if is_regional:
            regionales = file_info.get('available_regionales', [])
            if regionales:
                self.show_regional_selector(regionales)
                self.log_success(f"Formato REGIONAL detectado con {len(regionales)} regionales")
            
            self.predict_button.setEnabled(True)
            self.optimize_button.setEnabled(True)
            self.validate_button.setEnabled(True)
            self.overfitting_button.setEnabled(True)
            self.cv_button.setEnabled(True)
            self.simple_model_button.setEnabled(True)
            self.transformation_button.setEnabled(True)
        else:
            self.hide_regional_selector()
            
            self.predict_button.setEnabled(True)
            self.optimize_button.setEnabled(True)
            self.validate_button.setEnabled(True)
            self.overfitting_button.setEnabled(True)
            self.cv_button.setEnabled(True)
            self.simple_model_button.setEnabled(True)
            self.transformation_button.setEnabled(True)
            
            self.log_success("Formato TRADICIONAL detectado")

        details = "<div style='font-size: 11px;'>"
        details += f"<div style='padding: 10px; background-color: #fff; border-radius: 6px; margin-bottom: 8px;'>"
        details += f"<b style='color: #4CAF50; font-size: 12px;'> Archivo:</b> <span style='color: #333;'>{file_info.get('file_name', 'N/A')}</span><br>"
        details += f"<b style='color: #2196F3;'> Dimensiones:</b> <span style='color: #333;'>{file_info.get('rows', 0):,} filas × {file_info.get('columns', 0)} columnas</span><br>"
        
        if is_regional:
            details += f"<b style='color: #FF9800;'> Formato:</b> <span style='color: #333; font-weight: bold;'>Regional (multi-columna)</span><br>"
            details += f"<b style='color: #9C27B0;'> Regionales:</b> <span style='color: #333;'>{len(file_info.get('available_regionales', []))}</span><br>"
        else:
            details += f"<b style='color: #FF9800;'> Formato:</b> <span style='color: #333; font-weight: bold;'>Tradicional (columna única)</span><br>"
        
        details += f"</div>"
        
        details += f"<div style='padding: 10px; background-color: #f9f9f9; border-radius: 6px; border-left: 3px solid #2196F3;'>"
        details += f"<b style='color: #2196F3;'> Período:</b><br>"
        details += f"<span style='color: #666;'>Desde: <b>{file_info.get('date_range', {}).get('start', 'N/A')}</b></span><br>"
        details += f"<span style='color: #666;'>Hasta: <b>{file_info.get('date_range', {}).get('end', 'N/A')}</b></span><br>"
        details += f"<span style='color: #4CAF50; font-weight: bold;'>✓ Contiene SAIDI: {'Sí' if file_info.get('has_saidi', False) else 'No'}</span>"
        details += f"</div>"
        
        details += "</div>"
        
        self.excel_details_label.setText(details)
        
        self.log_success(f"Archivo Excel cargado: {file_info.get('file_name', 'N/A')}")
        self.log_message(f"Dimensiones: {file_info.get('rows', 0):,} filas × {file_info.get('columns', 0)} columnas")
        
        # Verificar disponibilidad de simulación
        if hasattr(self, 'enable_simulation_checkbox'):
            # La simulación requiere datos climáticos
            # Se habilitará cuando se carguen archivos climáticos
            self.enable_simulation_checkbox.setEnabled(False)
            self.enable_simulation_checkbox.setChecked(False)
            self.enable_simulation_checkbox.setToolTip(
                "Cargue datos climáticos de la regional para habilitar simulación"
            )
        
    def update_status(self, message):
        """Actualizar barra de estado"""
        self.status_bar.showMessage(f"✓ {message}")
        
    def show_progress(self, visible=True):
        """Mostrar/ocultar barra de progreso"""
        self.progress_bar.setVisible(visible)
        self.progress_label.setVisible(visible)
        if not visible:
            self.progress_bar.setValue(0)
        
    def update_progress(self, value, message=""):
        """Actualizar progreso"""
        self.progress_bar.setValue(value)
        if message:
            self.progress_label.setText(message)
            
    def set_buttons_enabled(self, enabled):
        """Habilitar/deshabilitar botones durante procesamiento"""
        self.load_excel_button.setEnabled(enabled)
        if hasattr(self, 'file_info_label') and self.file_info_label.text() != "No hay archivo cargado":
            self.predict_button.setEnabled(enabled)
            self.optimize_button.setEnabled(enabled)
            self.validate_button.setEnabled(enabled)
            self.overfitting_button.setEnabled(enabled)
            self.cv_button.setEnabled(enabled)
            self.simple_model_button.setEnabled(enabled)
            self.transformation_button.setEnabled(enabled)
            
            if hasattr(self, 'regional_combo'):
                self.regional_combo.setEnabled(enabled)
        
        for btn in self.climate_load_buttons.values():
            btn.setEnabled(enabled)
        
        if hasattr(self, 'enable_simulation_checkbox'):
            # Solo habilitar si hay datos climáticos
            if enabled and hasattr(self, 'climate_details_label'):
                # Verificar si hay texto de datos climáticos
                has_climate = "DATOS CLIMÁTICOS CARGADOS" in self.climate_details_label.text()
                self.enable_simulation_checkbox.setEnabled(has_climate)
            else:
                self.enable_simulation_checkbox.setEnabled(False)

    def enable_export_button(self, enabled: bool = True):
        """
        Habilitar/deshabilitar el botón de exportar
        
        Args:
            enabled: True para habilitar, False para deshabilitar
        """
        if hasattr(self, 'export_excel_button'):
            self.export_excel_button.setEnabled(enabled)
            
            if enabled:
                self.export_excel_button.setToolTip(
                    "Exportar últimas predicciones generadas a archivo Excel\n"
                    "Incluye intervalos de confianza y métricas del modelo"
                )
                # Cambiar estilo para indicar que está activo
                self.export_excel_button.setStyleSheet("""
                    QPushButton {
                        background-color: #4CAF50;
                        color: white;
                        border-radius: 6px;
                        padding: 8px 12px;
                        font-weight: bold;
                    }
                    QPushButton:hover {
                        background-color: #45a049;
                        transform: scale(1.02);
                    }
                    QPushButton:pressed {
                        background-color: #3d8b40;
                    }
                """)
            else:
                self.export_excel_button.setToolTip("")
                # Estilo deshabilitado
                self.export_excel_button.setStyleSheet("""
                    QPushButton {
                        background-color: #BDBDBD;
                        color: #757575;
                        border-radius: 6px;
                        padding: 8px 12px;
                    }
                """)

    def disable_export_button(self):
        """Deshabilitar botón de exportación"""
        self.enable_export_button(False)

    def on_export_clicked(self):
        """Emisión directa (por compatibilidad) — se usa la señal export_requested"""
        self.export_requested.emit()