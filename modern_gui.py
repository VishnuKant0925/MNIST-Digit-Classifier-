import os
import sys
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from pathlib import Path

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFileDialog, QMessageBox, QProgressBar, QFrame
)
from PyQt5.QtGui import QPixmap, QImage, QFont, QColor, QIcon
from PyQt5.QtCore import Qt, QSize, pyqtSlot, QThread, pyqtSignal
from PyQt5.QtCore import QMimeData

from main import Net

MODEL_PATH = "mnist_cnn.pt"


def load_model(device: torch.device) -> torch.nn.Module | None:
    if not os.path.exists(MODEL_PATH):
        return None
    
    # Load state dict to determine number of classes
    state_dict = torch.load(MODEL_PATH, map_location=device)
    num_classes = state_dict['fc2.bias'].shape[0]  # Infer from final layer bias
    
    model = Net(num_classes=num_classes).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model(device)

# Same preprocessing as in training script
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])


def predict_image(image_path: str) -> tuple[int, float, list]:
    if model is None:
        raise RuntimeError(
            f"Model file '{MODEL_PATH}' not found. Run 'python main.py --save-model' first."
        )

    img = Image.open(image_path).convert("L")  # ensure grayscale
    img = img.resize((28, 28))

    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)[0]
        pred = output.argmax(dim=1).item()
        confidence = float(probs[pred].item())
        all_probs = [float(p.item()) for p in probs]

    return pred, confidence, all_probs


class PredictionWorker(QThread):
    prediction_done = pyqtSignal(int, float, list)
    prediction_error = pyqtSignal(str)

    def __init__(self, image_path: str):
        super().__init__()
        self.image_path = image_path

    def run(self):
        try:
            pred, confidence, all_probs = predict_image(self.image_path)
            self.prediction_done.emit(pred, confidence, all_probs)
        except Exception as e:
            self.prediction_error.emit(str(e))


class MnistGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.current_image_path = None
        self.prediction_worker = None
        self.initUI()
        self.setAcceptDrops(True)

        if model is None:
            QMessageBox.critical(
                self,
                "Model Not Found",
                f"Could not find '{MODEL_PATH}'.\n\nPlease run:\npython main.py --save-model\n\nto train and save the model first.",
            )

    def initUI(self):
        self.setWindowTitle("MNIST Digit Classifier")
        self.setGeometry(100, 100, 900, 700)
        self.setStyleSheet(self.get_stylesheet())

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout()
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(30, 30, 30, 30)

        # Header
        header_label = QLabel("MNIST Digit Classifier")
        header_font = QFont()
        header_font.setPointSize(28)
        header_font.setBold(True)
        header_label.setFont(header_font)
        header_label.setAlignment(Qt.AlignCenter)
        header_label.setStyleSheet("color: #2c3e50; margin-bottom: 10px;")
        main_layout.addWidget(header_label)

        # Subtitle
        subtitle = QLabel("Upload or drag an image to predict the digit")
        subtitle_font = QFont()
        subtitle_font.setPointSize(11)
        subtitle.setFont(subtitle_font)
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("color: #7f8c8d; margin-bottom: 20px;")
        main_layout.addWidget(subtitle)

        # Content layout (two columns)
        content_layout = QHBoxLayout()
        content_layout.setSpacing(30)

        # Left side - Image display
        left_layout = QVBoxLayout()
        image_frame = QFrame()
        image_frame.setStyleSheet("""
            QFrame {
                background-color: #ecf0f1;
                border: 2px dashed #95a5a6;
                border-radius: 10px;
            }
        """)
        image_frame_layout = QVBoxLayout()
        image_frame_layout.setContentsMargins(0, 0, 0, 0)

        self.image_label = QLabel()
        self.image_label.setMinimumSize(300, 300)
        self.image_label.setMaximumSize(300, 300)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("""
            QLabel {
                background-color: white;
                border-radius: 8px;
                border: 1px solid #bdc3c7;
            }
        """)
        self.image_label.setText("No image loaded\n\nDrag and drop image here\nor click 'Load Image'")
        default_font = QFont()
        default_font.setPointSize(11)
        self.image_label.setFont(default_font)
        self.image_label.setStyleSheet("""
            QLabel {
                background-color: white;
                border-radius: 8px;
                border: 1px solid #bdc3c7;
                color: #95a5a6;
            }
        """)
        image_frame_layout.addWidget(self.image_label, 1, Qt.AlignCenter)
        image_frame.setLayout(image_frame_layout)
        left_layout.addWidget(image_frame)

        content_layout.addLayout(left_layout)

        # Right side - Results
        right_layout = QVBoxLayout()
        right_layout.setSpacing(15)

        # Prediction display
        prediction_frame = QFrame()
        prediction_frame.setStyleSheet("""
            QFrame {
                background-color: #ecf0f1;
                border-radius: 10px;
                border: 1px solid #bdc3c7;
            }
        """)
        prediction_layout = QVBoxLayout()
        prediction_layout.setContentsMargins(20, 20, 20, 20)

        pred_label = QLabel("Prediction")
        pred_label_font = QFont()
        pred_label_font.setPointSize(13)
        pred_label_font.setBold(True)
        pred_label.setFont(pred_label_font)
        pred_label.setStyleSheet("color: #2c3e50;")
        prediction_layout.addWidget(pred_label)

        self.prediction_display = QLabel("-")
        pred_display_font = QFont()
        pred_display_font.setPointSize(48)
        pred_display_font.setBold(True)
        self.prediction_display.setFont(pred_display_font)
        self.prediction_display.setAlignment(Qt.AlignCenter)
        self.prediction_display.setStyleSheet("color: #3498db;")
        prediction_layout.addWidget(self.prediction_display)

        self.confidence_label = QLabel("Confidence: -")
        conf_font = QFont()
        conf_font.setPointSize(11)
        self.confidence_label.setFont(conf_font)
        self.confidence_label.setAlignment(Qt.AlignCenter)
        self.confidence_label.setStyleSheet("color: #27ae60;")
        prediction_layout.addWidget(self.confidence_label)

        prediction_frame.setLayout(prediction_layout)
        right_layout.addWidget(prediction_frame)

        # Class probabilities
        probs_label = QLabel("Class Probabilities")
        probs_label_font = QFont()
        probs_label_font.setPointSize(12)
        probs_label_font.setBold(True)
        probs_label.setStyleSheet("color: #2c3e50;")
        probs_label.setFont(probs_label_font)
        right_layout.addWidget(probs_label)

        # Probability bars
        self.prob_labels = []
        self.prob_bars = []

        probs_frame = QFrame()
        probs_frame.setStyleSheet("""
            QFrame {
                background-color: white;
                border-radius: 8px;
                border: 1px solid #bdc3c7;
            }
        """)
        probs_layout = QVBoxLayout()
        probs_layout.setContentsMargins(15, 15, 15, 15)
        probs_layout.setSpacing(10)

        for i in range(10):
            row_layout = QHBoxLayout()
            row_layout.setSpacing(10)

            digit_label = QLabel(str(i))
            digit_label.setMinimumWidth(25)
            digit_font = QFont()
            digit_font.setBold(True)
            digit_label.setFont(digit_font)
            digit_label.setStyleSheet("color: #2c3e50;")
            row_layout.addWidget(digit_label)

            prob_bar = QProgressBar()
            prob_bar.setStyleSheet("""
                QProgressBar {
                    border: 1px solid #bdc3c7;
                    border-radius: 5px;
                    background-color: #ecf0f1;
                    height: 20px;
                    text-align: center;
                }
                QProgressBar::chunk {
                    background-color: #3498db;
                    border-radius: 4px;
                }
            """)
            prob_bar.setValue(0)
            row_layout.addWidget(prob_bar)

            prob_text = QLabel("0%")
            prob_text.setMinimumWidth(40)
            prob_text.setAlignment(Qt.AlignRight)
            prob_text_font = QFont()
            prob_text_font.setPointSize(9)
            prob_text.setFont(prob_text_font)
            prob_text.setStyleSheet("color: #7f8c8d;")
            row_layout.addWidget(prob_text)

            probs_layout.addLayout(row_layout)
            self.prob_labels.append((digit_label, prob_text))
            self.prob_bars.append(prob_bar)

        probs_frame.setLayout(probs_layout)
        right_layout.addWidget(probs_frame)

        content_layout.addLayout(right_layout)
        main_layout.addLayout(content_layout)

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.setSpacing(10)

        self.load_button = QPushButton("📁 Load Image")
        self.load_button.setMinimumHeight(45)
        self.load_button.setMinimumWidth(150)
        self.load_button.clicked.connect(self.load_image)
        self.load_button.setStyleSheet(self.get_button_stylesheet())
        button_layout.addWidget(self.load_button)

        self.clear_button = QPushButton("🔄 Clear")
        self.clear_button.setMinimumHeight(45)
        self.clear_button.setMinimumWidth(150)
        self.clear_button.clicked.connect(self.clear_all)
        self.clear_button.setStyleSheet(self.get_button_stylesheet())
        button_layout.addWidget(self.clear_button)

        button_layout.addStretch()

        self.quit_button = QPushButton("❌ Exit")
        self.quit_button.setMinimumHeight(45)
        self.quit_button.setMinimumWidth(150)
        self.quit_button.clicked.connect(self.close)
        self.quit_button.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                border: none;
                border-radius: 5px;
                font-weight: bold;
                font-size: 12px;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
            QPushButton:pressed {
                background-color: #a93226;
            }
        """)
        button_layout.addWidget(self.quit_button)

        main_layout.addLayout(button_layout)

        central_widget.setLayout(main_layout)

    @pyqtSlot()
    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select a digit image",
            "",
            "Image Files (*.png *.jpg *.jpeg *.bmp *.gif);;All Files (*)"
        )
        if file_path:
            self.process_image(file_path)

    def process_image(self, file_path: str):
        try:
            img = Image.open(file_path).convert("L")
            self.current_image_path = file_path

            # Display preview
            preview = img.resize((280, 280), Image.LANCZOS)
            q_image = QImage(preview.tobytes(), preview.width, preview.height, preview.width, QImage.Format_Grayscale8)
            pixmap = QPixmap.fromImage(q_image)
            self.image_label.setPixmap(pixmap)
            self.image_label.setText("")

            # Predict in background thread
            if self.prediction_worker is not None:
                self.prediction_worker.quit()
                self.prediction_worker.wait()

            self.prediction_worker = PredictionWorker(file_path)
            self.prediction_worker.prediction_done.connect(self.on_prediction_done)
            self.prediction_worker.prediction_error.connect(self.on_prediction_error)
            self.prediction_worker.start()

            self.load_button.setText("⏳ Predicting...")
            self.load_button.setEnabled(False)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load image:\n{str(e)}")

    @pyqtSlot(int, float, list)
    def on_prediction_done(self, pred: int, confidence: float, all_probs: list):
        self.prediction_display.setText(str(pred))
        self.confidence_label.setText(f"Confidence: {confidence*100:.2f}%")

        # Update probability bars
        for i, prob in enumerate(all_probs):
            self.prob_bars[i].setValue(int(prob * 100))
            self.prob_labels[i][1].setText(f"{prob*100:.1f}%")

        self.load_button.setText("📁 Load Image")
        self.load_button.setEnabled(True)

    @pyqtSlot(str)
    def on_prediction_error(self, error_msg: str):
        QMessageBox.critical(self, "Prediction Error", f"Failed to predict:\n{error_msg}")
        self.load_button.setText("📁 Load Image")
        self.load_button.setEnabled(True)

    def clear_all(self):
        self.image_label.setText("No image loaded\n\nDrag and drop image here\nor click 'Load Image'")
        self.image_label.setPixmap(QPixmap())
        self.prediction_display.setText("-")
        self.confidence_label.setText("Confidence: -")
        for prob_bar in self.prob_bars:
            prob_bar.setValue(0)
        for _, prob_text in self.prob_labels:
            prob_text.setText("0%")
        self.current_image_path = None

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            # Check if it's an image
            image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff'}
            if Path(file_path).suffix.lower() in image_extensions:
                self.process_image(file_path)
            else:
                QMessageBox.warning(self, "Invalid File", "Please drop an image file.")

    @staticmethod
    def get_stylesheet():
        return """
            QMainWindow {
                background-color: #f5f6fa;
            }
            QApplication {
                background-color: #f5f6fa;
            }
        """

    @staticmethod
    def get_button_stylesheet():
        return """
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                border-radius: 5px;
                font-weight: bold;
                font-size: 12px;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:pressed {
                background-color: #1f618d;
            }
            QPushButton:disabled {
                background-color: #95a5a6;
                color: #bdc3c7;
            }
        """


def main():
    app = QApplication(sys.argv)
    window = MnistGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
