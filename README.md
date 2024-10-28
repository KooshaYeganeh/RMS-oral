# Oral Disease Diagnosis Application

This application is designed to assist with the diagnosis of oral and dental diseases using advanced AI and deep learning techniques. Leveraging powerful frameworks like PyTorch and MONAI, the system is capable of analyzing oral disease images and providing accurate predictions to assist healthcare professionals in early diagnosis and treatment.

## Table of Contents
- [Features](#features)
- [Technology Stack](#technology-stack)
- [Installation](#installation)
- [Usage](#usage)
- [Building Executables](#building-executables)
- [Deep Learning & AI in Dental Healthcare](#deep-learning--ai-in-dental-healthcare)
- [License](#license)
- [Contributing](#contributing)

## Features
- **User-friendly Interface**: Easy navigation and interaction with the app.
- **Image Upload**: Allows users to upload oral disease images for analysis.
- **Real-Time Prediction**: Provides accurate predictions of oral diseases using advanced deep learning models.
- **Integration with MONAI**: The Medical Open Network for AI (MONAI) is utilized for deep learning workflows in the healthcare domain.
- **Cross-Platform Support**: Runs on Linux, Windows, and macOS.

## Technology Stack
- **UI Framework**: Kivy (for cross-platform graphical user interface)
- **Deep Learning Frameworks**: PyTorch & MONAI (for medical imaging and AI models)
- **Image Processing**: Pillow (for image handling)
- **Packaging**: PyInstaller (for creating standalone executables)


- **Note : Latest trained Data at : Mon 21 Oct 2024 08:24:36 PM**
- **Number of Data : 150**

## Installation

### Prerequisites
Ensure you have the following installed on your system:
- Python 3.11
- Git (optional, for cloning the repository)
- Virtualenv (recommended for isolating dependencies)

### Steps to Install and Run Locally

1. **Clone the repository**:
   ```bash
   git clone https://gitlab.com/katayoun_katebi/oral_disease.git
   cd oral_disease
   ```

2. **Set up a virtual environment (optional but recommended)**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**:
   ```bash
   python app/Desktop.py
   ```

### Additional Dependencies

Ensure you have the following Python libraries installed:

- **Kivy**: For building the UI.
- **Pillow**: For image handling and processing.
- **PyTorch**: For deep learning model development and predictions.
- **MONAI**: For advanced AI workflows in medical image processing and deep learning.

To install these dependencies, run:

```bash
pip install kivy pillow torch monai
```

### Installing PyInstaller

If you want to build a standalone executable, install `pyinstaller`:

```bash
pip install pyinstaller
```

## Usage

1. **Run the App**:
   After installing, you can run the application with:
   ```bash
   python app/Desktop.py
   ```

2. **Upload Images**:
   Use the upload button in the app to select oral disease images for analysis.

3. **Get Predictions**:
   After uploading an image, the app processes it using the trained CNN and MONAI models and provides predictions on the type of disease.

## Building Executables

To build a standalone executable for distribution:

1. **Install PyInstaller**:
   Make sure `pyinstaller` is installed:
   ```bash
   pip install pyinstaller
   ```

2. **Create the Executable**:
   Run the following command to package the application:
   ```bash
   pyinstaller --onefile --windowed app/Desktop.py
   ```

   This will generate a standalone executable in the `dist` folder that can be distributed to users without requiring Python or additional dependencies.

3. **Distribute the Executable**:
   Share the executable generated in the `dist` folder. Users can run the application without needing to install Python or other dependencies.

## Deep Learning & AI in Dental Healthcare

Artificial Intelligence (AI) is revolutionizing the healthcare industry, and dental care is no exception. With advances in **deep learning** and **medical imaging technologies** like **MONAI**, AI systems can now assist dentists in diagnosing complex oral diseases with high accuracy.

### Why Use AI in Dental Healthcare?
- **Early Detection**: AI can assist in detecting diseases at an early stage, which is crucial for successful treatment outcomes.
- **Accuracy**: Deep learning models, particularly Convolutional Neural Networks (CNNs), can analyze intricate patterns in oral disease images, surpassing human capabilities in many cases.
- **Efficiency**: AI tools provide instant analysis, saving valuable time for dental professionals.
  
### MONAI in Medical Imaging
MONAI (Medical Open Network for AI) is a specialized open-source framework for deep learning in healthcare. It integrates with PyTorch and provides optimized tools for training AI models on medical images, such as X-rays and dental scans. This project uses MONAI to improve the accuracy and robustness of oral disease predictions.

**Sample of Internet**
![Deep Learning for Dental Disease sample](https://media.springernature.com/lw1200/springer-static/image/art%3A10.1007%2Fs13721-024-00459-0/MediaObjects/13721_2024_459_Fig3_HTML.png)  


By combining CNN-based models with MONAI’s medical-specific deep learning tools, this app provides state-of-the-art predictions for various oral diseases. 

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contributing
We welcome contributions! If you'd like to contribute to this project, please fork the repository and submit a pull request with your changes.

### Steps to Contribute:
1. Fork the repository.
2. Create a new feature branch: `git checkout -b feature/my-feature`.
3. Commit your changes: `git commit -m 'Add some feature'`.
4. Push to the branch: `git push origin feature/my-feature`.
5. Open a pull request.


### Notes :

- The data related to machine learning is in the form of a sample, and if you want to use this software in a stable way, put your csv file and train it.
- The data related to Oral Disease Detection is very limited and cannot be guaranteed. Therefore, if needed, you can replace your data (The data will be updated regularly and the train file will be updated)
- Histopathologic Oral Cancer Detection data were taken from kaggle site.