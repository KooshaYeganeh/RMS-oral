import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os
from sklearn.preprocessing import OrdinalEncoder
import joblib
from sklearn.preprocessing import MinMaxScaler
# Image transformation for preprocessing
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend for matplotlib
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from flask import Flask, render_template, request, redirect, url_for , send_file , Response
import torch
from torch import nn
from torchvision import transforms
from PIL import Image
import os
import io
import joblib
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from skimage import color, filters, measure
from skimage import io as skio, color, filters, measure, morphology, segmentation
import matplotlib
import pickle
from io import BytesIO
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user

# Initialize Flask app
app = Flask(__name__)



upload_folder = os.path.join(f"uploads")
os.makedirs(upload_folder, exist_ok=True)



opg = os.path.join(f"uploads/opg")
os.makedirs(opg, exist_ok=True)



bitewing = os.path.join(f"uploads/bitewing")
os.makedirs(bitewing, exist_ok=True)



app.config.update(SECRET_KEY="rms-oral")










login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"


class User(UserMixin):
    def __init__(self, id):
        self.id = id

    def __repr__(self):
        return f"{self.id}"


# Sample hardcoded user data (username and plain text password)
users = {
    "katebi": "Katebi_102030"  # Plain text password
}


# Load user from the ID
@login_manager.user_loader
def load_user(userid):
    return User(userid)


@app.route("/")
@login_required
def dashboard():
    return render_template("dashboard.html", username=current_user.id)


@app.route("/login")
def login():
    return render_template("sign-in.html")


@app.route("/login", methods=["POST"])
def loggin():
    username = request.form["username"]
    password = request.form["password"]

    # Check if the username exists and password matches in plain text
    if username in users and users[username] == password:
        user = User(username)
        login_user(user)  # Log the user in
        return redirect(url_for("dashboard"))  # Redirect to the protected dashboard
    else:
        return render_template("sign-in.html", error="Username or password is invalid")


@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))


# Error handler for unauthorized access
@app.errorhandler(401)
def page_not_found(e):
    return Response("""
                    <html><center style='background-color:white;'>
                    <h2 style='color:red;'>Login failed</h2>
                    <h1>Error 401</h1>
                    </center></html>""")



""" This Route Get histopathologic_cancer Detection. Give Image Like Samples
to return Valid Response"""


@app.route("/histopathologic_oral_cancer")
def histopathologic_oral_cancer_get():
    return render_template("histopathologic_oral_cancer.html")


# Define the CNN model for cancer detection
class SimpleCNN_histopathologic(nn.Module):
    def __init__(self):
        super(SimpleCNN_histopathologic, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(16 * 112 * 112, 2)  # Output for 2 classes (Normal, Abnormal)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 16 * 112 * 112)  # Flatten the output
        x = self.fc1(x)
        return x

# Initialize model and load weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
histopathologic_cancer_model = SimpleCNN_histopathologic().to(device)
histopathologic_cancer_model.load_state_dict(torch.load('histopathologic_oral_cancer_model.pth', map_location=device))
histopathologic_cancer_model.eval()

# Transform for input image for cancer detection
histopathologic_cancer_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])



@app.route('/histopathologic_oral_cancer', methods=['POST'])
def histopathologic_oral_cancer_predict():
    file = request.files['histopathologic_file']
    if file.filename == '':
        return({'error': 'No selected file'})

    try:
        # Process image
        image = Image.open(file.stream)
        image = histopathologic_cancer_transform(image).unsqueeze(0).to(device)

        # Model prediction
        with torch.no_grad():
            outputs = histopathologic_cancer_model(image)
            _, predicted = torch.max(outputs, 1)

        # Map predicted index to class name
        class_names = ['Normal', 'Abnormal']
        predicted_class = class_names[predicted.item()]
        return render_template("histopathologic_oral_cancer.html" , prediction =  predicted_class)

    except Exception as e:
        return ({'error': str(e)})








"""
This Scopr is Oral Diease Detection model with dr Katayoun Katebi's Data
Upooad Oral Diease Image to return Result of Cancer
"""




@app.route('/oral_disease_detection')
def oral_disease_detection_get():
    return render_template('oral_disease_detection.html')


# Define device for model (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the transformation for images
oral_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize image to the model's input size
    transforms.ToTensor(),  # Convert image to PyTorch tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize tensor values
])

# Define class names for the 12 classes of oral diseases
oral_disease_class_names = [
    'aphtous_ulcer', 'denture_stomatitis', 'epulis_fissuratum',
    'erythroplakia', 'fordyce_granules', 'geographic_tongue',
    'herpes_labialis', 'intra_oral_herpes', 'leukoplakia',
    'oral_lichen_planus', 'squamous_cell_carcinoma', 'traumatic_ulcer'
]

# SimpleCNN Model Definition
class SimpleCNN_oral_disease(nn.Module):
    def __init__(self):
        super(SimpleCNN_oral_disease, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(32 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, 12)
        self.dropout = nn.Dropout(0.5)  # Dropout with 50% probability

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 56 * 56)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # Apply dropout after the first fully connected layer
        x = self.fc2(x)
        return x


# Initialize model
oral_disease_model = SimpleCNN_oral_disease().to(device)

# Load the pretrained model
oral_disease_model.load_state_dict(torch.load('oral_disease_monai_model.pth', map_location=device))
oral_disease_model.eval()

def process_image_oral(image_data):
    img = Image.open(io.BytesIO(image_data)).convert('RGB')  # Ensure image is in RGB mode
    img = oral_transform(img)  # Apply transformations
    img = img.unsqueeze(0)  # Add batch dimension
    return img.to(device)








@app.route('/oral_disease_detection', methods=['POST'])
def oral_disease_detection():
    file = request.files['toothfile']

    # Process the image
    image_data = file.read()
    image = process_image_oral(image_data)

    # Model prediction
    with torch.no_grad():
        outputs = oral_disease_model(image)
        _, predicted = torch.max(outputs, 1)

    # Map predicted index to class name
    predicted_class = oral_disease_class_names[predicted.item()]

    return render_template('oral_disease_detection.html', prediction=predicted_class)









## ML



@app.route("/oral_cancer_detection_with_ML")
def oral_cancer_detection_with_ML_get():
    return render_template("oral_cancer_detection_with_ML.html")



model = joblib.load("trained_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

@app.route("/oral_cancer_detection_with_ml", methods=["POST"])
def oral_cancer_detection_with_ml_predict():
    try:
        # Get data from request.form (convert to float or string as needed)
        input_data = {
            "age": [float(request.form.get("age"))],
            "gender": [request.form.get("gender")],
            "location": [request.form.get("location")],
            "size": [float(request.form.get("size"))],
            "color": [request.form.get("color")],
            "surface": [request.form.get("surface")],
            "texture": [request.form.get("texture")],
            "lymphnode_involvment": [request.form.get("lymphnode_involvment")],
            "lymphnode_location": [request.form.get("lymphnode_location")],
            "lymphnode_side": [request.form.get("lymphnode_side")],
            "lymphnode_texture": [request.form.get("lymphnode_texture")],
            "lymphnode_tenderness": [request.form.get("lymphnode_tenderness")],
            "lymphnode_mobility": [request.form.get("lymphnode_mobility")],
            "smoking": [request.form.get("smoking")],
            "alcohol": [request.form.get("alcohol")]
        }

        # Convert input data into a pandas DataFrame
        input_df = pd.DataFrame(input_data)

        # Preprocessing: Handle categorical variables for encoding
        cat_columns = ["gender", "location", "color", "surface", "texture", "lymphnode_involvment",
                       "lymphnode_location", "lymphnode_side", "lymphnode_texture",
                       "lymphnode_tenderness", "lymphnode_mobility", "smoking", "alcohol"]

        # Label encoding for categorical variables (if applicable)
        for col in cat_columns:
            if col in input_df.columns and input_df[col].dtype == 'object':  # Only encode if categorical
                # Check if the label exists in the encoder's classes_ and encode accordingly
                if input_df[col].iloc[0] not in label_encoder.classes_:
                    # Handle unseen labels by using the first class (or another appropriate value)
                    input_df[col] = label_encoder.transform([label_encoder.classes_[0]])  # Fallback to the first class
                else:
                    input_df[col] = label_encoder.transform(input_df[col])

        # Use the model pipeline to predict the input
        prediction = model.predict(input_df)[0]

        # Decode the prediction (if the result is a label encoding, decode it)
        prediction_label = label_encoder.inverse_transform([int(prediction)])[0]

        # Return the result using the template
        return render_template("oral_cancer_detection_with_ML.html", prediction=prediction_label)

    except Exception as e:
        # Handle exceptions and return a descriptive error message
        return {"error": str(e)}, 500









## Tooth Segments






@app.route("/tooth_segments")
def tooth_segmentation():
    return render_template("tooth_segmentation.html")




#@app.route("/tooth_segmentation", methods=["POST"])
#def tooth_decay_segments_post():
#    try:
#        # Get current date and time
#        image_file = request.files.get("bitwing2")
#
#        if not image_file or image_file.filename == '':
#            return("""
#                <html><body style='background-color:black;'>
#                    <center><h2 style='color:red;'>
#                        !!! No Selected File !!!
#                    <h2>
#                    <h1>Please check the information sent from the form and then try again</h1>
#                    <a href='/toothdecay'><button style='background-color: red; border: none;color: white;padding: 15px 32px;text-align: center;text-decoration: none;display: inline-block;font-size: 16px;'> Return  </button></a>
#                    </h2></center>
#                </body></html>""")
#
#        try:
#            # Save file to two locations
#            name = image_file.filename
#            goal_path = os.path.join('uploads', name)
#            image_file.save(goal_path)
#        except Exception as e:
#            return f"Error saving file: {str(e)}"
#
#        try:
#            # Read and preprocess image
#            image = skio.imread(goal_path)
#            gray_image = color.rgb2gray(image)
#
#            # Preprocessing: Enhance image for better segmentation
#            enhanced_image = filters.gaussian(gray_image, sigma=1.0)
#
#            # Segmentation: Thresholding and morphological operations
#            thresh = filters.threshold_otsu(enhanced_image)
#            binary_image = enhanced_image > thresh
#
#            # Remove small objects and perform morphological operations
#            cleaned_image = morphology.remove_small_objects(binary_image, min_size=500)
#            cleaned_image = morphology.remove_small_holes(cleaned_image, area_threshold=500)
#
#            # Label connected regions (individual teeth)
#            labels = measure.label(cleaned_image)
#
#            # Feature extraction and analysis
#            scaler = MinMaxScaler()
#            regions_info = []
#            for region in measure.regionprops(labels):
#                area = region.area
#                centroid = region.centroid
#                bbox = region.bbox
#                min_intensity = gray_image[bbox[0]:bbox[2], bbox[1]:bbox[3]].min()
#                max_intensity = gray_image[bbox[0]:bbox[2], bbox[1]:bbox[3]].max()
#                decay_info = max_intensity - min_intensity
#
#                regions_info.append({
#                    'area': area,
#                    'centroid': centroid,
#                    'bounding_box': bbox,
#                    'decay_info': decay_info
#                })
#
#                # Optional: Visualize decay detection
#                plt.contour(labels == region.label, colors='r')
#
#            # Normalize labels for visualization
#            labels_normalized = (labels - labels.min()) / (labels.max() - labels.min()) * 255
#            labels_normalized = labels_normalized.astype(np.uint8)
#
#            # Plot original and processed images side by side
#            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
#
#            # Original Image
#            axes[0].imshow(image)
#            axes[0].set_title('Original Image')
#            axes[0].axis('off')
#
#            # Processed Image with Segmentation
#            axes[1].imshow(gray_image, cmap='gray')
#            axes[1].imshow(labels, alpha=0.5, cmap='jet')
#            axes[1].set_title('Processed Image')
#            axes[1].axis('off')
#
#            # Save the plot
#            temp_img_path = os.path.join('uploads', f'{image_file.filename}')
#            plt.savefig(temp_img_path, bbox_inches='tight', pad_inches=0)
#            plt.close()  # Close plot to free resources
#
#            # Send file
#            return send_file(temp_img_path, mimetype='image/png')
#
#        except Exception as e:
#            return f"""
#                <html><body>
#                    <center>
#                        <h2 style='color:red;'>Error</h2>
#                        <h1>An error occurred while processing the image. Please try again.</h1>
#                    </center>
#                    </body></html>"""
#    except Exception as e:
#        return render_template("500_error.html")




@app.route("/tooth_segmentation", methods=["POST"])
def tooth_decay_detection():
    try:
        # Get the uploaded image file
        image_file = request.files.get("bitwing2")
        
        # Check if the file is present
        if not image_file or image_file.filename == '':
            return render_template("error.html", message="No file selected. Please check the information sent from the form and try again.")

        # Save the uploaded file
        upload_path = os.path.join('uploads', image_file.filename)
        image_file.save(upload_path)

        try:
            # Read and preprocess the image
            image = skio.imread(upload_path)
            gray_image = color.rgb2gray(image)  # Convert to grayscale
            
            # Enhance the image for better decay detection
            enhanced_image = filters.gaussian(gray_image, sigma=1.0)  # Smoothing
            
            # Detect decay regions using intensity thresholding
            # Decay regions are typically darker than healthy tooth regions
            decay_threshold = filters.threshold_otsu(enhanced_image)  # Automatic thresholding
            decay_mask = enhanced_image < decay_threshold  # Regions darker than the threshold
            
            # Remove small noise from the decay mask
            decay_mask = morphology.remove_small_objects(decay_mask, min_size=100)
            decay_mask = morphology.remove_small_holes(decay_mask, area_threshold=100)
            
            # Label the decay regions
            labeled_decay = measure.label(decay_mask)
            
            # Analyze decay regions
            decay_regions = []
            for region in measure.regionprops(labeled_decay):
                decay_regions.append({
                    'area': region.area,
                    'centroid': region.centroid,
                    'bounding_box': region.bbox
                })

            # Overlay decay regions on the original image
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            
            # Original Image
            axes[0].imshow(image)
            axes[0].set_title('Original Image')
            axes[0].axis('off')
            
            # Highlight Decay Regions
            axes[1].imshow(image)
            axes[1].imshow(decay_mask, alpha=0.3, cmap='Reds')  # Overlay decay mask
            axes[1].set_title('Decay Detection')
            axes[1].axis('off')

            # Save the plot
            output_path = os.path.join('uploads', f'decay_{image_file.filename}')
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
            plt.close()  # Close plot to free resources
            
            # Send the processed image back to the user
            return send_file(output_path, mimetype='image/png')

        except Exception as e:
            return render_template("error.html", message=f"An error occurred while processing the image: {str(e)}")

    except Exception as e:
        return render_template("500_error.html")






# Absolute path to the CSV file for tooth decay
decay_csv = 'decay.csv'

# Function to load and preprocess tooth decay data
def load_and_preprocess_tooth_data(decay_csv):
    if not os.path.exists(decay_csv):
        raise FileNotFoundError(f"The file {decay_csv} does not exist.")

    data = pd.read_csv(decay_csv)

    # Initialize LabelEncoders for categorical columns
    label_encoders = {}
    for column in ['Diet', 'Saliva_Flow', 'Fluoride_Exposure', 'Dental_History', 'Family_History', 'Socioeconomic']:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le

    # Encode the target variable
    data['decay'] = data['decay'].map({'Yes': 1, 'No': 0})

    X = data.drop('decay', axis=1)
    y = data['decay']

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return label_encoders, scaler, X, y

# Load and preprocess data
label_encoders_tooth, scaler_tooth, X_tooth, y_tooth = load_and_preprocess_tooth_data(decay_csv)

""" The classifier is trained once globally, which optimizes performance by avoiding re-training on each prediction request."""

# Initialize KNN classifier for tooth decay
knn_tooth = KNeighborsClassifier(n_neighbors=3)
knn_tooth.fit(X_tooth, y_tooth)  # Fit KNN classifier using all data


@app.route("/tooth_decay_ml")
def predict_tooth_get():
    return render_template("tooth_decay_detection.html")

"""
This POST route processes form data, applies label encoding and scaling, and uses the trained model to predict the risk.
It calculates the probabilities of both high and low risks, and renders the result.

"""


# Route to predict tooth decay
@app.route('/predict/tooth', methods=['POST'])
def predict_tooth():
    try:
        # Extract data from form submission
        age = int(request.form['age'])  # Assuming 'age' is a numerical field
        diet = request.form['diet']
        saliva_flow = request.form['saliva_flow']
        fluoride_exposure = request.form['fluoride_exposure']
        dental_history = request.form['dental_history']
        family_history = request.form['family_history']
        socioeconomic = request.form['socioeconomic']
        dental_visits = int(request.form['dental_visits'])
        dental_xrays = int(request.form['dental_xrays'])

        # Create a DataFrame with the input data
        input_data = pd.DataFrame({
            'Age': [age],
            'Diet': [diet],
            'Saliva_Flow': [saliva_flow],
            'Fluoride_Exposure': [fluoride_exposure],
            'Dental_History': [dental_history],
            'Family_History': [family_history],
            'Socioeconomic': [socioeconomic],
            'Dental_Visits': [dental_visits],
            'Dental_X-rays': [dental_xrays]
        })

        # Ensure categorical encoding matches what was used during training
        for column in ['Diet', 'Saliva_Flow', 'Fluoride_Exposure', 'Dental_History', 'Family_History', 'Socioeconomic']:
            input_data[column] = label_encoders_tooth[column].transform([input_data[column][0]])  # Transform using the same encoder

        # Scale numerical features using the same scaler
        input_data_scaled = scaler_tooth.transform(input_data)

        # Get prediction probabilities
        proba = knn_tooth.predict_proba(input_data_scaled)
        high_risk_proba = proba[0][1]  # Probability of high risk
        low_risk_proba = proba[0][0]   # Probability of low risk

        # Determine the prediction
        result = 'High Risk' if high_risk_proba > 0.5 else 'Low Risk'

        # Render the result template with prediction data
        return render_template(
            'tooth_decay_prediction_result.html',
            result=result,
            high_risk_proba=high_risk_proba,
            low_risk_proba=low_risk_proba
        )

    except Exception as e:
        return ({'error': str(e)})





@app.route("/about_us")
def about_us():
    return render_template("profile.html")


@app.route("/contact_us")
def contact_us():
    return render_template("contact_us.html")



@app.route("/tables")
def tables_view():
    return render_template("tables.html")


if __name__ == '__main__':
    app.run(debug=True , port = 5005)
