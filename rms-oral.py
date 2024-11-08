import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.filechooser import FileChooserIconView
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.core.window import Window
from kivy.uix.screenmanager import Screen
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.graphics import Color, Rectangle
from kivy.uix.scrollview import ScrollView
from kivy.uix.textinput import TextInput
from kivy.uix.popup import Popup
from kivy.uix.gridlayout import GridLayout
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os
from kivy.uix.image import Image as myimage
from sklearn.preprocessing import OrdinalEncoder
import joblib
from sklearn.preprocessing import MinMaxScaler






# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



Window.size = (1400, 800)

"""Input: RGB images of size 224x224 pixels.
Architecture:

    A single convolutional layer with 16 filters, followed by ReLU and max-pooling.
    A fully connected layer that outputs predictions for two classes.

Activation Function: ReLU (Rectified Linear Unit).
Output: A binary classification result (Normal/Abnormal). """

class SimpleCNN_cancer(nn.Module):
    def __init__(self):
        super(SimpleCNN_cancer, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(16 * 112 * 112, 2)  # Output for 2 classes (Normal, Abnormal)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 16 * 112 * 112)  # Flatten the output
        x = self.fc1(x)
        return x


"""
The SimpleCNNOralDisease model is designed for multi-class classification of 12 different oral diseases. 
It has three convolutional layers, each followed by a max-pooling layer, and two fully connected layers. """

# SimpleCNN Model Definition
class SimpleCNNOralDisease(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNNOralDisease, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 16 * 16)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x






"""
Input: RGB images resized to 128x128 pixels.
Architecture:

    Three convolutional layers with increasing filters (16, 32, 64).
    Max pooling after each convolution to reduce spatial dimensions.
    Two fully connected layers with ReLU activations.

Output: Probability scores for 12 different oral diseases. """

# Load the model and set it to evaluation mode
oral_disease_model = SimpleCNNOralDisease(num_classes=12)
oral_disease_model.load_state_dict(torch.load('oral_disease_model.pth', map_location=device))
oral_disease_model.eval()



""" 
The SimpleCNN_cancer model is a simple CNN architecture for binary classification of oral cancer. It takes an input image,
processes it through convolutional and pooling layers, and finally outputs a prediction for two classes (Normal or Abnormal)
"""
cancer_model = SimpleCNN_cancer().to(device)
cancer_model.load_state_dict(torch.load('histopathologic_oral_cancer_model.pth', map_location=device))
cancer_model.eval()


# Transform for input image
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])




# Transform for input image for cancer detection
cancer_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


# Define class names for the 12 classes of oral diseases
oral_disease_class_names = [
    'denture_stomatitis', 'desquamative_gingivitis', 'epulis_fissuratum', 
    'erosive_and_ulcerative_oral_lichen_planus', 'erythroplakia', 'geographic_tongue', 
    'leukoplakia', 'lichenoid_reaction', 'plaque_type_oral_lichen_planus', 
    'reticular_and_papular_oral_lichen_planus', 'squamous_cell_carcinoma', 'traumatic_ulcer'
]

# Kivy Layout for Sidebar Menu
class SideBar(BoxLayout):
    def __init__(self, app, **kwargs):
        super(SideBar, self).__init__(**kwargs)
        self.orientation = 'vertical'
        self.size_hint_x = None
        self.width = '200dp'
        self.spacing = 20
        self.padding = 20
        Color(0.1, 0.14, 0.13)

        # Add sidebar buttons with navigation
        self.add_widget(Button(text='Home', size_hint_y=None, height='50dp',
                               background_color =  [0, 1, 1, 0.5], 
                               color=(1, 1, 1, 1), 
                               on_press=lambda x: app.change_screen('home')))
        self.add_widget(Button(text='About', size_hint_y=None, height='50dp', 
                               background_color =  [0, 1, 1, 0.5], 
                               color=(1, 1, 1, 1), 
                               on_press=lambda x: app.change_screen('about')))
        self.add_widget(Button(text='Contact', size_hint_y=None, height='50dp', 
                               background_color =  [0, 1, 1, 0.5], 
                               color=(1, 1, 1, 1), 
                               on_press=lambda x: app.change_screen('contact')))

# File Manager Screen
IMG_SIZE = (64, 64)
class FileManagerScreen(Screen):
    def __init__(self, **kwargs):
        super(FileManagerScreen, self).__init__(**kwargs)
        layout = BoxLayout(orientation='horizontal')

        # Sidebar
        self.side_bar = SideBar(App.get_running_app())
        layout.add_widget(self.side_bar)

        # File chooser for selecting image
        content_layout = BoxLayout(orientation='vertical', padding=20, spacing=20)
        self.file_chooser = FileChooserIconView()
        self.file_chooser.size_hint = (1, 0.8)
        content_layout.add_widget(self.file_chooser)
        Color(0.1, 0.14, 0.13)

        # Upload button
        self.upload_button = Button(text="Upload and Predict", size_hint=(1, 0.1) ,  background_color =  [0, 1, 1, 0.5])
        self.upload_button.bind(on_press=self.upload_and_predict)
        content_layout.add_widget(self.upload_button)

        # TextInput to display prediction response
        self.result_box = TextInput(text="", readonly=True, size_hint=(1, 0.1))
        self.result_box.foreground_color = (1, 0, 0, 1)  # Set initial text color to red
        content_layout.add_widget(self.result_box)

        layout.add_widget(content_layout)
        self.add_widget(layout)

    def upload_and_predict(self, instance):
        selected_file = self.file_chooser.selection
        if selected_file:
            file_path = selected_file[0]
            try:
                # Load and process the image
                image = Image.open(file_path)
                image = image.resize(IMG_SIZE)  # Resize to 64x64
                image = transform(image).unsqueeze(0)  # Add batch dimension

                # Move image to the same device as the model (CPU or GPU)
                image = image.to(torch.device('cpu'))  # or .to(device) if using GPU

                # Set the model to evaluation mode
                oral_disease_model.eval()

                # Make prediction with PyTorch model
                with torch.no_grad():
                    output = oral_disease_model(image)  # Forward pass through the model
                    _, predicted = torch.max(output, 1)  # Get the predicted class

                predicted_class = oral_disease_class_names[predicted.item()]  # Map the predicted index to class name

                # Display result
                self.result_box.text = f"Predicted Oral Disease: {predicted_class}"

            except Exception as e:
                self.result_box.text = f"Error: {str(e)}"
        else:
            self.result_box.text = "Please select a file first."




class CancerDetectionScreen(Screen):
    def __init__(self, **kwargs):
        super(CancerDetectionScreen, self).__init__(**kwargs)
        layout = BoxLayout(orientation='horizontal')

        self.side_bar = SideBar(App.get_running_app())
        layout.add_widget(self.side_bar)

        # File chooser for selecting image
        content_layout = BoxLayout(orientation='vertical', padding=20, spacing=20)
        self.file_chooser = FileChooserIconView()
        self.file_chooser.size_hint = (1, 0.8)
        content_layout.add_widget(self.file_chooser)

        # Upload button
        self.upload_button = Button(text="Upload and Predict", size_hint=(1, 0.1) ,background_color =  [0, 1, 1, 0.5])
        self.upload_button.bind(on_press=self.upload_and_predict)
        content_layout.add_widget(self.upload_button)

        # TextInput to display prediction response
        self.result_box = TextInput(text="", readonly=True, size_hint=(1, 0.1))
        self.result_box.foreground_color = (1, 0, 0, 1)  # Set initial text color to red
        content_layout.add_widget(self.result_box)

        layout.add_widget(content_layout)
        self.add_widget(layout)

    def upload_and_predict(self, instance):
        selected_file = self.file_chooser.selection
        if selected_file:
            file_path = selected_file[0]
            try:
                # Load and process the image
                image = Image.open(file_path)
                image = cancer_transform(image).unsqueeze(0).to(device)

                # Model prediction
                with torch.no_grad():
                    outputs = cancer_model(image)
                    _, predicted = torch.max(outputs, 1)

                # Map predicted index to class name
                class_names = ['Normal', 'Abnormal']
                predicted_class = class_names[predicted.item()]
                self.result_box.text = f"Predicted: {predicted_class}"

            except Exception as e:
                self.result_box.text = f"Error: {str(e)}"
        else:
            self.result_box.text = "Please select a file first."






# Function to load the trained model



def load_model_oralml():
    try:
        # Use joblib to load the model, since it was saved with joblib
        model = joblib.load('oral_cancer_ml_random_forest_model.pkl')
        if hasattr(model, 'predict'):
            print("Model loaded successfully.")
            return model
        else:
            print("The model is not trained.")
            raise ValueError("The model is not trained.")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise


def parse_input(input_str):
    try:
        # Safely parse numeric input
        return float(input_str)  # For sizes or age as decimal numbers
    except ValueError:
        raise ValueError(f"Invalid input format: {input_str}")  # More informative error message










        

class CancerDetectionScreen_ML(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.build_ui()

        main_layout = BoxLayout(orientation='horizontal')




        # Initialize LabelEncoders
        self.age_scaler = joblib.load('age_scaler.pkl')

        # Fit LabelEncoders on predefined categories
        self.le_gender = OrdinalEncoder(categories=[['female', 'male']], handle_unknown='use_encoded_value', unknown_value=-1)
        self.le_gender.fit([['female'], ['male']])  # Fit on the gender categories

        self.le_location = OrdinalEncoder(categories=[['gingiva', 'ventral surface of tongue', 'hard palate', 'floor of the mouth', 'dorsal surface of tongue', 'lateral border of tongue']], handle_unknown='use_encoded_value', unknown_value=-1)
        self.le_location.fit([['gingiva'], ['ventral surface of tongue'], ['hard palate'], ['floor of the mouth'], ['dorsal surface of tongue'], ['lateral border of tongue']])  # Fit on location categories

        self.le_size = OrdinalEncoder(categories=[['2*3', '2*4', '3*3', '3*4', '3*2', '4*2']], handle_unknown='use_encoded_value', unknown_value=-1)
        self.le_size.fit([['2*3'], ['2*4'], ['3*3'], ['3*4'], ['3*2'], ['4*2']])  # Fit on size categories

        self.le_color = OrdinalEncoder(categories=[['red', 'white', 'pink', 'red-white']], handle_unknown='use_encoded_value', unknown_value=-1)
        self.le_color.fit([['red'], ['white'], ['pink'], ['red-white']])  # Fit on color categories

        self.le_surface = OrdinalEncoder(categories=[['Verrucous', 'Granuler', 'Ulcerated', 'Disfigured', 'papillomatosis', 'plaque', 'intact']], handle_unknown='use_encoded_value', unknown_value=-1)
        self.le_surface.fit([['Verrucous'], ['Granuler'], ['Ulcerated'], ['Disfigured'], ['papillomatosis'], ['plaque'], ['intact']])  # Fit on surface categories

        self.le_texture = OrdinalEncoder(categories=[['hard', 'firm', 'soft']], handle_unknown='use_encoded_value', unknown_value=-1)
        self.le_texture.fit([['hard'], ['firm'], ['soft']])  # Fit on texture categories

        self.le_lnh_involvement = OrdinalEncoder(categories=[['yes', 'no']], handle_unknown='use_encoded_value', unknown_value=-1)
        self.le_lnh_involvement.fit([['yes'], ['no']])  # Fit on lymph node involvement categories

        self.le_lnh_location = OrdinalEncoder(categories=[['submandibular', 'neck', 'parotid', 'None']], handle_unknown='use_encoded_value', unknown_value=-1)
        self.le_lnh_location.fit([['submandibular'], ['neck'], ['parotid'], ['None']])  # Fit on lymph node location categories

        self.le_lnh_side = OrdinalEncoder(categories=[['same as the lesion', 'Other side', 'Bilateral']], handle_unknown='use_encoded_value', unknown_value=-1)
        self.le_lnh_side.fit([['same as the lesion'], ['Other side'], ['Bilateral']])  # Fit on lymph node side categories

        self.le_lnh_texture = OrdinalEncoder(categories=[['hard', 'firm']], handle_unknown='use_encoded_value', unknown_value=-1)
        self.le_lnh_texture.fit([['hard'], ['firm']])  # Fit on lymph node texture categories

        self.le_lnh_tenderness = OrdinalEncoder(categories=[['hard', 'firm']], handle_unknown='use_encoded_value', unknown_value=-1)
        self.le_lnh_tenderness.fit([['hard'], ['firm']])  # Fit on lymph node tenderness categories

        self.le_lnh_mobility = OrdinalEncoder(categories=[['fixed', 'mobile']], handle_unknown='use_encoded_value', unknown_value=-1)
        self.le_lnh_mobility.fit([['fixed'], ['mobile']])  # Fit on lymph node mobility categories

        self.le_smoking = OrdinalEncoder(categories=[['pos', 'neg']], handle_unknown='use_encoded_value', unknown_value=-1)
        self.le_smoking.fit([['pos'], ['neg']])  # Fit on smoking categories

        self.le_alcohol = OrdinalEncoder(categories=[['pos', 'neg']], handle_unknown='use_encoded_value', unknown_value=-1)
        self.le_alcohol.fit([['pos'], ['neg']])  # Fit on alcohol categories

        # Fit the result encoder
        # self.result_encoder = OrdinalEncoder(categories=[['No cancer', 'Pre-cancerous', 'squamous cell carcinoma']])
        # self.result_encoder.fit([['No cancer'], ['Pre-cancerous'], ['squamous cell carcinoma']])  # Fit on result categories



        
        
        # Label encoder for decoding prediction result


    def build_ui(self):
        main_layout = BoxLayout(orientation='vertical', padding=0.2, spacing=0.2)


        self.header_label = Label(text="Oral Disease Detector with ML", font_size='28sp', color="#ffffff", size_hint=(1, 0.2))
        main_layout.add_widget(self.header_label)


        # Result label
        self.result_label = Label(
            text="Prediction Result: ",
            size_hint=(1, None),
            height='80dp',
            font_size='20sp',  # Increased font size
            color=(1, 0.647, 0)  # Set color to orange (RGBA format)
        )

        main_layout.add_widget(self.result_label)

        # ScrollView to make the form scrollable
        scroll_view = ScrollView(size_hint=(1, None), size=(Window.width, Window.height - 150))

        # Input grid layout inside the ScrollView
        input_grid = GridLayout(cols=2, spacing=15, size_hint_y=None, row_force_default=True, row_default_height=50)
        input_grid.bind(minimum_height=input_grid.setter('height'))
        navbar = BoxLayout(orientation='horizontal', size_hint_y=None, height='50dp', spacing=10, padding=10)
        button_home = Button(
            text="Home",
            font_size='18sp',
            size_hint=(None, 1),
            width=100,
            background_color=[0, 1, 1, 0.5],
            color=(1, 1, 1, 1),
            on_press=lambda x: App.get_running_app().change_screen('home')
        )
        navbar.add_widget(button_home)

        # Define input fields
        self.inputs = {
            'age': TextInput(hint_text='Age (e.g., 50)', multiline=False, size_hint=(None, None), size=(1000, 50)),
            'gender': TextInput(hint_text='Gender (male/female)', multiline=False, size_hint=(None, None), size=(1000, 50)),
            'location': TextInput(hint_text='lateral border of tongue, gingiva, buccal mucosa, palate, ventral surface of tongue , hard palate,floor of the mouth ,dorsal surface of tongue', multiline=False, size_hint=(None, None), size=(1000, 50)),
            'size': TextInput(hint_text='Size (Just Give Number Like 2 for : 2cm²)', multiline=False, size_hint=(None, None), size=(1000, 50)),
            'color': TextInput(hint_text='Color (red, white, red-white, pink)', multiline=False, size_hint=(None, None), size=(1000, 50)),
            'surface': TextInput(hint_text='Surface (Verrucous , Granuler , Ulcerated , Disfigured , papillomatosis , Disfigured , plaque , intact)', multiline=False, size_hint=(None, None), size=(1000, 50)),
            'texture': TextInput(hint_text='Texture (hard, soft, firm)', multiline=False, size_hint=(None, None), size=(1000, 50)),
            'lymphnode_involvement': TextInput(hint_text='Lymph Node Involvement (yes/no)', multiline=False, size_hint=(None, None), size=(1000, 50)),
            'lymphnode_location': TextInput(hint_text='Lymph Node Location (e.g., submandibular, neck)', multiline=False, size_hint=(None, None), size=(1000, 50)),
            'lymphnode_side': TextInput(hint_text='Lymph Node Side (same as the lesion, Other side, Bilateral)', multiline=False, size_hint=(None, None), size=(1000, 50)),
            'lymphnode_texture': TextInput(hint_text='Lymph Node Texture (firm, hard)', multiline=False, size_hint=(None, None), size=(1000, 50)),
            'lymphnode_tenderness': TextInput(hint_text='Lymph Node Tenderness (yes/no)', multiline=False, size_hint=(None, None), size=(1000, 50)),
            'lymphnode_mobility': TextInput(hint_text='Lymph Node Mobility (fixed, mobile)', multiline=False, size_hint=(None, None), size=(1000, 50)),
            'smoking': TextInput(hint_text='Smoking (pos/neg)', multiline=False, size_hint=(None, None), size=(1000, 50)),
            'alcohol': TextInput(hint_text='Alcohol (pos/neg)', multiline=False, size_hint=(None, None), size=(1000, 50)),
        }

        # Add labels and input fields to the grid
        for label_text, input_field in self.inputs.items():
            label = Label(text=label_text.capitalize(), size_hint=(None, None), size=(180, 40))
            input_grid.add_widget(label)
            input_grid.add_widget(input_field)

        scroll_view.add_widget(input_grid)
        main_layout.add_widget(scroll_view)

        # Prediction button
        self.predict_button = Button(text='Predict', size_hint=(1, None), height='60dp', font_size='20sp' , background_color =  [0, 1, 1, 0.5] , color=(1, 0.647, 0) )
        self.predict_button.bind(on_press=self.predict_cancer)
        main_layout.add_widget(self.predict_button)
        main_layout.add_widget(navbar)

                # Add sidebar buttons with navigation





        self.add_widget(main_layout)




    def predict_cancer(self, instance):
        # Check if any input is empty
        for field, input_field in self.inputs.items():
            if not input_field.text.strip():
                # If a field is empty, display an error message
                self.result_label.text = f"Error: {field.capitalize()} cannot be empty."
                return

        try:
            # Collect and transform input data into a dictionary
            input_data = {
                'age': self.age_scaler.transform([[parse_input(self.inputs['age'].text) if self.inputs['age'].text else 0]])[0, 0],
                'gender': self.le_gender.transform([[self.inputs['gender'].text.lower() if self.inputs['gender'].text else 'female']])[0, 0],
                'location': self.le_location.transform([[self.inputs['location'].text.lower() if self.inputs['location'].text else 'gingiva']])[0, 0],
                'size': self.age_scaler.transform([[parse_input(self.inputs['size'].text) if self.inputs['size'].text else 0]])[0, 0],
                'color': self.le_color.transform([[self.inputs['color'].text.lower() if self.inputs['color'].text else 'red']])[0, 0],
                'surface': self.le_surface.transform([[self.inputs['surface'].text.lower() if self.inputs['surface'].text else 'verrucous']])[0, 0],
                'texture': self.le_texture.transform([[self.inputs['texture'].text.lower() if self.inputs['texture'].text else 'hard']])[0, 0],
                'lymphnode_involvment': self.le_lnh_involvement.transform([[self.inputs['lymphnode_involvement'].text.lower() if self.inputs['lymphnode_involvement'].text else 'no']])[0, 0],
                'lymphnode_location': self.le_lnh_location.transform([[self.inputs['lymphnode_location'].text.lower() if self.inputs['lymphnode_location'].text else 'none']])[0, 0],
                'lymphnode_side': self.le_lnh_side.transform([[self.inputs['lymphnode_side'].text.lower() if self.inputs['lymphnode_side'].text else 'same as the lesion']])[0, 0],
                'lymphnode_texture': self.le_lnh_texture.transform([[self.inputs['lymphnode_texture'].text.lower() if self.inputs['lymphnode_texture'].text else 'firm']])[0, 0],
                'lymphnode_tenderness': self.le_lnh_tenderness.transform([[self.inputs['lymphnode_tenderness'].text.lower() if self.inputs['lymphnode_tenderness'].text else 'firm']])[0, 0],
                'lymphnode_mobility': self.le_lnh_mobility.transform([[self.inputs['lymphnode_mobility'].text.lower() if self.inputs['lymphnode_mobility'].text else 'fixed']])[0, 0],
                'smoking': self.le_smoking.transform([[self.inputs['smoking'].text.lower() if self.inputs['smoking'].text else 'neg']])[0, 0],
                'alcohol': self.le_alcohol.transform([[self.inputs['alcohol'].text.lower() if self.inputs['alcohol'].text else 'neg']])[0, 0]
            }
            
            # Convert the dictionary to a DataFrame with the correct feature names
            input_df = pd.DataFrame([input_data])
            
            # Further code to make the prediction...
            
        except Exception as e:
            # Handle any other errors that may occur during prediction
            self.result_label.text = f"Error: {str(e)}"












class MyApp(App):
    def build(self):
        sm = ScreenManager()
        sm.add_widget(CancerDetectionScreen_ML(name="oral_cancer_ml_screen"))
        return sm




class HomeScreen(Screen):
    def __init__(self, **kwargs):
        super(HomeScreen, self).__init__(**kwargs)


        # Create a FloatLayout for the background color
        root_layout = FloatLayout()

        # Background color using Canvas instructions
        with root_layout.canvas.before:
            Color(0.1, 0.14, 0.13)  # Dark gray background color
            self.rect = Rectangle(size=root_layout.size, pos=root_layout.pos)

        # Bind the size and position of the rectangle to the layout
        root_layout.bind(size=self.update_rect, pos=self.update_rect)

        # Main layout for the content
        main_layout = BoxLayout(orientation='horizontal')

        # Sidebar for navigation menu
        sidebar = BoxLayout(orientation='vertical', size_hint=(0.25, 1), padding=20, spacing=20)
        sidebar.canvas.before.clear()
        with sidebar.canvas.before:
            Color(0.2, 0.2, 0.2, 1)  # Sidebar background color
            self.sidebar_rect = Rectangle(size=sidebar.size, pos=sidebar.pos)

        # Sidebar buttons with larger size
        brand_image = myimage(source='opensuse-1737766358.png', size_hint_x=None, height=100 , width=200 , on_press=lambda x: App.get_running_app().change_screen('home'))
       
        button_brand = Button(
            text="KYGnus",
            size_hint_y=None,
            height=80,  # Increased height for larger buttons
            background_color =  [0, 1, 1, 0.5],
            color=(1, 1, 1, 1),
            font_size='16sp',
            on_press=lambda x: App.get_running_app().change_screen('home')
        )
        button_oral_cancer = Button(
            text="Histopathologic Oral Cancer",
            size_hint_y=None,
            height=80,  # Increased height for larger buttons
            background_color =  [0, 1, 1, 0.5],
            color=(1, 1, 1, 1),
            font_size='16sp',
            on_press=lambda x: App.get_running_app().change_screen('cancer_detection')
        )
        button_oral_disease = Button(
            text="Oral Disease Detection",
            size_hint_y=None,
            height=80,  # Increased height for larger buttons
            background_color =  [0, 1, 1, 0.5],
            color=(1, 1, 1, 1),
            font_size='16sp',
            on_press=lambda x: App.get_running_app().change_screen('file_manager')
        )
        button_oral_cancer_ml = Button(
            text="Oral Cancer Detection with ML",
            size_hint_y=None,
            height=80,  # Increased height for larger buttons
            background_color =  [0, 1, 1, 0.5],
            color=(1, 1, 1, 1),
            font_size='16sp',
            on_press=lambda x: App.get_running_app().change_screen('cancer_detection_ml')
        )
        button_about = Button(
            text="About Us",
            size_hint_y=None,
            height=80,  # Increased height for larger buttons
            background_color =  [0, 1, 1, 0.5],
            color=(1, 1, 1, 1),
            font_size='16sp',
            on_press=lambda x: App.get_running_app().change_screen('about')
        )
        button_contact = Button(
            text="Contact Us",
            size_hint_y=None,
            height=80,  # Increased height for larger buttons
            background_color =  [0, 1, 1, 0.5],
            color=(1, 1, 1, 1),
            font_size='16sp',
            on_press=lambda x: App.get_running_app().change_screen('contact')
        )
        button_home = Button(
            text="Home",
            size_hint_y=None,
            height=80,  # Increased height for larger buttons
            background_color =  [0, 1, 1, 0.5],
            color=(1, 1, 1, 1),
            font_size='16sp',
            on_press=lambda x: App.get_running_app().change_screen('home')
        )

        # Adding buttons to sidebar
        sidebar.add_widget(brand_image)
        sidebar.add_widget(button_brand)
        sidebar.add_widget(button_oral_cancer)
        sidebar.add_widget(button_oral_disease)
        sidebar.add_widget(button_oral_cancer_ml)
        sidebar.add_widget(button_about)
        sidebar.add_widget(button_contact)
        sidebar.add_widget(button_home)

        # Main content layout
        content_layout = BoxLayout(orientation='vertical', padding=[20, 30, 20, 30], spacing=15)

        # Main title
        self.header_label1 = Label(
            text="KYGnus",
            font_size='48sp',
            color=(1, 1, 1, 1),
            size_hint=(1, 0.15),
            bold=True
        )

        # Subtitle for the application purpose
        self.header_label = Label(
            text="Oral Disease Detector",
            font_size='36sp',
            color=(1, 1, 1, 1),
            size_hint=(1, 0.15),
            bold=True
        )

        # Additional subtitle with a softer color for contrast
        self.header_label2 = Label(
            text="Powered by AI and Advanced Medical Imaging",
            font_size='24sp',
            color=(0.8, 0.8, 0.8, 1),
            size_hint=(1, 0.1)
        )
        self.header_image = myimage(
            source='opg.png',  # Replace with your image file path
            size_hint=(1, 1),  # Adjust size hint as necessary
            allow_stretch=True  # Enable to scale image if needed
        )
        # Informational section - Overview
        self.info_label = Label(
            text="Integrating openSUSE with medical software creates a powerful and flexible environment tailored to the healthcare sector"
            "This combination leverages openSUSE’s stability and extensive software repositories to seamlessly develop and integrate medical applications, providing professionals with cutting-edge tools and a secure operating system."
            "This version has tried to have a new space for desktop users."
            "Using XFCE desktops and Plank and new icons can give the user a new Fresh space.",
            font_size='18sp',
            color=(0.9, 0.9, 0.9, 1),
            size_hint=(1, 0.25),
            halign='center',
            valign='middle'
        )
        self.info_label.bind(size=self.info_label.setter('text_size'))  # Center-align text within the label
        
        # Benefits section
        self.benefits_label = Label(
            text="[b]openSUSE Medical:[/b]\n•  RMS: Research Medical Software\n• Gmi: GNU Medical Imaging\n• Dicom Scanner\n• Dicom Optimizer\n• SUSE Package Manager\n• and More ...  "
                "Enhanced patient outcomes\n• User-friendly interface",
            markup=True,
            font_size='18sp',
            color=(0.7, 0.9, 0.7, 1),
            size_hint=(1, 0.2),
            halign='left',
            valign='top'
        )
        self.benefits_label.bind(size=self.benefits_label.setter('text_size'))

        # Adding components to content layout
        content_layout.add_widget(self.header_label1)
        content_layout.add_widget(self.header_label)
        content_layout.add_widget(self.header_label2)
        content_layout.add_widget(self.header_image)
        content_layout.add_widget(self.info_label)
        content_layout.add_widget(self.benefits_label)

        # Adding content layout to the main layout
        main_layout.add_widget(sidebar)  # Add sidebar to the main layout
        main_layout.add_widget(content_layout)  # Add content layout to the main layout
        root_layout.add_widget(main_layout)  # Add main layout to the root layout
        self.add_widget(root_layout)  # Add the root layout to the screen
    
    def update_rect(self, *args):
        self.rect.size = self.size
        self.rect.pos = self.pos

# Sample App Class
class MyApp(App):
    def build(self):
        return HomeScreen()





# About and Contact Screens (unchanged)
class AboutScreen(Screen):
    def __init__(self, **kwargs):
        super(AboutScreen, self).__init__(**kwargs)
        
        # Create the main layout with horizontal orientation
        layout = BoxLayout(orientation='horizontal')

        # Sidebar
        self.side_bar = SideBar(App.get_running_app())
        layout.add_widget(self.side_bar)

        # Content layout for the 'About Us' text
        content_layout = BoxLayout(orientation='vertical', padding=30, spacing=20)

        # Title label
        title_label = Label(text="About Us", font_size='32sp', size_hint=(1, 0.1), color=(0.2, 0.6, 0.8, 1))
        content_layout.add_widget(title_label)

        # About text wrapped in a ScrollView to make it scrollable for long content
        about_text = (
            "About the Application:\n\n"
            "This application is designed to diagnose oral and dental diseases using deep learning techniques.\n"
            "It leverages a Convolutional Neural Network (CNN) model trained on a variety of oral disease images\n"
            "to provide accurate predictions. The collection and organization of data was done by Dr. Ketayoun Katbi,\n"
            "who also contributed as a domain expert in this project.\n\n"
            "Key Features:\n"
            "- Portable\n"
            "- Easy to Use\n"
            "- Real-time predictions of oral diseases with high accuracy\n\n"
            "Technology Stack:\n"
            "- The application is built using Kivy for the UI, PyTorch for deep learning, and PIL for image processing.\n\n"
            "License:\n"
            "This software is released under the MIT License, allowing for flexibility in usage and modification.\n\n"
            "Project Repository:\n"
            "You can follow US project through the following links:\n\n"
            "> https://https://kooshayeganeh.github.io\n"
            "> https://github.com/KooshaYeganeh\n"
            "> https://gitlab.com/katayoun_katebi\n"
            "> Your contributions and feedback are welcome!"
        )

        # ScrollView for the About text
        scroll_view = ScrollView(size_hint=(1, 0.9))
        about_label = Label(text=about_text, font_size='18sp', size_hint_y=None, color=(1, 1, 1, 1))  # Changed text color to white
        about_label.bind(texture_size=about_label.setter('size'))
        scroll_view.add_widget(about_label)

        content_layout.add_widget(scroll_view)
        layout.add_widget(content_layout)

        # Add layout to the screen
        self.add_widget(layout)

        # Background color using Canvas instructions
        with self.canvas.before:
            Color(0.1, 0.14, 0.13)  # This is black in RGB
            self.rect = Rectangle(size=self.size, pos=self.pos)
            self.bind(size=self.update_rect, pos=self.update_rect)

    def update_rect(self, *args):
        self.rect.pos = self.pos
        self.rect.size = self.size

class MyApp(App):
    def build(self):
        sm = ScreenManager()
        sm.add_widget(AboutScreen(name='about'))
        return sm

class ContactScreen(Screen):
    def __init__(self, **kwargs):
        super(ContactScreen, self).__init__(**kwargs)
        layout = BoxLayout(orientation='horizontal')

        # Sidebar
        self.side_bar = SideBar(App.get_running_app())
        layout.add_widget(self.side_bar)
        Color(0.1, 0.14, 0.13)

        content_layout = BoxLayout(orientation='vertical', padding=20, spacing=20)
        content_layout.add_widget(Label(text="Contact Us", font_size='24sp', size_hint=(1, 0.1)))

        # Display contact information
        contact_info = "KYGnus \n\n ➜ Contact Us:\n\n ➜ website : https://kooshayeganeh.github.io \n\n ➜ Github : https://github.com/KooshaYeganeh \n\n ➜ Domain Expert and Consultant: k_katebi@yahoo.co.uk\n\n ➜ Developer and Maintainer : kooshakooshadv@gmail.com"
        content_layout.add_widget(Label(text=contact_info, font_size='18sp', size_hint=(1, 0.2)))

        layout.add_widget(content_layout)
        self.add_widget(layout)

# Main Kivy App
class OralPredictorApp(App):
    def build(self):
        self.screen_manager = ScreenManager()
        
        # Set the window size
        Color(0.1, 0.14, 0.13)
        # Center the window on the screen
        self.center_window()

        # Add screens to the screen manager
        self.screen_manager.add_widget(HomeScreen(name='home'))
        self.screen_manager.add_widget(CancerDetectionScreen(name='cancer_detection'))
        self.screen_manager.add_widget(CancerDetectionScreen_ML(name='cancer_detection_ml'))
        self.screen_manager.add_widget(FileManagerScreen(name='file_manager'))
        self.screen_manager.add_widget(AboutScreen(name='about'))
        self.screen_manager.add_widget(ContactScreen(name='contact'))

        return self.screen_manager  # Return only the screen manager

    # Function to switch between screens
    def change_screen(self, screen_name):
        self.screen_manager.current = screen_name

    def center_window(self):
        # Get the screen size
        screen_width = Window.system_size[1]
        screen_height = Window.system_size[1]

        # Get the window size
        window_width = Window.size[1]
        window_height = Window.size[1]

        # Calculate the x and y position to center the window
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2

        # Set the new position
        Window.left = x
        Window.top = y


# Run the app
if __name__ == '__main__':
    OralPredictorApp().run()
    
    
