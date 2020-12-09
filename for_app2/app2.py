import streamlit as st
import pandas as pd
from keras import models
from keras.models import model_from_json
from PIL import Image, ImageOps
import numpy as np
from keras.preprocessing import image

st.write('''# Welcome to the plant disease predictor!''')

st.write(
'''
The PlantVillage dataset was used to train a neural network.

'''
)

link = '[Click here for the dataset](https://data.mendeley.com/datasets/tywbtsjrjv/1)'


st.markdown(link, unsafe_allow_html = True)

@st.cache
def get_data(csv):
    return pd.read_csv(str(csv))

if __name__ == "__main__":
    # load the csv
    df = get_data('summaries.csv')

if __name__ == "__main__":
    # load the csv
    df2 = get_data('recommendations.csv')

@st.cache(allow_output_mutation=True)
def load_model():
    model_weights = "DenseNet169_weights.h5"
    model_json = "DenseNet169.json"
    with open(model_json) as json_file:
        loaded_model = model_from_json(json_file.read())
    loaded_model.load_weights(model_weights)
    loaded_model.summary()  # included to make it visible when model is reloaded
    return loaded_model


if __name__ == "__main__":
    # load the saved model
    model = load_model()

st.write('Neural network loaded. This model has a accuracy of 98.7%.')
st.write('List of diseases it could predict:')
list_names = []
for item in df2['Label']:
	item = item.replace('___',': ').replace('_', ' ')
	list_names.append(item)
st.write(list_names)

# Code adapted from https://www.pluralsight.com/guides/deploying-image-classification-on-the-web-with-streamlit-and-heroku
def predict(img, model_name):
	data = np.ndarray(shape = (1,150,150,3), dtype = np.float32)
	image = img
	size = (150,150)
	image = ImageOps.fit(image,size, Image.ANTIALIAS)

	image_array = np.asarray(image)
	normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
	data[0] = normalized_image_array
	prediction = model.predict(data)
	return np.argmax(prediction)

plant_disease = {0:'Apple___Apple_scab', 1: 'Apple___Black_rot', 2:'Apple___Cedar_apple_rust', 3:'Apple___healthy', 4:'Background_without_leaves', 5:'Blueberry___healthy', 6:'Cherry___Powdery_mildew', 7:'Cherry___healthy', 8:'Corn___Cercospora_leaf_spot Gray_leaf_spot', 9:'Corn___Common_rust', 10:'Corn___Northern_Leaf_Blight', 11:'Corn___healthy', 12:'Grape___Black_rot', 13:'Grape___Esca_(Black_Measles)', 14:'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 15:'Grape___healthy', 16:'Orange___Haunglongbing_(Citrus_greening)', 17:'Peach___Bacterial_spot', 18:'Peach___healthy', 19:'Pepper,_bell___Bacterial_spot', 20:'Pepper,_bell___healthy', 21:'Potato___Early_blight', 22:'Potato___Late_blight', 23:'Potato___healthy', 24:'Raspberry___healthy', 25:'Soybean___healthy', 26:'Squash___Powdery_mildew', 27:'Strawberry___Leaf_scorch', 28:'Strawberry___healthy', 29:'Tomato___Bacterial_spot', 30:'Tomato___Early_blight', 31:'Tomato___Late_blight', 32:'Tomato___Leaf_Mold', 33:'Tomato___Septoria_leaf_spot', 34:'Tomato___Spider_mites Two-spotted_spider_mite', 35:'Tomato___Target_Spot', 36:'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 37:'Tomato___Tomato_mosaic_virus', 38:'Tomato___healthy'}


file = st.file_uploader("Please upload an image file", type=["jpg","jpeg", "png"])
if file is not None:
	img = Image.open(file)
	st.image(img, caption='Uploaded Image.', use_column_width=False)
	if st.button('Predict '):
		label = predict(img, model)
		label = plant_disease[label]
		name = label.replace('___'," ").replace('_',' ')
		st.write("Prediction: " + name)
		try:
			input_1 = df[df['Label'] == label]
			summary = input_1['Summary'].iloc[0]
			st.write("## Summary of disease")
			st.write(summary)
			try:
				input_2 = df2[df2['Label']==label]
				recommendation = input_2['Recommendation'].iloc[0]
				st.write("## Recommendation to control the disease")
				st.write(recommendation)
				st.write("### Information obtained from Wikipedia.")
			except:
				pass
		except IndexError:
			pass





# if label == df['label']:
# 	st.write(df.label['recommendation'])

