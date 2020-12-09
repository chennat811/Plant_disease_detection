# Plant Disease Detector
A neural network that recognizes plant disease
### [Heroku App Link](http://plant-disease-detector-88.herokuapp.com/)
### Project 5 at Metis

### The goal of this project is use transfer learning to create a convolutional neural network that could predict plant disease.
Data obtained from the [PlantVillage dataset](https://data.mendeley.com/datasets/tywbtsjrjv/1)

Contents:
1. [PART 1 Cleaning and EDA](https://github.com/chennat811/Plant_disease_detection/blob/main/Project5_PART1_SplitData_Clean_EDA.ipynb)
2. [PART 2 First pass: Densenet training](https://github.com/chennat811/Plant_disease_detection/blob/main/Project5-PART2_First_pass_Densenet.ipynb)
3. [PART 3 Second pass: Densenet training](https://github.com/chennat811/Plant_disease_detection/blob/main/Project5_PART3_Second_Pass_Densenet.ipynb)
4. [PART 4 Third pass: Densenet training](https://github.com/chennat811/Plant_disease_detection/blob/main/Project5_PART4_Third_Pass_Densenet.ipynb)
5. [PART 5 Scraping wikipedia](https://github.com/chennat811/Plant_disease_detection/blob/main/Project5_PART5__Scrape_summaries_recommendations_Wiki.ipynb)
6. Used [helper functions](https://github.com/chennat811/Plant_disease_detection/blob/main/utilities.py)
7. Created a [Streamlit app](https://github.com/chennat811/Plant_disease_detection/blob/main/for_app2/app2.py) Deployed on [Heroku](http://plant-disease-detector-88.herokuapp.com/)

### Background
Plant infectious diseases are affecting crop yields all over the world. An average of 40% yield losses are experienced when a disease hits and in developing worlds they can experience up to 100% yield loss. It is being exacerbated by climate change because some pathogens favor warmer temperatures which helps them propagate more quickly.

As you can see in the image grid below, plant diseases are very hard to identify with the human eye.
![Imagegrid](https://github.com/chennat811/Plant_disease_detection/blob/main/image_grid.jpeg)
However, computer vision has been developing rapidly over the years. Farmers are now able to utilize these tools to help them identify the diseases and correctly eradicate the disease.

### Results
Several transfer models(InceptionV3, ResNet50, VGG16, DenseNet) were tested on the dataset and Densenet 169 was the best model of all.

DenseNet was created to solve the issue of the vanishing gradient, a problem found in a lot of neural networks. It has a simple connectivity pattern and it connects each layer to every other layer in a feed-forward fashion. It does well in selecting and reusing features and the relatively few parameters help the model load and train faster.

In the first pass, the transfer model was frozen and the subsequent layers used to classify the images were trained, the test accuracy was 94.3%. In the second pass, the transfer model was trained so that it improve its feature extraction and the subsequent classifier layers were frozen. The test accuracy was 96.2%. Finally, in the third pass, the transfer model was frozen and the classifier layers were trained. The final accuracy score was 98.7%.
![Process](https://github.com/chennat811/Plant_disease_detection/blob/main/Process.png)

### Discussion
There was slight imbalance in the dataset, 3 out of 39 classes had more images than the others. Precision and recall was examined but the imbalance does not seem to affect the results.

Corn: Northern Leaf Blight, tomato: late blight, and tomato: target spot had lower recall and higher precision, meaning that it does very well at identifying the true positives but may miss alot of them because the model is picky. Since farmers would want to be safe than sorry, the disease should always pick up the positives even if a couple healthy plants are killed, thus recall should be improved in these classes.

This model was deployed on Heroku, using Streamlit to build the app. [Heroku App](http://plant-disease-detector-88.herokuapp.com/)

### Tools Used
Pandas, Matplotlib, Keras, Streamlit, Heroku, Wikipedia API

### Impacts
Plant disease identification can help farmers recognize the types of diseases that their plant may have. This could help them make the right management decisions.  

### Future work
In the future, more images of a different class of diseases could be added to the dataset by scraping the internet. 

### References

- Huang, G., Liu, Z., Van Der Maaten, L.,  and Weinberger, K. Q. (2017). Densely Connected Convolutional 	Networks. 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Honolulu, 	HI, 2017, pp. 2261-2269, doi: 10.1109/CVPR.2017.243.
- Hughes, D., & Salathe, M. (2015). An open access repository of images on plant health to 	enable the development of mobile disease diagnostics through machine learning and 	crowdsourcing.
- Ken Pernezny, M. (2017, April 10). Guidelines for Identification and Management of Plant Disease 	Problems: Part II. Diagnosing Plant Diseases Caused by Fungi, Bacteria and Viruses. Retrieved 	December 08, 2020, from https://edis.ifas.ufl.edu/mg442
- Pandian J, A., Gopal, G. (2019) Data for: Identification of Plant Leaf Diseases Using a 9-layer Deep 	Convolutional Neural Network. Mendeley Data V1, doi: 10.17632/tywbtsjrjv.1
- Ruiz, P. (2018, October 18). Understanding and visualizing DenseNets. Retrieved December 08, 	2020, from https://towardsdatascience.com/understanding-and-visualizing-densenets-	7f688092391a
- Wikipedia: The free encyclopedia. (2004, July 22). FL: Wikimedia Foundation, Inc. Retrieved 	August 10, 2004, from https://www.wikipedia.org


