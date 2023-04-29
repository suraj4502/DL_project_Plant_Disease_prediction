
# Plant Disease Prediction Using Deep Learning
[_Webapp_](https://suraj4502-ppd.streamlit.app/ "Go to Streamlit app") --------------------------------------------------------------------------------------------------                                                            [_git Repo_](https://github.com/suraj4502/DL_project_Plant_Disease_prediction "Go to GitHub")
- The project aims to develop a system that can accurately identify  various diseases in plants using machine learning algorithms. 
- The system will be trained on a dataset of images of plants affected by different diseases and will use deep learning techniques to extract relevant features from the images. 
- The extracted features will then be used to train a model that can classify new images of plants based on the presence or absence of disease.
- The project will involve the use of deep learning frameworks such as **TensorFlow**, **Keras**, and will require the use of image processing techniques such as **convolutional neural networks (CNNs)** and **transfer learning**.
- Other technologies that are used to built this project are **React JS**, **Streamlit**,**javascript**, **FastAPI**,**Google Cloud Platforms**. 

- The ultimate goal of the project is to provide a tool that can help farmersand plant pathologists quickly identify and treat plant diseases, leading to higher crop yields and better food security.


## Dataset
The PlantVillage dataset is a collection of over 50,000 high-resolution images of 26 different plant species and 38 different plant diseases, created for advancing computer vision and machine learning techniques for plant disease diagnosis and classification. 

## Model architecture
- This project uses a **transfer learning** approach to tackle our image classification task, leveraging the pre-trained VGG16 model as our base.
- I added my own custom layers on top of the VGG16 architecture to fine-tune the model to my specific problem domain. 
- This allowed me to benefit from the powerful feature extraction capabilities of VGG16 while also tailoring the model to my unique data and classification needs.
- The architecture is as followed :

        1. Resizing layer
        2. VGG16 layers (only Cnn layers)
        3. Flattening layer
        4. A dense layer with 128 neurons and relu function
        5. A dense layer with 64 neurons and relu function
        6. Finally an output layer with Softmax function.
## Results
The Performance of the CNN model is as followed :

![image](https://user-images.githubusercontent.com/76464630/235282893-6e8fabb8-cd0b-4248-ad0a-c3ad4ce240ce.png)


![image](https://user-images.githubusercontent.com/76464630/235282919-14857144-7edc-4e1b-9bbf-272dca2538c6.png)
## Installation

Install my project on your device locally.
1. clone the entire Repository.
```bash
git clone https://github.com/suraj4502/DL_project_Plant_Disease_prediction/
```
2. go the api folder and install all the requirements from requirements.txt file and execute the main.py .
```bash
cd api
pip install -r requirements.txt
python main.py
```
3. go to frontend folder and install the requirements and run the reacts Js app. 
```bash
cd frontend
npm install --from-lock-json
npm audit fix
npm run start
```
4. To run the Streamlit App.
```bash
cd Streamlit_app
pip install -r requirements.txt
Streamlit run 🏠Home.py
```

    
## Demo

1. [Streamlit APP](https://suraj4502-ppd.streamlit.app/ "Go to Streamlit app") 

*The [app](https://github.com/suraj4502/Streamlit_app_PDP) is deployed on the Streamlit Cloud while the model is deployed on the GCP as a cloud function, a request is sent every time we want to predict the class of the plant and a response is received from GCP.*

- Login Page: 
![image](https://user-images.githubusercontent.com/76464630/235284574-5b2f39dc-420f-4e79-ac68-2397264a7746.png)

- Signing up:
![image](https://user-images.githubusercontent.com/76464630/235284601-cf123fc7-ada1-4a7a-80b5-d12acf5a7fa3.png)

- Retrieving Credentials:
![image](https://user-images.githubusercontent.com/76464630/235284650-0ed2077c-7966-4300-b006-0aa90d0d683a.png)

- Main section:

![image](https://user-images.githubusercontent.com/76464630/235284735-5da44624-a977-431a-9ba8-5c93d07ab55d.png)
- Uploading an image to get result:
![image](https://user-images.githubusercontent.com/76464630/235284686-1f56fb2e-0cd7-4f82-b18d-7f1df2d45061.png)

- Taking an image from camera to get results :
![image](https://user-images.githubusercontent.com/76464630/235284766-4e8781ab-6922-4d5c-a37b-43b9b5bb005e.png)
- Disease Information Page:

![image](https://user-images.githubusercontent.com/76464630/235284827-6bd5319e-78f2-4d8c-b3f8-23231e6ecb84.png)

![image](https://user-images.githubusercontent.com/76464630/235284833-67636905-aa4f-4470-8bcb-5a2a4704f2b2.png)

2. React JS App :  
![image](https://user-images.githubusercontent.com/76464630/235284878-e61c3939-a751-4dc0-b541-1a60a0c0373b.png)


![image](https://user-images.githubusercontent.com/76464630/235284919-596ee3f4-5767-47b0-8ba6-f2dd8948623d.png)

## Credits 
This project was created by [Surajkumar Yadav.](https://www.linkedin.com/in/surajkumar-yadav-6ab2011a4/)

--- 
