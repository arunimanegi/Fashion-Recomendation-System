# Fashion-Recomendation-System
Fashion Recommendation System that leverages deep learning and generative AI to provide personalized fashion product suggestions.

Abstract	
This project aims to develop a Fashion Recommendation System that leverages deep learning and generative AI to provide personalized fashion product suggestions. The system uses a combination of image similarity analysis via ResNet50 and descriptive captioning through the BLIP model, enhancing the shopping experience by recommending visually similar items along with unique descriptions.

Introduction	
With the rapid growth of e-commerce, fashion retailers are increasingly looking for innovative ways to enhance customer experience and engagement. A personalized recommendation system can significantly improve user satisfaction by suggesting relevant products based on user preferences and browsing history. This project combines state-of-the-art deep learning models for image feature extraction and natural language processing to offer an advanced recommendation system.	

Problem	Statement	
Existing recommendation systems often lack the ability to provide personalized, visually similar product suggestions with detailed descriptions. This project addresses the challenge by integrating image similarity analysis with descriptive captioning, providing users with a more comprehensive and engaging shopping experience.	
Objectives	
•  Develop an image feature extraction module using ResNet50 to capture visual similarities between fashion products.
•  Implement a Nearest Neighbors algorithm to identify and recommend visually similar items.
•  Integrate the BLIP model to generate unique descriptions for recommended products.
•  Build a user-friendly interface using Streamlit to display recommendations and descriptions.

Methodology	
Methods	and	Procedures
1.	Feature Extraction: Utilize the ResNet50 model pre-trained on ImageNet to extract features from fashion product images. The model's output is processed to generate normalized feature vectors representing each image.
2.	Similarity Analysis: Implement the Nearest Neighbors algorithm to find and recommend images that are visually similar to a user-uploaded image based on the extracted feature vectors.
3.	Caption Generation: Use the BLIP model to generate descriptive captions for the recommended images, providing users with detailed information about each product.
4.	User Interface: Develop a Streamlit application to facilitate user interaction, allowing users to upload images and view recommendations along with their descriptions.

	
Dataset	Information	

Description	
The growing e-commerce industry presents us with a large dataset waiting to be scraped and researched upon. In addition to professionally shot high resolution product images, we also have multiple label attributes describing the product which was manually entered while cataloging. To add to this, we also have descriptive text that comments on the product characteristics. 
 
Source	and	Format	
https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small	
![image](https://github.com/user-attachments/assets/44216bb1-13d2-417a-b7dc-3f572d368238)
