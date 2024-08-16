# ApparelRecomendationSystem-
CHAPTER 1 - ABSTRACT

This study pavilions a product recommendation system powered by multimodal data sources with image features, text titles, and brand descriptions. It uses CNN features for images and TF-IDF vectorization for text to compute the pairwise distances for determining similar products. It extracts bottleneck features from a pre-trained CNN model, which helps vectorize product titles and brand descriptions, followed by calculating weighted Euclidean distances for recommendation of visually and contextually similar products. The approach proposed here has been validated on a dataset of apparel, hence the ability of the system to exhibit diverse and effective recommendations. Major ingredients that go into it include integration of image data with textual information and use of pairwise distance metrics to enhance recommendation accuracy.












CHAPTER 2 - INTRODUCTION

On this note, the rapid growth of e-commerce has accommodated a volume of items available online that is overwhelming, making it hard for consumers to find products that best suit their preference. Recommendation systems have been incorporated as an important module in online retail platforms to offer personalized suggestions to the user. They use multi-source data that contains user behavior, product features, and context information to make recommendations.
In product recommendation systems, utilizing multi-modal data has been very effective. Multi-modal data is basically data consisting of text descriptions, images, and metadata—all giving broad knowledge about a product. Such integration of heterogeneous data types aids the system in finding similarities better and thus recommending more accurately.
The authors present a multi-modal recommendation system that combines image features from convolutional neural networks with textual features from product title and brand descriptions. This system can find and rank products that are visually and contextually similar to a given item through the use of a weighted combination of Euclidean distances calculated from these features.



Example : 
Source image : 



Recommendations:















CHAPTER 3 - THEORETICAL FRAMEWORK

Theoretical Framework
Developing a strong recommendation system requires the implementation of various theoretical concepts and methodologies that draw on machine learning and information retrieval. This section presents the theoretical background of the key techniques underlying the proposed multi-modal recommendation system.
1. Convolutional Neural Networks (CNNs) for Image Feature Extraction
Convolutional Neural Networks are one of the classes of deep learning models that work tremendously in the processing of visual data. Automatic extraction of hierarchical features from images is done by CNNs, which use convolutional layers equipped with learnable filters. These features will then capture key visual patterns, like edges and textures, and parts of objects. This makes CNNs quite suitable for any tasks on image classification and image similarity.
Key Concepts:
Convolutional Layers: Apply convolution operations on the input images to generate feature maps.
Pooling Layers: These are responsible for down-sampling the spatial dimensions of feature maps, retaining only the most prominent features, and reducing computational load. 
Fully Connected Layers: The features extracted by convolutional and pooling layers are combined to make a final prediction or generate feature vectors.
In the proposed system, pre-trained CNN models like VGG16 or ResNet can be employed for the extraction of bottleneck features from images of products. These would correspond to high-level representations of the visual content and hence would allow the computation of image similarities.
2. Text Vectorization Techniques for Title and Brand Descriptions
The textual data from product titles and brand descriptions would carry a lot of contextual information, complementing these visual features. In integrating textual data into the recommendation system, it becomes imperative to represent text in numerical forms so that machine learning algorithms can process them.
Key Techniques:
Count Vectorization: This is a method that will count the occurrence of tokens in each document and represent them in a matrix. Documents are represented through rows, while columns would represent unique tokens (words) in the corpus. The entry in the matrix represents the frequency of the token in the document.
TF-IDF Vectorization: It represents the text data by numerical vectors, taking into account term frequency and inverse document frequency. This method lessens the impact of the common words and provides higher importance to the very rare, but significant, terms.
In this paper, Count Vectorizer is applied to obtain the feature vectors from the product title and brand description. Such vectors would capture the textual content, enabling the computation of pairwise distances based on textual similarities.
3. Pairwise Distance Metrics
It is possible to recommend similar products only if the notion of 'similarity' between items, based on their features, could be quantified in some manner. Pairwise distance metrics quantify the dissimilarity between feature vectors.
Key Metric:
Euclidean Distance: A commonly used distance metric that calculates the straight-line distance between two points in a multi-dimensional space. For feature vectors a\mathbf{a}a and b\mathbf{b}b, the Euclidean distance is given by:
 
In this system, Euclidean distances are calculated for image features, title features, and brand description features separately. A weighted combination of these distances is then used to derive an overall similarity score.
4. Weighted Combination of Distance Metrics
In order to effectively combine the multi-modal data, a weighted combination of distances computed from these different feature sets is used. This will ensure that the information from both visual and context strictly gets embedded into the final similarity score.
Formula:


The weights w1​, w2, and w3​ can be adjusted based on the importance of each feature type in the recommendation process. This weighted combination approach allows for flexible and adaptive similarity computation.

















CHAPTER 4 - METHODOLOGY

The methodology of developing a multimodal recommendation system includes several major processes: data collection and preprocessing, extraction of features, computation of similarity, and generation of recommendations. In this section, details of the process and techniques used in every step will be given.
1. Data Collection and Preprocessing
The dataset used in this study consists of apparel items, each represented by images, titles, and brand descriptions. The data was collected from various sources and preprocessed to ensure consistency and usability.
Steps:
Data Loading: The dataset is loaded using pandas from a preprocessed pickle file (16k_apperal_data_preprocessed).
Handling Missing Values: Missing values in the dataset are identified and appropriately handled. For example, if an image URL is missing, the corresponding entry may be removed or flagged.
Data Type Conversion: Columns, such as brand, are converted to appropriate data types (e.g., string).
2. Feature Extraction
Feature extraction is performed on both image and textual data to convert them into numerical representations suitable for similarity computation.
a. Image Features:
Pre-trained CNN Model: A pre-trained CNN model, VGG16, is used to extract bottleneck features from product images. These features capture high-level visual patterns and are saved in a NumPy array (16k_data_cnn_features.npy).
Feature Extraction Process: Each image is passed through the CNN model, and the activations from one of the fully connected layers are extracted as feature vectors.

b. Textual Features:
Title Vectorization: Product titles are vectorized using CountVectorizer, which converts the text into a matrix of token counts.
Brand Description Vectorization: Similarly, brand descriptions are vectorized using CountVectorizer.
3. Similarity Computation
Similarity between products is computed using pairwise distance metrics for the extracted features.
Steps:
Image Distance Calculation: Euclidean distances between the image feature vectors are calculated using pairwise_distances from sklearn.metrics.
Title Distance Calculation: Euclidean distances between the title feature vectors are calculated.
Brand Distance Calculation: Euclidean distances between the brand description feature vectors are calculated.
4. Weighted Combination of Distances
To integrate the multi-modal features, a weighted combination of the distances is used. 
5. Recommendation Generation
Based on the combined distances, the system generates recommendations for a given product by identifying the most similar items.
Steps:
Sort Distances: For a given product, the combined distances to all other products are sorted in ascending order.
Select Top N Results: The top N closest products are selected as recommendations.
Display Results: For each recommended product, the title, image URL, and a link to the product page are displayed. The display_img function is used to visually present the product images.


CHAPTER 5 - RESULTS

Input Product:
Doc id:  1416
Asin:  B00JXQB5FQ
Product Title:  burnt umber tiger tshirt zebra stripes xl  xxl
Product Image: https://images-na.ssl-images-amazon.com/images/I/51a33K-9qfL._SL160_.jpg




Recommendation #1:
Doc id:  1413
Asin:  B00JXQASS6
Euclidean Distance from input image: 11.143011434099202
Product Title:  pink tiger tshirt zebra stripes xl  xxl
Product Image: https://images-na.ssl-images-amazon.com/images/I/51idp4BP50L._SL160_.jpg



Recommendation #2:
Doc id:  1421
Asin:  B00JXQCUIC
Euclidean Distance from input image: 14.29210744397706
Product Title:  yellow tiger tshirt tiger stripes  l
Product Image: https://images-na.ssl-images-amazon.com/images/I/511SmrC%2BS1L._SL160_.jpg 



Recommendation #3:
Doc id:  1422
Asin:  B00JXQCWTO
Euclidean Distance from input image: 15.10413710705545
Product Title:  brown  white tiger tshirt tiger stripes xl  xxl
Product Image: https://images-na.ssl-images-amazon.com/images/I/51tOiBaq5FL._SL160_.jpg















CHAPTER 6 - CONCLUSION 

In this work, a truly multimodal recommendation system has been developed that can effectively integrate visual and textual features for the generation of relevant and accurate recommendations for products. The system provides a deeper understanding of the products through the strengths of visual and textual data, providing better recommendations than single-modality models.
Implications and Future Work:
Broader Application: The performance of the multi-modal recommendation system indicates potential applications in many other e-commerce domains where products are described by image and textual features.
Feature Expansion: Future research could explore the inclusion of additional features such as customer reviews, ratings, aand more sophisticated natural language processing techniques to improve recommendation accuracy—is another potential avenue for future study.
Real-time Implementation: It would enhance user experience with real-time recommendations which are instant in personalized product suggestions.
In summary, the multi-modal approach in product recommendations is a very solid and efficient way to build the recommendation functionality of e-commerce applications aiming at increasing user satisfaction and driving better sales. The integration of such heterogeneous data is one promising direction for future research and development in recommendation systems.







CHAPTER 7 - REFERENCES
                                                          
YASHAR DELDJOO & Co.  arxiv.org/pdf/2202.02757 (Sep 2023)
     
U. C. De, S. Banerjee, M. K. Rath, T. Swain and T. Samant, "Content Based Apparel Recommendation for E-Commerce Stores," 2022 3rd International Conference for Emerging Technology (INCET), Belgaum, India, 2022, pp. 1-6, doi: 10.1109/INCET54531.2022.9824870. keywords: 

M. Tahir, R. N. Enam and S. M. Nabeel Mustafa, "E-commerce platform based on Machine Learning Recommendation System," 2021 6th International Multi-Topic ICT Conference (IMTIC), Jamshoro & Karachi, Pakistan, 2021, pp. 1-4, doi: 10.1109/IMTIC53841.2021.9719822.
                
