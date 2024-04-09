|![](Aspose.Words.96696cac-afe9-4f16-b5d4-0b03d3031a7b.001.png)||14|
| :- | :-: | -: |

# <a name="_toc161238029"></a>**Introduction**
Biomedical research regarding understanding cancer development and its mechanism has been its crucial point. With the advancement of technology in the medical field researchers now have vast amounts of data which hold valuable information about the molecular basis of cancer. In this report, we will analyze the gene data given to us to identify a pattern to build a predictive model for cancer classification. The Dataset that we have been provided contains information for 78 patients which are the rows in the data set and 4949 genes which are the columns in the dataset and the last column is the target column which tells us about the type of cancer, 1 indicate invasive cancer type and 2 indicates non-invasive cancer type. For this analysis we will be using several machine learning techniques to answer several research questions. 
# <a name="_toc161238030"></a>**Abstract**
Our analysis is structured into four main parts:
## <a name="_toc161238031"></a>**Data Cleaning (Statistical Imputation Methods)**
We identified 62 missing values in dataset which becomes a noise in model training. Hence we used Mean Imputation on the missing values to enrich the dataset and remove the noise.
## <a name="_toc161238032"></a>**Unsupervised and Supervised Dimension reduction**
Firstly, we will start out analysis by applying dimensionality reduction for supervised and unsupervised learning on our gene data. This is done to transform highly dimensional data to low dimension data to keep things simple for us. We will use Principal component analysis and T Test techniques to perform these actions.
## <a name="_toc161238033"></a>**Unsupervised learning model**
We will apply unsupervised learning techniques such as K-means clustering and hierarchical clustering, PCA, and explore the clusters groups of genes to identify the underlying structure.

**Supervised learning model**

We will use various supervised learning models such as logistic regression, Linear discriminant analysis, K-Nearest Neighbor, Random Forest and Support Vector Machine to classify cancer samples as invasive and non-invasive.

**Integration of clustering results**

We will examine the clustering results and integrate them into our best Machine Learning model to enhance classification performance.
# <a name="_toc161238034"></a>**Preliminary analysis:**
As mentioned above, we have 4949 Gene Expressions out of which we have made a subset of 2000 Gene Expressions, by setting seed as the highest Registration Number. We now must further reduce the dataset's dimensionality. So why do we need to do it?

- <a name="_toc161238035"></a>***Curse of Dimensionality***: The amount of data needed to accurately generalize increases exponentially with the number of features or dimensions in a dataset. We refer to this phenomena as the "curse of dimensionality." Dimensionality reduction helps to reduce this issue by lowering the number of dimensions, which improves the accuracy and efficiency of future analysis.
- <a name="_toc161238036"></a>***Computational Efficiency***: Algorithms' computational complexity can be markedly raised by high-dimensional data. By streamlining the dataset and reducing its dimensionality, computations become more efficient and easier to handle.
## <a name="_toc161238037"></a>**PCA (For Unsupervised Learning)**
We have performed unsupervised dimension reduction on a dataset containing gene expression values using Principal Component Analysis (PCA), a statistical procedure that uses orthogonal transformation to convert a set of observations of possibly correlated variables (in this case, gene expression values) into a set of values of linearly uncorrelated variables called principal components. These are the steps involves in performing PCA:

- <a name="_toc161238038"></a>***Perform PCA:*** The **prcomp** function is used on the gene expression dataset (excluding last columns, representing non-gene-expression data such as sample identifiers and class labels). 
- <a name="_toc161238039"></a>***Visualize PCA Results:*** It visualizes the importance of principal components through a summary and a plot, helping to understand how much variance each principal component captures from the dataset.
- <a name="_toc161238040"></a>***Determine Number of Components for Explaining 90% Variance:*** Cumulative Variance is calculated and It is used to find the minimum number of principal components needed to explain at least 90% of the total variance in the dataset. 
- <a name="_toc161238041"></a>***Identify Top N Genes Based on Loadings:*** It calculates the absolute values of the loadings to consider both positive and negative contributions equally and identifies the top N genes that contribute most to the variance captured by the principal components. 
- <a name="_toc161238042"></a>***Create a Filtered Dataset:*** Finally, we filter the original dataset to include only the top N genes and the class variable. This the dataset is dimensionally reduced. 
## <a name="_toc161238043"></a>**T-squared Test (For Supervised Learning)**
We have implemented supervised dimension reduction on a dataset of gene expression values using t-tests, a method that involves utilizing the class labels (or target variable) to identify the genes most relevant for distinguishing between classes. This is done as follows:

- <a name="_toc161238044"></a>***Define a Function for t-tests***: A function named **perform\_t\_test** is created to perform a t-test on a given gene column against the class labels, which performs t-test and returns the p-values. 
- <a name="_toc161238045"></a>***Apply t-tests Across All Genes:*** We apply the **perform\_t\_test** function across all identified gene columns using **sapply**. This step computes a t-test for each gene, comparing its expression levels between the different classes in the dataset, resulting in a vector of p-values corresponding to each gene.
- <a name="_toc161238046"></a>***Filter Genes Based on P-value Threshold:*** Genes with p-values less than 0.05 are considered statistically significant, and only relevant genes are kept. 
- <a name="_toc161238047"></a>***Create a Filtered Dataset with Significant Genes:*** The class label column is added back to the list of significant genes to ensure its included in the final filtered dataset which is reduced to lower dimensions. 

|**S.No**|**Supervised**|**Unsupervised**|
| :-: | :-: | :-: |
|**1**|Logistic Regression|PCA|
|**2**|LDA|K Means Clustering|
|**3**|QDA|Hierarchical Clustering|
|**4**|Random Forest|T-SNE Model (Extra)|
|**5**|SVM||
|**6**|KNN||
|**7**|GBM (Extra)||
|**8**|XGBoost (Extra)||
# <a name="_toc161238048"></a>**Reasons for choosing Gradient Boosting and Extreme Gradient Boosting**
Extreme Gradient Boosting (XGBoost) and Gradient Boosting Machines (GBM) are two powerful machine learning algorithms that are well-known for their remarkable efficiency, adaptability, and predicted accuracy. These algorithms prevent overfitting through regularization techniques, handle missing values effectively, and provide insights into the significance of features. XGBoost is notable for its scalability and competitive performance in machine learning contests. GBM and XGBoost continue to be essential resources for data scientists and researchers looking for innovative predictive modelling performance because of their vibrant communities and ongoing development.
# ` `<a name="_toc161238049"></a>**Reasons for choosing TSNE:**
The t-Distributed Stochastic Neighbor embedding (t-SNE) method is a popular dimensionality reduction technique that is especially useful for visualizing complicated datasets because of its capacity to maintain local structures in high-dimensional data. t-SNE facilitates the intuitive exploration and analysis of complex patterns and clusters within the data by highlighting the links between close data points. 

The Flowchart represents a high level process plan to use the given gene subset and conduct statistical test to reduce the dimension followed by supervised and unsupervised modelling to classify invasive and non-invasive cancer types

![C:\Users\farjaad\Downloads\flow.drawio.png](Aspose.Words.96696cac-afe9-4f16-b5d4-0b03d3031a7b.002.png)
# <a name="_toc161238050"></a>**Discussion**
## <a name="_toc161238051"></a>**Unsupervised learning model**
We will be accounting for 90% of total cumulative variation. This variation is given by first 49 principal components. We will now use unsupervised learning methods for making clusters:
- ### <a name="_toc161238052"></a>***K-means Clustering***
K-means randomly selects clusters and then classifies all the data points in the clusters based on the centroids. 

***Determining the number of clusters***

- **Elbow Method**

![A graph of a number of clusters

Description automatically generated](Aspose.Words.96696cac-afe9-4f16-b5d4-0b03d3031a7b.003.png)

The above graph tells us about the optimal number of clusters. Based on the WCSS values, we observe a notable change in the slope around k = 3 or k = 4, indicating that 3 or 4 clusters is a reasonable choice. 

- **Silhouette Score Method**:

![A graph with numbers and lines

Description automatically generated](Aspose.Words.96696cac-afe9-4f16-b5d4-0b03d3031a7b.004.png)

The silhouette score falls in –1 to 1. Any value which is closer to 1 is better. According to this, the maximum average silhouette widths are recorded for k = 3 (0.7522921). Breaking our PCA-reduced gene expression data into three clusters is expected to offer the most distinct and well-separated grouping. After k = 3, the scores start to decline, indicating that more clusters do not enhance the ability to distinguish between them. From above 2 methods, we are finalizing the value of **k=3.**

Following is the plot for the 3-clusters we got using K-means.

![A graph with numbers and dots

Description automatically generated](Aspose.Words.96696cac-afe9-4f16-b5d4-0b03d3031a7b.005.png)
- ### ![](Aspose.Words.96696cac-afe9-4f16-b5d4-0b03d3031a7b.006.png)<a name="_toc161238053"></a>***Hierarchical Clustering:***
In unsupervised learning, hierarchical clustering is a strategy that groups similar data points into clusters according to their similarities or distances from one another. It produces a hierarchy of clusters, each of which has a dendrogram connecting them. The distance metric that we will be using is ‘Euclidean’ and the linkage with ‘complete’.
### <a name="_toc161238054"></a>***TSNE Model:*** 
In machine learning and data visualisation, t-distributed Stochastic Neighbour Embedding (t-SNE) is a well-liked dimensionality reduction method. When high-dimensional data is visualised in lower-dimensional areas, usually 2D or 3D, it works especially well. In order to identify clusters, patterns, and correlations within the data, t-SNE is helpful since it maintains local commonalities between data points in the original high-dimensional space.

**Our result:**

![](Aspose.Words.96696cac-afe9-4f16-b5d4-0b03d3031a7b.007.png)

From above plot, we are getting 3 clusters.
## <a name="_toc161238055"></a>**Supervised Learning Model:**
We have used ‘caret’ library in r to the model training. Furthermore, we have made use of “TrControl” parameter in caret library for performing cross-validation and “TuneGrid” parameter for performing hyperparameter tuning. The results obtained have then been converted into a resampled list, summary of which tells us about various evaluation metrics such as the ‘Accuracy’, ‘F-1 score’, ‘Recall’, ‘Precision’, ‘Sensitivity’, ‘Specificity’. By resampling we get the above-mentioned metrics in the form of minimum, 1st quartile, median, mean, 3rd quartile and maximum metrics. We have decided to with median as our final evaluation metric as it is more suitable and accurate than mean.

We won’t be using Quadratic Determinant Analysis as it is not suitable for our  datasetAccording to the book "Introduction to Statistical Learning with R", the basis on which QDA or LDA should be selected is the Bias-Variance tradeoff.LDA tends to be a better bet than QDA if there are relatively few training observations and so reducing variance is crucial. In contrast, QDA is recommended if the training set is very large, so that the variance of the classifer is not a major concern.
## <a name="_toc161238056"></a>**Model Training and Hyperparameter Tuning:**
- ### <a name="_toc161238057"></a>***Logistic Regression:***
The most common classification method. We used the “glmnet” package to train the data using logistic regression.

![A screenshot of a computer code

Description automatically generated](Aspose.Words.96696cac-afe9-4f16-b5d4-0b03d3031a7b.008.png)

Lr\_hyperparameters, contains the parameters inside the logistic regression model which are used for tuning.

Logistic regression gives us a median accuracy of 0.833
- ### <a name="_toc161238058"></a>***Linear Discriminant Analysis***
LDA is primarily a dimensionality reduction technique which helps us train complex high dimensional problems. The code for LDA under the “caret” package is below.The median accuracy we get is 0.714.

![A computer code with black text

Description automatically generated with medium confidence](Aspose.Words.96696cac-afe9-4f16-b5d4-0b03d3031a7b.009.png)
- ### <a name="_toc161238059"></a>***K-Nearest Neighbors***
KNN considers the nearest neighbors and their class for classification of any new datapoint. The number of neighbors is the main parameter which is used to tune. Accuracy obtained by KNN is 0.75

![A screenshot of a computer code

Description automatically generated](Aspose.Words.96696cac-afe9-4f16-b5d4-0b03d3031a7b.010.png)
- ### <a name="_toc161238060"></a>***Random Forest***
Random Forest is an ensemble technique which allows to capture complex relationships. 

![A screenshot of a computer program

Description automatically generated](Aspose.Words.96696cac-afe9-4f16-b5d4-0b03d3031a7b.011.png)

For our dataset, Random forest was leading to over fit, so we have selected cross validation only. The median accuracy is 0.714
- ### <a name="_toc161238061"></a>***Support Vector Classifier***
![A screenshot of a computer code

Description automatically generated](Aspose.Words.96696cac-afe9-4f16-b5d4-0b03d3031a7b.012.png)

SVM demonstrated competitive performance, particularly in scenarios with high-dimensional data and complex decision boundaries. Its ability to handle non-linear data through kernel tricks made it a valuable model for various classification tasks. The hyperparameter tuning for SVM contains the optimization factor. The accuracy is 0.833
- ### <a name="_toc161238062"></a>***Gradient Boosting Model***
![A computer code with black text

Description automatically generated](Aspose.Words.96696cac-afe9-4f16-b5d4-0b03d3031a7b.013.png)

GBM can handle heterogeneous features and is robust to outliers due to its ensemble nature. It typically provides high predictive accuracy and can capture complex interactions between variables.

The median accuracy for GBM is  0.845.
- ### <a name="_toc161238063"></a>***Extreme Gradient* Boosting**  
![A screenshot of a computer code

Description automatically generated](Aspose.Words.96696cac-afe9-4f16-b5d4-0b03d3031a7b.014.png)

XGboost is one of the most recent boosting methods and is widely used in Machine Learning. One of the best advantages of XGboost is that it stays robust even for the data set with missing values.

![A screenshot of a computer

Description automatically generated](Aspose.Words.96696cac-afe9-4f16-b5d4-0b03d3031a7b.015.png)

There are various parameters that can be used for hyperparameter tuning of XGboost, like, the learning rate, maximum depth, etc. The median accuracy of xgboost is  0.69. Following are best parameters after hyperparameter.
## <a name="_toc161238064"></a>**Evaluating the results of Resampling:**
- ### <a name="_toc161238065"></a>***Accuracy***
**GBM** (Gradient Boosting Machine) shows the highest mean accuracy (0.7824), indicating its strong overall performance. **Logistic Regression (LogReg)** and **SVM** also perform well, with mean accuracies of 0.7452 and 0.7762, respectively.
- ### <a name="_toc161238066"></a>***F1 Score***
**LogReg** has the highest mean F1 score (0.7314), with **GBM** close behind (0.7695).F1 scores suggest	 that **LogReg** and **GBM** might balance precision and recall better than the others.
- ### <a name="_toc161238067"></a>***Kappa:*** 
**GBM** shows the highest mean Kappa (0.5662), followed by **SVM** (0.5475).Kappa scores indicate that **GBM** and **SVM** might be better at accounting for chance agreement than other models.
- ### <a name="_toc161238068"></a>***Recall / Sensitivity***
**GBM** has the highest mean recall/sensitivity (0.8333), indicating its strength in identifying all relevant cases.
- ### <a name="_toc161238069"></a>***Specificity***
  XGboost exhibits the highest mean specificity (0.7917), closely followed by RandomForest (0.7750). High specificity in these models indicates their strength in correctly identifying negatives.
#### **Overall Results:**

|**Model**|**Accuracy**|**Precision**|**Recall**|**Sensitivity**|**Specificity**|
| :-: | :-: | :-: | :-: | :-: | :-: |
|**Logistic**|0\.833|0\.750|1|1|0\.75|
|**LDA**|0\.714|0\.666|0\.666|0\.666|0\.708|
|**GBM**|0\.845|0\.75|1|1|0\.75|
|**KNN**|0\.75|0\.708|0\.666|0\.666|0\.75|
|**RF**|0\.714|0\.666|0\.666|0\.666|0\.75|
|**SVM**|0\.833|0\.708|0\.833|0\.833|0\.75|
|**XGB**|0\.690|0\.708|0\.667|0\.667|0\.75|

![](Aspose.Words.96696cac-afe9-4f16-b5d4-0b03d3031a7b.016.png)

So, from the above resampling done on the supervised learning model, we can see that Gradient Boosting Model gives us the best accuracy. It performs well across a range of metrics, suggesting that it can effectively manage the subtleties of your dataset.
# <a name="_toc161238070"></a>**Conclusion**
We will now examine the accuracies of the model trained using cluster labels with those of the original model. A rise in these measures can mean that the unsupervised model-created clusters have been successful in gathering more data that is useful for the prediction task. So for that, we will be dividing our dataset into train and test sets. Our train set will consist of 80% of the dataset and 20% will be in the test set. Train-Test ensures that the model does not overfit. We have also added cross-validation with 10 folds. Cross-validation ensures that whole data is properly considered for training and testing, so that biases are avoided. Gradient Boosting, our best performing model, is considered here along with the cluster from the unsupervised clustering, We now do our predictions on the test set. There is a rise of 0.154 (15%) in accuracy. 
# <a name="_toc161238071"></a>**References**
[1] Caret Package Documentation [**https://www.rdocumentation.org/packages/caret/versions/6.0-94**](https://www.rdocumentation.org/packages/caret/versions/6.0-94)

[2] XGBoost Documentation [**https://xgboost.readthedocs.io/en/stable/**](https://xgboost.readthedocs.io/en/stable/)

[3] TSNE - Distill [**https://distill.pub/**](https://distill.pub/)

[4] T-squared - An Introduction to Multivariate Statistical Analysis Third Edition T. W. ANDERSON Stanford University Department of Statistics, Stanford, CA

[5] QDA Gareth James - Daniela Witten - Trevor Hastie - Robert Tibshirani An Introduction to Statistical Learning with Applications in R Second Edition




# <a name="_toc161238072"></a>**Appendix**
**MA321-7 Team Project Assignment**

11-Mar-2024

*# Load necessary library*
**suppressWarnings**({ 
*# Code that generates warning messages* 
`  `**library**(stats)
**library**(pls)
**library**(caret)
**library**(Rtsne)
**library**(tidymodels)
**library**(themis)
**library**(tidyverse)
**library**(ggplot2)
**library**(MASS)
**library**(gbm)
**library**(class)
**library**(randomForest)
**library**(e1071)
**library**(nnet)
**library**(dplyr)
**library**(xgboost)
**library**(glmnet)
**library**(cluster)
**library**(knitr) 
})



*# Loading data from CSV*
InitialData <- **read.csv**(file="gene-expression-invasive-vs-noninvasive-cancer.csv")

*# Declare variable for to store seed value*
seed\_val = 2312489

*# Set seed for reproducing random subset (Based on largest registration number within group)*
**set.seed**(seed\_val)

*# Picking the random 2000 Genes from dataset*
team.gene.subset <- **rank**(**runif**(1**:**4948))[1**:**2000]

*#Creating a Vector to store the 2000 genes and also add last columns i.e. Class variables*
team.gene.subset <- **c**(team.gene.subset, 4949)
team.gene.subset <- InitialData[, team.gene.subset]

*#Data Preprocessing*

*# Display the performance improvement*
**cat**("NA Count (Before data pre processing):", **sum**(**is.na**(team.gene.subset)), "**\n**")

\## NA Count (Before data pre processing): 62

*# Imputation: Replacing NA with Mean Value*
team.gene.subset <- team.gene.subset **%>%**
`  `**mutate**(**across**(**-ncol**(.), **~ifelse**(**is.na**(.), **mean**(., na.rm = TRUE), .)))

*# Display the performance improvement*
**cat**("NA Count (After data pre processing):", **sum**(**is.na**(team.gene.subset)), "**\n**")

\## NA Count (After data pre processing): 0



*# Task 1A - Unsupervised Dimension Reduction using PCA*
*# Perform PCA on gene expression data*
pca\_result <- **prcomp**(team.gene.subset[,**-c**(1, **ncol**(team.gene.subset))], center = TRUE, scale. = TRUE)

*# Visualize the importance of principal components*
**summary**(pca\_result)

\## Importance of components:
\##                            PC1      PC2      PC3      PC4     PC5    PC6
\## Standard deviation     16.5765 12.95982 11.20009 10.58652 9.27941 8.9306
\## Proportion of Variance  0.1375  0.08402  0.06275  0.05607 0.04308 0.0399
\## Cumulative Proportion   0.1375  0.22148  0.28423  0.34030 0.38337 0.4233
\##                            PC7     PC8     PC9    PC10    PC11    PC12    PC13
\## Standard deviation     7.95729 7.30949 6.83297 6.40533 6.14182 6.05823 5.73208
\## Proportion of Variance 0.03168 0.02673 0.02336 0.02052 0.01887 0.01836 0.01644
\## Cumulative Proportion  0.45495 0.48167 0.50503 0.52555 0.54442 0.56278 0.57922
\##                           PC14    PC15    PC16    PC17    PC18    PC19   PC20
\## Standard deviation     5.61688 5.55578 5.44368 5.13554 5.12191 5.02259 4.9577
\## Proportion of Variance 0.01578 0.01544 0.01482 0.01319 0.01312 0.01262 0.0123
\## Cumulative Proportion  0.59500 0.61044 0.62527 0.63846 0.65159 0.66421 0.6765
\##                           PC21    PC22    PC23    PC24    PC25    PC26    PC27
\## Standard deviation     4.88272 4.75169 4.64017 4.47587 4.43705 4.39974 4.34091
\## Proportion of Variance 0.01193 0.01129 0.01077 0.01002 0.00985 0.00968 0.00943
\## Cumulative Proportion  0.68843 0.69972 0.71049 0.72052 0.73036 0.74005 0.74947
\##                           PC28    PC29    PC30    PC31    PC32    PC33    PC34
\## Standard deviation     4.24351 4.20580 4.14419 4.05506 4.02693 3.95670 3.91848
\## Proportion of Variance 0.00901 0.00885 0.00859 0.00823 0.00811 0.00783 0.00768
\## Cumulative Proportion  0.75848 0.76733 0.77592 0.78415 0.79226 0.80009 0.80777
\##                           PC35    PC36    PC37    PC38    PC39    PC40    PC41
\## Standard deviation     3.89480 3.78168 3.74654 3.72383 3.62545 3.60817 3.59856
\## Proportion of Variance 0.00759 0.00715 0.00702 0.00694 0.00658 0.00651 0.00648
\## Cumulative Proportion  0.81536 0.82252 0.82954 0.83647 0.84305 0.84956 0.85604
\##                           PC42    PC43   PC44    PC45    PC46    PC47    PC48
\## Standard deviation     3.51838 3.47743 3.4329 3.34775 3.32284 3.26594 3.24951
\## Proportion of Variance 0.00619 0.00605 0.0059 0.00561 0.00552 0.00534 0.00528
\## Cumulative Proportion  0.86223 0.86828 0.8742 0.87978 0.88531 0.89064 0.89593
\##                          PC49    PC50    PC51    PC52    PC53    PC54    PC55
\## Standard deviation     3.2237 3.20880 3.13421 3.10077 3.08483 2.97915 2.96331
\## Proportion of Variance 0.0052 0.00515 0.00491 0.00481 0.00476 0.00444 0.00439
\## Cumulative Proportion  0.9011 0.90628 0.91119 0.91600 0.92076 0.92520 0.92959
\##                           PC56    PC57    PC58    PC59    PC60    PC61    PC62
\## Standard deviation     2.94354 2.90832 2.87750 2.84365 2.78711 2.77397 2.68873
\## Proportion of Variance 0.00433 0.00423 0.00414 0.00405 0.00389 0.00385 0.00362
\## Cumulative Proportion  0.93393 0.93816 0.94230 0.94635 0.95023 0.95408 0.95770
\##                           PC63    PC64    PC65    PC66   PC67    PC68    PC69
\## Standard deviation     2.67973 2.64081 2.58534 2.53737 2.4888 2.45345 2.40374
\## Proportion of Variance 0.00359 0.00349 0.00334 0.00322 0.0031 0.00301 0.00289
\## Cumulative Proportion  0.96129 0.96478 0.96812 0.97134 0.9744 0.97745 0.98034
\##                           PC70    PC71    PC72    PC73    PC74    PC75    PC76
\## Standard deviation     2.40171 2.32793 2.30163 2.27081 2.19329 2.16163 2.06800
\## Proportion of Variance 0.00289 0.00271 0.00265 0.00258 0.00241 0.00234 0.00214
\## Cumulative Proportion  0.98323 0.98594 0.98859 0.99117 0.99358 0.99591 0.99805
\##                           PC77      PC78
\## Standard deviation     1.97315 7.361e-15
\## Proportion of Variance 0.00195 0.000e+00
\## Cumulative Proportion  1.00000 1.000e+00

**plot**(pca\_result)

*# Calculate cumulative variance explained*
cumulative\_variance <- **cumsum**(pca\_result**$**sdev**^**2) **/** **sum**(pca\_result**$**sdev**^**2)

*# Find number of components for 90% variance*
num\_components <- **which**(cumulative\_variance **>=** 0.90)[1]

*# Extract loadings (contributions) of the first principal component*
loadings <- pca\_result**$**rotation[,1]

*# Get the absolute values of the loadings to consider both positive and negative contributions*
abs\_loadings <- **abs**(loadings)

*# Identify the top N genes based on their loadings' absolute values*
top\_N\_genes\_indices <- **order**(abs\_loadings, decreasing = TRUE)[1**:**num\_components]
top\_N\_genes\_names <- **names**(abs\_loadings)[top\_N\_genes\_indices]

*# Add the class variable column name to the list of top genes to keep it*
top\_N\_genes\_with\_class <- **c**(top\_N\_genes\_names, "Class")

*# Filter the original dataset to include only the top N genes and the class variable*
team.gene.subset\_unsup <- team.gene.subset[, top\_N\_genes\_with\_class]

*# Task 1B - Supervised Dimension Reduction using T Test*

*# Function to perform t-test and return p-value*
perform\_t\_test <- **function**(data, gene\_column) {
`  `t\_result <- **t.test**(data[[gene\_column]] **~** data**$**Class)
`  `**return**(t\_result**$**p.value)
}

*# Apply t-test for each gene (Excluding ID and Class label)*
gene\_columns <- **colnames**(team.gene.subset)[**-c**(1, **ncol**(team.gene.subset))] 
p\_values <- **sapply**(gene\_columns, perform\_t\_test, data = team.gene.subset)

*# Filter genes based on p-value < 0.05*
significant\_genes <- **names**(p\_values)[p\_values **<** 0.05]

*# Add the class label column to the list of significant genes*
significant\_genes\_with\_label <- **c**(significant\_genes, "Class") 

*# Filter the original dataset to keep only the significant genes and the class label*
team.gene.subset\_sup <- team.gene.subset[, significant\_genes\_with\_label]



*# Perform PCA on the dataset excluding Class Variable*
pca\_results <- **prcomp**(team.gene.subset\_unsup[, **-ncol**(team.gene.subset\_unsup)], center = TRUE, scale. = TRUE)

*# Plot PCA to visualize the first two principal components*
**plot**(pca\_results**$**x[, 1**:**2], xlab = "PC1", ylab = "PC2", main = "PCA of Gene Expression Data")

*# Elbow method to determine K Value*
*# Calculate WCSS for a range of k values*
**set.seed**(seed\_val)  *# Set seed to ensure reproducibility*

wcss <- **numeric**(10)  *# Array to store value*

**for** (k **in** 1**:**10) {  *# Evaluating k from 1 to 10*
`  `**set.seed**(seed\_val)  *# Reset the seed for each iteration*
`  `kmeans\_result <- **kmeans**(pca\_results**$**x[, 1**:**2], centers = k, nstart = 25)
`  `wcss[k] <- kmeans\_result**$**tot.withinss
}

*# Plot the WCSS to visualize the elbow*
**plot**(1**:**10, wcss, type = "b", xlab = "Number of Clusters k", ylab = "Within-Cluster Sum of Squares (WCSS)", main = "Elbow Method for Optimal k")

*#Based on the WCSS values, we observe a notable change in the slope around* 
*#k = 3 or k = 4, indicating that 3 or 4 clusters could be a reasonable choice.*
*#The exact elbow point can sometimes be subjective and depends on how sharply the slope changes,*
*#so it's often useful to consider other factors (such as domain knowledge or additional validation metrics) alongside the Elbow Method to make a final decision on the optimal number of clusters.*

*# Silhouette method to determine K Value*
avg\_sil\_width <- **numeric**(10)  *# Array to store value*

**for** (k **in** 2**:**10) {  *# Starting from 2 because silhouette score requires at least 2 clusters*
`  `**set.seed**(seed\_val)  *# Set seed to ensure reproducibility*
`  `km\_res <- **kmeans**(pca\_results**$**x[, 1**:**2], centers = k, nstart = 25)
`  `silhouette\_res <- **silhouette**(km\_res**$**cluster, **dist**(pca\_results**$**x[, 1**:**2]))
`  `avg\_sil\_width[k] <- **mean**(silhouette\_res[, "sil\_width"])
}

*# Plot the average silhouette width for each k*
**plot**(2**:**10, avg\_sil\_width[2**:**10], type = "b", xlab = "Number of Clusters k", ylab = "Average Silhouette Width", main = "Silhouette Method for Optimal k")

*#The highest average silhouette widths are observed for k = 3 (0.7522921), indicating that dividing your PCA-reduced gene expression data into 3 clusters likely provides the most distinct and well-separated grouping according to this method.*
*#The scores decrease after k = 3, suggesting that additional clusters do not improve the distinction between them.*

*# Determine the optimal number of clusters*
**set.seed**(seed\_val) *# Set seed to ensure reproducibility*
k <- 3 *#Based on Optimal K from Elbow and Silhouette Method*
kmeans\_result <- **kmeans**(pca\_results**$**x[, 1**:**2], centers = k)

*# Plot the clusters*
**plot**(pca\_results**$**x[, 1**:**2], col = kmeans\_result**$**cluster, xlab = "PC1", ylab = "PC2", main = "k-means Clustering on PCA-reduced Data")
**points**(kmeans\_result**$**centers, col = 1**:**k, pch = 8, cex = 2)

*# Use Euclidean distance and complete linkage for hierarchical clustering*
dist\_mat <- **dist**(**t**(team.gene.subset\_unsup[, **-ncol**(team.gene.subset\_unsup)])) *# Compute distance matrix*
hc\_result <- **hclust**(dist\_mat, method = "complete")

*# Plot the dendrogram*
**plot**(hc\_result, main = "Hierarchical Clustering of Gene Expression Data", sub = "", xlab = "")

*#t-SNE test*
*# Select the first N principal components for t-SNE*
pca\_N <- pca\_results**$**x[, 1**:**num\_components]

*# pca\_N is the PCA-reduced dataset ready for t-SNE*
**set.seed**(seed\_val) *# Set seed to ensure reproducibility*
adjusted\_perplexity <- 5 *# Adjusted to a lower value since it is a small datasets*

*# Perform t-SNE with the adjusted perplexity value*
tsne\_results <- **Rtsne**(pca\_N, dims = 2, perplexity = adjusted\_perplexity, verbose = TRUE)

\## Performing PCA
\## Read the 78 x 49 data matrix successfully!
\## Using no\_dims = 2, perplexity = 5.000000, and theta = 0.500000
*# Plot the t-SNE results*
**plot**(tsne\_results**$**Y[,1], tsne\_results**$**Y[,2], main = "t-SNE on Gene Expression Data", xlab = "", ylab = "", pch = 20, col = **rainbow**(**length**(**unique**(kmeans\_result**$**cluster)))[kmeans\_result**$**cluster])
**legend**("topright", legend = **unique**(kmeans\_result**$**cluster), col = **rainbow**(**length**(**unique**(kmeans\_result**$**cluster))), pch = 20)



*# Supervised - Logistic Regression, LDA, QDA, k-NN, Random Forest and SVM*

**set.seed**(seed\_val)  *# Set seed to ensure reproducibility*

*# Convert class labels to a factor for classification*
Y <- **as.factor**(team.gene.subset\_sup[, **ncol**(team.gene.subset\_sup)])
X <- team.gene.subset\_sup[, **-ncol**(team.gene.subset\_sup)]

*# Splitting the data into training and testing sets*
trainIndex <- **createDataPartition**(Y, p = .8, list = FALSE)
X\_train <- X[trainIndex, ]
Y\_train <- Y[trainIndex]
X\_test <- X[**-**trainIndex, ]
Y\_test <- Y[**-**trainIndex]

*# Cleaning factor levels of Y\_train to ensure they are valid R variable names*
Y\_train=**ifelse**(Y\_train**==**1,0,1)
Y\_test=**ifelse**(Y\_test**==**1,0,1)
Y\_train <- **factor**(Y\_train)
Y\_test <- **factor**(Y\_test)
**levels**(Y\_train)

\## [1] "0" "1"

*# Set up cross-validation control with class probabilities*
control <- **trainControl**(method = "cv", 
`                        `number = 10, 
`                        `summaryFunction = multiClassSummary, 
`                        `savePredictions = TRUE)

*# Defining the target metric for evaluation*
metric <- "Accuracy"

*# Logistic Regression*
*# Setting Hyper Parameters for tuning*
lr\_hyperparamters= **expand.grid**(alpha=**seq**(0,1,0.1), 
`                               `lambda=**seq**(0.001, 0.1, 
`                                          `length.out=10))

*# Executing Logistic Regression*
model\_log <- **train**(Y\_train **~** .,
`                   `data = **data.frame**(X\_train, Y\_train),
`                   `method = "glmnet", family = "binomial",
`                   `trControl = control,
`                   `tuneGrid=lr\_hyperparamters,
`                   `verbosity=0)

\## Warning in nominalTrainWorkflow(x = x, y = y, wts = weights, info = trainInfo,
\## : There were missing values in resampled performance measures.

*# Make predictions on the test data*
predictions\_log <- **predict**(model\_log, newdata = X\_test)

*# Create a confusion matrix*
confusion\_matrix\_log <- **table**(predictions\_log, Y\_test)
confusion\_matrix\_log

\##                Y\_test
\## predictions\_log 0 1
\##               0 4 0
\##               1 2 8

*# Calculate accuracy*
acc\_log=**sum**(**diag**(confusion\_matrix\_log)) **/** **sum**(confusion\_matrix\_log)
acc\_log

\## [1] 0.8571429

*# The best parameters for logistic regression model are as follows:*
*# Check the best parameters*
best\_params\_log <- model\_log**$**bestTune
**print**(best\_params\_log)

\##    alpha lambda
\## 10     0    0.1

*# alpha is 0 means ridge regression is the best hyperparameter*

*# LDA*
model\_lda <- **train**(Y\_train **~** ., 
`                   `data = **data.frame**(X\_train, Y\_train), 
`                   `method = "lda", 
`                   `trControl = control, 
`                   `metric = metric,
`                   `verbose = FALSE)

*# GBM*
model\_gbm <- **train**(Y\_train **~** ., 
`                   `data = **data.frame**(X\_train, Y\_train), 
`                   `method = "gbm", 
`                   `trControl = control, 
`                   `metric = metric, tuneLength = 5,
`                   `verbose = FALSE
`                   `)

\## Warning in nominalTrainWorkflow(x = x, y = y, wts = weights, info = trainInfo,
\## : There were missing values in resampled performance measures.

*# SVM*
*# Setting Hyper Parameters for tuning*
svm\_hyperparameters <- **expand.grid**(C = **c**(0.1, 1, 10)) 
Y\_train=**factor**(Y\_train)

*# Executing SVM*
model\_svm <- **train**(Y\_train **~** ., 
`                   `data=**data.frame**(X\_train, Y\_train), 
`                   `method = "svmRadial",
`                   `trControl = control, 
`                   `metric = "Accuracy", 
`                   `TuneGrid=svm\_hyperparameters,
`                   `tuneLength = 3,
`                   `verbose = FALSE)

\## Warning in nominalTrainWorkflow(x = x, y = y, wts = weights, info = trainInfo,
\## : There were missing values in resampled performance measures.

Y\_pred\_svm = **predict**(model\_svm, X\_test)
cm\_svm = **table**(Y\_pred\_svm,Y\_test)
svm\_acc=**sum**(**diag**(cm\_svm)) **/** **sum**(cm\_svm)
svm\_acc

\## [1] 0.8571429

best\_params\_svm <- model\_svm**$**bestTune
best\_params\_svm

\##         sigma   C
\## 2 0.001880866 0.5

*# KNN*
*# Setting Hyper Parameters for tuning*
knn\_hyperparameters=**expand.grid**(
`  `k=**c**(1,2,3,4,5,6,7,8)
)

*# Executing KNN*
model\_knn <- **train**(Y\_train **~** .,
`                   `data = **data.frame**(X\_train, Y\_train), 
`                   `method = "knn", 
`                   `trControl = control,
`                   `tuneGrid=knn\_hyperparameters
)

Y\_pred\_knn=**predict**(model\_knn, **as.matrix**(X\_test))
cm\_knn=**table**(Y\_pred\_knn,Y\_test)
knn\_acc=**sum**(**diag**(cm\_knn)) **/** **sum**(cm\_knn)
knn\_acc

\## [1] 0.8571429

best\_params\_knn <- model\_knn**$**bestTune
best\_params\_knn

\##   k
\## 1 1

*# Random Forest*
model\_rf <- **train**(Y\_train **~** ., 
`                  `data = **data.frame**(X\_train, Y\_train), 
`                  `method = "rf", 
`                  `trControl = control, 
`                  `metric = metric, tuneLength = 3,
`                  `verbose = FALSE)

\## Warning in nominalTrainWorkflow(x = x, y = y, wts = weights, info = trainInfo,
\## : There were missing values in resampled performance measures.

*#XGboost*
model\_xgb <- **xgboost**(data = **as.matrix**(X\_train), label = **as.numeric**(**as.character**(Y\_train)), max.depth = 4, eta = 1, nthread = 2, nrounds = 2, objective = "binary:logistic", verbosity=0)

\## [1]  train-logloss:0.262535 
\## [2]  train-logloss:0.141991

Y\_pred\_xgb=**predict**(model\_xgb, **as.matrix**(X\_test))
Y\_pred\_xgb\_bin=**ifelse**(Y\_pred\_xgb**>**0.5,1,0)
cm\_xgb=**table**(Y\_pred\_xgb\_bin,Y\_test)
cm\_xgb

\##               Y\_test
\## Y\_pred\_xgb\_bin 0 1
\##              0 3 2
\##              1 3 6

xgb\_acc=**sum**(**diag**(cm\_xgb)) **/** **sum**(cm\_xgb)
xgb\_acc

\## [1] 0.6428571

*# XGboost hyperparameter tuned.*
xgb\_hyperparameter <- **expand.grid**(
`  `nrounds = **c**(50, 100, 200),
`  `max\_depth = **c**(2, 4, 6),
`  `eta = **c**(0.01, 0.1, 0.3),

`  `gamma = 0,
`  `colsample\_bytree = 1,
`  `min\_child\_weight = 1,
`  `subsample = 1
)
Y\_train <- **as.numeric**(**as.character**(Y\_train))

model\_xgb\_hyp=**train**(**as.matrix**(X\_train), **factor**(Y\_train), 
`                    `method="xgbTree", 
`                    `trControl=control, 
`                    `tuneGrid=xgb\_hyperparameter,
`                    `tuneLength = 5,
`                    `verbose = FALSE)

\## Warning in nominalTrainWorkflow(x = x, y = y, wts = weights, info = trainInfo,
\## : There were missing values in resampled performance measures.

Y\_pred\_xgb\_hyp=**predict**(model\_xgb\_hyp, **as.matrix**(X\_test))
**levels**(Y\_pred\_xgb\_hyp)

\## [1] "0" "1"

Y\_pred\_xgb\_hyp\_bin=**ifelse**(Y\_pred\_xgb\_hyp**>**0.5,1,0)

\## Warning in Ops.factor(Y\_pred\_xgb\_hyp, 0.5): '>' not meaningful for factors

cm\_xgb\_hyp=**table**(Y\_pred\_xgb\_hyp,Y\_test)
cm\_xgb

\##               Y\_test
\## Y\_pred\_xgb\_bin 0 1
\##              0 3 2
\##              1 3 6

xgb\_acc\_hyp=**sum**(**diag**(cm\_xgb\_hyp) )**/** **sum**(cm\_xgb\_hyp)
xgb\_acc\_hyp

\## [1] 0.7857143

*#best\_xgb\_parameters*
best\_xgb\_params= model\_xgb\_hyp**$**bestTune
best\_xgb\_params

\##    nrounds max\_depth eta gamma colsample\_bytree min\_child\_weight subsample
\## 19      50         2 0.3     0                1                1         1

*# Create a list of model objects*
model\_list <- **list**(LogReg = model\_log, LDA = model\_lda, GBM = model\_gbm, KNN = model\_knn, RandomForest = model\_rf, SVM = model\_svm,XGboost=model\_xgb\_hyp)

*# Create tmodel\_log# Create the resamples object*
results <- **resamples**(model\_list)

*# Analyze the results*
**summary**(results)

\## 
\## Call:
\## summary.resamples(object = results)
\## 
\## Models: LogReg, LDA, GBM, KNN, RandomForest, SVM, XGboost 
\## Number of resamples: 10 
\## 
\## Accuracy 
\##                   Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
\## LogReg       0.3333333 0.6666667 0.8333333 0.7452381 0.8511905 1.0000000    0
\## LDA          0.5000000 0.6666667 0.7142857 0.7190476 0.8333333 0.8571429    0
\## GBM          0.3333333 0.7357143 0.8452381 0.7823810 0.8571429 1.0000000    0
\## KNN          0.2857143 0.6666667 0.7500000 0.7404762 0.8571429 1.0000000    0
\## RandomForest 0.5000000 0.7142857 0.7142857 0.7490476 0.8250000 1.0000000    0
\## SVM          0.5000000 0.6785714 0.8333333 0.7761905 0.8571429 1.0000000    0
\## XGboost      0.5000000 0.5714286 0.6904762 0.7252381 0.8428571 1.0000000    0
\## 
\## Balanced\_Accuracy 
\##                   Min.   1st Qu.    Median      Mean   3rd Qu.  Max. NA's
\## LogReg       0.4166667 0.6666667 0.8333333 0.7666667 0.8750000 1.000    0
\## LDA          0.5000000 0.6041667 0.7083333 0.7083333 0.8333333 0.875    0
\## GBM          0.3333333 0.7395833 0.8541667 0.7916667 0.8750000 1.000    0
\## KNN          0.3333333 0.6354167 0.7500000 0.7416667 0.8645833 1.000    0
\## RandomForest 0.3750000 0.7083333 0.7083333 0.7375000 0.8333333 1.000    0
\## SVM          0.5000000 0.6770833 0.8333333 0.7750000 0.8645833 1.000    0
\## XGboost      0.5000000 0.5833333 0.6875000 0.7208333 0.8437500 1.000    0
\## 
\## Detection\_Rate 
\##                   Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
\## LogReg       0.1428571 0.3333333 0.3333333 0.3428571 0.4285714 0.5000000    0
\## LDA          0.0000000 0.2857143 0.3095238 0.3119048 0.4047619 0.5000000    0
\## GBM          0.1666667 0.2976190 0.4000000 0.3585714 0.4285714 0.4285714    0
\## KNN          0.1666667 0.2857143 0.3333333 0.3447619 0.4214286 0.5000000    0
\## RandomForest 0.0000000 0.2857143 0.3095238 0.3138095 0.3833333 0.5000000    0
\## SVM          0.1666667 0.2857143 0.3095238 0.3452381 0.4821429 0.5000000    0
\## XGboost      0.1428571 0.1750000 0.2857143 0.2795238 0.3214286 0.5000000    0
\## 
\## F1 
\##                   Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
\## LogReg       0.3333333 0.5416667 0.8000000 0.7314286 0.8571429 1.0000000    0
\## LDA          0.4000000 0.6666667 0.6666667 0.7047619 0.8571429 0.8571429    1
\## GBM          0.3333333 0.7000000 0.8285714 0.7695238 0.8571429 1.0000000    0
\## KNN          0.4444444 0.5952381 0.7750000 0.7389683 0.8428571 1.0000000    0
\## RandomForest 0.5714286 0.6666667 0.6666667 0.7597884 0.8000000 1.0000000    1
\## SVM          0.4000000 0.6875000 0.8000000 0.7473810 0.8428571 1.0000000    0
\## XGboost      0.4000000 0.5178571 0.6190476 0.6633333 0.8095238 1.0000000    0
\## 
\## Kappa 
\##                    Min.   1st Qu.    Median      Mean   3rd Qu. Max. NA's
\## LogReg       -0.1666667 0.3333333 0.6666667 0.5220000 0.7066667 1.00    0
\## LDA           0.0000000 0.2033333 0.4166667 0.4046667 0.6666667 0.72    0
\## GBM          -0.3333333 0.4663462 0.6933333 0.5662051 0.7200000 1.00    0
\## KNN          -0.2962963 0.2708333 0.5000000 0.4862689 0.7139130 1.00    0
\## RandomForest -0.2857143 0.4166667 0.4166667 0.4663004 0.6538462 1.00    0
\## SVM           0.0000000 0.3541667 0.6666667 0.5474638 0.6956522 1.00    0
\## XGboost       0.0000000 0.1600000 0.3750000 0.4422411 0.6763636 1.00    0
\## 
\## Neg\_Pred\_Value 
\##                   Min.   1st Qu.    Median      Mean 3rd Qu. Max. NA's
\## LogReg       0.5000000 0.6666667 1.0000000 0.8351852  1.0000    1    1
\## LDA          0.5000000 0.6666667 0.7500000 0.8000000  1.0000    1    0
\## GBM          0.3333333 0.7500000 1.0000000 0.8583333  1.0000    1    0
\## KNN          0.0000000 0.6875000 0.7750000 0.7633333  1.0000    1    0
\## RandomForest 0.5000000 0.7500000 0.7500000 0.7850000  0.9375    1    0
\## SVM          0.5000000 0.7625000 0.9000000 0.8516667  1.0000    1    0
\## XGboost      0.5000000 0.6166667 0.7083333 0.7533333  0.9375    1    0
\## 
\## Pos\_Pred\_Value 
\##                   Min.   1st Qu.    Median      Mean   3rd Qu. Max. NA's
\## LogReg       0.3333333 0.6666667 0.7500000 0.7500000 1.0000000 1.00    0
\## LDA          0.5000000 0.6666667 0.6666667 0.6574074 0.7500000 0.75    1
\## GBM          0.3333333 0.6666667 0.7500000 0.7333333 0.7500000 1.00    0
\## KNN          0.3333333 0.5250000 0.7083333 0.7350000 1.0000000 1.00    0
\## RandomForest 0.0000000 0.6666667 0.6666667 0.6833333 0.9166667 1.00    0
\## SVM          0.3333333 0.6166667 0.7083333 0.7516667 1.0000000 1.00    0
\## XGboost      0.5000000 0.5000000 0.7083333 0.7416667 1.0000000 1.00    0
\## 
\## Precision 
\##                   Min.   1st Qu.    Median      Mean   3rd Qu. Max. NA's
\## LogReg       0.3333333 0.6666667 0.7500000 0.7500000 1.0000000 1.00    0
\## LDA          0.5000000 0.6666667 0.6666667 0.6574074 0.7500000 0.75    1
\## GBM          0.3333333 0.6666667 0.7500000 0.7333333 0.7500000 1.00    0
\## KNN          0.3333333 0.5250000 0.7083333 0.7350000 1.0000000 1.00    0
\## RandomForest 0.0000000 0.6666667 0.6666667 0.6833333 0.9166667 1.00    0
\## SVM          0.3333333 0.6166667 0.7083333 0.7516667 1.0000000 1.00    0
\## XGboost      0.5000000 0.5000000 0.7083333 0.7416667 1.0000000 1.00    0
\## 
\## Recall 
\##                   Min.   1st Qu.    Median      Mean   3rd Qu. Max. NA's
\## LogReg       0.3333333 0.6666667 1.0000000 0.8000000 1.0000000    1    0
\## LDA          0.0000000 0.6666667 0.6666667 0.7000000 1.0000000    1    0
\## GBM          0.3333333 0.6666667 1.0000000 0.8333333 1.0000000    1    0
\## KNN          0.5000000 0.6666667 0.6666667 0.7833333 1.0000000    1    0
\## RandomForest 0.0000000 0.6666667 0.6666667 0.7000000 0.9166667    1    0
\## SVM          0.3333333 0.6666667 0.8333333 0.7833333 1.0000000    1    0
\## XGboost      0.3333333 0.3750000 0.6666667 0.6500000 0.9166667    1    0
\## 
\## Sensitivity 
\##                   Min.   1st Qu.    Median      Mean   3rd Qu. Max. NA's
\## LogReg       0.3333333 0.6666667 1.0000000 0.8000000 1.0000000    1    0
\## LDA          0.0000000 0.6666667 0.6666667 0.7000000 1.0000000    1    0
\## GBM          0.3333333 0.6666667 1.0000000 0.8333333 1.0000000    1    0
\## KNN          0.5000000 0.6666667 0.6666667 0.7833333 1.0000000    1    0
\## RandomForest 0.0000000 0.6666667 0.6666667 0.7000000 0.9166667    1    0
\## SVM          0.3333333 0.6666667 0.8333333 0.7833333 1.0000000    1    0
\## XGboost      0.3333333 0.3750000 0.6666667 0.6500000 0.9166667    1    0
\## 
\## Specificity 
\##                   Min.   1st Qu.    Median      Mean 3rd Qu. Max. NA's
\## LogReg       0.0000000 0.6666667 0.7500000 0.7333333  1.0000    1    0
\## LDA          0.5000000 0.6666667 0.7083333 0.7166667  0.7500    1    0
\## GBM          0.3333333 0.7500000 0.7500000 0.7500000  0.7500    1    0
\## KNN          0.0000000 0.5416667 0.7500000 0.7000000  1.0000    1    0
\## RandomForest 0.3333333 0.7500000 0.7500000 0.7750000  0.9375    1    0
\## SVM          0.3333333 0.6666667 0.7500000 0.7666667  1.0000    1    0
\## XGboost      0.5000000 0.6875000 0.7500000 0.7916667  1.0000    1    0

*# Convert to factor and ensure both have the same levels*
unique\_classes <- **sort**(**unique**(**c**(Y\_train, Y\_test)))

*# Evaluate the GBM model on the test set*
predictions\_baseline <- **predict**(model\_gbm, newdata = X\_test)

predictions\_baseline <- **factor**(predictions\_baseline, levels = unique\_classes)
Y\_test <- **factor**(Y\_test, levels = unique\_classes)

*# Generating confusion matrix*
confusionMatrix\_baseline <- **confusionMatrix**(predictions\_baseline, Y\_test)
performance\_baseline <- confusionMatrix\_baseline**$**overall['Accuracy']

*# Assuming pca\_results and k are already defined as before*
kmeans\_result <- **kmeans**(pca\_results**$**x[, 1**:**2], centers = k)

*# Add the cluster labels to your original dataset as a new feature*
team.gene.subset\_sup**$**ClusterLabel <- **factor**(kmeans\_result**$**cluster)

*# Assuming you're using the same split as before*
X <- **cbind**(team.gene.subset\_sup[, **-ncol**(team.gene.subset\_sup)], ClusterLabel = team.gene.subset\_sup**$**ClusterLabel)
Y <- team.gene.subset\_sup[, **ncol**(team.gene.subset\_sup)]

*# Splitting the data into training and testing sets again to include ClusterLabel*
**set.seed**(seed\_val)  *# Ensure reproducibility*
trainIndex <- **createDataPartition**(Y, p = .8, list = FALSE)
X\_train <- X[trainIndex, ]
Y\_train <- Y[trainIndex]
X\_test <- X[**-**trainIndex, ]
Y\_test <- Y[**-**trainIndex]

*# control settings remain the same*
model\_gbm\_with\_clusters <- **train**(Y\_train **~** ., data = **data.frame**(X\_train, Y\_train), 
`                                 `method = "gbm", trControl = control, metric = metric, tuneLength = 5,verbose = FALSE)

*# Predict and evaluate*
predictions\_with\_clusters <- **predict**(model\_gbm\_with\_clusters, newdata = X\_test)
confusionMatrix\_with\_clusters <- **confusionMatrix**(predictions\_with\_clusters, Y\_test)

*# Ensure you extract accuracy as numeric values*
accuracy\_baseline <- confusionMatrix\_baseline**$**overall["Accuracy"]
accuracy\_with\_clusters <- confusionMatrix\_with\_clusters**$**overall["Accuracy"]

*# Calculate performance improvement for accuracy*
performance\_improvement <- accuracy\_with\_clusters **-** accuracy\_baseline

*# Display the performance improvement*
**cat**("Performance Improvement (Accuracy):", performance\_improvement, "**\n**")

\## Performance Improvement (Accuracy): 0.154
#
**MA321-7-SP**

