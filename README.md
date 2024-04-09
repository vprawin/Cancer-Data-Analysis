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

![C:\Users\farjaad\Downloads\flow.drawio.png](Image/Aspose.Words.96696cac-afe9-4f16-b5d4-0b03d3031a7b.002.png)
# <a name="_toc161238050"></a>**Discussion**
## <a name="_toc161238051"></a>**Unsupervised learning model**
We will be accounting for 90% of total cumulative variation. This variation is given by first 49 principal components. We will now use unsupervised learning methods for making clusters:
- ### <a name="_toc161238052"></a>***K-means Clustering***
K-means randomly selects clusters and then classifies all the data points in the clusters based on the centroids. 

***Determining the number of clusters***

- **Elbow Method**

![A graph of a number of clusters

Description automatically generated](Image/Aspose.Words.96696cac-afe9-4f16-b5d4-0b03d3031a7b.003.png)

The above graph tells us about the optimal number of clusters. Based on the WCSS values, we observe a notable change in the slope around k = 3 or k = 4, indicating that 3 or 4 clusters is a reasonable choice. 

- **Silhouette Score Method**:

![A graph with numbers and lines

Description automatically generated](Image/Aspose.Words.96696cac-afe9-4f16-b5d4-0b03d3031a7b.004.png)

The silhouette score falls in –1 to 1. Any value which is closer to 1 is better. According to this, the maximum average silhouette widths are recorded for k = 3 (0.7522921). Breaking our PCA-reduced gene expression data into three clusters is expected to offer the most distinct and well-separated grouping. After k = 3, the scores start to decline, indicating that more clusters do not enhance the ability to distinguish between them. From above 2 methods, we are finalizing the value of **k=3.**

Following is the plot for the 3-clusters we got using K-means.

![A graph with numbers and dots

Description automatically generated](Image/Aspose.Words.96696cac-afe9-4f16-b5d4-0b03d3031a7b.005.png)
- ### ![](Image/Aspose.Words.96696cac-afe9-4f16-b5d4-0b03d3031a7b.006.png)<a name="_toc161238053"></a>***Hierarchical Clustering:***
In unsupervised learning, hierarchical clustering is a strategy that groups similar data points into clusters according to their similarities or distances from one another. It produces a hierarchy of clusters, each of which has a dendrogram connecting them. The distance metric that we will be using is ‘Euclidean’ and the linkage with ‘complete’.
### <a name="_toc161238054"></a>***TSNE Model:*** 
In machine learning and data visualisation, t-distributed Stochastic Neighbour Embedding (t-SNE) is a well-liked dimensionality reduction method. When high-dimensional data is visualised in lower-dimensional areas, usually 2D or 3D, it works especially well. In order to identify clusters, patterns, and correlations within the data, t-SNE is helpful since it maintains local commonalities between data points in the original high-dimensional space.

**Our result:**

![](Image/Aspose.Words.96696cac-afe9-4f16-b5d4-0b03d3031a7b.007.png)

From above plot, we are getting 3 clusters.
## <a name="_toc161238055"></a>**Supervised Learning Model:**
We have used ‘caret’ library in r to the model training. Furthermore, we have made use of “TrControl” parameter in caret library for performing cross-validation and “TuneGrid” parameter for performing hyperparameter tuning. The results obtained have then been converted into a resampled list, summary of which tells us about various evaluation metrics such as the ‘Accuracy’, ‘F-1 score’, ‘Recall’, ‘Precision’, ‘Sensitivity’, ‘Specificity’. By resampling we get the above-mentioned metrics in the form of minimum, 1st quartile, median, mean, 3rd quartile and maximum metrics. We have decided to with median as our final evaluation metric as it is more suitable and accurate than mean.

We won’t be using Quadratic Determinant Analysis as it is not suitable for our  datasetAccording to the book "Introduction to Statistical Learning with R", the basis on which QDA or LDA should be selected is the Bias-Variance tradeoff.LDA tends to be a better bet than QDA if there are relatively few training observations and so reducing variance is crucial. In contrast, QDA is recommended if the training set is very large, so that the variance of the classifer is not a major concern.
## <a name="_toc161238056"></a>**Model Training and Hyperparameter Tuning:**
- ### <a name="_toc161238057"></a>***Logistic Regression:***
The most common classification method. We used the “glmnet” package to train the data using logistic regression.

![A screenshot of a computer code

Description automatically generated](Image/Aspose.Words.96696cac-afe9-4f16-b5d4-0b03d3031a7b.008.png)

Lr\_hyperparameters, contains the parameters inside the logistic regression model which are used for tuning.

Logistic regression gives us a median accuracy of 0.833
- ### <a name="_toc161238058"></a>***Linear Discriminant Analysis***
LDA is primarily a dimensionality reduction technique which helps us train complex high dimensional problems. The code for LDA under the “caret” package is below.The median accuracy we get is 0.714.

![Pic](Image/Aspose.Words.96696cac-afe9-4f16-b5d4-0b03d3031a7b.009.png)
- ### <a name="_toc161238059"></a>***K-Nearest Neighbors***
KNN considers the nearest neighbors and their class for classification of any new datapoint. The number of neighbors is the main parameter which is used to tune. Accuracy obtained by KNN is 0.75

![Pic](Image/Aspose.Words.96696cac-afe9-4f16-b5d4-0b03d3031a7b.010.png)
- ### <a name="_toc161238060"></a>***Random Forest***
Random Forest is an ensemble technique which allows to capture complex relationships. 

![A screenshot of a computer program

Description automatically generated](Image/Aspose.Words.96696cac-afe9-4f16-b5d4-0b03d3031a7b.011.png)

For our dataset, Random forest was leading to over fit, so we have selected cross validation only. The median accuracy is 0.714
- ### <a name="_toc161238061"></a>***Support Vector Classifier***
![A screenshot of a computer code

Description automatically generated](Image/Aspose.Words.96696cac-afe9-4f16-b5d4-0b03d3031a7b.012.png)

SVM demonstrated competitive performance, particularly in scenarios with high-dimensional data and complex decision boundaries. Its ability to handle non-linear data through kernel tricks made it a valuable model for various classification tasks. The hyperparameter tuning for SVM contains the optimization factor. The accuracy is 0.833
- ### <a name="_toc161238062"></a>***Gradient Boosting Model***
![A computer code with black text

Description automatically generated](Image/Aspose.Words.96696cac-afe9-4f16-b5d4-0b03d3031a7b.013.png)

GBM can handle heterogeneous features and is robust to outliers due to its ensemble nature. It typically provides high predictive accuracy and can capture complex interactions between variables.

The median accuracy for GBM is  0.845.
- ### <a name="_toc161238063"></a>***Extreme Gradient* Boosting**  
![A screenshot of a computer code

Description automatically generated](Image/Aspose.Words.96696cac-afe9-4f16-b5d4-0b03d3031a7b.014.png)

XGboost is one of the most recent boosting methods and is widely used in Machine Learning. One of the best advantages of XGboost is that it stays robust even for the data set with missing values.

![A screenshot of a computer

Description automatically generated](Image/Aspose.Words.96696cac-afe9-4f16-b5d4-0b03d3031a7b.015.png)

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

![](Image/Aspose.Words.96696cac-afe9-4f16-b5d4-0b03d3031a7b.016.png)

So, from the above resampling done on the supervised learning model, we can see that Gradient Boosting Model gives us the best accuracy. It performs well across a range of metrics, suggesting that it can effectively manage the subtleties of your dataset.
# <a name="_toc161238070"></a>**Conclusion**
We will now examine the accuracies of the model trained using cluster labels with those of the original model. A rise in these measures can mean that the unsupervised model-created clusters have been successful in gathering more data that is useful for the prediction task. So for that, we will be dividing our dataset into train and test sets. Our train set will consist of 80% of the dataset and 20% will be in the test set. Train-Test ensures that the model does not overfit. We have also added cross-validation with 10 folds. Cross-validation ensures that whole data is properly considered for training and testing, so that biases are avoided. Gradient Boosting, our best performing model, is considered here along with the cluster from the unsupervised clustering, We now do our predictions on the test set. There is a rise of 0.154 (15%) in accuracy. 
# <a name="_toc161238071"></a>**References**
[1] Caret Package Documentation [**https://www.rdocumentation.org/packages/caret/versions/6.0-94**](https://www.rdocumentation.org/packages/caret/versions/6.0-94)

[2] XGBoost Documentation [**https://xgboost.readthedocs.io/en/stable/**](https://xgboost.readthedocs.io/en/stable/)

[3] TSNE - Distill [**https://distill.pub/**](https://distill.pub/)

[4] T-squared - An Introduction to Multivariate Statistical Analysis Third Edition T. W. ANDERSON Stanford University Department of Statistics, Stanford, CA

[5] QDA Gareth James - Daniela Witten - Trevor Hastie - Robert Tibshirani An Introduction to Statistical Learning with Applications in R Second Edition
