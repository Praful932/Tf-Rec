
![Tf-rec](https://user-images.githubusercontent.com/45713796/101985944-ee9b5500-3cb0-11eb-882c-ae5145434b80.png)

**Tf-Rec** is a python💻 package for building⚒ Recommender Systems. It is built on top of **Keras** and **Tensorflow 2** to utilize *GPU Acceleration* during training.
# Contents
- Why Tf-Rec? 🧐
- Installation ⚡
- Documentation 📝
- Supported Algorithms 🎯
- Benchmark 🔥
- Contribute 😇

### Why Tf-Rec? 🧐
There are several open source libraries which implement popular recommender algorithms in, infact this library is inspired by them - **Surprise**, **Funk-SVD** and more. However, there is bottleneck in training time, when the training data is huge. This can be solved by using ready frameworks like **Tensorflow 2** & **Keras** which support running computations on GPU thus delivering speed and higher throughput. Building on top of such frameworks also provide us with off the shelf capabilities, such as in case of Tensorflow 2 - Exporting Model, Data API and much more.

### Installation ⚡
The package is available on PyPi:

``pip install tfrec``

### Documentation 📝


### Supported Algorithms 🎯
Currently the library supports these algorithms:
- **SVD** - Despite the Name, it is different from the Matrix Eigen Decomposition of Assymmetric Matrices. In a gist, it approximates a vector for each user and each item. The vector contains latent factors which signify for brevity sake, if the item is a movie the movie vector would represent - how much the movie contains action or romance likewise. Similarly for the user.
The predicted rating is given by:
![](https://latex.codecogs.com/png.latex?\hat{r}_{u,&space;i}&space;=&space;\bar{r}&space;&plus;&space;b_{u}&space;&plus;&space;b_{i}&space;&plus;&space;\sum_{f=1}^{F}&space;p_{u,&space;f}&space;*&space;q_{i,&space;f})

- **SVD++** - This is an extension of SVD which incorporates implicit feedback, by also taking into account the interactions between the user and the item by involving another factor. More Precisely, it takes into account the fact that the user has rated an item itself as a preference than an item which the user has not rated.
The predicted rating is given by:
![image](https://user-images.githubusercontent.com/45713796/101982506-6ca03180-3c9a-11eb-8285-f9f243ab877c.png)

### Benchmark 🔥
Both of the algorithms were tested on Google Collab using a GPU Runtime. The dataset used was the MovieLens-100k. Default parameters were used for intilization of Model. Optimizer used was **Adam** and batch size used was **128**. 
These are the 5-Fold Cross Validation Scores:

|   Algorithm  | Mean MAE | Mean RMSE | Time per Epoch |
| ------------ | -------- | --------  | -------------- |
| **SVD**      | 0.6701   | 0.8694    |  < 3 sec       |
| **SVD++**    | 0.6838   | 0.8862    |  < 45 sec      |
### Contribute 😇 
