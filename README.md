# cs181-homework4
Spring 2021 Machine Learning ([CS 181](https://harvard-ml-courses.github.io/cs181-web-2021/)) [Homework 4](https://github.com/harvard-ml-courses/cs181-s21-homeworks/tree/main/hw4)

## Problem Topics

> Solutions contained in the `personal-solutions` folder

1. Fitting a Support Vector Machine (SVM) by solving for the discriminant function and decision boundary
2. Implementing and comparing K-Means Clustering and Hierarchical Agglomerative Clustering (HAC) for classifying images
3. Computer Science ethics covering discrimination and fairness in machine learning settings
4. Computer Science ethics covering the moral obligation of machine learning practitioners in combatting algorithmic discrimination

---

## Code

> Implementation contained in the `code` folder

#### problem1-Plot-Basis-Transformed-Data.py

- Plots basis transformed training data and the optimal decision boundary

#### problem2-KMeans-HAC-Clustering.py

- Implements K-Means and HAC to classify images from the [MNIST dataset] (http://yann.lecun.com/exdb/mnist/) (a collection of handwritten digits used as a benchmark for image recognition)
- Implements 3 K-Means Classifiers where K = 5, 10, 20
- Implements 3 HAC Classifiers using min, max, and centroid-based linkages
