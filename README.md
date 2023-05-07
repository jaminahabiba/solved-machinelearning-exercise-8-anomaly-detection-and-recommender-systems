Download Link: https://assignmentchef.com/product/solved-machinelearning-exercise-8-anomaly-detection-and-recommender-systems
<br>
<h1>Introduction</h1>

In this exercise, you will implement the anomaly detection algorithm and apply it to detect failing servers on a network. In the second part, you will use collaborative filtering to build a recommender system for movies. Before starting on the programming exercise, we strongly recommend watching the video lectures and completing the review questions for the associated topics.

To get started with the exercise, you will need to download the starter code and unzip its contents to the directory where you wish to complete the exercise. If needed, use the cd command in Octave/MATLAB to change to this directory before starting this exercise.

You can also find instructions for installing Octave/MATLAB in the “Environment Setup Instructions” of the course website.

<h2>Files included in this exercise</h2>

ex8.m – Octave/MATLAB script for first part of exercise ex8 cofi.m – Octave/MATLAB script for second part of exercise ex8data1.mat – First example Dataset for anomaly detection ex8data2.mat – Second example Dataset for anomaly detection ex8 movies.mat – Movie Review Dataset ex8 movieParams.mat – Parameters provided for debugging multivariateGaussian.m – Computes the probability density function for a Gaussian distribution visualizeFit.m – 2D plot of a Gaussian distribution and a dataset checkCostFunction.m – Gradient checking for collaborative filtering computeNumericalGradient.m – Numerically compute gradients fmincg.m – Function minimization routine (similar to fminunc) loadMovieList.m – Loads the list of movies into a cell-array movie ids.txt – List of movies normalizeRatings.m – Mean normalization for collaborative filtering submit.m – Submission script that sends your solutions to our servers [<em>?</em>] estimateGaussian.m – Estimate the parameters of a Gaussian distribution with a diagonal covariance matrix

[<em>?</em>] selectThreshold.m – Find a threshold for anomaly detection [<em>?</em>] cofiCostFunc.m – Implement the cost function for collaborative filtering

<em>? </em>indicates files you will need to complete

Throughout the first part of the exercise (anomaly detection) you will be using the script ex8.m. For the second part of collaborative filtering, you will use ex8 cofi.m. These scripts set up the dataset for the problems and make calls to functions that you will write. You are only required to modify functions in other files, by following the instructions in this assignment.

<h2>Where to get help</h2>

The exercises in this course use Octave<a href="#_ftn1" name="_ftnref1"><sup>[1]</sup></a> or MATLAB, a high-level programming language well-suited for numerical computations. If you do not have Octave or MATLAB installed, please refer to the installation instructions in the “Environment Setup Instructions” of the course website.

At the Octave/MATLAB command line, typing help followed by a function name displays documentation for a built-in function. For example, help plot will bring up help information for plotting. Further documentation for Octave functions can be found at the <a href="https://www.gnu.org/software/octave/doc/interpreter/">Octave documentation pages</a><a href="https://www.gnu.org/software/octave/doc/interpreter/">.</a> MATLAB documentation can be found at the <a href="https://www.mathworks.com/help/matlab/?refresh=true">MATLAB documentation pages</a><a href="https://www.mathworks.com/help/matlab/?refresh=true">.</a>

We also strongly encourage using the online <strong>Discussions </strong>to discuss exercises with other students. However, do not look at any source code written by others or share your source code with others.

<h1>1          Anomaly detection</h1>

In this exercise, you will implement an anomaly detection algorithm to detect anomalous behavior in server computers. The features measure the throughput (mb/s) and latency (ms) of response of each server. While your servers were operating, you collected <em>m </em>= 307 examples of how they were behaving, and thus have an unlabeled dataset {<em>x</em><sup>(1)</sup><em>,…,x</em><sup>(<em>m</em>)</sup>}. You suspect that the vast majority of these examples are “normal” (non-anomalous) examples of the servers operating normally, but there might also be some examples of servers acting anomalously within this dataset.

You will use a Gaussian model to detect anomalous examples in your dataset. You will first start on a 2D dataset that will allow you to visualize what the algorithm is doing. On that dataset you will fit a Gaussian distribution and then find values that have very low probability and hence can be considered anomalies. After that, you will apply the anomaly detection algorithm to a larger dataset with many dimensions. You will be using ex8.m for this part of the exercise.

The first part of ex8.m will visualize the dataset as shown in Figure 1.

Figure 1: The first dataset.

<h2>1.1        Gaussian distribution</h2>

To perform anomaly detection, you will first need to fit a model to the data’s distribution.

Given a training set {<em>x</em><sup>(1)</sup><em>,…,x</em><sup>(<em>m</em>)</sup>} (where <em>x</em><sup>(<em>i</em>) </sup>∈ R<em><sup>n</sup></em>), you want to estimate the Gaussian distribution for each of the features <em>x<sub>i</sub></em>. For each feature <em>i </em>= 1<em>…n</em>, you need to find parameters <em>µ<sub>i </sub></em>and <em>σ<sub>i</sub></em><sup>2 </sup>that fit the data in the <em>i</em>-th dimension -th dimension of each example).

The Gaussian distribution is given by

<em>,</em>

where <em>µ </em>is the mean and <em>σ</em><sup>2 </sup>controls the variance.

<h2>1.2        Estimating parameters for a Gaussian</h2>

You can estimate the parameters, (<em>µ<sub>i</sub></em>, <em>σ<sub>i</sub></em><sup>2</sup>), of the <em>i</em>-th feature by using the following equations. To estimate the mean, you will use:

<em>,                                               </em>(1)

and for the variance you will use:

<em>.                                        </em>(2)

Your task is to complete the code in estimateGaussian.m. This function takes as input the data matrix X and should output an <em>n</em>-dimension vector mu that holds the mean of all the <em>n </em>features and another <em>n</em>-dimension vector sigma2 that holds the variances of all the features. You can implement this using a for-loop over every feature and every training example (though a vectorized implementation might be more efficient; feel free to use a vectorized implementation if you prefer). Note that in Octave/MATLAB, the var function will (by default) use, instead of , when computing <em>σ<sub>i</sub></em><sup>2</sup>.

Once you have completed the code in estimateGaussian.m, the next part of ex8.m will visualize the contours of the fitted Gaussian distribution. You should get a plot similar to Figure 2. From your plot, you can see that most of the examples are in the region with the highest probability, while the anomalous examples are in the regions with lower probabilities.

<em>You should now submit your solutions.</em>

Figure 2: The Gaussian distribution contours of the distribution fit to the dataset.

<h2>1.3        Selecting the threshold, <em>ε</em></h2>

Now that you have estimated the Gaussian parameters, you can investigate which examples have a very high probability given this distribution and which examples have a very low probability. The low probability examples are more likely to be the anomalies in our dataset. One way to determine which examples are anomalies is to select a threshold based on a cross validation set. In this part of the exercise, you will implement an algorithm to select the threshold <em>ε </em>using the <em>F</em><sub>1 </sub>score on a cross validation set.

You should now complete the code in selectThreshold.m. For this, we will use a cross validation set , where the label <em>y </em>= 1 corresponds to an anomalous example, and <em>y </em>= 0 corresponds to a normal example. For each cross validation example, we will compute). The vector of all of these probabilities) is passed to selectThreshold.m in the vector pval. The corresponding labels <em>y</em><sub>cv</sub><sup>(1)</sup><em>,…,y</em><sub>cv</sub><sup>(<em>m</em></sup><sup>cv) </sup>is passed to the same function in the vector yval.

The function selectThreshold.m should return two values; the first is the selected threshold <em>ε</em>. If an example <em>x </em>has a low probability <em>p</em>(<em>x</em>) <em>&lt; ε</em>, then it is considered to be an anomaly. The function should also return the <em>F</em><sub>1 </sub>score, which tells you how well you’re doing on finding the ground truth anomalies given a certain threshold. For many different values of <em>ε</em>, you will compute the resulting <em>F</em><sub>1 </sub>score by computing how many examples the current threshold classifies correctly and incorrectly.

The <em>F</em><sub>1 </sub>score is computed using precision (<em>prec</em>) and recall (<em>rec</em>):

<em>,                                            </em>(3)

You compute precision and recall by:

(4)

<em>,                                            </em>(5)

where

<ul>

 <li><em>tp </em>is the number of true positives: the ground truth label says it’s an anomaly and our algorithm correctly classified it as an anomaly.</li>

 <li><em>fp </em>is the number of false positives: the ground truth label says it’s not an anomaly, but our algorithm incorrectly classified it as an anomaly.</li>

 <li><em>fn </em>is the number of false negatives: the ground truth label says it’s an anomaly, but our algorithm incorrectly classified it as not being anomalous.</li>

</ul>

In the provided code selectThreshold.m, there is already a loop that will try many different values of <em>ε </em>and select the best <em>ε </em>based on the <em>F</em><sub>1 </sub>score.

You should now complete the code in selectThreshold.m. You can implement the computation of the F1 score using a for-loop over all the cross validation examples (to compute the values <em>tp</em>, <em>fp</em>, <em>fn</em>). You should see a value for epsilon of about 8.99e-05.

<table width="516">

 <tbody>

  <tr>

   <td width="516"><strong>Implementation Note: </strong>In order to compute <em>tp</em>, <em>fp </em>and <em>fn</em>, you may be able to use a vectorized implementation rather than loop over all the examples. This can be implemented by Octave/MATLAB’s equality test between a vector and a single number. If you have several binary values in an <em>n</em>-dimensional binary vector <em>v </em>∈ {0<em>,</em>1}<em><sup>n</sup></em>, you can find out how many values in this vector are 0 by using: sum(<em>v </em>== 0). You can also apply a logical and operator to such binary vectors. For instance, let cvPredictions be a binary vector of the size of your number of cross validation set, where the <em>i</em>-th element is 1 if your algorithm considers an anomaly, and 0 otherwise. You can then, for example, compute the number of false positives using: fp = sum((cvPredictions == 1) &amp; (yval == 0)).</td>

  </tr>

 </tbody>

</table>

Figure 3: The classified anomalies.

Once you have completed the code in selectThreshold.m, the next step in ex8.m will run your anomaly detection code and circle the anomalies in the plot (Figure 3).

<em>You should now submit your solutions.</em>

<h2>1.4        High dimensional dataset</h2>

The last part of the script ex8.m will run the anomaly detection algorithm you implemented on a more realistic and much harder dataset. In this dataset, each example is described by 11 features, capturing many more properties of your compute servers.

The script will use your code to estimate the Gaussian parameters (<em>µ<sub>i </sub></em>and <em>σ<sub>i</sub></em><sup>2</sup>), evaluate the probabilities for both the training data X from which you estimated the Gaussian parameters, and do so for the the cross-validation set Xval. Finally, it will use selectThreshold to find the best threshold <em>ε</em>.

You should see a value epsilon of about 1.38e-18, and 117 anomalies found.

<h1>2          Recommender Systems</h1>

In this part of the exercise, you will implement the collaborative filtering learning algorithm and apply it to a dataset of movie ratings.<a href="#_ftn2" name="_ftnref2"><sup>[2]</sup></a> This dataset consists of ratings on a scale of 1 to 5. The dataset has <em>n<sub>u </sub></em>= 943 users, and <em>n<sub>m </sub></em>= 1682 movies. For this part of the exercise, you will be working with the script ex8 cofi.m.

In the next parts of this exercise, you will implement the function cofiCostFunc.m that computes the collaborative fitlering objective function and gradient. After implementing the cost function and gradient, you will use fmincg.m to learn the parameters for collaborative filtering.

<h2>2.1        Movie ratings dataset</h2>

The first part of the script ex8 cofi.m will load the dataset ex8 movies.mat, providing the variables Y and R in your Octave/MATLAB environment.

The matrix <em>Y </em>(a num movies × num users matrix) stores the ratings <em>y</em><sup>(<em>i,j</em>)</sup>

(from 1 to 5). The matrix <em>R </em>is an binary-valued indicator matrix, where <em>R</em>(<em>i,j</em>) = 1 if user <em>j </em>gave a rating to movie <em>i</em>, and <em>R</em>(<em>i,j</em>) = 0 otherwise. The objective of collaborative filtering is to predict movie ratings for the movies that users have not yet rated, that is, the entries with <em>R</em>(<em>i,j</em>) = 0. This will allow us to recommend the movies with the highest predicted ratings to the user.

To help you understand the matrix Y, the script ex8 cofi.m will compute the average movie rating for the first movie (Toy Story) and output the average rating to the screen.

Throughout this part of the exercise, you will also be working with the matrices, X and Theta:

<table width="388">

 <tbody>

  <tr>

   <td width="189">                     (1))<em>T </em>— — (<em>x</em> — (<em>x</em>(2))<em>T </em>—  X = <sub>       </sub>…          <sub></sub><em>,</em>— (<em>x</em>(<em>n</em><em>m</em>))<em>T </em>—</td>

   <td width="199">                                 (1))<em>T </em>— — (<em>θ</em> — (<em>θ</em>(2))<em>T </em>—  Theta = <sub>     </sub>…          <sub></sub><em>.</em>— (<em>θ</em>(<em>n</em><em>u</em>))<em>T </em>—</td>

  </tr>

 </tbody>

</table>

The <em>i</em>-th row of X corresponds to the feature vector <em>x</em><sup>(<em>i</em>) </sup>for the <em>i</em>-th movie, and the <em>j</em>-th row of Theta corresponds to one parameter vector <em>θ</em><sup>(<em>j</em>)</sup>, for the <em>j</em>-th user. Both <em>x</em><sup>(<em>i</em>) </sup>and <em>θ</em><sup>(<em>j</em>) </sup>are <em>n</em>-dimensional vectors. For the purposes of this exercise, you will use <em>n </em>= 100, and therefore, <em>x</em><sup>(<em>i</em>) </sup>∈ R<sup>100 </sup>and <em>θ</em><sup>(<em>j</em>) </sup>∈ R<sup>100</sup>. Correspondingly, X is a <em>n<sub>m </sub></em>× 100 matrix and Theta is a <em>n<sub>u </sub></em>× 100 matrix.

<h2>2.2        Collaborative filtering learning algorithm</h2>

Now, you will start implementing the collaborative filtering learning algorithm. You will start by implementing the cost function (without regularization).

The collaborative filtering algorithm in the setting of movie recommendations considers a set of <em>n</em>-dimensional parameter vectors <em>x</em><sup>(1)</sup><em>,…,x</em><sup>(<em>n</em></sup><em><sup>m</sup></em><sup>) </sup>and <em>θ</em><sup>(1)</sup><em>,…,θ</em><sup>(<em>n</em></sup><em><sup>u</sup></em><sup>)</sup>, where the model predicts the rating for movie <em>i </em>by user <em>j </em>as <em>y</em><sup>(<em>i,j</em>) </sup>= (<em>θ</em><sup>(<em>j</em>)</sup>)<em><sup>T </sup>x</em><sup>(<em>i</em>)</sup>. Given a dataset that consists of a set of ratings produced by some users on some movies, you wish to learn the parameter vectors

<em>x</em><sup>(1)</sup><em>,…,x</em><sup>(<em>n</em></sup><em><sup>m</sup></em><sup>)</sup><em>,θ</em><sup>(1)</sup><em>,…,θ</em><sup>(<em>n</em></sup><em><sup>u</sup></em><sup>) </sup>that produce the best fit (minimizes the squared error).

You will complete the code in cofiCostFunc.m to compute the cost function and gradient for collaborative filtering. Note that the parameters to the function (i.e., the values that you are trying to learn) are X and Theta. In order to use an off-the-shelf minimizer such as fmincg, the cost function has been set up to unroll the parameters into a single vector params. You had previously used the same vector unrolling method in the neural networks programming exercise.

<h3>2.2.1       Collaborative filtering cost function</h3>

The collaborative filtering cost function (without regularization) is given by

<em>.</em>

You should now modify cofiCostFunc.m to return this cost in the variable J. Note that you should be accumulating the cost for user <em>j </em>and movie <em>i </em>only if <em>R</em>(<em>i,j</em>) = 1.

After you have completed the function, the script ex8 cofi.m will run your cost function. You should expect to see an output of 22<em>.</em>22.

<em>You should now submit your solutions.</em>

<strong>Implementation Note: </strong>We strongly encourage you to use a vectorized implementation to compute <em>J</em>, since it will later by called many times by the optimization package fmincg. As usual, it might be easiest to first write a non-vectorized implementation (to make sure you have the right answer), and the modify it to become a vectorized implementation (checking that the vectorization steps don’t change your algorithm’s output). To come up with a vectorized implementation, the following tip might be helpful: You can use the R matrix to set selected entries to 0. For example, R .* M will do an element-wise multiplication between M and R; since R only has elements with values either 0 or 1, this has the effect of setting the elements of M to 0 only when the corresponding value in R is 0. Hence, sum(sum(R.*M)) is the sum of all the elements of M for which the corresponding element in R equals 1.

<h3>2.2.2       Collaborative filtering gradient</h3>

Now, you should implement the gradient (without regularization). Specifically, you should complete the code in cofiCostFunc.m to return the variables X grad and Theta grad. Note that X grad should be a matrix of the same size as X and similarly, Theta grad is a matrix of the same size as Theta. The gradients of the cost function is given by:

<em>.</em>

Note that the function returns the gradient for both sets of variables by unrolling them into a single vector. After you have completed the code to compute the gradients, the script ex8 cofi.m will run a gradient check (checkCostFunction) to numerically check the implementation of your gradients.<a href="#_ftn3" name="_ftnref3"><sup>[3]</sup></a> If your implementation is correct, you should find that the analytical and numerical gradients match up closely.

<em>You should now submit your solutions.</em>




<table width="516">

 <tbody>

  <tr>

   <td width="516"><strong>Implementation Note: </strong>You can get full credit for this assignment without using a vectorized implementation, but your code will run much more slowly (a small number of hours), and so we recommend that you try to vectorize your implementation.To get started, you can implement the gradient with a for-loop over movies (for computing ) and a for-loop over users (for computing ). When you first implement the gradient, you might start with an unvectorized version, by implementing another inner for-loop that computes each element in the summation. After you have completed the gradient computation this way, you should try to vectorize your implementation (vectorize the inner for-loops), so that you’re left with only two for-loops (one for looping over movies to computefor each movie, and one for looping over users to compute  for each user).</td>

  </tr>

 </tbody>

</table>







<table width="516">

 <tbody>

  <tr>

   <td width="516"><strong>Implementation Tip: </strong>To perform the vectorization, you might find this helpful: You should come up with a way to compute all the derivatives associated with(i.e., the derivative terms associated with the feature vector <em>x</em><sup>(<em>i</em>)</sup>) at the same time. Let us define the derivatives for the feature vector of the <em>i</em>-th movie as:(XgradTo vectorize the above expression, you can start by indexing into Theta and Y to select only the elements of interests (that is, those with <em>r</em>(<em>i,j</em>) = 1). Intuitively, when you consider the features for the <em>i</em>-th movie, you only need to be concern about the users who had given ratings to the movie, and this allows you to remove all the other users from Theta and Y.Concretely, you can set idx = find(R(i, :)==1) to be a list of all the users that have rated movie <em>i</em>. This will allow you to create the temporary matrices Theta<sub>temp </sub>= Theta(idx<em>,</em>&#x1f642; and Y<sub>temp </sub>= Y(i<em>,</em>idx) that index into Theta and Y to give you only the set of users which have rated the <em>i</em>-th movie. This will allow you to write the derivatives as:Xgrad(i<em>,</em>&#x1f642; = (X(i<em>,</em>&#x1f642; ∗ ThetaTtemp − Ytemp) ∗ Thetatemp<em>.</em>(Note: The vectorized computation above returns a row-vector instead.)After you have vectorized the computations of the derivatives with respect to <em>x</em><sup>(<em>i</em>)</sup>, you should use a similar method to vectorize the derivatives with respect to <em>θ</em><sup>(<em>j</em>) </sup>as well.</td>

  </tr>

 </tbody>

</table>

<h3>2.2.3       Regularized cost function</h3>

The cost function for collaborative filtering with regularization is given by <em> .</em>

You should now add regularization to your original computations of the cost function, <em>J</em>. After you are done, the script ex8 cofi.m will run your regularized cost function, and you should expect to see a cost of about 31.34.

<em>You should now submit your solutions.</em>

<h3>2.2.4       Regularized gradient</h3>

Now that you have implemented the regularized cost function, you should proceed to implement regularization for the gradient. You should add to your implementation in cofiCostFunc.m to return the regularized gradient by adding the contributions from the regularization terms. Note that the gradients for the regularized cost function is given by:

<em>.</em>

This means that you just need to add <em>λx</em><sup>(<em>i</em>) </sup>to the X grad(i,:) variable described earlier, and add <em>λθ</em><sup>(<em>j</em>) </sup>to the Theta grad(j,:) variable described earlier.

After you have completed the code to compute the gradients, the script ex8 cofi.m will run another gradient check (checkCostFunction) to numerically check the implementation of your gradients.

<em>You should now submit your solutions.</em>

<h2>2.3        Learning movie recommendations</h2>

After you have finished implementing the collaborative filtering cost function and gradient, you can now start training your algorithm to make movie recommendations for yourself. In the next part of the ex8 cofi.m script, you can enter your own movie preferences, so that later when the algorithm runs, you can get your own movie recommendations! We have filled out some values according to our own preferences, but you should change this according to your own tastes. The list of all movies and their number in the dataset can be found listed in the file movie idx.txt.

<h3>2.3.1       Recommendations</h3>

Top recommendations for you:

Predicting rating 9.0 for movie Titanic (1997)

Predicting rating 8.9 for movie Star Wars (1977)

Predicting rating 8.8 for movie Shawshank Redemption, The (1994)

Predicting rating 8.5 for movie As Good As It Gets (1997)

Predicting rating 8.5 for movie Good Will Hunting (1997)

Predicting rating 8.5 for movie Usual Suspects, The (1995)

Predicting rating 8.5 for movie Schindler’s List (1993)

Predicting rating 8.4 for movie Raiders of the Lost Ark (1981)

Predicting rating 8.4 for movie Empire Strikes Back, The (1980)

Predicting rating 8.4 for movie Braveheart (1995)

Original ratings provided:

Rated 4 for Toy Story (1995)

Rated 3 for Twelve Monkeys (1995)

Rated 5 for Usual Suspects, The (1995)

Rated 4 for Outbreak (1995)

Rated 5 for Shawshank Redemption, The (1994)

Rated 3 for While You Were Sleeping (1995)

Rated 5 for Forrest Gump (1994)

Rated 2 for Silence of the Lambs, The (1991)

Rated 4 for Alien (1979)

Rated 5 for Die Hard 2 (1990)

Rated 5 for Sphere (1998)

Figure 4: Movie recommendations

After the additional ratings have been added to the dataset, the script will proceed to train the collaborative filtering model. This will learn the parameters X and Theta. To predict the rating of movie <em>i </em>for user <em>j</em>, you need to compute (<em>θ</em><sup>(<em>j</em>)</sup>)<em><sup>T </sup>x</em><sup>(<em>i</em>)</sup>. The next part of the script computes the ratings for all the movies and users and displays the movies that it recommends (Figure 4), according to ratings that were entered earlier in the script. Note that you might obtain a different set of the predictions due to different random initializations.




<a href="#_ftnref1" name="_ftn1">[1]</a> Octave is a free alternative to MATLAB. For the programming exercises, you are free to use either Octave or MATLAB.

<a href="#_ftnref2" name="_ftn2">[2]</a> <a href="http://www.grouplens.org/node/73/">MovieLens 100k Dataset</a> from GroupLens Research.

<a href="#_ftnref3" name="_ftn3">[3]</a> This is similar to the numerical check that you used in the neural networks exercise.<img decoding="async" data-recalc-dims="1" data-src="https://i0.wp.com/www.ankitcodinghub.com/wp-content/uploads/2022/04/511.png?w=980&amp;ssl=1" class="lazyload" src="data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==">

 <noscript>

  <img decoding="async" src="https://i0.wp.com/www.ankitcodinghub.com/wp-content/uploads/2022/04/511.png?w=980&amp;ssl=1" data-recalc-dims="1">

 </noscript>