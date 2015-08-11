---
title: 
layout: default
---

# A Beginner's Guide to Eigenvectors, PCA, Covariance and Entropy

Content: 

* [Linear Transformations](#linear)
* [Principal Component Analysis (PCA)](#principal)
* [Covariance Matrix](#covariance)
* [Entropy & Information Gain](#entropy)
* [Resources](#resources)

This post introduces eigenvectors and their relationship to matrices in plain language and without a great deal of math. 

The *eigen* in eigenvector comes from German, and it means something like “very own.” For example, in German, “mein eigenes Auto” means “my very own car.” So eigen denotes a special relationship between two things. Something particular, characteristic and definitive. This car, or this vector, is mine and not someone else’s.

Matrices, in linear algebra, are simply rectangular arrays of numbers, a collection of scalar values between brackets, like a spreadsheet. Most square matrices (e.g. 2 x 2 or 3 x 3) have eigenvectors, and they have a very special relationship with them, a bit like Germans have with their cars. 

## Linear Transformations

We’ll define that relationship after a brief detour into what matrices do, and how they relate to other numbers.

Matrices are useful because you can do things with them like add and multiply. If you multiply a vector *v* by a matrix *A*, you get another vector *b*, and you could say that the matrix performed a linear transformation on the input vector. 

*Av = b* 

It [maps](https://en.wikipedia.org/wiki/Linear_map) one set of numbers *v* to another, *b*.  We’ll illustrate with a concrete example. 

![Alt text](../img/eigen_matrix.png)

So *A* turned *v* into *b*. In a two-dimensional plane, the coordinates of the vector changed. In the graph below, the short, low line is *v*, and the long, high one is *b*.

![Alt text](../img/two_vectors.png)

You could feed one vector after another into matrix A, and each would be projected onto a new space that stretches higher and farther to the right. 

Imagine that all the input vectors *v* live in a normal grid, like this:

![Alt text](../img/space_1.png)

And all the matrix projects them into a new space like the one below, which holds the output vectors *b*:

![Alt text](../img/space_2.png)

Here you can see the two spaces juxtaposed:

![Alt text](../img/two_spaces.png)

(*Credit: William Gould, Stata Blog*)

And here’s an animation that shows the matrix's work transforming one space to another (the blue lines are eigenvectors):

<iframe src="https://upload.wikimedia.org/wikipedia/commons/0/06/Eigenvectors.gif" width="100%" height="300px;" style="border:none;"></iframe>

You can imagine a matrix like a gust of wind, an invisible force that produces a visible result. And a gust of wind must blow in a certain direction. The eigenvector tells you the direction the matrix is blowing in. 

![Alt text](../img/mona_lisa_eigenvector.png)

So out of all the vectors affected by a matrix blowing through one space, which one is the eigenvector? It’s the one that doesn’t change direction; that is, the eigenvector is already pointing in the same direction that the matrix is pushing all vectors toward. An eigenvector is a like a weathervane, pointing in the direction of the wind. 

You could also say that eigenvectors are axes along which linear transformation acts, stretching or compressing input vectors. They are the lines of change that represent the action of the larger matrix.

Notice we’re using the plural – axes and lines. Just as a German may have a Volkswagen for grocery shopping, a Mercedes for business travel, and a Porsche for joy rides, square matrices have as many eigenvectors as they have linearly independent dimensions; i.e. a 2 x 2 matrix would have two eigenvectors, a 3 x 3 matrix three, and an n x n matrix would have n eigenvectors, each one representing its line of action in one dimension. 

Because eigenvectors distill the axes of principal force that a matrix moves along, they are useful in matrix decomposition, or matrix vectorization; i.e. the representation of a matrix by a vector. In that sense, they perform the same task as autoencoders. 

To quote Yoshua Bengio:

    Many mathematical objects can be understood better by breaking them into constituent parts, or ﬁnding some properties of them that are universal, not caused by the way we choose to represent them.
    
    For example, integers can be decomposed into prime factors. The way we represent the number 12 will change depending on whether we write it in base ten or in binary, but it will always be true that 12 = 2 × 2 × 3. 
    
    From this representation we can conclude useful properties, such as that 12 is not divisible by 5, or that any integer multiple of 12 will be divisible by 3.
    
    Much as we can discover something about the true nature of an integer by decomposing it into prime factors, we can also decompose matrices in ways that show us information about their functional properties that is not obvious from the representation of the matrix as an array of elements.
    
    One of the most widely used kinds of matrix decomposition is called eigen-decomposition, in which we decompose a matrix into a set of eigenvectors and eigenvalues.

## Principal Component Analysis (PCA)

PCA is a tool for finding patterns in high-dimensional data such as images. Old-school machine-learning aficionados occasionally use PCA to preprocess data for their neural networks. By centering,  rotating and scaling data, PCA reduces its dimensionality and can improve the neural network's convergence speed and the overall quality of results. 

To get to PCA, we’re going to quickly gloss some basic statistical ideas -- mean, standard deviation, variance and covariance -- so we can weave them together later. Their equations are closely related.

*Mean* is simply the average value of *x* in the set X, which is found by dividing the sum of all data points by the number of data points, *n*.

![Alt text](../img/mean.png)

*Standard deviation* is "the average distance from the mean of the data set to a point." In the equation below, we have the sum of the differences between each datapoint and the mean in the numerator, and number of data points (minus one) in the denominator, producing the average.

![Alt text](../img/standard_deviation.png)

Variance is the measure of the data's spread. If I take a team of Dutch basketball players and measure their height, those measurements won’t have a lot of variance. They’ll all be grouped above six feet. 

But if I throw in a classroom of psychotic kindergartners as well as a few CIA spies that have been carefully selected for appearing average in every way, then the combined group’s height measurements will have a lot of variance. Variance is the spread, or the amount of difference that data expresses. 

Variance is simply standard deviation squared, and is often expressed as *s^2*.

![Alt text](../img/variance.png)

Let's assume you plotted the age (x axis) and height (y axis) of those indivuals and came up with an oblong scatterplot:

![Alt text](../img/scatterplot.png)

PCA draws straight, explanatory lines through data, like linear regression. 

Each straight line represents a "principal component," or a relationship between an independent and dependent variable. While there are as many principal components as there are dimensions in the data, PCA's role is to prioritize them. 

The first principal component bisects a scatterplot with a straight line in a way that explains the most variance; that is, it follows the longest dimension of the data. (This happens to coincide with the least error, as expressed by the red lines...) In the graph below, it slices down the length of the "baguette."

![Alt text](../img/scatterplot_line.png)

The second principal component cuts through the data nearly perpendicular to the first, fitting the errors produced by the first. The third fits the errors from the first and second principal components and so forth. 

## Covariance Matrix

While we introduced matrices as something that transformed one set of vectors into another, another way to think about them is as a description of data that captures the the forces at work upon it, the forces by which two variables might relate to each other as expressed by their variance and covariance.

Imagine that we compose a square matrix of numbers that describe the variance of the data, and the covariance among variables. This is the *covariance matrix*. Like many many other matrices, it has its very own eigenvectors. 

Finding the eigenvectors and eigenvalues of the covariance matrix is the equivalent of fitting those straight, principal-component lines to the variance of the data. 

*Eigenvalues*, another term you will encounter, are simply the coefficients attached to eigenvectors, which give the axes magnitude. In this case, they are the measure of the data's covariance. 

For a 2 x 2 matrix, a covariance matrix might look like this:

![Alt text](../img/covariance_matrix.png)

The numbers on the upper left and lower right represent the variance of the x and y variables, respectively, while the identical numbers on the lower left and upper right represent the covariance between x and y. Because of that identity, such matrices are known as symmetrical. As you can see, the covariance is positive, since the graph near the top of the PCA section points up and to the right. 

If two variables increase and decrease together (a line going up and to the right), they have a positive covariance, and if one decreases while the other increases, they have a negative covariance (a line going down and to the right). 

![Alt text](../img/covariances.png)

(*Credit: Vincent Spruyt*)

Notice that when one variable or the other doesn't move at all, and the graph shows no diagonal motion, there is no covariance whatsoever. Also, in the equation below, you'll notice that there is only a small difference between covariance and variance.

![Alt text](../img/covariance.png)

vs.

![Alt text](../img/variance.png)

The great thing about being able to calculate covariance is that, in a high-dimensional space where you can't eyeball intervariable relationships, you will know by the positive, negative or non-existent character of covariance how two variables move together. 

The main difference between covariance and *correlation* is that correlation also tracks the magnitude of the change in two variables, so two variables with a correlation of 1 always move the same distance in the same direction.

To sum up, the covariance matrix defines the shape of the data. Diagonal spread (along eigenvectors) is expressed by the covariance, while x-and-y-axis-aligned spread is expressed by the variance. 

While not entirely accurate, it may help to think of each component as a causal force in the Dutch basketball player example above, with the first principal component being age; the second possibly gender; the third nationality (implying nations' differing healthcare systems), and each of those occupying its own dimension in relation to height. Each acts on height to different degrees.

## Entropy & Information Gain

In information theory, the term *entropy* refers to information we don't have (normally people define "information" as what they know, and jargon has triumphed once again in turning plain language on its head to the detriment of the uninitiated). The information we don't have about a system, its entropy, is related to its unpredictability: how much it can surprise us.

If you know that a certain coin has heads embossed on both sides, then flipping the coin gives you absolutely no information, because it will be heads every time. You don't have to flip it to know. 

A balanced, two-sided coin does contain an element of surprise with each coin toss. And a six-sided die, by the same argument, contains even more surprise with each roll, which could produce any one of six results with equal frequency. Both those objects contain *information* in the technical sense.

Now let's imagine the die is loaded, it comes up "three" on five out of six rolls, and we figure out the game is rigged. Suddenly the amount of surprise produced with each roll by this die is greatly reduced. We understand a trend in the die's behavior that gives us greater predictive capacity. 

That transfer of information, from *what we don't know* about the system to *what we know*, represents a change in entropy. Insight decreases the entropy of the system. Get information, reduce entropy. This is information gain. And yes, this type of entropy is subjective, in that it depends on what we know about the system at hand. (Fwiw, [information gain](https://en.wikipedia.org/wiki/Information_gain_in_decision_trees) is synonymous with Kullback-Leibler divergence, which we explored briefly in this tutorial on [restricted Boltzmann machines](../restrictedboltzmannmachine.html).) 

So each principal component cutting through the scatterplot represents a decrease in the system's entropy, in its unpredictability. It so happens that explaining the shape of the data one principal component at a time, beginning with the component that accounts for the most variance, is similar to walking data through a decision tree. 

The first component of PCA, like the first split in a decision tree, should be along the dimension that reduces unpredictability the most. 

## Other Resources

* [A Tutorial on Principal Components Analysis](http://www.cs.otago.ac.nz/cosc453/student_tutorials/principal_components.pdf)
* [What is the importance of eigenvalues/eigenvectors?](http://math.stackexchange.com/a/23325)
* [Making Sense of PCA, Eigenvectors and Eigenvalues](http://stats.stackexchange.com/a/140579/85518)
* [A Geometric Interpretation of the Covariance matrix](http://www.visiondummy.com/2014/04/geometric-interpretation-covariance-matrix/)
* [Introduction to Eigenvectors & Eigenvalues Part 1](https://www.youtube.com/watch?v=G4N8vJpf7hM) (Video)
* [(Another) Introduction to Eigenvectors & Eigenvalues](https://www.youtube.com/watch?v=8UX82qVJzYI) (Video)
* [Information & Entropy, Unit 1, Lecture 2](https://www.youtube.com/watch?v=phxsQrZQupo) (MIT OpenCourseWare)
