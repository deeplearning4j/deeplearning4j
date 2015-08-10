---
title: 
layout: default
---

# Introduction to Eigenvectors

This post introduces eigenvectors and their relationship to matrices in plain language and without a great deal of math. 

The *eigen* in eigenvector comes from German, and it means something like “very own.” For example, in German, “mein eigenes Auto” means “my very own car.” So eigen denotes a special relationship between two things. Something particular, characteristic and definitive. This car, or this vector, is mine and not someone else’s.

Matrices, in linear algebra, are simply rectangular arrays of numbers, a collection of scalar values between brackets, like a spreadsheet. Most square matrices (e.g. 2 x 2 or 3 x 3) have eigenvectors, and they have a very special relationship with them, a bit like Germans have with their cars.

## Linear Transformations

We’ll define that relationship after a brief detour into what matrices do, and how they relate to other numbers.

Matrices are useful because you can do things with them like add and multiply. If you multiply a vector *v* by a matrix *A*, you get another vector *b*, and you could say that the matrix performed a linear transformation on the input vector. 

*Av = b* 

It [maps](https://en.wikipedia.org/wiki/Linear_map) one set of numbers *v* to another, *b*.  We’ll illustrate with a concrete example. 

![Alt text](../img/eigen_matrix.png)

So *A* turned *v* into *b*. In a two-dimensional plane, the coordinates of the vector changed.

![Alt text](../img/two_vectors.png)

You could feed one vector after another into matrix A, and each would be projected onto a new space that stretches higher and father to the right. 

Imagine that all the input vectors *v* live in a normal grid, like this:

![Alt text](../img/space_1.png)

And all the matrix projects them into a new space like this holding output vectors *b*:

![Alt text](../img/space_2.png)

Here are the two spaces juxtaposed:

![Alt text](../img/two_spaces.png)

(*Credit: William Gould, Stata Blog*)

And here’s an animation that shows the matrix's work transforming one space to another:

<iframe src="https://upload.wikimedia.org/wikipedia/commons/0/06/Eigenvectors.gif" width="100%" height="300px;" style="border:none;"></iframe>

You can imagine a matrix like a gust of wind, an invisible force that produce a  visible result. And a gust of wind blows in a certain direction. The eigenvector tells you the direction the matrix is blowing in. 

![Alt text](../img/mona_lisa_eigenvector.png)

So out of all the vectors affected by a matrix blowing through one space, which one is the eigenvector? It’s the one that doesn’t change direction; that is, the eigenvector is already pointing in the same direction that the matrix is pushing all vectors toward. An eigenvector is a like a weathervane, pointing in the direction of the wind. 

![Alt text](../img/weathervane.jpg)

You could also say that eigenvectors are axes along which linear transformation acts, stretching or compressing input vectors. They are the lines of change that represent the action of the larger matrix.

Notice we’re using the plural – axes and lines – because square matrices have as many eigenvectors as they have dimensions; i.e. a 2 x 2 matrix has two eigenvectors, a 3 x 3 matrix has three, and an n x n has n eigenvectors, each one representing its line of action in one dimension. 

Because eigenvectors distill the axes of principal force that a matrix moves along, they are useful in matrix decomposition, or matrix vectorization; i.e. the representation of a matrix by a vector. In that sense, they perform the same task as autoencoders. 

## Principle Component Analysis (PCA)

PCA is a tool for finding patterns in high-dimensional data such as images. 

To get to PCA, we’re going to quickly gloss some basic statistical ideas so we can weave them together later. The first is *variance*, and variance is a property of data. 

If I take a team of Dutch basketball players and measure their height, those measurements won’t have a lot of variance. They’ll all be grouped above six feet. 

But if I throw in a classroom of psychotic kindergartners as well as a few CIA spies that have been selected for appearing utterly normal in every way, then the combined group’s height measurements will have a lot of variance. Variance is the spread, or the amount of difference that data expresses. 

## Other Resources

* [A Tutorial on Principal Components Analysis](http://www.cs.otago.ac.nz/cosc453/student_tutorials/principal_components.pdf)
