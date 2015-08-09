---
title: 
layout: default
---

# Introduction to Eigenvectors & Eigenvalues

This post introduces eigenvectors and their relationship to matrices.

The *eigen* in eigenvector comes from German, and it means something like “very own.” For example, in German, “mein eigenes Auto” means “my very own car.” So eigen denotes a special relationship between two things. Something particular, characteristic and definitive. This car, or this vector, is mine and not someone else’s.

Matrices, in linear algebra, are simply rectangular arrays of numbers, a collection of scalar values between brackets, like a spreadsheet. Most square matrices (e.g. 2 x 2 or 3 x 3) have eigenvectors, and they have a very special relationship with them, a bit like Germans have with their cars.

## Linear Transformations

We’ll define that relationship after a brief detour into what matrices do, and how they relate to other numbers.

Matrices are useful because you can do things with them like add and multiply. If you multiply a vector v by a matrix A, you get another vector b, and you could say that the matrix performed a linear transformation on the input vector. 

"Av = b" 

It map one set of numbers v to another, b.  We’ll illustrate with a concrete example. 

Let the matrix A be 

[■(2&1@1.5&2)]

and let the input vector v be 

■(0.75@0.25)

Put them together to get output vector b

[■(2&1@1.5&2)]* ■(0.75@0.25)= ■(1.75@1.625)

So A turned v into b. In a two-dimensional plane, the coordinates of the vector changed.

IMAGE two_vectors

You could feed one vector after another into matrix A, and each would be projected onto a new space that stretches higher and father to the right. 

Imagine that all the input vectors live in a normal grid, like this:

IMAGE space_1

And all the matrix projects them into a new space like this

IMAGE space_2

Here are the two spaces juxtaposed:

IMAGE two_spaces

And here’s an animation:

IMAGE Wikipedia.gif

You can almost imagine a matrix like a gust of wind, an invisible force that produce a  visible result. And a gust of wind blows in a certain direction. The eigenvector tells you the direction the matrix is blowing in. 

IMAGE Mona_Lisa_eigenvector

So out of all the vectors affected by a matrix blowing through one space, which one is the eigenvector? It’s the one that doesn’t change direction; that is, the eigenvector is already pointing in the same direction that the matrix is pushing all vectors toward. An eigenvector is a like a weathervane, pointing in the direction of the wind. 

IMAGE weathervane.jpg

You could also say that eigenvectors are axes along which linear transformation acts, stretching or compressing input vectors. They are the lines of change that represent the action of the larger matrix.

Notice we’re using the plural – axes, lines – because square matrices have as many eigenvectors as they have dimensions; i.e. a 2 x 2 matrix has two eigenvectors, a 3 x 3 matrix has three, and an n x n has n eigenvectors, each one representing its line of action in one dimension. 
