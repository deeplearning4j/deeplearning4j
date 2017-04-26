---
title: What Is Machine Learning?
layout: default
---

# What Is Machine Learning?

To define machine learning, we first need to define some of its components. 

The *machine* in machine learning refers to an algorithm, or a method of computation. You could say that an algorithm combines math and logic. Here's an example in code of an algorithm that finds the biggest number in a set of numbers:

    def find_max (L):
        max = 0
        for x in L:
            if x > max:
                max = x
        return max

The algorithm above isn't a learning algorithm. It finds the maximum value the same way every time no matter what data it is exposed to. It doesn't change how it works in response to the data.

What does it mean for an alorithm to learn? It means that the algorithm alters itself over time as it is exposed to data. It learns how to make better decisions based on that data, just as a human is said to learn when they manage to make better decisions based on experience. When humans learn, they alter the way they relate to the world. When algorithms learn, they alter the way they process data. 

