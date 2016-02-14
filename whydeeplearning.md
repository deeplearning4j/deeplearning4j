---
title: When to Choose Deep Learning
layout: default
---

# When to Choose Deep Learning

Where does deep learning fit into machine learning and other optimization algorithms, and what kind of problems can it solve? These are the first questions to answer as you choose the tools to tackle whatever data problem you have. 

Deep learning is synonymous with machine learning, and simply an advanced subset of that larger field. Technically speaking, deep learning is an umbrella term for a set of neural nets that consist of three or more layers; i.e. at least one hidden layer, and the visible layers of input and output. 

So what is deep learning capable and incapable of? Deep learning can recognizing pattern and classify them. Put another way, it can tell your machine what it's looking at, hearing, or being fed as a stream of numbers. 

That very basic function, which occurs in our minds before we have the chance to reflect, is a task of enormous complexity for a computer, only accomplished after much calculation. Confronted with an array of pixels, no computer inherently knows the difference between a house, a tree and a cat. Deep learning is the guy with the paint brush. 

![Alt text](../img/that_should_clear.jpg)

### Prerequisites & Considerations

For deep learning to work well, it needs lots of data. So first of all, you should have that. (Deep learning has distinct advantages over machine learning when processing **unstructured data**. It doesn't need you to label everything to discover patterns.)

Next, you need a question you want answered. What does this unstructured data consist of? Are there images that require labels? Voices that need names? Speech that should be matched with written text? Video with multiple objects that you want to analyze? Or texts you would like to group by sentiment or content? Those are all concrete questions deep learning can help with. If those match your problem, then you should [choose your algorithm](../neuralnetworktable.html). 

In short: Does your problem require that some object or phenomenon be identified amid a sea of input? Do you need to isolate some event before you can make a decision? Answering "yes" to these questions may lead you to choose deep learning. 

### Similar, different

By measuring similarities between, say, images, a deep learning algorithm can find all the pictures of one person's face, and collect them on a page a la Facebook. Conversely, by measuring the differences, it may be able to tell you if an unknown face appears, say, in the entryway of your house at night. It groups what is similar, and highlights what is different.

Highlighting differences is known as **anomaly detection**. Since deep learning works best with unstructured media like text, sound and images, we'll use a visual anomaly. 

Doctors search CT scans for tumors everyday -- sometimes they find them, and sometimes they don't. Hospitals possess enormous amounts of images labeled as cancer, and an equally enormous number where they do not know if it is present. The labeled data could serve as the training set, and the as-yet-unlabeled images could be classified with deep-learning nets. 

Fed enough labeled instances of cancer, a deep learning net can begin to identify more subtle patterns, which doctors themselves may hardly be aware of. 

Now imagine that deep learning can analyze with the same accuracy video, sounds, text and time-series data. It can identify faces, voices, similar documents, and signals in the movement of a stock. It might spot a suspicious face in an airport terminal, or identify the latent audio artefacts of a fraudulent phone call.

With time series data and sound, it is capable of analyzing the beginnings of a pattern and filling in the rest, whether the rest is a complete word, or the next hiccup in the market. It does not care whether the pattern exists wholly in the present, like a face, or partially in the future, like a sentence being spoken. 

### Feature Extraction

One of deep learning's main advantages over all previous neural nets and other machine-learning algorithms is its capacity to extrapolate new features from a limited set of features contained in a training set. That is, it will search for and find other features that correlate to those already known. It will discover new ways of hearing the signal in the noise. 

The ability of deep learning to create features without being explicitly told means that data scientists can save sometimes months of work by relying on these networks. It also means that data scientists can work with more complex feature sets than they might have with machine learning tools. 

### Lacking Feature Introspection

While you can debug a neural net, you cannot map decisions back to individual features as you can with random forest decision trees. 

Deep learning is incapable of telling you why it reached a certain conclusion. Its hidden nets are creating their own features, extrapolating on the manual features contained in the training set. Those features it creates it may not have names for. 

In machine learning, the features are manually created by engineers who know exactly how much they will contribute to the algorithm's final decision. 

Deep learning lacks feature introspection. This is important if you are trying to classify events as fraudulent or non-fraudulent, and have to document and justify how you reached your conclusions. If you reject a client's order as fraudulent and your boss asks you why, she may not like the answer "I don't know."


