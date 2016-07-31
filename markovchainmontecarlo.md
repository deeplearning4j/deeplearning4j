---
title: Markov Chain Monte Carlo and Deep Learning
layout: default
---

# Monte Carlo Methods, Markov Chains and Neural Nets

Let's say you're a horrific alien looking for the perfect planet to colonize. 

You have been instructed by a distant galactic empress to find a tolerable globe covered with water and dirt, but which has more of the first than the second. Round and juicy. Your space ship is hovering over the planet Earth, somewhere in the stratosphere, but the planet is covered in clouds... You peer through the cloud cover onto humanity's modest ball of mud and waves, squinting to see if it is suitable for conquest. 

That's where the socks come in. 

You have no way of calculating precisely how much of the Earth's surface is water, and how much is dirt, because you can't see all of it. So instead of attempting to plot the geometry of the oceans and continents, you decide you're just going to drop a bunch of socks randomly all over the planet and haul them back up. The ratio of salty, wet socks to dry socks will represent the ratio of sea water to land, and give you a rough approximation of the total surface area of land and water. You drop 100,000 socks, and you get 70,573 sopping, salty socks back. 

It's time to prepare the invasion. 

## Monte Carlo 

That's a Monte Carlo simulation. With a little more jargon, you might say it's a mathematical simulation using a pseudo-random number generator that will produce data yielding a normally distributed, bell-shaped curve over all possible outcomes. The method is called Monte Carlo, because the capital of Monaco, which borders southern France, is known for its casinos and games of chance, where winning is a matter of probabilities. This is James Bond math. 

You can drop a sock on anything. It's called *sampling*. And randomly distributing those socks over a limited, pre-ordained set of possibilities, which together add up to 100% of the earth's surface, is called the Monte Carlo method. 

We're stuck behind a veil of ignorance (the clouds) everywhere we turn, unable to gauge reality around us with much precision. You gamble and you roll the die, not knowing if the die is fair or loaded. Monte Carlo looks at the results of rolling the die many times. It is an inductive method; it draws from experience.  

Another example: Perhaps during your conquest you stumble across the quaint language of English, and wonder if any particular letters are more important that others. By randomly sampling letters from the works of an infinite library, you would soon discover that "e" is in fact the [most commonly used letter](https://en.wikipedia.org/wiki/Letter_frequency#Relative_frequencies_of_letters_in_the_English_language) in the alphabet. (Unless you are sampling from the works of the novelist Georges Perec, who wrote [a fairly long novel](https://en.wikipedia.org/wiki/A_Void) without our hardest-working vowel.) You order your team of alien linguists to produce a report on "e" to better communicate with the natives. 

[An origin story](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.13.7133&rep=rep1&type=pdf): 

```
While convalescing from an illness in 1946, Stan Ulam was playing solitaire. It, then, occurred to him to try to compute the chances that a particular solitaire laid out with 52 cards would come out successfully (Eckhard, 1987). After attempting exhaustive combinatorial calculations, he decided to go for the more practical approach of laying out several solitaires at random and then observing and counting the number of successful plays. This idea of selecting a statistical sample to approximate a hard combinatorial problem by a much simpler problem is at the heart of modern Monte Carlo simulation.
```

## Markov Chains

A system is a set of things connected together. You might say it's a set of states, where each state is a condition of the system. But what are states? 

* Cities on a map are states. A road trip strings them together in transitions.
* Words in a language are states. A sentence is a series of transitions.
* Genes on a chromosome are states. 
* Web pages on the Internet are states. Links are the transitions. 
* Bank accounts are states. Transactions are the transitions.
* Emotions are states. Mood swings are the transitions. 
* Social media profiles are states in the network. Follows, likes and friending are the transitions. 
* Ocean and land are states in geography. Only amphibians know the transitions. 

So states are an abstraction used to describe all these discrete, or separable, things. A group of those states bound together by transitions is a system. And those systems have structure, in that some states are more likely to occur than others (ocean, land), or that some states are more likely to follow others (we are more like to read the sequence Paris -> France than Paris -> Texas, although both series exist, just as we are more likely to drive from Los Angeles to Las Vegas than from Los Angeles to [Slab City](https://www.google.com/maps/place/Slab+City,+CA+92233/@33.2579686,-117.7035463,7z/data=!4m5!3m4!1s0x80d0b20527ca5ebf:0xa7f292448cbd1988!8m2!3d33.2579703!4d-115.4623352), although both are nearby). A list of all possible states is known as the "state space." The more states you have, the larger the space gets, and the more complex your combinatorial problem becomes. 

Now, a Markov chain is just a system of states that tells you how to transition between them. Markov chains have a particular property, and that is oblivion, or forgetting. 

That is, they have no memory; they know nothing beyond the present, which means that the only factor determining the transition to a future state is a chain's current state. You could say the "m" in Markov stands for "memoryless": A woman with amnesia pacing through the rooms of a house without know why. For an excellent interactive demo of Markov Chains, [see the visual explanation on this site](http://setosa.io/ev/markov-chains/).

So imagine the current state as the input data, and the distribution of future states as the dependent data, or the output. From each state in the system, by sampling you can determine the probability of what will happen next, doing so recursively at each step of the walk through the system's states.

## Markov Chains and Neural Networks

Now the nodes of a neural network are states in a system, and the weights between those nodes are the transitions, continuing all the way through to the output layer. Remember, the output layer of a classifier, for example, might be image labels like cat, dog or human. The activations of the layer just before the classifications are connected to those labels by weights, and those weights are essentially saying: "If you see these activations, then in all likelihood the input was an image of a cat." 

You're not sampling with a God's-eye view any more, like a conquering alien. You are in the middle of things, groping your way toward one of several possible future states. 
