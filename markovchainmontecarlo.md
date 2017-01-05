---
title: Markov Chain Monte Carlo and Deep Learning
layout: default
---

# Monte Carlo Methods, Markov Chains and Deep Learning

Let's say you're a horrific alien looking for the perfect planet to colonize. 

You have been instructed by a distant galactic empress to find a tolerable globe covered with water and dirt, but which has more of the first than the second. Round and juicy. Your space ship is hovering over the planet Earth, somewhere in the stratosphere, but the planet is covered in clouds... You peer through the cloud cover onto humanity's modest ball of mud and waves, squinting to see if it is suitable for conquest. 

That's where the socks come in. 

You have no way of calculating precisely how much of the Earth's surface is water, and how much is dirt, because you can't see all of it. So instead of attempting to plot the geometry of the oceans and continents, you decide you're just going to drop a bunch of socks randomly all over the planet and haul them back up. The ratio of wet, salty socks to dry will represent the ratio of sea water to land, and give you a rough approximation of the total surface area of each. You drop 100,000 socks, randomly zipping your space ship around the earth, and you get 70,573 sopping, salty socks back. 

It's time to prepare the invasion. 

## Monte Carlo Methods

That's Monte Carlo: A mathematical method used to predict the probability of future events or an unknown distribution of states. (In this case, the distribution of land and water.)

With a little more jargon, you might say it's a simulation using a pseudo-random number generator (for the placement of socks) to produce data yielding a normally distributed, bell-shaped curve over all possible outcomes of a given system. The method goes by the name "Monte Carlo" because the capital of Monaco, which borders southern France, is known for its casinos and games of chance, where winning and losing are a matter of probabilities. In a word, it's James Bond math. 

You can drop a sock on anything. It's also called *sampling*. Sampling permits us to approximate data without exhaustively analyzing all of it, because some datasets are too large to compute. Randomly sending down those socks over a bounded set of possibilities, which together add up to 100% of the earth's surface, is an example of the Monte Carlo method. 

Like the alien, we're often stuck behind a veil of ignorance (the clouds), unable to gauge reality around us with much precision. So we sample. 

Or forget aliens. You're a gambler in the saloon of a Gold Rush town and you roll a die without knowing if it is fair or loaded. You roll that mysterious, six-sided die a thousand times, count the number of times you roll a four, and divide by a thousand. That gives you the probability of four in the total distribution. If it's close to 167 (1/6 * 1000), the die is probably fair. 

Monte Carlo looks at the results of rolling the die many times and tallies the results to determine the probabilities of different states. It is inductive method, drawing from experience. The die has a state space of six, one for each side; the earth under the alien has a state space of two, land and water.

The states in question can vary. Instead of surf and turf, they might be letters in the Roman alphabet, which has a state space of 26. ("e" happens to be the most frequently occurring letter in the English language....) They might be stock prices, or the weather (rainy, sunny, overcast), or notes on a scale. These are all systems of discrete states that occur in seriatim, one after another. 

[An origin story](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.13.7133&rep=rep1&type=pdf): 

```
While convalescing from an illness in 1946, Stan Ulam was playing solitaire. It, then, occurred to him to try to compute the chances that a particular solitaire laid out with 52 cards would come out successfully (Eckhard, 1987). After attempting exhaustive combinatorial calculations, he decided to go for the more practical approach of laying out several solitaires at random and then observing and counting the number of successful plays. This idea of selecting a statistical sample to approximate a hard combinatorial problem by a much simpler problem is at the heart of modern Monte Carlo simulation.
```

## Systems and States

At a more abstract level, where words hardly mean anything at all, a system is a set of things connected together. You might say it's a set of states, where each state is a condition of the system. But what are states? 

* Cities on a map are states. A road trip strings them together in transitions. The map represents the system.
* Words in a language are states. A sentence is a series of transitions.
* Genes on a chromosome are states. To read them and create amino acids is to go through their transitions. 
* Web pages on the Internet are states. Links are the transitions. 
* Bank accounts in a financial system are states. Transactions are the transitions.
* Emotions are states in the system of the psyche. Mood swings are the transitions. 
* Social media profiles are states in the network. Follows, likes and friending are the transitions. 
* Ocean and land are states in geography. The shoreline is the transition. 

So states are an abstraction used to describe all these discrete, separable, things. A group of those states bound together by transitions is a system. And those systems have structure, in that some states are more likely to occur than others (ocean, land), or that some states are more likely to follow others. 

We are more like to read the sequence Paris -> France than Paris -> Texas, although both series exist, just as we are more likely to drive from Los Angeles to Las Vegas than from Los Angeles to [Slab City](https://www.google.com/maps/place/Slab+City,+CA+92233/@33.2579686,-117.7035463,7z/data=!4m5!3m4!1s0x80d0b20527ca5ebf:0xa7f292448cbd1988!8m2!3d33.2579703!4d-115.4623352), although both are nearby. 

A list of all possible states is known as the "state space." The more states you have, the larger the space gets, and the more complex your combinatorial problem becomes. 

## Markov Chains

Since states can occur one after another, it may make sense to traverse the state space, moving from one to the next rather than sampling them independently from an alien ship. That's where Markov chains come in. 

A Markov chain is a probabilistic way to traverse a system of states. It traces a series of transitions from one state to another. Each current state may have a set of possible future states that differs from any other. For example, you can't drive straight from Atlanta to Seattle - you'll need to hit other states in between. We are always in such corridors of probabilities; from each state, we face an array of possible future states, and those change with each step. New possibilites open up, others closing behind us. 

You're not sampling with a God's-eye view any more, like a conquering alien. You are in the middle of things, groping your way toward one of several possible future states step by probabilistic step, through a Markov Chain. 

While our journeys across a state space may seem unique, like road trips across America, an infinite number of road trips would slowly give us a picture of the country as a whole, and the network that links its cities together. 

## Markovian Time

Markov chains have a particular property, and that is oblivion, or forgetting. 

That is, they have no memory; they know nothing beyond the present, which means that the only factor determining the transition to a future state is a chain's current state. You could say the "m" in Markov stands for "memoryless": A woman with amnesia pacing through the rooms of a house without know why. You might say that Markov Chains assume the entirety of the past is encoded in the present, so we don't need to know anything more than where we are to infer where we will be next. 

For an excellent interactive demo of Markov Chains, [see the visual explanation on this site](http://setosa.io/ev/markov-chains/). 

So imagine the current state as the input data, and the distribution of future states as the dependent data, or the output. From each state in the system, by sampling you can determine the probability of what will happen next, doing so recursively at each step of the walk through the system's states.

## MCMCL Markov Chain Monte Carlo and Marco Polo

Markov Chains allow us to traverse a space, sampling as we go, with each new sample *dependent* on the one before. 

Imagine a Mongol emperor, Kublai Khan, enthroned in a distant palace in a city now known as Beijing. He rules a vast empire whose boundaries, inhabitants and vassal states he hardly knows, bordered by rival Khans whose strength he needs to assess. 

He has heard of strange lands from the Venetian adventurer, Marco Polo, and he determines to gauge the extent of his conquests by sending out a thousand explorers, each of them under orders to observe, each day, the name of the Khanate they are passing through. Unlike an alien dropping socks, these explorers are land-bound. The states they can reach on any given day depend on where they were the day before. Given that they are explorers uncertain of their destination, the direction they take each day is somewhat random. They are taking a random walk across 13th-century Asia, and recording it for their emperor, so that he may know the size of his lands as opposed to his neighbors. 

![Alt text](./img/Mongol_Empire.jpg) 

## Probability as Space

When they call it a state space, they're not joking. You can picture it, just like you can picture land and water, or khanates, each one of them a probability as much as they are a physical thing. Unfold a six-sided die and you have a flattened state space in six equal pieces, shapes on a plane. Line up the letters by their frequency for 11 different languages, and you get 11 different state spaces:

![Alt text](./img/letter_frequency_multilang.png) 

Five letters account for half of all characters occurring in Italian, but only a third of Swedish, if you're just dropping socks from the sky. 

If you wanted to look at the English language alone, you would get this set of histograms. Here, probabilities are defined by a line traced across the top, and the area under the line can be measured with a calculus operation called integration, the opposite of a derivative.  

![Alt text](./img/english_letter_dist.png) 

## Neural Networks Mapping Transitions

Neural networks map inputs to outputs. They might treat the current state as the input and map it to an output; that is, they could describe the transitions from one set of states to the next, or from raw data to the final results of a classifier.

That is, the nodes of a neural network are states in a system, and the weights between those nodes affect the transitions as information passes through the net, continuing all the way through to the output layer, where it is crystallized in a decision about the data. 

Remember, the output layer of a classifier, for example, might be image labels like cat, dog or human. The activations of the layer just before the classifications are connected to those labels by weights, and those weights are essentially saying: "If you see these activations, then in all likelihood the input was an image of a cat." 

## MCMC and Deep Reinforcement Learning

MCMC can be used in the context of deep reinforcement learning to sample from the array of possible moves available in any given state. This section is a work in progress. 

## Further Deeplearning4j Tutorials

* [Regression & Neural Networks](./linear-regression.html)
* [Word2vec: Extracting Relations From Raw Text](./word2vec.html)
* [Restricted Boltzmann Machines: The Building Blocks of Deep-Belief Networks](./restrictedboltzmannmachine.html)
* [Recurrent Networks and Long Short-Term Memory Units](./lstm.html)
* [Reinforcement Learning](./reinforcementlearning.html)

## Further Reading on Markov Chain Monte Carlo 

* [Markov Chain Monte Carlo Without all the Bullshit](https://jeremykun.com/2015/04/06/markov-chain-monte-carlo-without-all-the-bullshit/)
* [Hamiltonian Monte Carlo explained](http://arogozhnikov.github.io/2016/12/19/markov_chain_monte_carlo.html)
