---
title: How Can We Control AI?
layout: default
---

# The Skynet Issue: How Can We Control AI?

AI is inherently hard to regulate, since it is nothing more than math and code running on a few chips, which any programmer, in the freedom of their garage, can work with in order to advance the state of the art.

In general, regulations need the events they regulate to be accessible and visible to them.  This is probably not a reasonable expectation when you think about imposing constraints on math and code; i.e. how will you know when they are happening? How will you tell the difference between good math and evil math?

It may not be impossible to monitor AI research, where advances are being made toward an artificial general intelligence, but it will definitely be costly to all involved and potentially onerous for the researchers.

This is complicated by the fact that AI research is happening around the world in many different nations. Those nations see short-term advantages in establishing an edge in AI. This makes them unlikely to accept rules imposed by other nations or international bodies.

Even if they theoretically accepted the argument that we should globally regulate AI, it is difficult to convince many, rival nation states to agree on complex subjects, as we have seen with proposed climate change regulation, or even OPEC's price-fixing meetings.

So we should not be so naive as to think it will be easy to impose constraints on AI research, even if we speculate on what the ideal constraints should be.

<p align="center">
<a href="http://deeplearning4j.org/quickstart" class="btn btn-custom" onClick="ga('send', 'event', ‘quickstart', 'click');">Get Started With Deeplearning4j</a>
</p>

Here are a few thoughts about what we might request of the makers of AI software:

1) Give us a heartbeat. A heartbeat is a small bit of code that "phones home." That is, it lets its makers know that the code is being used. So for any program designated as AI, this constraint would be to notify an AI's makers (or some central, authorized body) that it's active.

2) A unique ID is a corollary to the heartbeat. Let's constrain AI by making sure each AI agent has a unique identifier.

3) Each uniquely identified agent should be the responsibility of a human maker and/or owner. That is, each AI should contain and reveal information about who owns and built it, so that if something goes wrong, the people responsible for the code can be contacted and involved in solving whatever problem arises.

4) A kill switch. The greatest fears around superintelligence involve an intelligence far surpassing human capacities, which is capable of further improving itself and does not care about the human race. Regulators could insist that a backdoor be built into all AIs that allows certain people (a government regulator, or even the owner?) to access the AI and send a message that shuts it down.

5) Impose soft or hard ceilings on recursion and adversarial training. In the paper that DeepMind published showing that its algorithms can beat a Go grandmaster, they described a process by which they trained autonomous agents to play against each other (that is, AIs learned from one another's responses, or AI itself made AI smarter). Recursively improving AI is probably the fastest route to superintelligence, so watch recursion. I'm not saying this is doable, but I'm pointing to the areas where regulators should pay attention.

6) Monitor and control the sales of superpowerful chips and their ensembles. Deep artificial neural networks and other algorithms are computationally intensive. They require a LOT of hardware (or a lot of time) to train themselves to make accurate predictions. That hardware takes the form of specialized chips organized in large groups over which you run optimized software to get results in a reasonable amount of time. So follow the chips. It's actually much harder to make faster, better chips than it is to make better software, and it would be easier for regulators to track and constrain such physical objects than it would be to track the code. Items 1-4 could also be baked into chips. (I do not believe that any chips today pose a threat, although the ability to make them scale linearly one day when processing together may change that. When quantum computing meets AI, though, we will see huge advances in the ability and complexity of our artificially intelligent autonomous agents.)

7) Track the careers of machine learning PhDs, and pay attention to companies where they concentrate.

8) For drones and other vehicles that contain AI, add a sensor that recognizes signs of human death, injury or distress (this is actually quite feasible). The militaries of the world already build drones to kill people. That is nothing new. In all likelihood, they will make drones kill people in smarter and smarter ways. What I am suggesting is that we automate the way in which we recognize how AI-enabled drones harm humans. By testing that system in drones authorized to exercise lethal force, we may come up with something we want to add to all AI-enabled machines, to make sure they don't stray.

One way to thinking about controlling AIs, at least AIs of a certain level of intelligence, is to imagine animal control. People have pets and they are supposed to control and constrain their pets in certain ways: they shouldn't let them poop on the sidewalk, they should get them spade or neutered, they should teach them not to hurt or attack other humans in most cases. But not every pet owner does control or constrain their pets. Sometimes those pets go feral and breed uncontrollably, sometimes they damage property and persons, yet we still benefit from animal control laws, all in all.

* - Chris Nicholson, July 2016*

### <a name="beginner">Other Deeplearning4j Posts</a>
* [Word2vec: Neural Embeddings for Java](./word2vec)
* [Introduction to Neural Networks](./neuralnet-overview)
* [Restricted Boltzmann Machines](./restrictedboltzmannmachine)
* [Eigenvectors, Covariance, PCA and Entropy](./eigenvector)
* [LSTMs and Recurrent Networks](./lstm)
* [Neural Networks and Regression](./linear-regression)
* [Convolutional Networks](./convolutionalnets)
