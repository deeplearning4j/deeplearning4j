---
title: A Beginner's Guide to Deep Reinforcement Learning (for Java and Scala)
layout: default
redirect_from: reinforcementlearning
---

# A Beginner's Guide to Deep Reinforcement Learning 

```
When it is not in our power to determine what is true, we ought to act in accordance with what is most probable. - Descartes
```
Contents

* <a href="#define">Reinforcement Learning Definitions</a>
* <a href="#neural">Neural Networks and Deep Reinforcement Learning</a>
* <a href="#code">Just Show Me the Code</a>
* <a href="#reading">Further Reading</a>

Neural networks are responsible for recent breakthroughs in problems like computer vision, machine translation and time series prediction – but they can also combine with other algorithms like reinforcement learning to create something like [AlphaGo](https://deepmind.com/blog/alphago-zero-learning-scratch/). 

Reinforcement learning is a set of goal-oriented algorithms. Reinforcement learning algorithms learn how to attain a complex objective (goal) or maximize along a particular dimension over many steps; for example, maximize the points won in a game over many moves.  They can start from a blank slate, and under the right conditions they achieve superhuman performance. 

Reinforcement algorithms that incorporate deep learning can beat world champions at the [game of Go](https://deeplearning4j.org/deep-learning-and-the-game-of-go#) as well as human experts playing numerous [Atari video games](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf). While that may sound trivial, it’s a vast improvement over their previous accomplishments, and the state of the art is progressing rapidly. 

Two reinforcement learning algorithms - Deep-Q learning and A3C - have been implemented in a Deeplearning4j library called [RL4J](https://github.com/deeplearning4j/rl4j). It can [already play Doom](https://www.youtube.com/watch?v=Pgktl6PWa-o).

Reinforcement learning solves the difficult problem of correlating immediate actions with the delayed returns they produce. Like humans, reinforcement learning algorithms sometimes have to wait a while to see the fruit of their decisions. They operate in a delayed return environment, where it can be difficult to understand which action leads to which outcome over many time steps. 

Reinforcement learning algorithms can be expected to perform better and better in more ambiguous, real-life environments while choosing from an arbitrary number of possible actions, rather than from the limited options of a video game. That is, with time we expect them to be valuable to achieve goals in the real world. 

<p align="center">
<a href="https://docs.skymind.ai/docs" type="button" class="btn btn-lg btn-success" onClick="ga('send', 'event', ‘quickstart', 'click');">GET STARTED WITH REINFORCEMENT LEARNING</a>
</p>

## <a name="define">Reinforcement Learning Definitions</a>

Reinforcement learning can be understand using the concepts of agents, environments, states, actions and rewards, all of which we’ll explain below. Capital letters tend to denote sets of things, and lower-case letters denote a specific instance of that thing; e.g. `A` is all possible actions, while `a` is a specific action contained in the set. 

* Agent: An **agent** takes actions; for example, a drone making a delivery, or Super Mario navigating a video game. The algorithm is the agent. In life, the agent is you.<sup>[1](#one)</sup>  
* Action (A): `A` is the set of all possible moves the agent can make. An **action** is almost self-explanatory, but it should be noted that agents choose among a list of possible actions. In video games, the list might include running right or left, jumping high or low, crouching or standing still. In the stock markets, the list might include buying, selling or holding any one of an array of securities and their derivatives. When handling aerial drones, alternatives would include many different velocities and accelerations in 3D space. 
* Environment: The world through which the agent moves. The environment takes the agent's current state and action as input, and returns as output the agent's reward and next state. If you are the agent, the environment could be the laws of physics and the rules of society that process your actions and determine the consequences of them.
* State (S): A **state** is the concrete and immediate situation in which the agent finds itself; i.e. a specific place and moment, an instantaneous configuration that puts the agent in relation to other significant things such as tools, obstacles, enemies or prizes. It is the current situation returned by the environment. Were you in the wrong place at the wrong time? That's a state. 
* Reward (R): A **reward** is the feedback by which we measure the success or failure of an agent’s actions. For example, in a video game, when Mario touches a coin, he wins points. From any given state, an agent sends output in the form of actions to the environment, and the environment returns the agent’s new state (which resulted from acting on the previous state) as well as rewards, if there are any. Rewards can be immediate or delayed. They effectively evaluate the agent's action. 
* Policy (π): The policy is the strategy that the agent employs to determine the next action based on the current state.
* Value (V): The expected long-term return with discount, as opposed to the short-term reward R. `Vπ(s)` is defined as the expected long-term return of the current state under policy π.
* Q-value or action-value (Q): Q-value is similar to Value, except that it takes an extra parameter, the current action a. `Qπ(s, a)` refers to the long-term return of the current state s, taking action a under policy π. 

So environments are functions that transform an action taken in the current state into the next state and a reward; agents are functions that transform the new state and reward into the next action. We can know the agent's function, but we cannot know the function of the environment. It is a black box where we only see the inputs and outputs. Reinforcement learning represents an agent's attempt to approximate the environment's function, such that we can send actions into the black-box environment that maximize the rewards it spits out. 

![Alt text](./img/simple_RL_schema.png)

In the feedback loop above, the subscripts denote the time steps `t` and `t+1`, each of which refer to different states: the state at moment `t`, and the state at moment `t+1`. Unlike other forms of machine learning – such as supervised and unsupervised learning -- reinforcement learning can only be thought about sequentially in terms of state-action pairs that occur one after the other. 

Reinforcement learning judges actions by the results they produce. It is goal oriented, and its aim is to learn sequences of actions that will lead an agent to achieve its goal. 

* In video games, the goal is to finish the game with the most points, so each additional point obtained throughout the game will affect the agent’s subsequent behavior; i.e. the agent may learn that it should shoot battleships, touch coins or dodge meteors to maximize its score. 
* In the real world, the goal might be for a robot to travel from point A to point B, and every inch the robot is able to move closer to point B could be counted like points. 

Reinforcement learning differs from both supervised and unsupervised learning by how it interprets inputs. We can illustrate their difference by describing what they learn about a "thing." 

* Unsupervised learning: That thing is like this other thing. (The algorithms learn similarities w/o names, and by extension they can spot the inverse and perform anomaly detection by recognizing what is unusual or dissimilar)
* Supervised learning: That thing is a “double bacon cheese burger”. (Labels, putting names to faces...) These algorithms learn the correlations between data instances and their labels; that is, they require a labelled dataset. Those labels are used to "supervise" and correct the algorithm as it makes wrong guesses when predicting labels. 
* Reinforcement learning: Eat that thing because it tastes good and will keep you alive. (Actions based on short- and long-term rewards.) Reinforcement learning can be thought of as supervised learning in an environment of sparse feedback. 

<iframe width="420" height="315" src="https://www.youtube.com/embed/-uXVu0l8guo" frameborder="0" allowfullscreen></iframe>

One way to imagine an autonomous reinforcement learning agent would be as a blind person attempting to navigate the world with only their ears and a white cane. Agents have small windows that allow them to perceive their environment, and those windows may not even be the most appropriate way for them to perceive what's around them. 

(In fact, deciding *which types* of input and feedback your agent should pay attention to is a hard problem to solve. This is known as domain selection. Algorithms that are learning how to play video games can mostly ignore this problem, since the environment is man-made and strictly limited. Thus, video games provide the sterile environment of the lab, where ideas about reinforcement learning can be tested. Domain selection requires human decisions, usually based on knowledge or theories about the problem to be solved; e.g. selecting the domain of input for an algorithm in a self-driving car might include choosing to include radar sensors in addition to cameras and GPS data.)

![Alt text](./img/rat_wired.jpg)

The goal of reinforcement learning is to pick the best known action for any given state, which means the actions have to be ranked, and assigned values relative to one another. Since those actions are state-dependent, what we are really gauging is the value of state-action pairs; i.e. an action taken from a certain state, something you did somewhere. Here are a few examples to demonstrate that the value and meaning of an action is contingent upon the state in which it is taken:

* If the action is marrying someone, then marrying a 35-year-old when you’re 18 probably means something different than marrying a 35-year-old when you’re 90, and those two outcomes probably have different motivations and lead to different outcomes. 

* If the action is yelling "Fire!", then performing the action a crowded theater should mean something different from performing the action next to a squad of men with rifles. We can’t predict an action’s outcome without knowing the context.

We map state-action pairs to the values we expect them to produce with the Q function, described above. The Q function takes as its input an agent’s state and action, and maps them to probable rewards. 

Reinforcement learning is the process of running the agent through sequences of state-action pairs, observing the rewards that result, and adapting the predictions of the Q function to those rewards until it accurately predicts the best path for the agent to take. That prediction is known as a policy. 

Reinforcement learning is an attempt to model a complex probability distribution of rewards in relation to a very large number of state-action pairs. This is one reason reinforcement learning is paired with, say, a [Markov decision process](./markovchainmontecarlo), a method to sample from a complex distribution to infer its properties. It closely resembles the problem that inspired [Stan Ulam to invent the Monte Carlo method](http://permalink.lanl.gov/object/tr?what=info:lanl-repo/lareport/LA-UR-88-9068); namely, trying to infer the chances that a given hand of solitaire will turn out successful.

Any statistical approach is essentially a confession of ignorance. The immense complexity of some phenomena (biological, political, sociological, or related to board games) make it impossible to reason from first principles. The only way to study them is through statistics, measuring superficial events and attempting to establish correlations between them, even when we do not understand the mechanism by which they relate. Reinforcement learning, like deep neural networks, is one such strategy, relying on sampling to extract information from data.  

Reinforcement learning is iterative. In its most interesting applications, it doesn’t begin by knowing which rewards state-action pairs will produce. It learns those relations by running through states again and again, like athletes or musicians iterate through states in an attempt to improve their performance.

You could say that reinforcement learning algorithms have a different relationship to time than humans do. We are able to run the algorithms through the same states over and over again while experimenting with different actions, until we can infer which actions are best from which states. Effectively, we give algorithms their very own [Groundhog Day](http://www.imdb.com/title/tt0107048/), where they start out as dumb jerks and slowly get wise. 

Since humans never experience Groundhog Day outside the movie, reinforcement learning algorithms have the potential to learn more, and better, than humans. You could say that the true advantage of these algorithms over humans stems not so much from their inherent nature, but from their ability to live in parallel on many chips at once, to train night and day without fatigue, and therefore to learn more. An algorithm trained on the game of Go, such as AlphaGo, will have played many more games of Go than any human could hope to complete in 100 lifetimes.<sup>[2](#two)</sup> 

## <a name="neural">Neural Networks and Deep Reinforcement Learning</a>

Where do neural networks fit in? Neural networks are the agent that learns to map state-action pairs to rewards. Like all neural networks, they use coefficients to approximate the function relating inputs to outputs, and their learning consists to finding the right coefficients, or weights, by iteratively adjusting those weights along gradients that promise less error.  

In reinforcement learning, convolutional networks can be used to recognize an agent’s state; e.g. the screen that Mario is on, or the terrain before a drone. That is, they perform their typical task of image recognition. 

But convolutional networks derive different interpretations from images in reinforcement learning than in supervised learning. In supervised learning, the network applies a label to an image; that is, it matches names to pixels. 

![Alt text](./img/conv_classifier.png)

In fact, it will rank the labels that best fit the image in terms of their probabilities. Shown an image of a donkey, it might decide the picture is 80% likely to be a donkey, 50% likely to be a horse, and 30% likely to be a dog. 

In reinforcement learning, given an image that represents a state, a convolutional net can rank the actions possible to perform in that state; for example, it might predict that running right will return 5 points, jumping 7, and running left none. 

![Alt text](./img/conv_agent.png)

Having assigned values to the expected rewards, the Q function simply selects the state-action pair with the highest so-called Q value. 

At the beginning of reinforcement learning, the neural network coefficients may be initialized stochastically, or randomly. Using feedback from the environment, the neural net can use the difference between its expected reward and the ground-truth reward to adjust its weights and improve its interpretation of state-action pairs. 

This feedback loop is analogous to the backpropagation of error in supervised learning. However, supervised learning begins with knowledge of the ground-truth labels the neural network is trying to predict. Its goal is to create a model that maps different images to their respective names. 

Reinforcement learning relies on the environment to send it a scalar number in response to each new action. The rewards returned by the environment can be varied, delayed or affected by unknown variables, introducing noise to the feedback loop. 

This leads us to a more complete expression of the Q function, which takes into account not only the immediate rewards produced by an action, but also the delayed rewards that may be returned several time steps deeper in the sequence.

Like human beings, the Q function is recursive. Just as calling the wetware method human() contains within it another method human(), of which we are all the fruit, calling the Q function on a given state-action pair requires us to call a nested Q function to predict the value of the next state, which in turn depends on the Q function of the state after that, and so forth. 

## <a name="code">Just Show Me the Code</a>

[RL4J examples are available here](https://github.com/deeplearning4j/dl4j-examples/tree/master/rl4j-examples).

```
package org.deeplearning4j.examples.rl4j;

import java.io.IOException;
import org.deeplearning4j.rl4j.learning.HistoryProcessor;
import org.deeplearning4j.rl4j.learning.async.a3c.discrete.A3CDiscrete;
import org.deeplearning4j.rl4j.learning.async.a3c.discrete.A3CDiscreteConv;
import org.deeplearning4j.rl4j.mdp.ale.ALEMDP;
import org.deeplearning4j.rl4j.network.ac.ActorCriticFactoryCompGraphStdConv;
import org.deeplearning4j.rl4j.util.DataManager;

/**
 * @author saudet
 *
 * Main example for A3C with The Arcade Learning Environment (ALE)
 *
 */
public class A3CALE {

    public static HistoryProcessor.Configuration ALE_HP =
            new HistoryProcessor.Configuration(
                    4,       //History length
                    84,      //resize width
                    110,     //resize height
                    84,      //crop width
                    84,      //crop height
                    0,       //cropping x offset
                    0,       //cropping y offset
                    4        //skip mod (one frame is picked every x
            );

    public static A3CDiscrete.A3CConfiguration ALE_A3C =
            new A3CDiscrete.A3CConfiguration(
                    123,            //Random seed
                    10000,          //Max step By epoch
                    8000000,        //Max step
                    8,              //Number of threads
                    32,             //t_max
                    500,            //num step noop warmup
                    0.1,            //reward scaling
                    0.99,           //gamma
                    10.0            //td-error clipping
            );

    public static final ActorCriticFactoryCompGraphStdConv.Configuration ALE_NET_A3C =
            new ActorCriticFactoryCompGraphStdConv.Configuration(
                    0.00025, //learning rate
                    0.000,   //l2 regularization
                    null, null, false
            );

    public static void main(String[] args) throws IOException {

        //record the training data in rl4j-data in a new folder
        DataManager manager = new DataManager(true);

        //setup the emulation environment through ALE, you will need a ROM file
        ALEMDP mdp = null;
        try {
            mdp = new ALEMDP("pong.bin");
        } catch (UnsatisfiedLinkError e) {
            System.out.println("To run this example, uncomment the \"ale-platform\" dependency in the pom.xml file.");
        }

        //setup the training
        A3CDiscreteConv<ALEMDP.GameScreen> a3c = new A3CDiscreteConv(mdp, ALE_NET_A3C, ALE_HP, ALE_A3C, manager);

        //start the training
        a3c.train();

        //save the model at the end
        a3c.getPolicy().save("ale-a3c.model");

        //close the ALE env
        mdp.close();
    }
}

```

## <a name="reading">Further Reading</a>

* [RL4J: Reinforcement Learning in Java](https://github.com/deeplearning4j/rl4j)
* Richard S. Sutton and Andrew G. Barto's [Reinforcement Learning: An Introduction](http://incompleteideas.net/sutton/book/the-book-2nd.html)
* Andrej Karpathy's [ConvNetJS Deep Q Learning Demo](https://cs.stanford.edu/people/karpathy/convnetjs/demo/rldemo.html)
* [Brown-UMBC Reinforcement Learning and Planning (BURLAP)](http://burlap.cs.brown.edu/)(Apache 2.0 Licensed as of June 2016)
* [Glossary of Terms in Reinforcement Learning](http://www-anw.cs.umass.edu/rlr/terms.html)
* [Reinforcement Learning and DQN, learning to play from pixels](https://rubenfiszel.github.io/posts/rl4j/2016-08-24-Reinforcement-Learning-and-DQN.html)
* Video: [Richard Sutton on Temporal Difference Learning](https://www.youtube.com/watch?v=EeMCEQa85tw)
* [A Brief Survey of Deep Reinforcement Learning](https://arxiv.org/pdf/1708.05866.pdf)

### <a name="beginner">Other Deep Learning Tutorials</a>
* [LSTMs and Recurrent Networks](./lstm)
* [Introduction to Deep Neural Networks](./neuralnet-overview)
* [Convolutional Networks](./convolutionalnets)
* [Restricted Boltzmann Machines](./restrictedboltzmannmachine)
* [Eigenvectors, Covariance, PCA and Entropy](./eigenvector)
* [Deeplearning4j Quickstart Examples](./quickstart)
* [ND4J: A Tensor Library for the JVM](http://nd4j.org)
* [MNIST for Beginners](./mnist-for-beginners.html)
* [Glossary of Deep-Learning and Neural-Net Terms](./glossary.html)
* [Generative Adversarial Networks (GANs)](./generative-adversarial-network)
* [Word2vec and Natural-Language Processing](./word2vec.html)

<a name="one">1)</a> *It might be helpful to imagine a reinforcement learning algorithm in action, to paint it visually. Let's say the algorithm is learning to play the video game Super Mario. It's trying to get Mario through the game and acquire the most points. To do that, we can spin up lots of different Marios in parallel and run them through the space of all possible game states. It's as though you have 1,000 Marios all tunnelling through a mountain, and as they dig (e.g. as they decide again and again which action to take to affect the game environment), their tunnels branch like the intricate and fractal twigs of a tree. The Marios' experience-tunnels are corridors of light cutting through the mountain. And as in life itself, one successful action may precede another in a larger decision flow, propelling the winning Marios onward. You might also imagine that in front of each Mario is a heat map tracking the rewards they can associate with state-action pairs. Because the algorithm starts ignorant and many of the paths through the game-state space are unexplored, the heat maps will reflect their lack of experience; i.e. there could be blank holes in the heatmap of the rewards they imagine, or they might just start with some default assumptions about rewards that will be adjusted with experience. The Marios are essentially reward-seeking missiles, and the more times they run through the game, the more accurate their heatmap of potential future reward becomes. Those heatmaps are basically probability distributions of reward over the state-action pairs possible from the Mario's current state.*

<a name="two">2)</a> *The correct analogy may actually be that a learning algorithm is like a species. Each simulation the algorithm runs as it learns could be considered an individual of the species. Just as knowledge from the algorithm's runs through the game is collected in the algorithm's model of the world, the individual humans of any group will report back via language, allowing the collective's model of the world, embodied in its texts, records and oral traditions, to become more intelligent (At least in the ideal case. The subversion and noise introduced into our collective models is a topic for another post, and probably for another website entirely.). This puts a finer point on why the contest between algorithms and individual humans, even when the humans are world champions, is unfair. We are pitting a civilization against a single sack of flesh.*
