# RL4J

RL4J is a reinforcement learning framework integrated with deeplearning4j and released under an Apache 2.0 open-source license. By contributing code to this repository, you agree to make your contribution available under an Apache 2.0 license.

* DQN (Deep Q Learning with double DQN)
* Async RL (A3C, Async NStepQlearning)

Both for Low-Dimensional (array of info) and high-dimensional (pixels) input.


![DOOM](doom.gif)


![Cartpole](cartpole.gif)


Here is a useful blog post I wrote to introduce you to reinforcement learning, DQN and Async RL:


[Blog post](https://rubenfiszel.github.io/posts/rl4j/2016-08-24-Reinforcement-Learning-and-DQN.html)

[temporary example repo](https://github.com/rubenfiszel/rl4j-examples)

[Cartpole example](https://github.com/rubenfiszel/rl4j-examples/blob/master/src/main/java/org/deeplearning4j/rl4j/Cartpole.java)

# Disclaimer

This is a tech preview and distributed as is.
Comments are welcome on our gitter channel:
[gitter](https://gitter.im/deeplearning4j/deeplearning4j)


# Quickstart

** INSTALL rl4j-api before installing all (see below)!**

* mvn install -pl rl4j-api
* [if you want rl4j-gym too] Download and mvn install: [gym-java-client](https://github.com/deeplearning4j/gym-java-client)
* mvn install

# Visualisation

[webapp-rl4j](https://github.com/rubenfiszel/webapp-rl4j)

# Quicktry cartpole:

* Install [gym-http-api](https://github.com/openai/gym-http-api).
* launch http api server.
* run with this [main](https://github.com/rubenfiszel/rl4j-examples/blob/master/src/main/java/org/deeplearning4j/rl4j/Cartpole.java)

# Doom

Doom is not ready yet but you can make it work if you feel adventurous with some additional steps:

* You will need vizdoom, compile the native lib and move it into the root of your project in a folder
* export MAVEN_OPTS=-Djava.library.path=THEFOLDEROFTHELIB
* mvn compile exec:java -Dexec.mainClass="YOURMAINCLASS"

# WIP

* Documentation
* Serialization/Deserialization (load save)
* Compression of pixels in order to store 1M state in a reasonnable amount of memory
* Async learning: A3C and nstep learning (requires some missing features from dl4j (calc and apply gradients)).

# Author

[Ruben Fiszel](http://rubenfiszel.github.io/)

# Proposed contribution area:

* Continuous control
* Policy Gradient
* Update gym-java-client when gym-http-api gets compatible with pixels environments to play with Pong, Doom, etc ..
