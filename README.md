# RL4J

Reinforcement learning framework integrated with deeplearning4j.

GIFS are coming

![DOOM](doom.gif)

[temporary example repo](https://github.com/rubenfiszel/rl4j-examples)

[Cartpole example](https://github.com/rubenfiszel/rl4j-examples/blob/master/src/main/java/org/deeplearning4j/rl4j/Cartpole.java)

# Disclaimer

This is a tech preview and distributed as is.
Comments are welcome on our gitter channel:
[gitter](https://gitter.im/deeplearning4j/deeplearning4j)


# Quickstart

* mvn install -pl rl4j-api
* [if you want rl4j-gym too] Download and mvn install: [gym-java-client](https://github.com/deeplearning4j/gym-java-client)
* mvn install


# Quicktry cartpole:

* Install [gym-http-api](https://github.com/openai/gym-http-api).
* launch http api server.
* run with this [gist](https://gist.github.com/rubenfiszel/5d16aa2fc2c1e8cc9bdbf4ead5b3ab43)

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
