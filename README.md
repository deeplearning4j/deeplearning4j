# reinforcementlearning4j

Reinforcement learning framework integrated with deeplearning4j

# Quickstart

* Install [ViZDoom](https://github.com/Marqt/ViZDoom/)
* Install [gym-java-client](https://github.com/deeplearning4j/gym-java-client):
    mvn install
* move the files ../ViZDoom/bin/java/libvizdoom.so and ../ViZDoom/bin/java/vizdoom to ../rl4j/vizdoom/
* make the vizdoom native folder accessible to maven:

On Linux:
    export MAVEN_OPTS=-Djava.library.path=vizdoom

To execute (Currently doom current prediction):
    mvn compile exec:java


# WIP

* Dense DQN (current environment tested: gym CartPole-v0)
* Conv DQN (ViZDoom predict_position)

# TODO

* have working example for each mdp if solvable
* Serialization/Deserialization (load save)
* adapt visualisation to async (in external webapp)
* check if actor critic work
* finish boltzmannQ
* Continuous control
