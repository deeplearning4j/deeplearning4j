---
title: Saving and Loading a Neural Network
layout: default
---

# Saving and Loading a Neural Network

Saving a network in DL4J is typically done by saving, in separate files, a copy of:

* The network configuration, in JSON format
* The network parameters, as a flat vector (in a binary format)
* (Optionally) the network updater (see: [A Note on Updaters](#updaters))

Loading the network is also done in two (or three) steps.

## <a name="vector">Save an Interoperable Vector of All Weights</a>

See the code sample below, for an example of how to save and reload a model.

        //Write the network parameters:
        try(DataOutputStream dos = new DataOutputStream(Files.newOutputStream(Paths.get("coefficients.bin")))){
            Nd4j.write(net.params(),dos);
        }

        //Write the network configuration:
        FileUtils.write(new File("conf.json"), net.getLayerWiseConfigurations().toJson());

        //Load network configuration from disk:
        MultiLayerConfiguration confFromJson = MultiLayerConfiguration.fromJson(FileUtils.readFileToString(new File("conf.json")));

        //Load parameters from disk:
        INDArray newParams;
        try(DataInputStream dis = new DataInputStream(new FileInputStream("coefficients.bin"))){
            newParams = Nd4j.read(dis);
        }

        //Create a MultiLayerNetwork from the saved configuration and parameters
        MultiLayerNetwork savedNetwork = new MultiLayerNetwork(confFromJson);
        savedNetwork.init();
        savedNetwork.setParameters(newParams);

## <a name="updaters">A Note on Updaters</a>
If you want to continue training after loading a network, it is usually advisable to save an additional part of the network, the updater in addition to the configuration and parameters. If no further training of the network is required, you do not need to save and load the updater.

When training a network, we often use training mechanisms such as momentum, AdaGrad and RMSProp. These are known as 'updaters' in DL4J as they modify (update) the raw network gradients used in training, before being used to adjust the network parameters.
It is important to note here that most of these updaters contain internal state (the SGD updater does not). For example, RMSProp and AdaGrad contain a type of history of past gradients, which are used to modify the next gradients. If we continue training without this updater history, the magnitudes of the updates can be different (usually much larger) than if we had the updater history. This can adversely affect network training after reloading the network.

To save an updater, you can use the following processes (in addition to the previous code above)

        //Save the updater:
        try(ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream("updater.bin"))){
            oos.writeObject(model.getUpdater());
        }
        
        //Load the updater:
        org.deeplearning4j.nn.api.Updater updater;
        try(ObjectInputStream ois = new ObjectInputStream(new FileInputStream("updater.bin"))){
            updater = (org.deeplearning4j.nn.api.Updater) ois.readObject();
        }
        
        //Set the updater in the network
        model.setUpdater(updater);

## Model Utils

Here are several ways to save and load data using the ModelUtils class:

<script src="http://gist-it.appspot.com/https://github.com/deeplearning4j/ImageNet-Example/blob/master/src/main/java/imagenet/Utils/ModelUtils.java?slice=22:118"></script>
