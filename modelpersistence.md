---
title: 
layout: default
---

# Saving and Loading a Neural Network

Here are two ways to save and load models with Deeplearning4j.

## <a name="vector">Save an Interoperable Vector of All Weights</a>

Please see [this code sample](https://github.com/deeplearning4j/dl4j-0.4-examples/blob/master/src/main/java/org/deeplearning4j/examples/deepbelief/DBNIrisExample.java#L127) for an example of how to save and reload a model.

        OutputStream fos = Files.newOutputStream(Paths.get("coefficients.bin"));
        DataOutputStream dos = new DataOutputStream(fos);
        Nd4j.write(model.params(), dos);
        dos.flush();
        dos.close();
        FileUtils.write(new File("conf.json"), model.getLayerWiseConfigurations().toJson());
        
        MultiLayerConfiguration confFromJson = MultiLayerConfiguration.fromJson(FileUtils.readFileToString(new File("conf.json")));
        DataInputStream dis = new DataInputStream(new FileInputStream("coefficients.bin"));
        INDArray newParams = Nd4j.read(dis);
        dis.close();
        MultiLayerNetwork savedNetwork = new MultiLayerNetwork(confFromJson);
        savedNetwork.init();
        savedNetwork.setParameters(newParams);
        System.out.println("Original network params " + model.params());
        System.out.println(savedNetwork.params());

## Model Utils

Here are several ways to save and load data using the ModelUtils class:

<script src="http://gist-it.appspot.com/https://github.com/deeplearning4j/ImageNet-Example/blob/master/src/main/java/imagenet/Utils/ModelUtils.java?slice=22:118"></script>
