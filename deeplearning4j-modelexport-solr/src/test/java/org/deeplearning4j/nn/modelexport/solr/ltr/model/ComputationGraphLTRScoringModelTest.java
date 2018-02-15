package org.deeplearning4j.nn.modelexport.solr.ltr.model;

import java.util.List;

import org.apache.solr.ltr.feature.Feature;
import org.apache.solr.ltr.model.AdapterModel;
import org.apache.solr.ltr.norm.Normalizer;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.api.NeuralNetwork;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class ComputationGraphLTRScoringModelTest extends NeuralNetworkLTRScoringModelTest {

  protected NeuralNetwork buildAndSaveModel(int numFeatures, String serializedModelFileName) throws Exception {

    final ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
        .graphBuilder()
        .addInputs("inputLayer")
        .addLayer("outputLayer",
          new OutputLayer.Builder().nIn(numFeatures).nOut(1).activation(Activation.IDENTITY).build(),
          "inputLayer")
        .setOutputs("outputLayer")
        .build();

    final ComputationGraph model = new ComputationGraph(conf);
    model.init();

    final float[] floats = new float[numFeatures+1];
    float base = 1f;
    for (int ii=0; ii<floats.length; ++ii)
    {
      base *= 2;
      floats[ii] = base;
    }

    final INDArray params = Nd4j.create(floats);
    model.setParams(params);

    ModelSerializer.writeModel(model, serializedModelFileName, false);
    return model;
  }

  protected NeuralNetwork restoreModel(String serializedModelFileName) throws Exception {
    return ModelSerializer.restoreComputationGraph(serializedModelFileName);
  }

  protected AdapterModel newAdapterModel(String name, List<Feature> features, List<Normalizer> norms, String serializedModelFileName) throws Exception {
    final ComputationGraphLTRScoringModel model = new ComputationGraphLTRScoringModel(
          name, features, norms, null, null, null);
    model.setSerializedModelFileName(serializedModelFileName);
    return model;
  }

  protected float score(NeuralNetwork neuralNetwork, float[] vals) throws Exception {
    final INDArray input = Nd4j.create(vals);
    final INDArray[] output = ((ComputationGraph)neuralNetwork).output(input);
    return output[0].getFloat(0);
  }

}
