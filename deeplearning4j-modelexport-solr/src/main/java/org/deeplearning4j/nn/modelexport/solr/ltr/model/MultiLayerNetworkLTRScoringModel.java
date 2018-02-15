package org.deeplearning4j.nn.modelexport.solr.ltr.model;

import java.io.InputStream;
import java.io.IOException;
import java.util.List;
import java.util.Map;

import org.apache.solr.ltr.feature.Feature;
import org.apache.solr.ltr.norm.Normalizer;
import org.deeplearning4j.nn.api.NeuralNetwork;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

// TODO: change 7_2 to 7_3 once that is released
/**
 * An <a href="https://lucene.apache.org/solr/7_2_0/solr-ltr/org/apache/solr/ltr/model/LTRScoringModel.html">
 * org.apache.solr.ltr.model.LTRScoringModel</a> that computes scores using a {@link MultiLayerNetwork}.
 * <p>
 * Example configuration (snippeet):
 * <pre>{
  "class": "org.deeplearning4j.nn.modelexport.solr.ltr.model.MultiLayerNetworkLTRScoringModel",
  "name": "myMultiLayerNetwork",
  "features" : [
    { "name" : "documentRecency", ... },
    { "name" : "isBook", ... },
    { "name" : "originalScore", ... }
  ],
  "params": {
    "serializedModelFileName": "mySerializedMultiLayerNetwork"
  }
}</pre>
 */
public class MultiLayerNetworkLTRScoringModel extends NeuralNetworkLTRScoringModel {

  public MultiLayerNetworkLTRScoringModel(String name, List<Feature> features, List<Normalizer> norms, String featureStoreName,
      List<Feature> allFeatures, Map<String,Object> params) {
    super(name, features, norms, featureStoreName, allFeatures, params);
  }

  protected NeuralNetwork restoreNeuralNetwork(InputStream inputStream) throws IOException {
    return ModelSerializer.restoreMultiLayerNetwork(inputStream);
  }

  @Override
  public float score(float[] modelFeatureValuesNormalized) {
    final MultiLayerNetwork multiLayerNetwork = (MultiLayerNetwork)neuralNetwork;
    final INDArray input = Nd4j.create(modelFeatureValuesNormalized);
    final INDArray output = multiLayerNetwork.output(input);
    return output.getFloat(0);
  }

}
