package org.deeplearning4j.nn.modelexport.solr.ltr.model;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import org.apache.lucene.search.Explanation;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.solr.core.SolrResourceLoader;
import org.apache.solr.ltr.feature.Feature;
import org.apache.solr.ltr.feature.FeatureException;
import org.apache.solr.ltr.model.ModelException;
import org.apache.solr.ltr.norm.IdentityNormalizer;
import org.apache.solr.ltr.norm.Normalizer;
import org.apache.solr.request.SolrQueryRequest;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelGuesser;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class ScoringModelTest {

  protected static class DummyFeature extends Feature {

    public DummyFeature(String name) {
      super(name, Collections.EMPTY_MAP);
    }

    @Override
    protected void validate() throws FeatureException {
    }

    @Override
    public FeatureWeight createWeight(IndexSearcher searcher, boolean needsScores, SolrQueryRequest request,
        Query originalQuery, Map<String,String[]> efi) throws IOException {
      return null;
    }

    @Override
    public LinkedHashMap<String,Object> paramsToMap() {
      return null;
    }

  };

  protected List<Feature> featuresList(int numFeatures) throws Exception {
    final ArrayList<Feature> features = new ArrayList<Feature>();
    for (int ii=1; ii<=numFeatures; ++ii)
    {
      features.add(new DummyFeature("dummy"+ii));
    }
    return features;
  }

  protected List<Normalizer> normalizersList(int numNormalizers) throws Exception {
    final ArrayList<Normalizer> normalizers = new ArrayList<Normalizer>();
    for (int ii=1; ii<=numNormalizers; ++ii)
    {
      normalizers.add(IdentityNormalizer.INSTANCE);
    }
    return normalizers;
  }

  protected List<float[]> floatsList(int numFloats) {
    final List<float[]> floatsList = new ArrayList<float[]>();
    final float[] floats0 = new float[numFloats];
    final float[] floats1 = new float[numFloats];
    for (int ii=0; ii<numFloats; ++ii) {
      floats0[ii] = 0f;
      floats1[ii] = 1f;
    }
    floatsList.add(floats0);
    floatsList.add(floats1);
    return floatsList;
  }

  @Test
  public void test() throws Exception {
    for (int numFeatures = 3; numFeatures <= 5; ++numFeatures) {

      for (Model model : new Model[]{
          buildMultiLayerNetworkModel(numFeatures),
          buildComputationGraphModel(numFeatures)
        }) {

        doTest(model, numFeatures);

      }
    }
  }

  private void doTest(Model originalModel, int numFeatures) throws Exception {

    final Path tempDirPath = Files.createTempDirectory(null);
    final File tempDirFile = tempDirPath.toFile();
    tempDirFile.deleteOnExit();

    final SolrResourceLoader solrResourceLoader = new SolrResourceLoader(tempDirPath);

    final File tempFile = File.createTempFile("prefix", "suffix", tempDirFile);
    tempFile.deleteOnExit();

    final String serializedModelFileName = tempFile.getPath();

    ModelSerializer.writeModel(originalModel, serializedModelFileName, false);

    final Model restoredModel = ModelGuesser.loadModelGuess(serializedModelFileName);

    final ScoringModel ltrModel = new ScoringModel(
        "myModel", featuresList(numFeatures), normalizersList(numFeatures), null, null, null);
    ltrModel.setSerializedModelFileName(serializedModelFileName);
    ltrModel.init(solrResourceLoader);

    for (final float[] floats : floatsList(numFeatures)) {

      final float originalScore = ScoringModel.outputScore((Model)originalModel, floats);
      final float restoredScore = ScoringModel.outputScore((Model)restoredModel, floats);
      final float ltrScore = ltrModel.score(floats);

      assertEquals(originalScore, restoredScore, 0f);
      assertEquals(originalScore, ltrScore, 0f);

      if (3 == numFeatures) {
        final List<Explanation> explanations = new ArrayList<Explanation>();
        explanations.add(Explanation.match(floats[0], ""));
        explanations.add(Explanation.match(floats[1], ""));
        explanations.add(Explanation.match(floats[2], ""));

        final Explanation explanation = ltrModel.explain(null, 0, ltrScore, explanations);
        assertEquals(ltrScore+" = (name=myModel"+
            ",class="+ltrModel.getClass().getSimpleName()+
            ",featureValues="+
            "[dummy1="+Float.toString(floats[0])+
            ",dummy2="+Float.toString(floats[1])+
            ",dummy3="+Float.toString(floats[2])+
            "])\n",
            explanation.toString());
      }
    }

    final ScoringModel invalidLtrModel = new ScoringModel(
        "invalidModel", featuresList(numFeatures+1), normalizersList(numFeatures+1), null, null, null);
    invalidLtrModel.setSerializedModelFileName(serializedModelFileName);
    try {
      invalidLtrModel.init(solrResourceLoader);
      fail("expected to exception from invalid model init");
    } catch (ModelException me) {
      assertTrue(me.getMessage().startsWith("score(...) test failed for model "));
    }

  }

  protected Model buildMultiLayerNetworkModel(int numFeatures) throws Exception {

    final MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
        .list(
            new OutputLayer.Builder().nIn(numFeatures).nOut(1).activation(Activation.IDENTITY).build()
            )
        .build();

    final MultiLayerNetwork model = new MultiLayerNetwork(conf);
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

    return model;
  }

  protected Model buildComputationGraphModel(int numFeatures) throws Exception {

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

    return model;
  }

}
