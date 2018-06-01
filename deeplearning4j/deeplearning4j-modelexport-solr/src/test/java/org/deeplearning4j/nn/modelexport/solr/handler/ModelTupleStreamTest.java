package org.deeplearning4j.nn.modelexport.solr.handler;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import org.apache.solr.client.solrj.io.stream.expr.StreamFactory;
import org.apache.solr.client.solrj.io.stream.StreamContext;
import org.apache.solr.client.solrj.io.stream.TupleStream;
import org.apache.solr.client.solrj.io.SolrClientCache;
import org.apache.solr.client.solrj.io.Tuple;
import org.apache.solr.core.SolrResourceLoader;
import org.apache.solr.handler.SolrDefaultStreamFactory;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelexport.solr.ltr.model.ScoringModel; // TODO: temporary only
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelGuesser;
import org.deeplearning4j.util.ModelSerializer;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.assertNotNull;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class ModelTupleStreamTest {

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
    for (int numInputs = 1; numInputs <= 5; ++numInputs) {
      for (int numOutputs = 1; numOutputs <= 5; ++numOutputs) {

        for (Model model : new Model[]{
            buildMultiLayerNetworkModel(numInputs, numOutputs),
            buildComputationGraphModel(numInputs, numOutputs)
          }) {

          doTest(model, numInputs, numOutputs);

        }
      }
    }
  }

  private void doTest(Model originalModel, int numInputs, int numOutputs) throws Exception {

    final Path tempDirPath = Files.createTempDirectory(null);
    final File tempDirFile = tempDirPath.toFile();
    tempDirFile.deleteOnExit();

    final SolrResourceLoader solrResourceLoader = new SolrResourceLoader(tempDirPath);

    final File tempFile = File.createTempFile("prefix", "suffix", tempDirFile);
    tempFile.deleteOnExit();

    final String serializedModelFileName = tempFile.getPath();

    ModelSerializer.writeModel(originalModel, serializedModelFileName, false);

    final Model restoredModel = ModelGuesser.loadModelGuess(serializedModelFileName);

    final StreamContext streamContext = new StreamContext();
    final SolrClientCache solrClientCache = new SolrClientCache();
    streamContext.setSolrClientCache(solrClientCache);

    final String[] inputKeys = new String[numInputs];
    final String inputKeysList = fillArray(inputKeys, "input", ",");

    final String[] outputKeys = new String[numOutputs];
    final String outputKeysList = fillArray(outputKeys, "output", ",");

    for (final float[] floats : floatsList(numInputs)) {

      final String inputValuesList;
      {
        final StringBuilder sb = new StringBuilder();
        for (int ii=0; ii<inputKeys.length; ++ii) {
          if (0 < ii) sb.append(',');
          sb.append(inputKeys[ii]).append('=').append(floats[ii]);
        }
        inputValuesList = sb.toString();
      }

      final String expressionClause = "model("
        + " tuple(" + inputValuesList + ")"
        + ",serializedModelFileName=\"" + serializedModelFileName + "\""
        + ",inputKeys=\"" + inputKeysList + "\""
        + ",outputKeys=\"" + outputKeysList + "\""
        + ")";

      final StreamFactory streamFactory = new SolrDefaultStreamFactory()
          .withSolrResourceLoader(solrResourceLoader)
          .withFunctionName("model", ModelTupleStream.class);

      final TupleStream tupleStream = streamFactory.constructStream(expressionClause);
      tupleStream.setStreamContext(streamContext);

      assertTrue(tupleStream instanceof ModelTupleStream);
      final ModelTupleStream modelTupleStream = (ModelTupleStream)tupleStream;

      modelTupleStream.open();
      {
        final Tuple tuple1 = modelTupleStream.read();
        assertNotNull(tuple1);
        assertFalse(tuple1.EOF);

        for (int ii=0; ii<outputKeys.length; ++ii)
        {
          final double originalScore = ScoringModel.output((Model)originalModel, Nd4j.create(floats)).getDouble(ii);
          final double restoredScore = ScoringModel.output((Model)restoredModel, Nd4j.create(floats)).getDouble(ii);
          assertEquals(
            originalModel.getClass().getSimpleName()+" (originalScore-restoredScore)="+(originalScore-restoredScore),
            originalScore, restoredScore, 1e-5);

          final Double outputValue = tuple1.getDouble(outputKeys[ii]);
          assertNotNull(outputValue);
          final double tupleScore = outputValue.doubleValue();
          assertEquals(
            originalModel.getClass().getSimpleName()+" (originalScore-tupleScore["+ii+"])="+(originalScore-outputValue.doubleValue()),
            originalScore, outputValue.doubleValue(), 1e-5);
        }

        final Tuple tuple2 = modelTupleStream.read();
        assertNotNull(tuple2);
        assertTrue(tuple2.EOF);
      }
      modelTupleStream.close();
    }

  }

  /**
   * Fills an existing array using prefix and delimiter, e.g.
   * input: arr = [ "", "", "" ] prefix="value" delimiter=","
   * output: arr = [ "value1", "value2", "value3" ]
   * return: "value1,value2,value3"
   */
  private static String fillArray(String[] arr, final String prefix, final String delimiter) {
    final StringBuilder sb = new StringBuilder();
    for (int ii=0; ii<arr.length; ++ii) {
      arr[ii] = prefix + Integer.toString(ii+1);
      if (0 < ii) sb.append(delimiter);
      sb.append(arr[ii]);
    }
    return sb.toString();
  }

  protected Model buildMultiLayerNetworkModel(int numInputs, int numOutputs) throws Exception {

    final MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
        .list(
            new OutputLayer.Builder().nIn(numInputs).nOut(numOutputs).activation(Activation.IDENTITY).build()
            )
        .build();

    final MultiLayerNetwork model = new MultiLayerNetwork(conf);
    model.init();

    final float[] floats = new float[(numInputs+1)*numOutputs];
    final float base0 = 0.01f;
    float base = base0;
    for (int ii=0; ii<floats.length; ++ii)
    {
      base *= 2;
      if (base > 1/base0) base = base0;
      floats[ii] = base;
    }

    final INDArray params = Nd4j.create(floats);
    model.setParams(params);

    return model;
  }

  protected Model buildComputationGraphModel(int numInputs, int numOutputs) throws Exception {

    final ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
        .graphBuilder()
        .addInputs("inputLayer")
        .addLayer("outputLayer",
          new OutputLayer.Builder().nIn(numInputs).nOut(numOutputs).activation(Activation.IDENTITY).build(),
          "inputLayer")
        .setOutputs("outputLayer")
        .build();

    final ComputationGraph model = new ComputationGraph(conf);
    model.init();

    final float[] floats = new float[(numInputs+1)*numOutputs];
    final float base0 = 0.01f;
    float base = base0;
    for (int ii=0; ii<floats.length; ++ii)
    {
      base *= 2;
      if (base > 1/base0) base = base0;
      floats[ii] = base;
    }

    final INDArray params = Nd4j.create(floats);
    model.setParams(params);

    return model;
  }

}
