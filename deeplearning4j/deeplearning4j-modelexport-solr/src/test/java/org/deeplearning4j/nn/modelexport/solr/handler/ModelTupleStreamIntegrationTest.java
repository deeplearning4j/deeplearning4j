package org.deeplearning4j.nn.modelexport.solr.handler;

import java.io.File;
import java.nio.file.Path;
import org.apache.solr.client.solrj.io.Tuple;
import org.apache.solr.client.solrj.io.stream.SolrStream;
import org.apache.solr.client.solrj.io.stream.StreamContext;
import org.apache.solr.client.solrj.io.stream.TupleStream;
import org.apache.solr.client.solrj.request.CollectionAdminRequest;
import org.apache.solr.client.solrj.request.UpdateRequest;
import org.apache.solr.cloud.SolrCloudTestCase;
import org.apache.solr.common.params.ModifiableSolrParams;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.junit.BeforeClass;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class ModelTupleStreamIntegrationTest extends SolrCloudTestCase {

  final private static String MY_COLLECTION_NAME = "mySolrCollection";
  final private static String MY_SERIALIZED_MODEL_FILENAME = "mySerializedModel";

  @BeforeClass
  public static void setupCluster() throws Exception {

    final Path configsetPath = configset("mini-expressible");

    // create and serialize model
    {
      final Model model = buildModel();
      final File serializedModelFile = configsetPath
        .resolve(MY_SERIALIZED_MODEL_FILENAME)
        .toFile();
      ModelSerializer.writeModel(model, serializedModelFile.getPath(), false);
    }

    final String configName = "conf";
    final int numShards = 2;
    final int numReplicas = 2;
    final int maxShardsPerNode = 1;
    final int nodeCount = (numShards*numReplicas + (maxShardsPerNode-1))/maxShardsPerNode;

    // create and configure cluster
    configureCluster(nodeCount)
        .addConfig(configName, configsetPath)
        .configure();

    // create an empty collection
    CollectionAdminRequest.createCollection(MY_COLLECTION_NAME, configName, numShards, numReplicas)
        .setMaxShardsPerNode(maxShardsPerNode)
        .process(cluster.getSolrClient());

    // compose an update request
    final UpdateRequest updateRequest = new UpdateRequest();

    // add some documents
    updateRequest.add(
      sdoc("id", "green",
        "channel_b_f", "0",
        "channel_g_f", "255",
        "channel_r_f", "0"));
    updateRequest.add(
      sdoc("id", "black",
        "channel_b_f", "0",
        "channel_g_f", "0",
        "channel_r_f", "0"));
    updateRequest.add(
      sdoc("id", "yellow",
        "channel_b_f", "0",
        "channel_g_f", "255",
        "channel_r_f", "255"));

    // make the update request
    updateRequest.commit(cluster.getSolrClient(), MY_COLLECTION_NAME);
  }

  private static Model buildModel() throws Exception {

    final int numInputs = 3;
    final int numOutputs = 2;

    final MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
        .list(
            new OutputLayer.Builder().nIn(numInputs).nOut(numOutputs).activation(Activation.IDENTITY).build()
            )
        .build();

    final MultiLayerNetwork model = new MultiLayerNetwork(conf);
    model.init();

    final float[] floats = new float[]{ +1, +1, +1, -1, -1, -1, 0, 0 };
    // positive weight for first output, negative weight for second output, no biases
    assertEquals((numInputs+1)*numOutputs, floats.length);

    final INDArray params = Nd4j.create(floats);
    model.setParams(params);

    return model;
  }

  private void doTest(String expr, String[] expectedIds, Object[] expectedLefts, Object[] expectedRights) throws Exception {
    ModifiableSolrParams paramsLoc = new ModifiableSolrParams();
    paramsLoc.set("expr", expr);
    paramsLoc.set("qt", "/stream");

    String url = cluster.getRandomJetty(random()).getBaseUrl().toString()+"/"+MY_COLLECTION_NAME;


    TupleStream tupleStream = new SolrStream(url, paramsLoc);

    StreamContext context = new StreamContext();
    tupleStream.setStreamContext(context);

    try {
      tupleStream.open();

      for (int ii=0; ii<expectedIds.length; ++ii) {
        final Tuple tuple = tupleStream.read();
        assertFalse(tuple.EOF);

        final String expectedId = expectedIds[ii];
        final String actualId = tuple.getString("id");
        assertEquals(expectedId, actualId);

        if (expectedLefts != null) {
          final Object expectedLeft = expectedLefts[ii];
          final String actualLeft = tuple.getString("left");
          assertEquals(tuple.getMap().toString(), expectedLeft, actualLeft);
        }

        if (expectedRights != null) {
          final Object expectedRight = expectedRights[ii];
          final String actualRight = tuple.getString("right");
          assertEquals(tuple.getMap().toString(), expectedRight, actualRight);
        }
      }

      final Tuple lastTuple = tupleStream.read();
      assertTrue(lastTuple.EOF);

    } finally {
      tupleStream.close();
    }
  }

  @Test
  public void searchTest() throws Exception {

    final String searchExpr =
      "search("+MY_COLLECTION_NAME+"," +
      "zkHost=\""+cluster.getZkClient().getZkServerAddress() + "\"," +
      "q=\"*:*\"," +
      "fl=\"id,channel_b_f,channel_g_f,channel_r_f\"," +
      "qt=\"/export\"," +
      "sort=\"id asc\")";

    final String modelTupleExpr =
      "modelTuple("+searchExpr+"," +
      "serializedModelFileName=\""+MY_SERIALIZED_MODEL_FILENAME+"\"," +
      "inputKeys=\"channel_b_f,channel_g_f,channel_r_f\"," +
      "outputKeys=\"left,right\")";

    final String[] expectedIds = new String[]{ "black", "green", "yellow" };

    {
      final String[] expectedLefts = null;
      final String[] expectedRights = null;
      doTest(searchExpr, expectedIds, expectedLefts, expectedRights);
    }

    {
      final String[] expectedLefts = new String[]{ "0.0", "255.0", "510.0" };
      final String[] expectedRights = new String[]{ "0.0", "-255.0", "-510.0" };
      doTest(modelTupleExpr, expectedIds, expectedLefts, expectedRights);
    }
  }

}
