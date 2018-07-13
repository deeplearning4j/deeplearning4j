package org.deeplearning4j.nn.dataimport.solr.client.solrj.io.stream;

import com.carrotsearch.randomizedtesting.annotations.ThreadLeakFilters;
import com.carrotsearch.randomizedtesting.ThreadFilter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import org.apache.solr.client.solrj.request.CollectionAdminRequest;
import org.apache.solr.client.solrj.request.UpdateRequest;
import org.apache.solr.cloud.SolrCloudTestCase;
import org.apache.solr.common.SolrInputDocument;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.junit.BeforeClass;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.nd4j.linalg.memory.provider.BasicWorkspaceManager;
import org.nd4j.rng.deallocator.NativeRandomDeallocator;

@ThreadLeakFilters(defaultFilters = true, filters = {
  TupleStreamDataSetIteratorTest.PrivateDeallocatorThreadsFilter.class
})
public class TupleStreamDataSetIteratorTest extends SolrCloudTestCase {

  public static class PrivateDeallocatorThreadsFilter implements ThreadFilter {
    /**
     * Reject deallocator threads over whose cleanup this test has no control.
     */
    @Override
    public boolean reject(Thread thread) {
      final ThreadGroup threadGroup = thread.getThreadGroup();
      final String threadGroupName = (threadGroup == null ? null : threadGroup.getName());

      if (threadGroupName != null &&
          threadGroupName.endsWith(TupleStreamDataSetIteratorTest.class.getSimpleName())) {

        final String threadName = thread.getName();
/*
        if (threadName.startsWith(NativeRandomDeallocator.DeallocatorThreadNamePrefix) ||
            threadName.equals("JavaCPP Deallocator") ||
            threadName.equals(BasicWorkspaceManager.WorkspaceDeallocatorThreadName)) {
          return true;
        }
*/
        if (threadName.startsWith("NativeRandomDeallocator thread ") ||
            threadName.equals("JavaCPP Deallocator") ||
            threadName.equals("Workspace deallocator thread")) {
          return true;
        }
      }

      return false;
    }
  }

  private static int numDocs = 0;

  @BeforeClass
  public static void setupCluster() throws Exception {

    final int numShards = 2;
    final int numReplicas = 2;
    final int maxShardsPerNode = 1;
    final int nodeCount = (numShards*numReplicas + (maxShardsPerNode-1))/maxShardsPerNode;

    // create and configure cluster
    configureCluster(nodeCount)
        .addConfig("conf", configset("mini"))
        .configure();

    // create an empty collection
    CollectionAdminRequest.createCollection("mySolrCollection", "conf", numShards, numReplicas)
        .setMaxShardsPerNode(maxShardsPerNode)
        .process(cluster.getSolrClient());

    // compose an update request
    final UpdateRequest updateRequest = new UpdateRequest();

    final List<Integer> docIds = new ArrayList<Integer>();
    for (int phase = 1; phase <= 2; ++phase) {
      int docIdsIdx = 0;

      if (phase == 2) {
        Collections.shuffle(docIds);
      }

      final int increment = 32;

      for (int b = 0; b <= 256; b += increment) {
        if (256 == b) b--;
        for (int g = 0; g <= 256; g += increment) {
          if (256 == g) g--;
          for (int r = 0; r <= 256; r += increment) {
            if (256 == r) r--;

            if (phase == 1) {
              docIds.add(docIds.size()+1);
              continue;
            }

            final float luminance = (b*0.0722f + g*0.7152f + r*0.2126f)/(255*3.0f); // https://en.wikipedia.org/wiki/Luma_(video)

            final SolrInputDocument doc = sdoc("id", Integer.toString(docIds.get(docIdsIdx++)),
              "channel_b_f", Float.toString(b/255f),
              "channel_g_f", Float.toString(g/255f),
              "channel_r_f", Float.toString(r/255f),
              "luminance_f", Float.toString(luminance));

            updateRequest.add(doc);
            ++numDocs;

          }
        }
      }
    }

    // make the update request
    updateRequest.commit(cluster.getSolrClient(), "mySolrCollection");
  }

  private static class CountingIterationListener extends ScoreIterationListener {

    private int numIterationsDone = 0;

    public CountingIterationListener() {
      super(1);
    }

    public int numIterationsDone() {
      return numIterationsDone;
    }

    @Override
    public void iterationDone(Model model, int iteration, int epoch) {
      super.iterationDone(model, iteration, epoch);
      ++numIterationsDone;
    }

  }

  @Test
  public void iterateTest() throws Exception {
    doIterateTest(true);
    doIterateTest(false);
  }

  private void doIterateTest(boolean withIdKey) throws Exception {

      try (final TupleStreamDataSetIterator
      tsdsi = new TupleStreamDataSetIterator(
        123 /* batch */,
        (withIdKey ? "greeting" : null) /* idKey */,
        new String[] { "pie" },
        new String[] { "answer" },
        "tuple(greeting=\"hello world\",pie=3.14,answer=42)",
        null)) {

      assertTrue(tsdsi.hasNext());
      final DataSet ds = tsdsi.next();

      assertEquals(1, ds.getFeatures().length());
      assertEquals(3.14f, ds.getFeatures().getFloat(0), 0.0f);

      assertEquals(1, ds.getLabels().length());
      assertEquals(42f, ds.getLabels().getFloat(0), 0.0f);

      assertFalse(tsdsi.hasNext());
    }
  }

  @Test
  public void modelFitTest() throws Exception {

    final MultiLayerNetwork model = new MultiLayerNetwork(
      new NeuralNetConfiguration.Builder()
        .list(
          new OutputLayer.Builder(LossFunction.MSE)
            .nIn(3)
            .nOut(1)
            .weightInit(WeightInit.ONES)
            .activation(Activation.IDENTITY)
            .build()
          )
        .pretrain(false)
        .backprop(true)
        .build()
      );
    model.init();

    int batch = 1;
    for (int ii=1; ii<=5; ++ii) {
      final CountingIterationListener listener = new CountingIterationListener();
      model.setListeners(listener);
      batch *= 2;

      try (final TupleStreamDataSetIterator tsdsi =
          new TupleStreamDataSetIterator(
            batch,
            "id" /* idKey */,
            new String[] { "channel_b_f", "channel_g_f", "channel_r_f" },
            new String[] { "luminance_f" },
            "search(mySolrCollection," +
            "q=\"id:*\"," +
            "fl=\"id,channel_b_f,channel_g_f,channel_r_f,luminance_f\"," +
            "sort=\"id asc\"," +
            "qt=\"/export\")",
            cluster.getZkClient().getZkServerAddress())) {

        model.fit(tsdsi);
      }

      assertEquals("numIterationsDone="+listener.numIterationsDone()+" numDocs="+numDocs+" batch="+batch,
                   (numDocs+(batch-1))/batch, listener.numIterationsDone());
    }
  }

}
