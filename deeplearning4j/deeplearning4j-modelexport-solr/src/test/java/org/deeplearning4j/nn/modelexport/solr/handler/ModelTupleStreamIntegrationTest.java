/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */
package org.deeplearning4j.nn.modelexport.solr.handler;

import java.io.File;
import java.nio.file.Path;
import java.security.SecureRandom;
import com.carrotsearch.randomizedtesting.ThreadFilter;
import com.carrotsearch.randomizedtesting.annotations.ThreadLeakFilters;
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
import org.junit.jupiter.api.*;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.api.memory.provider.BasicWorkspaceManager;
import org.nd4j.rng.deallocator.NativeRandomDeallocator;
import org.junit.jupiter.api.extension.ExtendWith;

@ThreadLeakFilters(defaultFilters = true, filters = { ModelTupleStreamIntegrationTest.PrivateDeallocatorThreadsFilter.class })
@DisplayName("Model Tuple Stream Integration Test")
@Disabled("Timeout issue")
@Tag(TagNames.SOLR)
@Tag(TagNames.DIST_SYSTEMS)
class ModelTupleStreamIntegrationTest extends SolrCloudTestCase {

    static {
        /*
    This is a hack around the backend-dependent nature of secure random implementations
    though we can set the secure random algorithm in our pom.xml files (via maven surefire and test.solr.allowed.securerandom)
    there isn't a mechanism that is completely platform independent.
    By setting it there (for example, to NativePRNG) that makes it pass on some platforms like Linux but fails on some JVMs on Windows
    For testing purposes, we don't need strict guarantees around RNG, hence we don't want to enforce the RNG algorithm
     */
        String algorithm = new SecureRandom().getAlgorithm();
        System.setProperty("test.solr.allowed.securerandom", algorithm);
    }

    @DisplayName("Private Deallocator Threads Filter")
    static class PrivateDeallocatorThreadsFilter implements ThreadFilter {

        /**
         * Reject deallocator threads over whose cleanup this test has no control.
         */
        @Override
        public boolean reject(Thread thread) {
            final ThreadGroup threadGroup = thread.getThreadGroup();
            final String threadGroupName = (threadGroup == null ? null : threadGroup.getName());
            if (threadGroupName != null && threadGroupName.endsWith(ModelTupleStreamIntegrationTest.class.getSimpleName())) {
                final String threadName = thread.getName();
                if (threadName.startsWith(NativeRandomDeallocator.DeallocatorThreadNamePrefix) || threadName.toLowerCase().contains("deallocator") || threadName.equals(BasicWorkspaceManager.WorkspaceDeallocatorThreadName)) {
                    return true;
                }
            }
            return false;
        }
    }

    final private static String MY_COLLECTION_NAME = "mySolrCollection";

    final private static String MY_SERIALIZED_MODEL_FILENAME = "mySerializedModel";

    @BeforeAll
    static void setupCluster() throws Exception {
        final Path configsetPath = configset("mini-expressible");
        // create and serialize model
        {
            final Model model = buildModel();
            final File serializedModelFile = configsetPath.resolve(MY_SERIALIZED_MODEL_FILENAME).toFile();
            ModelSerializer.writeModel(model, serializedModelFile.getPath(), false);
        }
        final String configName = "conf";
        final int numShards = 2;
        final int numReplicas = 2;
        final int maxShardsPerNode = 1;
        final int nodeCount = (numShards * numReplicas + (maxShardsPerNode - 1)) / maxShardsPerNode;
        // create and configure cluster
        configureCluster(nodeCount).addConfig(configName, configsetPath).configure();
        // create an empty collection
        CollectionAdminRequest.createCollection(MY_COLLECTION_NAME, configName, numShards, numReplicas).setMaxShardsPerNode(maxShardsPerNode).process(cluster.getSolrClient());
        // compose an update request
        final UpdateRequest updateRequest = new UpdateRequest();
        // add some documents
        updateRequest.add(sdoc("id", "green", "channel_b_f", "0", "channel_g_f", "255", "channel_r_f", "0"));
        updateRequest.add(sdoc("id", "black", "channel_b_f", "0", "channel_g_f", "0", "channel_r_f", "0"));
        updateRequest.add(sdoc("id", "yellow", "channel_b_f", "0", "channel_g_f", "255", "channel_r_f", "255"));
        // make the update request
        updateRequest.commit(cluster.getSolrClient(), MY_COLLECTION_NAME);
    }

    private static Model buildModel() throws Exception {
        final int numInputs = 3;
        final int numOutputs = 2;
        final MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().list(new OutputLayer.Builder().nIn(numInputs).nOut(numOutputs).activation(Activation.IDENTITY).lossFunction(LossFunctions.LossFunction.MSE).build()).build();
        final MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        final float[] floats = new float[] { +1, +1, +1, -1, -1, -1, 0, 0 };
        // positive weight for first output, negative weight for second output, no biases
        assertEquals((numInputs + 1) * numOutputs, floats.length);
        final INDArray params = Nd4j.create(floats);
        model.setParams(params);
        return model;
    }

    private void doTest(String expr, String[] expectedIds, Object[] expectedLefts, Object[] expectedRights) throws Exception {
        ModifiableSolrParams paramsLoc = new ModifiableSolrParams();
        paramsLoc.set("expr", expr);
        paramsLoc.set("qt", "/stream");
        String url = cluster.getRandomJetty(random()).getBaseUrl().toString() + "/" + MY_COLLECTION_NAME;
        TupleStream tupleStream = new SolrStream(url, paramsLoc);
        StreamContext context = new StreamContext();
        tupleStream.setStreamContext(context);
        try {
            tupleStream.open();
            for (int ii = 0; ii < expectedIds.length; ++ii) {
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
    @DisplayName("Search Test")
    void searchTest() throws Exception {
        int testsCount = 0;
        final String searchExpr = "search(" + MY_COLLECTION_NAME + "," + "zkHost=\"" + cluster.getZkClient().getZkServerAddress() + "\"," + "q=\"*:*\"," + "fl=\"id,channel_b_f,channel_g_f,channel_r_f\"," + "qt=\"/export\"," + "sort=\"id asc\")";
        final String modelTupleExpr = "modelTuple(" + searchExpr + "," + "serializedModelFileName=\"" + MY_SERIALIZED_MODEL_FILENAME + "\"," + "inputKeys=\"channel_b_f,channel_g_f,channel_r_f\"," + "outputKeys=\"left,right\")";
        final String[] expectedIds = new String[] { "black", "green", "yellow" };
        {
            final String[] expectedLefts = null;
            final String[] expectedRights = null;
            doTest(searchExpr, expectedIds, expectedLefts, expectedRights);
            ++testsCount;
        }
        {
            final String[] expectedLefts = new String[] { "0.0", "255.0", "510.0" };
            final String[] expectedRights = new String[] { "0.0", "-255.0", "-510.0" };
            doTest(modelTupleExpr, expectedIds, expectedLefts, expectedRights);
            ++testsCount;
        }
        assertEquals(2, testsCount);
    }
}
