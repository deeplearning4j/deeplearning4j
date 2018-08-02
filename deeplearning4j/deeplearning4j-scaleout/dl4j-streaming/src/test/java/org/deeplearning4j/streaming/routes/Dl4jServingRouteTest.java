/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.streaming.routes;

import com.google.common.io.Files;
import org.apache.camel.*;
import org.apache.camel.builder.RouteBuilder;
import org.apache.camel.component.kafka.KafkaConstants;
import org.apache.camel.model.ProcessorDefinition;
import org.apache.camel.test.junit4.CamelTestSupport;
import org.apache.commons.io.FileUtils;
import org.apache.commons.net.util.Base64;
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.streaming.embedded.EmbeddedKafkaCluster;
import org.deeplearning4j.streaming.embedded.EmbeddedZookeeper;
import org.deeplearning4j.streaming.embedded.TestUtils;
import org.deeplearning4j.util.ModelSerializer;
import org.junit.After;
import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.util.UUID;


/**
 * Created by agibsonccc on 6/12/16.
 */
public class Dl4jServingRouteTest extends CamelTestSupport {
    private static EmbeddedZookeeper zookeeper;
    private static EmbeddedKafkaCluster kafkaCluster;
    private static int zkPort;
    public final static String LOCALHOST = "localhost";
    private File dir = Files.createTempDir();
    private DataSet next;
    private static String topicName = "predict";

    @BeforeClass
    public static void init() throws Exception {
        zkPort = TestUtils.getAvailablePort();
        zookeeper = new EmbeddedZookeeper(zkPort);
        zookeeper.startup();
        kafkaCluster = new EmbeddedKafkaCluster(LOCALHOST + ":" + zkPort);
        kafkaCluster.startup();
        kafkaCluster.createTopics(topicName);
    }

    @AfterClass
    public static void after2() {
        kafkaCluster.shutdown();
        zookeeper.shutdown();
    }

    @Override
    protected RouteBuilder createRouteBuilder() throws Exception {
        DataSetIterator iter = new IrisDataSetIterator(150, 150);
        next = iter.next();
        next.normalizeZeroMeanZeroUnitVariance();

        return new RouteBuilder() {
            @Override
            public void configure() throws Exception {
                final String kafkaUri = String.format("kafka:%s?topic=%s&groupId=dl4j-serving",
                                kafkaCluster.getBrokerList(), topicName);
                from("direct:start").process(new Processor() {
                    @Override
                    public void process(Exchange exchange) throws Exception {
                        final INDArray arr = next.getFeatures();
                        ByteArrayOutputStream bos = new ByteArrayOutputStream();
                        DataOutputStream dos = new DataOutputStream(bos);
                        Nd4j.write(arr, dos);
                        byte[] bytes = bos.toByteArray();
                        String base64 = Base64.encodeBase64String(bytes);
                        exchange.getIn().setBody(base64, String.class);
                        exchange.getIn().setHeader(KafkaConstants.KEY, UUID.randomUUID().toString());
                        exchange.getIn().setHeader(KafkaConstants.PARTITION_KEY, "1");
                    }
                }).to(kafkaUri);
            }
        };
    }


    @Override
    public boolean isUseDebugger() {
        // must enable debugger
        return true;
    }


    @Override
    protected void debugBefore(Exchange exchange, Processor processor, ProcessorDefinition<?> definition, String id,
                    String shortName) {
        // this method is invoked before we are about to enter the given processor
        // from your Java editor you can just add a breakpoint in the code line below
        log.info("Before " + definition + " with body " + exchange.getIn().getBody());
    }

    @After
    public void after() throws Exception {
        FileUtils.deleteDirectory(dir);
    }

    @Test
    public void testServingRoute() throws Exception {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                        .optimizationAlgo(OptimizationAlgorithm.LINE_GRADIENT_DESCENT).seed(123).list()
                        .layer(0, new DenseLayer.Builder().nIn(4).nOut(3).weightInit(WeightInit.XAVIER)
                                        .activation(Activation.TANH).build())
                        .layer(1, new DenseLayer.Builder().nIn(3).nOut(2).weightInit(WeightInit.XAVIER)
                                        .activation(Activation.TANH).build())
                        .layer(2, new org.deeplearning4j.nn.conf.layers.OutputLayer.Builder(
                                        LossFunctions.LossFunction.MCXENT).weightInit(WeightInit.XAVIER)
                                                        .activation(Activation.SOFTMAX).nIn(2).nOut(3).build())
                        .backprop(true).pretrain(false).build();


        MultiLayerNetwork network = new MultiLayerNetwork(conf);
        network.init();
        network.setListeners(new ScoreIterationListener(1));


        network.fit(next);
        String outputPath = "networktest.zip";
        dir.mkdirs();
        File tmp = new File(dir, "tmp.txt");
        tmp.createNewFile();
        tmp.deleteOnExit();

        ModelSerializer.writeModel(network, outputPath, false);
        final boolean computationGraph = false;
        final String uri = String.format("file://%s?fileName=tmp.txt", dir.getAbsolutePath());
        context.addRoutes(DL4jServeRouteBuilder.builder().computationGraph(computationGraph)
                        .zooKeeperPort(zookeeper.getPort()).kafkaBroker(kafkaCluster.getBrokerList())
                        .consumingTopic(topicName).modelUri(outputPath).outputUri(uri).finalProcessor(new Processor() {
                            @Override
                            public void process(Exchange exchange) throws Exception {
                                exchange.getIn().setBody(exchange.getIn().getBody().toString());

                            }
                        }).build());
        context.startAllRoutes();

        Endpoint endpoint = context.getRoutes().get(1).getConsumer().getEndpoint();
        ConsumerTemplate consumerTemplate = context.createConsumerTemplate();
        ProducerTemplate producerTemplate = context.createProducerTemplate();
        producerTemplate.sendBody("direct:start", "hello");
        consumerTemplate.receiveBody(endpoint, 3000, String.class);
        String contents = FileUtils.readFileToString(new File(dir, "tmp.txt"));
    }

}
