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

package org.deeplearning4j.nn.modelimport.keras.weights;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.KerasLayer;
import org.deeplearning4j.nn.modelimport.keras.KerasModel;
import org.deeplearning4j.nn.modelimport.keras.layers.convolutional.KerasSpaceToDepth;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;
import java.util.Arrays;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

@Slf4j
public class KerasWeightSettingTests {

    @Rule
    public final TemporaryFolder testDir = new TemporaryFolder();

    @Test
    public void testSimpleLayersWithWeights() throws Exception {
        int[] kerasVersions = new int[]{1, 2};
        String[] backends = new String[]{"tensorflow", "theano"};

        for (int version : kerasVersions) {
            for (String backend : backends) {
                String densePath = "modelimport/keras/weights/dense_" + backend + "_" + version + ".h5";
                importDense(densePath);

                String conv2dPath = "modelimport/keras/weights/conv2d_" + backend + "_" + version + ".h5";
                importConv2D(conv2dPath);

                if (version == 2 && backend.equals("tensorflow")) { // TODO should work for theano
                    String conv2dReshapePath = "modelimport/keras/weights/conv2d_reshape_"
                            + backend + "_" + version + ".h5";
                    System.out.println(backend + "_" + version);
                    importConv2DReshape(conv2dReshapePath);
                }

                if (version == 2) {
                    String conv1dFlattenPath = "modelimport/keras/weights/embedding_conv1d_flatten_"
                            + backend + "_" + version + ".h5";
                    importConv1DFlatten(conv1dFlattenPath);
                }

                String lstmPath = "modelimport/keras/weights/lstm_" + backend + "_" + version + ".h5";
                importLstm(lstmPath);

                String embeddingLstmPath = "modelimport/keras/weights/embedding_lstm_"
                        + backend + "_" + version + ".h5";
                importEmbeddingLstm(embeddingLstmPath);


                if (version == 2) {
                    String embeddingConv1dExtendedPath = "modelimport/keras/weights/embedding_conv1d_extended_"
                            + backend + "_" + version + ".h5";
                    importEmbeddingConv1DExtended(embeddingConv1dExtendedPath);
                }

                if (version == 2) {
                    String embeddingConv1dPath = "modelimport/keras/weights/embedding_conv1d_"
                            + backend + "_" + version + ".h5";
                    importEmbeddingConv1D(embeddingConv1dPath);
                }

                String simpleRnnPath = "modelimport/keras/weights/simple_rnn_" + backend + "_" + version + ".h5";
                importSimpleRnn(simpleRnnPath);

                String bidirectionalLstmPath = "modelimport/keras/weights/bidirectional_lstm_"
                        + backend + "_" + version + ".h5";
                importBidirectionalLstm(bidirectionalLstmPath);

                String bidirectionalLstmNoSequencesPath =
                        "modelimport/keras/weights/bidirectional_lstm_no_return_sequences_"
                                + backend + "_" + version + ".h5";
                importBidirectionalLstm(bidirectionalLstmNoSequencesPath);

                if (version == 2 && backend.equals("tensorflow")) {
                    String batchToConv2dPath = "modelimport/keras/weights/batch_to_conv2d_"
                            + backend + "_" + version + ".h5";
                    importBatchNormToConv2D(batchToConv2dPath);
                }

                if (backend.equals("tensorflow") && version == 2) { // TODO should work for theano
                    String simpleSpaceToBatchPath = "modelimport/keras/weights/space_to_depth_simple_"
                            + backend + "_" + version + ".h5";
                    importSimpleSpaceToDepth(simpleSpaceToBatchPath);
                }

                if (backend.equals("tensorflow") && version == 2) {
                    String graphSpaceToBatchPath = "modelimport/keras/weights/space_to_depth_graph_"
                            + backend + "_" + version + ".h5";
                    importGraphSpaceToDepth(graphSpaceToBatchPath);
                }

                if (backend.equals("tensorflow") && version == 2) {
                    String sepConvPath = "modelimport/keras/weights/sepconv2d_" + backend + "_" + version + ".h5";
                    importSepConv2D(sepConvPath);
                }
            }
        }
    }

    private void logSuccess(String modelPath) {
        log.info("***** Successfully imported " + modelPath);
    }

    private void importDense(String modelPath) throws Exception {
        MultiLayerNetwork model = loadMultiLayerNetwork(modelPath, true);

        INDArray weights = model.getLayer(0).getParam("W");
        val weightShape = weights.shape();
        assertEquals(4, weightShape[0]);
        assertEquals(6, weightShape[1]);

        INDArray bias = model.getLayer(0).getParam("b");
        assertEquals(6, bias.length());
        logSuccess(modelPath);
    }

    private void importSepConv2D(String modelPath) throws Exception {
        MultiLayerNetwork model = loadMultiLayerNetwork(modelPath, false);

        INDArray depthWeights = model.getLayer(0).getParam("W");
        val depthWeightShape = depthWeights.shape();

        long depthMult = 2;
        long kernel = 3;
        long nIn = 5;
        long nOut = 6;

        assertEquals(depthMult, depthWeightShape[0]);
        assertEquals(nIn, depthWeightShape[1]);
        assertEquals(kernel, depthWeightShape[2]);
        assertEquals(kernel, depthWeightShape[3]);

        INDArray weights = model.getLayer(0).getParam("pW");
        val weightShape = weights.shape();


        assertEquals(nOut, weightShape[0]);
        assertEquals(nIn * depthMult, weightShape[1]);
        assertEquals(1, weightShape[2]);
        assertEquals(1, weightShape[3]);

        INDArray bias = model.getLayer(0).getParam("b");
        assertEquals(6, bias.length());

        INDArray input = Nd4j.ones(1, 5, 3, 4);
        INDArray output = model.output(input);

        assertArrayEquals(new long[] {1, 6, 1, 2}, output.shape());

        logSuccess(modelPath);
    }

    private void importConv2D(String modelPath) throws Exception {
        MultiLayerNetwork model = loadMultiLayerNetwork(modelPath, false);

        INDArray weights = model.getLayer(0).getParam("W");
        val weightShape = weights.shape();
        assertEquals(6, weightShape[0]);
        assertEquals(5, weightShape[1]);
        assertEquals(3, weightShape[2]);
        assertEquals(3, weightShape[3]);

        INDArray bias = model.getLayer(0).getParam("b");
        assertEquals(6,bias.length());
        logSuccess(modelPath);
    }


    private void importConv2DReshape(String modelPath) throws Exception {
        MultiLayerNetwork model = loadMultiLayerNetwork(modelPath, false);


        int nOut = 12;
        int mb = 10;
        ;
        int[] inShape = new int[]{5, 5, 5};
        INDArray input = Nd4j.zeros(mb, inShape[0], inShape[1], inShape[2]);
        INDArray output = model.output(input);
        assertArrayEquals(new long[]{mb, nOut}, output.shape());
        logSuccess(modelPath);
    }

    private void importConv1DFlatten(String modelPath) throws Exception {
        MultiLayerNetwork model = loadMultiLayerNetwork(modelPath, false);

        int nOut = 6;
        int inputLength = 10;
        int mb = 42;
        int kernel = 3;

        INDArray input = Nd4j.zeros(mb, inputLength);
        INDArray output = model.output(input);
        assertArrayEquals(new long[]{mb, nOut, inputLength - kernel + 1}, output.shape());
        logSuccess(modelPath);
    }

    private void importBatchNormToConv2D(String modelPath) throws Exception {
        MultiLayerNetwork model = loadMultiLayerNetwork(modelPath, false);
        model.summary();
        logSuccess(modelPath);
    }

    private void importSimpleSpaceToDepth(String modelPath) throws Exception {
        KerasLayer.registerCustomLayer("Lambda", KerasSpaceToDepth.class);
        MultiLayerNetwork model = loadMultiLayerNetwork(modelPath, false);

        INDArray input = Nd4j.zeros(10, 4, 6, 6);
        INDArray output = model.output(input);
        assertArrayEquals(new long[]{10, 16, 3, 3}, output.shape());
        logSuccess(modelPath);
    }

    private void importGraphSpaceToDepth(String modelPath) throws Exception {
        KerasLayer.registerCustomLayer("Lambda", KerasSpaceToDepth.class);
        ComputationGraph model = loadComputationalGraph(modelPath, false);

        INDArray input[] = new INDArray[]{Nd4j.zeros(10, 4, 6, 6), Nd4j.zeros(10, 16, 3, 3)};
        INDArray[] output = model.output(input);
        log.info(Arrays.toString(output[0].shape()));
        assertArrayEquals(new long[]{10, 32, 3, 3}, output[0].shape());
        logSuccess(modelPath);
    }

    private void importLstm(String modelPath) throws Exception {
        MultiLayerNetwork model = loadMultiLayerNetwork(modelPath, false);
        model.summary();
        // TODO: check weights
        logSuccess(modelPath);
    }

    private void importEmbeddingLstm(String modelPath) throws Exception {
        MultiLayerNetwork model = loadMultiLayerNetwork(modelPath, false);

        int nIn = 4;
        int nOut = 6;
        int outputDim = 5;
        int inputLength = 10;
        int mb = 42;

        INDArray embeddingWeight = model.getLayer(0).getParam("W");
        val embeddingWeightShape = embeddingWeight.shape();
        assertEquals(nIn, embeddingWeightShape[0]);
        assertEquals(outputDim, embeddingWeightShape[1]);

        INDArray inEmbedding = Nd4j.zeros(mb, inputLength);
        INDArray output = model.output(inEmbedding);
        assertArrayEquals(new long[]{mb, nOut, inputLength}, output.shape());
        logSuccess(modelPath);
    }

    private void importEmbeddingConv1DExtended(String modelPath) throws Exception {
        MultiLayerNetwork model = loadMultiLayerNetwork(modelPath, false);
        logSuccess(modelPath);
    }

    private void importEmbeddingConv1D(String modelPath) throws Exception {
        MultiLayerNetwork model = loadMultiLayerNetwork(modelPath, false);

        int nIn = 4;
        int nOut = 6;
        int outputDim = 5;
        int inputLength = 10;
        int kernel = 3;
        int mb = 42;

        INDArray embeddingWeight = model.getLayer(0).getParam("W");
        val embeddingWeightShape = embeddingWeight.shape();
        assertEquals(nIn, embeddingWeightShape[0]);
        assertEquals(outputDim, embeddingWeightShape[1]);

        INDArray inEmbedding = Nd4j.zeros(mb, inputLength);
        INDArray output = model.output(inEmbedding);
        assertArrayEquals(new long[]{mb, nOut, inputLength - kernel + 1}, output.shape());
        logSuccess(modelPath);
    }

    private void importSimpleRnn(String modelPath) throws Exception {
        MultiLayerNetwork model = loadMultiLayerNetwork(modelPath, false);
        model.summary();
        logSuccess(modelPath);
        // TODO: check weights
    }

    private void importBidirectionalLstm(String modelPath) throws Exception {
        MultiLayerNetwork model = loadMultiLayerNetwork(modelPath, false);
        model.summary();
        logSuccess(modelPath);
        // TODO: check weights
    }

    private MultiLayerNetwork loadMultiLayerNetwork(String modelPath, boolean training) throws Exception {
        ClassPathResource modelResource = new ClassPathResource(modelPath,
                KerasWeightSettingTests.class.getClassLoader());
        File modelFile = createTempFile("temp", ".h5");
        Files.copy(modelResource.getInputStream(), modelFile.toPath(), StandardCopyOption.REPLACE_EXISTING);
        return new KerasModel().modelBuilder().modelHdf5Filename(modelFile.getAbsolutePath())
                .enforceTrainingConfig(training).buildSequential().getMultiLayerNetwork();
    }

    private ComputationGraph loadComputationalGraph(String modelPath, boolean training) throws Exception {
        ClassPathResource modelResource = new ClassPathResource(modelPath,
                KerasWeightSettingTests.class.getClassLoader());
        File modelFile = createTempFile("temp", ".h5");
        Files.copy(modelResource.getInputStream(), modelFile.toPath(), StandardCopyOption.REPLACE_EXISTING);
        return new KerasModel().modelBuilder().modelHdf5Filename(modelFile.getAbsolutePath())
                .enforceTrainingConfig(training).buildModel().getComputationGraph();
    }

    private File createTempFile(String prefix, String suffix) throws IOException {
        return testDir.newFile(prefix + "-" + System.nanoTime() + suffix);
    }

}
