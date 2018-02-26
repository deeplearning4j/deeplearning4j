package org.deeplearning4j.nn.modelimport.keras.weights;

import org.deeplearning4j.nn.modelimport.keras.KerasModel;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;
import java.util.Arrays;
import java.util.List;

public class KerasWeightSettingTests {

    @Test
    public void testSimpleLayersWithWeights() throws Exception {
        int[] kerasVersions = new int[]{1, 2};
        String[] backends = new String[]{"tensorflow", "theano"};

        for (int version : kerasVersions) {
            for (String backend : backends) {
//                String densePath = "weights/dense_" + backend + "_" + version + ".h5";
//                importDense(densePath);
//                System.out.println("***** Successfully imported " + densePath);
//
//                String conv2dPath = "weights/conv2d_" + backend + "_" + version + ".h5";
//                importConv2D(conv2dPath);
//                System.out.println("***** Successfully imported " + conv2dPath);
//
//                String lstmPath = "weights/lstm_" + backend + "_" + version + ".h5";
//                importLstm(lstmPath);
//                System.out.println("***** Successfully imported " + lstmPath);
//
//                String embeddingLstmPath = "weights/embedding_lstm_" + backend + "_" + version + ".h5";
//                importEmbeddingLstm(embeddingLstmPath);
//                System.out.println("***** Successfully imported " + embeddingLstmPath);
//
//                if (version == 2) {
//                    String embeddingConv1dPath = "weights/embedding_conv1d_" + backend + "_" + version + ".h5";
//                    importEmbeddingConv1D(embeddingConv1dPath);
//                    System.out.println("***** Successfully imported " + embeddingConv1dPath);
//                }
//
//                String simpleRnnPath = "weights/simple_rnn_" + backend + "_" + version + ".h5";
//                importSimpleRnn(simpleRnnPath);
//                System.out.println("***** Successfully imported " + simpleRnnPath);
//
//                String bidirectionalLstmPath = "weights/bidirectional_lstm_" + backend + "_" + version + ".h5";
//                importBidirectionalLstm(bidirectionalLstmPath);
//                System.out.println("***** Successfully imported " + bidirectionalLstmPath);

                String batchToConv2dPath = "weights/batch_to_conv2d_" + backend + "_" + version + ".h5";
                importBatchNormToConv2D(batchToConv2dPath);
                System.out.println("***** Successfully imported " + batchToConv2dPath);
            }
        }
    }

    private static void importDense(String modelPath) throws Exception {
        MultiLayerNetwork model = loadMultiLayerNetwork(modelPath, true);

        INDArray weights = model.getLayer(0).getParam("W");
        int[] weightShape = weights.shape();
        assert (weightShape[0] == 4);
        assert (weightShape[1] == 6);

        INDArray bias = model.getLayer(0).getParam("b");
        assert (bias.length() == 6);
    }

    private static void importConv2D(String modelPath) throws Exception {
        MultiLayerNetwork model = loadMultiLayerNetwork(modelPath, false);

        INDArray weights = model.getLayer(0).getParam("W");
        int[] weightShape = weights.shape();
        assert (weightShape[0] == 6);
        assert (weightShape[1] == 5);
        assert (weightShape[2] == 3);
        assert (weightShape[3] == 3);

        INDArray bias = model.getLayer(0).getParam("b");
        assert (bias.length() == 6);
    }

    private static void importBatchNormToConv2D(String modelPath) throws Exception {
        MultiLayerNetwork model = loadMultiLayerNetwork(modelPath, false);
    }

    private static void importLstm(String modelPath) throws Exception {
        MultiLayerNetwork model = loadMultiLayerNetwork(modelPath, false);
        // TODO: check weights
    }

    private static void importEmbeddingLstm(String modelPath) throws Exception {
        MultiLayerNetwork model = loadMultiLayerNetwork(modelPath, false);

        int nIn = 4;
        int nOut = 6;
        int outputDim = 5;
        int inputLength = 10;
        int mb = 42;

        INDArray embeddingWeight = model.getLayer(0).getParam("W");
        int[] embeddingWeightShape = embeddingWeight.shape();
        assert (embeddingWeightShape[0] == nIn);
        assert (embeddingWeightShape[1] == outputDim);

        INDArray inEmbedding = Nd4j.zeros(mb, 1, inputLength);
        INDArray output = model.output(inEmbedding);
        assert Arrays.equals(output.shape(), new int[]{mb, nOut, inputLength});

    }

    private static void importEmbeddingConv1D(String modelPath) throws Exception {
        MultiLayerNetwork model = loadMultiLayerNetwork(modelPath, false);

        int nIn = 4;
        int nOut = 6;
        int outputDim = 5;
        int inputLength = 10;
        int kernel = 3;
        int mb = 42;

        INDArray embeddingWeight = model.getLayer(0).getParam("W");
        int[] embeddingWeightShape = embeddingWeight.shape();
        assert (embeddingWeightShape[0] == nIn);
        assert (embeddingWeightShape[1] == outputDim);

        INDArray inEmbedding = Nd4j.zeros(mb, 1, inputLength);
        INDArray output = model.output(inEmbedding);
        assert Arrays.equals(output.shape(), new int[]{mb, nOut, inputLength - kernel + 1});

    }

    private static void importSimpleRnn(String modelPath) throws Exception {
        MultiLayerNetwork model = loadMultiLayerNetwork(modelPath, false);
        // TODO: check weights
    }

    private static void importBidirectionalLstm(String modelPath) throws Exception {
        MultiLayerNetwork model = loadMultiLayerNetwork(modelPath, false);
        // TODO: check weights
    }

    private static MultiLayerNetwork loadMultiLayerNetwork(String modelPath, boolean training) throws Exception {
        ClassPathResource modelResource = new ClassPathResource(modelPath,
                KerasWeightSettingTests.class.getClassLoader());
        File modelFile = File.createTempFile("temp", ".h5");
        Files.copy(modelResource.getInputStream(), modelFile.toPath(), StandardCopyOption.REPLACE_EXISTING);
        return new KerasModel().modelBuilder().modelHdf5Filename(modelFile.getAbsolutePath())
                .enforceTrainingConfig(training).buildSequential().getMultiLayerNetwork();
    }

}
