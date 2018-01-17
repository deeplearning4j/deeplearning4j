package org.deeplearning4j.nn.modelimport.keras.weights;

import org.deeplearning4j.nn.modelimport.keras.KerasModel;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.io.ClassPathResource;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;

public class KerasWeightSettingTests {

    @Test
    public void testSimpleLayersWithWeights() throws Exception {
        int[] kerasVersions = new int[]{1, 2};
        String[] backends = new String[]{"tensorflow", "theano"};

        for (int version : kerasVersions) {
            for (String backend : backends) {
                String densePath = "weights/dense_" + backend + "_" + version + ".h5";
                importDense(densePath);
                System.out.println("***** Successfully imported " + densePath);

                String conv2dPath = "weights/conv2d_" + backend + "_" + version + ".h5";
                importConv2D(conv2dPath);
                System.out.println("***** Successfully imported " + conv2dPath);

                String lstmPath = "weights/lstm_" + backend + "_" + version + ".h5";
                importLstm(lstmPath);
                System.out.println("***** Successfully imported " + lstmPath);
            }
        }
    }

    private static void importDense(String modelPath) throws Exception {
        MultiLayerNetwork model = loadMultiLayerNetwork(modelPath);

        INDArray weights = model.getLayer(0).getParam("W");
        int[] weightShape = weights.shape();
        assert (weightShape[0] == 4);
        assert (weightShape[1] == 6);

        INDArray bias = model.getLayer(0).getParam("b");
        assert (bias.length() == 6);
    }

    private static void importConv2D(String modelPath) throws Exception {
        MultiLayerNetwork model = loadMultiLayerNetwork(modelPath);

        INDArray weights = model.getLayer(0).getParam("W");
        int[] weightShape = weights.shape();
        assert (weightShape[0] == 6);
        assert (weightShape[1] == 5);
        assert (weightShape[2] == 3);
        assert (weightShape[3] == 3);

        INDArray bias = model.getLayer(0).getParam("b");
        assert (bias.length() == 6);
    }

    private static void importLstm(String modelPath) throws Exception {
        MultiLayerNetwork model = loadMultiLayerNetwork(modelPath);
        // TODO: check weights
    }

        private static MultiLayerNetwork loadMultiLayerNetwork(String modelPath) throws Exception {
        ClassPathResource modelResource = new ClassPathResource(modelPath,
                KerasWeightSettingTests.class.getClassLoader());
        File modelFile = File.createTempFile("temp", ".h5");
        Files.copy(modelResource.getInputStream(), modelFile.toPath(), StandardCopyOption.REPLACE_EXISTING);
        return new KerasModel().modelBuilder().modelHdf5Filename(modelFile.getAbsolutePath())
                .enforceTrainingConfig(false).buildSequential().getMultiLayerNetwork();
    }

}
