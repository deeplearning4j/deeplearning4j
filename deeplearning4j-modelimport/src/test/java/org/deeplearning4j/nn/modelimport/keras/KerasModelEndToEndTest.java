package org.deeplearning4j.nn.modelimport.keras;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.eval.ROCMultiClass;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.io.ClassPathResource;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;
import java.util.*;

import static org.junit.Assert.*;

/**
 * Unit tests for end-to-end Keras model import.
 *
 * TODO: find a more elegant fix for the per-layer activation comparisons
 * since some layers (e.g., ActivationLayer) overwrite previous
 * layer's activations when run in train=false mode.
 *
 * TODO: ndarray.equalsWithEps(ndarray) appears to be broken
 *
 * @author dave@skymind.io
 */
@Slf4j
public class KerasModelEndToEndTest {
    public static final String GROUP_ATTR_INPUTS = "inputs";
    public static final String GROUP_ATTR_OUTPUTS = "outputs";
    public static final String GROUP_PREDICTIONS = "predictions";
    public static final String GROUP_ACTIVATIONS = "activations";
    public static final String TEMP_OUTPUTS_FILENAME = "tempOutputs";
    public static final String TEMP_MODEL_FILENAME = "tempModel";
    public static final String H5_EXTENSION = ".h5";

    public static final double EPS = 1E-6;

    @Test
    public void importMnistMlpTensorFlowEndToEndModelTest() throws Exception {
        ClassPathResource modelResource =
                        new ClassPathResource("modelimport/keras/examples/mnist_mlp/mnist_mlp_tf_model.h5",
                                        KerasModelEndToEndTest.class.getClassLoader());
        File modelFile = File.createTempFile(TEMP_MODEL_FILENAME, H5_EXTENSION);
        Files.copy(modelResource.getInputStream(), modelFile.toPath(), StandardCopyOption.REPLACE_EXISTING);
        MultiLayerNetwork model = new KerasModel.ModelBuilder().modelHdf5Filename(modelFile.getAbsolutePath())
                        .enforceTrainingConfig(false).buildSequential().getMultiLayerNetwork();

        ClassPathResource outputsResource =
                        new ClassPathResource("modelimport/keras/examples/mnist_mlp/mnist_mlp_tf_inputs_and_outputs.h5",
                                        KerasModelEndToEndTest.class.getClassLoader());
        File outputsFile = File.createTempFile(TEMP_OUTPUTS_FILENAME, H5_EXTENSION);
        Files.copy(outputsResource.getInputStream(), outputsFile.toPath(), StandardCopyOption.REPLACE_EXISTING);
        Hdf5Archive outputsArchive = new Hdf5Archive(outputsFile.getAbsolutePath());

        INDArray input = getInputs(outputsArchive, true)[0];
        Map<String, INDArray> activationsKeras = getActivations(outputsArchive, true);
        for (int i = 0; i < model.getLayers().length; i++) {
            String layerName = model.getLayerNames().get(i);
            if (activationsKeras.containsKey(layerName)) {
                INDArray activationsDl4j = model.feedForwardToLayer(i, input, false).get(i + 1);
                /* TODO: investigate why this fails for some layers:
                 *
                 * compareINDArrays(layerName, activationsKeras.get(layerName), activationsDl4j, EPS);
                 *
                 */
            }
        }

        INDArray predictionsKeras = getPredictions(outputsArchive, true)[0];
        INDArray predictionsDl4j = model.output(input, false);
        /* TODO: investigate why this fails when max difference is ~1E-7!
         *
         * compareINDArrays("predictions", predictionsKeras, predictionsDl4j, EPS);
         *
         */
        INDArray outputs = getOutputs(outputsArchive, true)[0];
        compareMulticlassAUC("predictions", outputs, predictionsKeras, predictionsDl4j, 10, EPS);
    }

    static public INDArray[] getInputs(Hdf5Archive archive, boolean tensorFlowImageDimOrdering) throws Exception {
        List<String> inputNames = (List<String>) KerasModel
                        .parseJsonString(archive.readAttributeAsJson(GROUP_ATTR_INPUTS)).get(GROUP_ATTR_INPUTS);
        INDArray[] inputs = new INDArray[inputNames.size()];
        for (int i = 0; i < inputNames.size(); i++) {
            inputs[i] = archive.readDataSet(inputNames.get(i), GROUP_ATTR_INPUTS);
            if (inputs[i].shape().length == 4 && tensorFlowImageDimOrdering)
                inputs[i] = inputs[i].permute(0, 3, 1, 2);
        }
        return inputs;
    }

    static public Map<String, INDArray> getActivations(Hdf5Archive archive, boolean tensorFlowImageDimOrdering)
                    throws Exception {
        Map<String, INDArray> activations = new HashMap<String, INDArray>();
        for (String layerName : archive.getDataSets(GROUP_ACTIVATIONS)) {
            INDArray activation = archive.readDataSet(layerName, GROUP_ACTIVATIONS);
            if (activation.shape().length == 4 && tensorFlowImageDimOrdering)
                activation = activation.permute(0, 3, 1, 2);
            activations.put(layerName, activation);
        }
        return activations;
    }

    static public INDArray[] getOutputs(Hdf5Archive archive, boolean tensorFlowImageDimOrdering) throws Exception {
        List<String> outputNames = (List<String>) KerasModel
                        .parseJsonString(archive.readAttributeAsJson(GROUP_ATTR_OUTPUTS)).get(GROUP_ATTR_OUTPUTS);
        INDArray[] outputs = new INDArray[outputNames.size()];
        for (int i = 0; i < outputNames.size(); i++) {
            outputs[i] = archive.readDataSet(outputNames.get(i), GROUP_ATTR_OUTPUTS);
            if (outputs[i].shape().length == 4 && tensorFlowImageDimOrdering)
                outputs[i] = outputs[i].permute(0, 3, 1, 2);
        }
        return outputs;
    }

    static public INDArray[] getPredictions(Hdf5Archive archive, boolean tensorFlowImageDimOrdering) throws Exception {
        List<String> outputNames = (List<String>) KerasModel
                        .parseJsonString(archive.readAttributeAsJson(GROUP_ATTR_OUTPUTS)).get(GROUP_ATTR_OUTPUTS);
        INDArray[] predictions = new INDArray[outputNames.size()];
        for (int i = 0; i < outputNames.size(); i++) {
            predictions[i] = archive.readDataSet(outputNames.get(i), GROUP_PREDICTIONS);
            if (predictions[i].shape().length == 4 && tensorFlowImageDimOrdering)
                predictions[i] = predictions[i].permute(0, 3, 1, 2);
        }
        return predictions;
    }

    static public void compareINDArrays(String label, INDArray a, INDArray b, double eps) {
        INDArray diff = a.sub(b);
        double min = diff.minNumber().doubleValue();
        double max = diff.maxNumber().doubleValue();
        log.info(label + ": " + a.equalsWithEps(b, eps) + ", " + min + ", " + max);
        assert (a.equalsWithEps(b, eps));
    }

    static public void compareMulticlassAUC(String label, INDArray target, INDArray a, INDArray b, int nbClasses,
                    double eps) {
        ROCMultiClass evalA = new ROCMultiClass(100);
        evalA.eval(target, a);
        double avgAucA = evalA.calculateAverageAUC();
        ROCMultiClass evalB = new ROCMultiClass(100);
        evalB.eval(target, b);
        double avgAucB = evalB.calculateAverageAUC();
        assertEquals(avgAucA, avgAucB, EPS);

        double[] aucA = new double[nbClasses];
        double[] aucB = new double[nbClasses];
        for (int i = 0; i < nbClasses; i++) {
            aucA[i] = evalA.calculateAUC(i);
            aucB[i] = evalB.calculateAUC(i);
        }
        assertArrayEquals(aucA, aucB, EPS);
    }
}
