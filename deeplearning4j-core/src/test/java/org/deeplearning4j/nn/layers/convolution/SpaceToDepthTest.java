package org.deeplearning4j.nn.layers.convolution;

import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.nn.conf.layers.SpaceToDepthLayer;

import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;

import java.util.Arrays;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

public class SpaceToDepthTest extends BaseDL4JTest {

    private int mb = 1;
    private int inDepth = 2;
    private int inputWidth = 2;
    private int inputHeight = 2;

    private int blockSize = 2;
    private SpaceToDepthLayer.DataFormat dataFormat = SpaceToDepthLayer.DataFormat.NCHW;

    private int outDepth = inDepth * blockSize * blockSize;
    private int outputHeight = inputHeight / blockSize;
    private int outputWidth = inputWidth / blockSize;


    private INDArray getContainedData() {
        return Nd4j.create(new double[] {1., 2., 3., 4., 5., 6., 7., 8.},
                new int[] {mb, inDepth, inputHeight, inputWidth}, 'c');
    }

    private INDArray getContainedOutput() {
        return Nd4j.create(new double[] {1., 5., 2., 6., 3., 7., 4., 8.},
                new int[] {mb,  outDepth, outputHeight, outputWidth}, 'c');
    }

    private Layer getSpaceToDepthLayer() {
        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer).seed(123)
                .layer(new SpaceToDepthLayer.Builder(blockSize, dataFormat).build()).build();
        return conf.getLayer().instantiate(conf, null, 0, null, true);
    }

    @Test
    public void testSpaceToDepthForward() throws Exception {
        INDArray containedInput = getContainedData();
        INDArray containedExpectedOut = getContainedOutput();
        Layer std = getSpaceToDepthLayer();
        INDArray containedOutput = std.activate(containedInput, false, LayerWorkspaceMgr.noWorkspaces());

        assertTrue(Arrays.equals(containedExpectedOut.shape(), containedOutput.shape()));
        assertEquals(containedExpectedOut, containedOutput);
    }

    @Test
    public void testSpaceToDepthBackward() throws Exception {
        INDArray containedInputEpsilon = getContainedOutput();

        INDArray containedExpectedOut = getContainedData();
        Layer std = getSpaceToDepthLayer();

        std.setInput(getContainedData(), LayerWorkspaceMgr.noWorkspaces());
        INDArray containedOutput = std.backpropGradient(containedInputEpsilon, LayerWorkspaceMgr.noWorkspaces()).getRight();

        assertTrue(Arrays.equals(containedExpectedOut.shape(), containedOutput.shape()));
        assertEquals(containedExpectedOut, containedOutput);
    }
}