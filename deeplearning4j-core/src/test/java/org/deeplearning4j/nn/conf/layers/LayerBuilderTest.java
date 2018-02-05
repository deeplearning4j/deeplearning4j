package org.deeplearning4j.nn.conf.layers;

import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.Distribution;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.dropout.Dropout;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.activations.impl.ActivationSoftmax;
import org.nd4j.linalg.activations.impl.ActivationTanH;
import org.nd4j.linalg.convolution.Convolution;
import org.nd4j.linalg.learning.config.AdaGrad;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

import java.io.*;

import static org.junit.Assert.*;

/**
 * @author Jeffrey Tang.
 */
public class LayerBuilderTest extends BaseDL4JTest {
    final double DELTA = 1e-15;

    int numIn = 10;
    int numOut = 5;
    double drop = 0.3;
    IActivation act = new ActivationSoftmax();
    PoolingType poolType = PoolingType.MAX;
    int[] kernelSize = new int[] {2, 2};
    int[] stride = new int[] {2, 2};
    int[] padding = new int[] {1, 1};
    int k = 1;
    Convolution.Type convType = Convolution.Type.VALID;
    LossFunction loss = LossFunction.MCXENT;
    WeightInit weight = WeightInit.XAVIER;
    double corrupt = 0.4;
    double sparsity = 0.3;
    double corruptionLevel = 0.5;
    Distribution dist = new NormalDistribution(1.0, 0.1);
    double dropOut = 0.1;
    IUpdater updater = new AdaGrad();
    GradientNormalization gradNorm = GradientNormalization.ClipL2PerParamType;
    double gradNormThreshold = 8;

    @Test
    public void testLayer() throws Exception {
        DenseLayer layer = new DenseLayer.Builder().activation(act).weightInit(weight).dist(dist).dropOut(dropOut)
                        .updater(updater).gradientNormalization(gradNorm)
                        .gradientNormalizationThreshold(gradNormThreshold).build();

        checkSerialization(layer);

        assertEquals(act, layer.getActivationFn());
        assertEquals(weight, layer.getWeightInit());
        assertEquals(dist, layer.getDist());
        assertEquals(new Dropout(dropOut), layer.getIDropout());
        assertEquals(updater, layer.getIUpdater());
        assertEquals(gradNorm, layer.getGradientNormalization());
        assertEquals(gradNormThreshold, layer.getGradientNormalizationThreshold(), 0.0);
    }

    @Test
    public void testFeedForwardLayer() throws Exception {
        DenseLayer ff = new DenseLayer.Builder().nIn(numIn).nOut(numOut).build();

        checkSerialization(ff);

        assertEquals(numIn, ff.getNIn());
        assertEquals(numOut, ff.getNOut());
    }

    @Test
    public void testConvolutionLayer() throws Exception {
        ConvolutionLayer conv = new ConvolutionLayer.Builder(kernelSize, stride, padding).build();

        checkSerialization(conv);

        //        assertEquals(convType, conv.getConvolutionType());
        assertArrayEquals(kernelSize, conv.getKernelSize());
        assertArrayEquals(stride, conv.getStride());
        assertArrayEquals(padding, conv.getPadding());
    }

    @Test
    public void testSubsamplingLayer() throws Exception {
        SubsamplingLayer sample =
                        new SubsamplingLayer.Builder(poolType, stride).kernelSize(kernelSize).padding(padding).build();

        checkSerialization(sample);

        assertArrayEquals(padding, sample.getPadding());
        assertArrayEquals(kernelSize, sample.getKernelSize());
        assertEquals(poolType, sample.getPoolingType());
        assertArrayEquals(stride, sample.getStride());
    }

    @Test
    public void testOutputLayer() throws Exception {
        OutputLayer out = new OutputLayer.Builder(loss).build();

        checkSerialization(out);
    }

    @Test
    public void testRnnOutputLayer() throws Exception {
        RnnOutputLayer out = new RnnOutputLayer.Builder(loss).build();

        checkSerialization(out);
    }

    @Test
    public void testAutoEncoder() throws Exception {
        AutoEncoder enc = new AutoEncoder.Builder().corruptionLevel(corruptionLevel).sparsity(sparsity).build();

        checkSerialization(enc);

        assertEquals(corruptionLevel, enc.getCorruptionLevel(), DELTA);
        assertEquals(sparsity, enc.getSparsity(), DELTA);
    }

    @Test
    public void testGravesLSTM() throws Exception {
        GravesLSTM glstm = new GravesLSTM.Builder().forgetGateBiasInit(1.5).activation(Activation.TANH).nIn(numIn)
                        .nOut(numOut).build();

        checkSerialization(glstm);

        assertEquals(glstm.getForgetGateBiasInit(), 1.5, 0.0);
        assertEquals(glstm.nIn, numIn);
        assertEquals(glstm.nOut, numOut);
        assertTrue(glstm.getActivationFn() instanceof ActivationTanH);
    }

    @Test
    public void testGravesBidirectionalLSTM() throws Exception {
        final GravesBidirectionalLSTM glstm = new GravesBidirectionalLSTM.Builder().forgetGateBiasInit(1.5)
                        .activation(Activation.TANH).nIn(numIn).nOut(numOut).build();

        checkSerialization(glstm);

        assertEquals(glstm.getForgetGateBiasInit(), 1.5, 0.0);
        assertEquals(glstm.nIn, numIn);
        assertEquals(glstm.nOut, numOut);
        assertTrue(glstm.getActivationFn() instanceof ActivationTanH);
    }

    @Test
    public void testEmbeddingLayer() throws Exception {
        EmbeddingLayer el = new EmbeddingLayer.Builder().nIn(10).nOut(5).build();
        checkSerialization(el);

        assertEquals(10, el.getNIn());
        assertEquals(5, el.getNOut());
    }

    @Test
    public void testBatchNormLayer() throws Exception {
        BatchNormalization bN = new BatchNormalization.Builder().nIn(numIn).nOut(numOut).gamma(2).beta(1).decay(0.5)
                        .lockGammaBeta(true).build();

        checkSerialization(bN);

        assertEquals(numIn, bN.nIn);
        assertEquals(numOut, bN.nOut);
        assertEquals(true, bN.isLockGammaBeta());
        assertEquals(0.5, bN.decay, 1e-4);
        assertEquals(2, bN.gamma, 1e-4);
        assertEquals(1, bN.beta, 1e-4);
    }

    @Test
    public void testActivationLayer() throws Exception {
        ActivationLayer activationLayer = new ActivationLayer.Builder().activation(act).build();

        checkSerialization(activationLayer);

        assertEquals(act, activationLayer.activationFn);
    }

    private void checkSerialization(Layer layer) throws Exception {
        NeuralNetConfiguration confExpected = new NeuralNetConfiguration.Builder().layer(layer).build();
        NeuralNetConfiguration confActual;

        // check Java serialization
        byte[] data;
        try (ByteArrayOutputStream bos = new ByteArrayOutputStream(); ObjectOutput out = new ObjectOutputStream(bos)) {
            out.writeObject(confExpected);
            data = bos.toByteArray();
        }
        try (ByteArrayInputStream bis = new ByteArrayInputStream(data); ObjectInput in = new ObjectInputStream(bis)) {
            confActual = (NeuralNetConfiguration) in.readObject();
        }
        assertEquals("unequal Java serialization", confExpected.getLayer(), confActual.getLayer());

        // check JSON
        String json = confExpected.toJson();
        confActual = NeuralNetConfiguration.fromJson(json);
        assertEquals("unequal JSON serialization", confExpected.getLayer(), confActual.getLayer());

        // check YAML
        String yaml = confExpected.toYaml();
        confActual = NeuralNetConfiguration.fromYaml(yaml);
        assertEquals("unequal YAML serialization", confExpected.getLayer(), confActual.getLayer());

        // check the layer's use of callSuper on equals method
        confActual.getLayer().setIDropout(new Dropout(new java.util.Random().nextDouble()));
        assertNotEquals("broken equals method (missing callSuper?)", confExpected.getLayer(), confActual.getLayer());
    }

}
