package org.deeplearning4j.nn.conf.layers;

import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.distribution.Distribution;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.RBM.*;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer.PoolingType;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Test;
import static org.junit.Assert.*;

import org.nd4j.linalg.convolution.Convolution;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

import java.io.*;

/**
 * @author Jeffrey Tang.
 */
public class LayerBuilderTest {
    final double DELTA = 1e-15;

    int numIn = 10;
    int numOut = 5;
    double drop = 0.3;
    String act = "softmax";
    PoolingType poolType = PoolingType.MAX;
    int[] kernelSize = new int[]{2, 2};
    int[] stride = new int[]{2, 2};
    int[] padding = new int[]{1,1};
    HiddenUnit hidden = HiddenUnit.RECTIFIED;
    VisibleUnit visible = VisibleUnit.GAUSSIAN;
    int k  = 1;
    Convolution.Type convType = Convolution.Type.VALID;
    LossFunction loss = LossFunction.MCXENT;
    WeightInit weight = WeightInit.XAVIER;
    double corrupt = 0.4;
    double sparsity = 0.3;
    double corruptionLevel = 0.5;
    Distribution dist = new NormalDistribution(1.0, 0.1);
    double dropOut = 0.1;
    Updater updater = Updater.ADAGRAD;
    GradientNormalization gradNorm = GradientNormalization.ClipL2PerParamType;
    double gradNormThreshold = 8;

    @Test
    public void testLayer() throws Exception {
        RecursiveAutoEncoder layer = new RecursiveAutoEncoder.Builder()
            .activation(act).weightInit(weight).dist(dist).dropOut(dropOut).updater(updater)
            .gradientNormalization(gradNorm).gradientNormalizationThreshold(gradNormThreshold)
            .build();

        checkSerialization(layer);

        assertEquals(act, layer.getActivationFunction());
        assertEquals(weight, layer.getWeightInit());
        assertEquals(dist, layer.getDist());
        assertEquals(dropOut, layer.getDropOut(), DELTA);
        assertEquals(updater, layer.getUpdater());
        assertEquals(gradNorm, layer.getGradientNormalization());
        assertEquals(gradNormThreshold, layer.getGradientNormalizationThreshold(), 0.0);
    }

    @Test
    public void testFeedForwardLayer() throws Exception {
        RecursiveAutoEncoder ff = new RecursiveAutoEncoder.Builder().nIn(numIn).nOut(numOut).build();

        checkSerialization(ff);

        assertEquals(numIn, ff.getNIn());
        assertEquals(numOut, ff.getNOut());
    }
    @Test
    public void testConvolutionLayer() throws Exception {
        ConvolutionLayer conv = new ConvolutionLayer.Builder(kernelSize, stride, padding)
                .convolutionType(convType).build();

        checkSerialization(conv);

        assertEquals(convType, conv.getConvolutionType());
        assertArrayEquals(kernelSize, conv.getKernelSize());
        assertArrayEquals(stride, conv.getStride());
        assertArrayEquals(padding, conv.getPadding());
    }

    @Test
    public void testRBM() throws Exception {
        RBM rbm = new RBM.Builder(hidden, visible).sparsity(sparsity).k(k).build();

        checkSerialization(rbm);

        assertEquals(hidden, rbm.getHiddenUnit());
        assertEquals(visible, rbm.getVisibleUnit());
        assertEquals(k, rbm.getK());
        assertEquals(sparsity, rbm.getSparsity(), DELTA);
    }

    @Test
    public void testSubsamplingLayer() throws Exception {
        SubsamplingLayer sample = new SubsamplingLayer.Builder(poolType, stride)
                .kernelSize(kernelSize)
                .padding(padding)
                .build();

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

        assertEquals(loss, out.getLossFunction());
    }
    
    @Test
    public void testRnnOutputLayer() throws Exception {
    	RnnOutputLayer out = new RnnOutputLayer.Builder(loss).build();
    	
    	checkSerialization(out);
    	
    	assertEquals(loss, out.getLossFunction());
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
    	GravesLSTM glstm = new GravesLSTM.Builder()
                .forgetGateBiasInit(1.5)
                .activation("tanh")
    			.nIn(numIn).nOut(numOut).build();
    	
    	checkSerialization(glstm);

        assertEquals(glstm.getForgetGateBiasInit(),1.5,0.0);
    	assertEquals(glstm.nIn, numIn);
    	assertEquals(glstm.nOut,numOut);
    	assertEquals(glstm.activationFunction,"tanh");
    }

    @Test
    public void testGravesBidirectionalLSTM() throws Exception {
        final GravesBidirectionalLSTM glstm = new GravesBidirectionalLSTM.Builder()
                .forgetGateBiasInit(1.5)
                .activation("tanh")
                .nIn(numIn).nOut(numOut).build();

        checkSerialization(glstm);

        assertEquals(glstm.getForgetGateBiasInit(),1.5,0.0);
        assertEquals(glstm.nIn,numIn);
        assertEquals(glstm.nOut,numOut);
        assertEquals(glstm.activationFunction,"tanh");
    }
    
    @Test
    public void testGRU() throws Exception {
    	GRU gru = new GRU.Builder().activation("tanh")
    			.nIn(numIn).nOut(numOut).build();
    	
    	checkSerialization(gru);
    	
    	assertEquals(gru.nIn,numIn);
    	assertEquals(gru.nOut,numOut);
    	assertEquals(gru.activationFunction,"tanh");
    }

    @Test
    public void testEmbeddingLayer() throws Exception {
        EmbeddingLayer el = new EmbeddingLayer.Builder().nIn(10).nOut(5).build();
        checkSerialization(el);

        assertEquals(10,el.getNIn());
        assertEquals(5,el.getNOut());
    }

    @Test
    public void testBatchNormLayer() throws Exception {
        BatchNormalization bN = new BatchNormalization.Builder()
                .nIn(numIn).nOut(numOut)
                .gamma(2).beta(1).decay(0.5).lockGammaBeta(true).build();

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
        ActivationLayer activationLayer = new ActivationLayer.Builder()
                .nIn(numIn).nOut(numOut).activation(act).build();

        checkSerialization(activationLayer);

        assertEquals(numIn, activationLayer.nIn);
        assertEquals(numOut, activationLayer.nOut);
        assertEquals(act, activationLayer.activationFunction);
    }

    private void checkSerialization(Layer layer) throws Exception {
        NeuralNetConfiguration confExpected = new NeuralNetConfiguration.Builder()
                .layer(layer)
                .build();
        NeuralNetConfiguration confActual;

        // check Java serialization
        byte[] data;
        try (ByteArrayOutputStream bos = new ByteArrayOutputStream();
             ObjectOutput out = new ObjectOutputStream(bos)) {
            out.writeObject(confExpected);
            data = bos.toByteArray();
        }
        try (ByteArrayInputStream bis = new ByteArrayInputStream(data);
             ObjectInput in = new ObjectInputStream(bis)) {
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
        confActual.getLayer().setDropOut(new java.util.Random().nextDouble());
        assertNotEquals("broken equals method (missing callSuper?)", confExpected.getLayer(), confActual.getLayer());
    }

}
