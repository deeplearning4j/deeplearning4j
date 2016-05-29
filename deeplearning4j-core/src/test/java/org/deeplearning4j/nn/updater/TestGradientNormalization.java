package org.deeplearning4j.nn.updater;

import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.Updater;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.factory.LayerFactories;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertTrue;

public class TestGradientNormalization {

    @Test
    public void testRenormalizatonPerLayer(){
        Nd4j.getRandom().setSeed(12345);

        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                .layer(new DenseLayer.Builder().nIn(10).nOut(20)
                        .updater(org.deeplearning4j.nn.conf.Updater.NONE)
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                .build()).build();

        int numParams = LayerFactories.getFactory(conf).initializer().numParams(conf,true);
        INDArray params = Nd4j.create(1, numParams);
        Layer layer =  LayerFactories.getFactory(conf).create(conf, null, 0, params);
        Updater updater = UpdaterCreator.getUpdater(layer);
        INDArray weightGrad = Nd4j.rand(10, 20);
        INDArray biasGrad = Nd4j.rand(1, 10);
        INDArray weightGradCopy = weightGrad.dup();
        INDArray biasGradCopy = biasGrad.dup();
        Gradient gradient = new DefaultGradient();
        gradient.setGradientFor(DefaultParamInitializer.WEIGHT_KEY, weightGrad);
        gradient.setGradientFor(DefaultParamInitializer.BIAS_KEY, biasGrad);

        updater.update(layer, gradient, 0, 1);

        assertNotEquals(weightGradCopy, weightGrad);
        assertNotEquals(biasGradCopy, biasGrad);

        double sumSquaresWeight = weightGradCopy.mul(weightGradCopy).sumNumber().doubleValue();
        double sumSquaresBias = biasGradCopy.mul(biasGradCopy).sumNumber().doubleValue();
        double sumSquares = sumSquaresWeight + sumSquaresBias;
        double l2Layer = Math.sqrt(sumSquares);

        INDArray normWeightsExpected = weightGradCopy.div(l2Layer);
        INDArray normBiasExpected = biasGradCopy.div(l2Layer);

        double l2Weight = gradient.getGradientFor(DefaultParamInitializer.WEIGHT_KEY).norm2Number().doubleValue();
        double l2Bias = gradient.getGradientFor(DefaultParamInitializer.BIAS_KEY).norm2Number().doubleValue();
        assertTrue(!Double.isNaN(l2Weight) && l2Weight > 0.0 );
        assertTrue(!Double.isNaN(l2Bias) && l2Bias > 0.0 );
        assertEquals(normWeightsExpected, gradient.getGradientFor(DefaultParamInitializer.WEIGHT_KEY));
        assertEquals(normBiasExpected, gradient.getGradientFor(DefaultParamInitializer.BIAS_KEY));
    }

    @Test
    public void testRenormalizationPerParamType(){
        Nd4j.getRandom().setSeed(12345);

        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                .layer(new DenseLayer.Builder().nIn(10).nOut(20)
                        .updater(org.deeplearning4j.nn.conf.Updater.NONE)
                        .gradientNormalization(GradientNormalization.RenormalizeL2PerParamType)
                        .build()).build();

        int numParams = LayerFactories.getFactory(conf).initializer().numParams(conf,true);
        INDArray params = Nd4j.create(1, numParams);
        Layer layer =  LayerFactories.getFactory(conf).create(conf, null, 0, params);
        Updater updater = UpdaterCreator.getUpdater(layer);
        INDArray weightGrad = Nd4j.rand(10, 20);
        INDArray biasGrad = Nd4j.rand(1, 10);
        INDArray weightGradCopy = weightGrad.dup();
        INDArray biasGradCopy = biasGrad.dup();
        Gradient gradient = new DefaultGradient();
        gradient.setGradientFor(DefaultParamInitializer.WEIGHT_KEY, weightGrad);
        gradient.setGradientFor(DefaultParamInitializer.BIAS_KEY, biasGrad);

        updater.update(layer, gradient, 0, 1);

        INDArray normWeightsExpected = weightGradCopy.div(weightGradCopy.norm2Number());
        INDArray normBiasExpected = biasGradCopy.div(biasGradCopy.norm2Number());

        assertEquals(normWeightsExpected, gradient.getGradientFor(DefaultParamInitializer.WEIGHT_KEY));
        assertEquals(normBiasExpected, gradient.getGradientFor(DefaultParamInitializer.BIAS_KEY));
    }

    @Test
    public void testAbsValueClippingPerElement(){
        Nd4j.getRandom().setSeed(12345);
        double threshold = 3;

        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                .layer(new DenseLayer.Builder().nIn(10).nOut(20)
                        .updater(org.deeplearning4j.nn.conf.Updater.NONE)
                        .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                        .gradientNormalizationThreshold(threshold)
                        .build()).build();

        int numParams = LayerFactories.getFactory(conf).initializer().numParams(conf,true);
        INDArray params = Nd4j.create(1, numParams);
        Layer layer =  LayerFactories.getFactory(conf).create(conf, null, 0, params);
        Updater updater = UpdaterCreator.getUpdater(layer);
        INDArray weightGrad = Nd4j.rand(10, 20).muli(10).subi(5);
        INDArray biasGrad = Nd4j.rand(1, 10).muli(10).subi(5);
        INDArray weightGradCopy = weightGrad.dup();
        INDArray biasGradCopy = biasGrad.dup();
        Gradient gradient = new DefaultGradient();
        gradient.setGradientFor(DefaultParamInitializer.WEIGHT_KEY, weightGrad);
        gradient.setGradientFor(DefaultParamInitializer.BIAS_KEY, biasGrad);

        updater.update(layer, gradient, 0, 1);

        assertNotEquals(weightGradCopy, weightGrad);
        assertNotEquals(biasGradCopy, biasGrad);

        INDArray expectedWeightGrad = weightGradCopy.dup();
        for( int i=0; i<expectedWeightGrad.length(); i++ ){
            double d = expectedWeightGrad.getDouble(i);
            if(d>threshold) expectedWeightGrad.putScalar(i,threshold);
            else if(d<-threshold) expectedWeightGrad.putScalar(i,-threshold);
        }
        INDArray expectedBiasGrad = biasGradCopy.dup();
        for( int i=0; i<expectedBiasGrad.length(); i++ ){
            double d = expectedBiasGrad.getDouble(i);
            if(d>threshold) expectedBiasGrad.putScalar(i,threshold);
            else if(d<-threshold) expectedBiasGrad.putScalar(i,-threshold);
        }

        assertEquals(expectedWeightGrad,gradient.getGradientFor(DefaultParamInitializer.WEIGHT_KEY));
        assertEquals(expectedBiasGrad,gradient.getGradientFor(DefaultParamInitializer.BIAS_KEY));
    }

    @Test
    public void testL2ClippingPerLayer(){
        Nd4j.getRandom().setSeed(12345);
        double threshold = 3;

        for( int t=0; t<2; t++ ) {
            //t=0: small -> no clipping
            //t=1: large -> clipping

            NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                    .layer(new DenseLayer.Builder().nIn(10).nOut(20)
                            .updater(org.deeplearning4j.nn.conf.Updater.NONE)
                            .gradientNormalization(GradientNormalization.ClipL2PerLayer)
                            .gradientNormalizationThreshold(threshold)
                            .build()).build();

            int numParams = LayerFactories.getFactory(conf).initializer().numParams(conf,true);
            INDArray params = Nd4j.create(1, numParams);
            Layer layer =  LayerFactories.getFactory(conf).create(conf, null, 0, params);
            Updater updater = UpdaterCreator.getUpdater(layer);
            INDArray weightGrad = Nd4j.rand(10, 20).muli((t==0 ? 0.05 : 10));
            INDArray biasGrad = Nd4j.rand(1, 10).muli((t==0 ? 0.05 : 10));
            INDArray weightGradCopy = weightGrad.dup();
            INDArray biasGradCopy = biasGrad.dup();
            Gradient gradient = new DefaultGradient();
            gradient.setGradientFor(DefaultParamInitializer.WEIGHT_KEY, weightGrad);
            gradient.setGradientFor(DefaultParamInitializer.BIAS_KEY, biasGrad);

            double layerGradL2 = gradient.gradient().norm2Number().doubleValue();
            if(t==0) assertTrue(layerGradL2 < threshold);
            else assertTrue(layerGradL2 > threshold);

            updater.update(layer, gradient, 0, 1);

            if(t==0) {
                //norm2 < threshold -> no change
                assertEquals(weightGradCopy, weightGrad);
                assertEquals(biasGradCopy, biasGrad);
                continue;
            } else {
                //norm2 > threshold -> rescale
                assertNotEquals(weightGradCopy, weightGrad);
                assertNotEquals(biasGradCopy, biasGrad);
            }

            //for above threshold only...
            double scalingFactor = threshold / layerGradL2;
            INDArray expectedWeightGrad = weightGradCopy.mul(scalingFactor);
            INDArray expectedBiasGrad = biasGradCopy.mul(scalingFactor);
            assertEquals(expectedWeightGrad, gradient.getGradientFor(DefaultParamInitializer.WEIGHT_KEY));
            assertEquals(expectedBiasGrad, gradient.getGradientFor(DefaultParamInitializer.BIAS_KEY));
        }
    }

    @Test
    public void testL2ClippingPerParamType(){
        Nd4j.getRandom().setSeed(12345);
        double threshold = 3;

        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                .layer(new DenseLayer.Builder().nIn(10).nOut(20)
                        .updater(org.deeplearning4j.nn.conf.Updater.NONE)
                        .gradientNormalization(GradientNormalization.ClipL2PerParamType)
                        .gradientNormalizationThreshold(threshold)
                        .build()).build();

        int numParams = LayerFactories.getFactory(conf).initializer().numParams(conf,true);
        INDArray params = Nd4j.create(1, numParams);
        Layer layer =  LayerFactories.getFactory(conf).create(conf, null, 0, params);
        Updater updater = UpdaterCreator.getUpdater(layer);
        INDArray weightGrad = Nd4j.rand(10, 20).muli(0.05);
        INDArray biasGrad = Nd4j.rand(1, 10).muli(10);
        INDArray weightGradCopy = weightGrad.dup();
        INDArray biasGradCopy = biasGrad.dup();
        Gradient gradient = new DefaultGradient();
        gradient.setGradientFor(DefaultParamInitializer.WEIGHT_KEY, weightGrad);
        gradient.setGradientFor(DefaultParamInitializer.BIAS_KEY, biasGrad);

        double weightL2 = weightGrad.norm2Number().doubleValue();
        double biasL2 = biasGrad.norm2Number().doubleValue();
        assertTrue(weightL2 < threshold);
        assertTrue(biasL2 > threshold);

        updater.update(layer, gradient, 0, 1);

        assertEquals(weightGradCopy, weightGrad);   //weight norm2 < threshold -> no change
        assertNotEquals(biasGradCopy, biasGrad);    //bias norm2 > threshold -> rescale


        double biasScalingFactor = threshold / biasL2;
        INDArray expectedBiasGrad = biasGradCopy.mul(biasScalingFactor);
        assertEquals(expectedBiasGrad, gradient.getGradientFor(DefaultParamInitializer.BIAS_KEY));
    }
}
