/*
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */

package org.deeplearning4j.plot;

import org.canova.image.loader.ImageLoader;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.RBM;
import org.deeplearning4j.nn.layers.factory.LayerFactories;
import org.deeplearning4j.nn.params.PretrainParamInitializer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;

/**
 * Created by agibsoncccc on 11/19/15.
 */
public class ImageRenderTest {

    @Test
    public void testImageRender() throws Exception {
        DataSetIterator mnist = new MnistDataSetIterator(1,100);
        INDArray image = mnist.next().getFeatureMatrix().reshape(28,28);
        File tmp = new File(System.getProperty("java.io.tmpdir"),"render.png");
        ImageRender.render(image,tmp.getAbsolutePath());
        tmp.deleteOnExit();


        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.LBFGS).weightInit(WeightInit.RELU)
                .updater(Updater.ADAM).activation("sigmoid").iterations(10).regularization(true)
                .l2(1e-1).gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                .learningRate(1e-1).optimizationAlgo(OptimizationAlgorithm.CONJUGATE_GRADIENT)
                .layer(new org.deeplearning4j.nn.conf.layers.RBM.Builder(RBM.HiddenUnit.BINARY, RBM.VisibleUnit.BINARY)
                        .nIn(784).nOut(400)
                        .lossFunction(LossFunctions.LossFunction.RMSE_XENT).build())
                .build();


        int numParams = LayerFactories.getFactory(conf).initializer().numParams(conf,true);
        INDArray params = Nd4j.create(1, numParams);
        org.deeplearning4j.nn.layers.feedforward.rbm.RBM da = LayerFactories.getFactory(conf.getLayer()).create(conf, null, 0, params);
        da.setListeners(new ScoreIterationListener(1));
        mnist = new MnistDataSetIterator(1000,1000);
        da.fit(mnist.next().getFeatureMatrix());
        File autoEncoderWeights = new File(System.getProperty("java.io.tmpdir"),"renderautoencoder.png");
        PlotFilters filters = new PlotFilters(da.getParam(PretrainParamInitializer.WEIGHT_KEY).transpose(),new int[]{10,10},new int[]{0,0},new int[]{28,28});
        filters.plot();
        INDArray weightFilter = filters.getPlot();
        ImageRender.render(weightFilter,autoEncoderWeights.getAbsolutePath());
        autoEncoderWeights.deleteOnExit();


        ImageLoader loader = new ImageLoader(56,56,3);
        INDArray arr = loader.toBgr(new ClassPathResource("rendertest.jpg").getFile()).reshape(3,56,56);
        File tmp2 = new File(System.getProperty("java.io.tmpdir"),"rendercolor.png");
        ImageRender.render(arr,tmp2.getAbsolutePath());
        tmp2.deleteOnExit();

    }



}
