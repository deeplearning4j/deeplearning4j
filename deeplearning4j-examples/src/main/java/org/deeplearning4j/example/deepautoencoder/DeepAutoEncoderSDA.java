package org.deeplearning4j.example.deepautoencoder;

import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.deeplearning4j.autoencoder.DeepAutoEncoder;
import org.deeplearning4j.datasets.DataSet;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.MultipleEpochsIterator;
import org.deeplearning4j.datasets.iterator.ReconstructionDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.NeuralNetwork;
import org.deeplearning4j.nn.OutputLayer;
import org.deeplearning4j.nn.activation.Activations;
import org.deeplearning4j.plot.DeepAutoEncoderDataSetReconstructionRender;
import org.deeplearning4j.plot.MultiLayerNetworkReconstructionRender;
import org.deeplearning4j.rbm.RBM;
import org.deeplearning4j.sda.StackedDenoisingAutoEncoder;
import org.deeplearning4j.transformation.MatrixTransformations;
import org.deeplearning4j.util.RBMUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

/**
 * Created by agibsonccc on 6/12/14.
 */
public class DeepAutoEncoderSDA {


    private static Logger log = LoggerFactory.getLogger(DeepAutoEncoderExample.class);

    public static void main(String[] args) throws Exception {
        DataSetIterator iter = new MultipleEpochsIterator(50,new ReconstructionDataSetIterator(new MnistDataSetIterator(100,100,false)));

        int codeLayer = 3;

        /*
          Reduction of dimensionality with neural nets Hinton 2006
         */
        Map<Integer,Double> layerLearningRates = new HashMap<>();
        layerLearningRates.put(codeLayer,1e-1);
        RandomGenerator rng = new MersenneTwister(123);


        StackedDenoisingAutoEncoder dbn = new StackedDenoisingAutoEncoder.Builder()
                .learningRateForLayer(layerLearningRates).constrainGradientToUnitNorm(true)
                .hiddenLayerSizes(new int[]{1000, 500, 250, 30}).withRng(rng)
                .activateForLayer(Collections.singletonMap(3, Activations.sigmoid()))
                .numberOfInputs(784).sampleFromHiddenActivations(false)
                .lineSearchBackProp(false).useRegularization(true).forceEpochs()
                .withL2(2e-5).resetAdaGradIterations(10)
                .withOutputActivationFunction(Activations.sigmoid())
                .numberOfOutPuts(784).withOutputLossFunction(OutputLayer.LossFunction.SQUARED_LOSS)
                .build();

        //log.info("Training with layers of " + RBMUtil.architecture(dbn));
        //log.info("Begin training ");


        dbn.pretrain(iter,new Object[]{0.3,1e-1,10});

        DeepAutoEncoder a = new DeepAutoEncoder.Builder().withEncoder(dbn).build();

        a.finetune(iter,1e-2,10);

        iter.reset();

        while (iter.hasNext()) {
            DataSet data = iter.next();



            DeepAutoEncoderDataSetReconstructionRender r = new DeepAutoEncoderDataSetReconstructionRender(data.iterator(data.numExamples()),a,28,28);
            r.setPicDraw(MatrixTransformations.multiplyScalar(255));
            r.draw();
        }


    }


}
