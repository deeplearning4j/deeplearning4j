package org.deeplearning4j.example.deepautoencoder;

import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.deeplearning4j.autoencoder.DeepAutoEncoder;
import org.deeplearning4j.datasets.DataSet;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
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
        DataSetIterator iter = new MnistDataSetIterator(10,10,false);

        int codeLayer = 3;

        /*
          Reduction of dimensionality with neural nets Hinton 2006
         */
        Map<Integer,Double> layerLearningRates = new HashMap<>();
        layerLearningRates.put(codeLayer,1e-3);
        RandomGenerator rng = new MersenneTwister(123);


        StackedDenoisingAutoEncoder dbn = new StackedDenoisingAutoEncoder.Builder()
                .learningRateForLayer(layerLearningRates)
                .hiddenLayerSizes(new int[]{500, 250,100,50,25,10}).withRng(rng)
                .activateForLayer(Collections.singletonMap(3, Activations.sigmoid()))
                .numberOfInputs(784).sampleFromHiddenActivations(true)
                .lineSearchBackProp(false).useRegularization(true).forceEpochs()
                .withL2(2e-4)
                .withOutputActivationFunction(Activations.sigmoid())
                .numberOfOutPuts(784).withOutputLossFunction(OutputLayer.LossFunction.XENT)
                .build();

        //log.info("Training with layers of " + RBMUtil.architecture(dbn));
        //log.info("Begin training ");


        while(iter.hasNext()) {
            DataSet next = iter.next();
            dbn.pretrain(next.getFirst(),new Object[]{0.3,1e-1,1000});

        }


        DeepAutoEncoder a = new DeepAutoEncoder.Builder().withEncoder(dbn).build();



        iter.reset();


         while(iter.hasNext()) {
             DataSet next = iter.next();
             a.finetune(next.getFirst(),1e-1,1000);
         }

         iter.reset();

        while (iter.hasNext()) {
            DataSet data = iter.next();



            DeepAutoEncoderDataSetReconstructionRender r = new DeepAutoEncoderDataSetReconstructionRender(data.iterator(data.numExamples()),a,28,28);
            r.setPicDraw(MatrixTransformations.multiplyScalar(255));
            r.draw();
        }


    }


}
