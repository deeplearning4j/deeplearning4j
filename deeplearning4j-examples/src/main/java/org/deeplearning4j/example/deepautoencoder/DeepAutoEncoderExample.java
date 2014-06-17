package org.deeplearning4j.example.deepautoencoder;

import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.deeplearning4j.autoencoder.DeepAutoEncoder;
import org.deeplearning4j.datasets.DataSet;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.dbn.DBN;
import org.deeplearning4j.distributions.Distributions;
import org.deeplearning4j.nn.NeuralNetwork;
import org.deeplearning4j.nn.OutputLayer;
import org.deeplearning4j.nn.activation.Activations;
import org.deeplearning4j.plot.DeepAutoEncoderDataSetReconstructionRender;
import org.deeplearning4j.rbm.RBM;
import org.deeplearning4j.transformation.MatrixTransformations;
import org.deeplearning4j.util.RBMUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;

/**
 * Demonstrates a DeepAutoEncoder reconstructions with
 * the MNIST digits
 */
public class DeepAutoEncoderExample {

    private static Logger log = LoggerFactory.getLogger(DeepAutoEncoderExample.class);

    public static void main(String[] args) throws Exception {
        DataSetIterator iter =  new MnistDataSetIterator(10,10,false);


        int codeLayer = 7;

        /*
          Reduction of dimensionality with neural nets Hinton 2006
         */
        Map<Integer,Double> layerLearningRates = new HashMap<>();
        layerLearningRates.put(codeLayer,1e-2);
        RandomGenerator rng = new MersenneTwister(123);


        DBN dbn = new DBN.Builder()
                .learningRateForLayer(layerLearningRates)
                .hiddenLayerSizes(new int[]{1000, 500, 250,125,100,50,10}).withRng(rng)
                .useRBMPropUpAsActivation(true).withDist(Distributions.normal(rng,0.1))
                .activateForLayer(Collections.singletonMap(3, Activations.linear()))
                .withHiddenUnitsByLayer(Collections.singletonMap(codeLayer, RBM.HiddenUnit.GAUSSIAN))
                .numberOfInputs(784).lossFunctionByLayer(Collections.singletonMap(codeLayer, NeuralNetwork.LossFunction.RECONSTRUCTION_CROSSENTROPY))
                .sampleFromHiddenActivations(true)
                .useRegularization(true).withL2(2e-3)
                .numberOfOutPuts(784).withMomentum(0.5)
                .momentumAfter(Collections.singletonMap(10,0.9))
                .build();

        log.info("Training with layers of " + RBMUtil.architecture(dbn));
        log.info("Begin training ");



        while(iter.hasNext()) {
            DataSet next = iter.next();
            dbn.pretrain(next.getFirst(),1,1e-1,1000);
        }



        DeepAutoEncoder encoder = new DeepAutoEncoder.Builder().withEncoder(dbn).build();
        //log.info("Arch " + RBMUtil.architecture(encoder));


        iter.reset();




        while (iter.hasNext()) {
            DataSet data = iter.next();
            for(int i = 0; i < 10; i++)
                encoder.finetune(data.getFirst(),1e-2,1000);

        }

        iter.reset();







        while (iter.hasNext()) {
            DataSet data = iter.next();



            DeepAutoEncoderDataSetReconstructionRender r = new DeepAutoEncoderDataSetReconstructionRender(data.iterator(data.numExamples()),encoder,28,28);
            r.setPicDraw(MatrixTransformations.multiplyScalar(255));
            r.draw();
        }



    }

}
