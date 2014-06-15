package org.deeplearning4j.example.deepautoencoder;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.deeplearning4j.autoencoder.DeepAutoEncoder;
import org.deeplearning4j.datasets.DataSet;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.MultipleEpochsIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.dbn.DBN;
import org.deeplearning4j.nn.OutputLayer;
import org.deeplearning4j.nn.activation.Activations;
import org.deeplearning4j.plot.DeepAutoEncoderDataSetReconstructionRender;
import org.deeplearning4j.plot.FilterRenderer;
import org.deeplearning4j.plot.MultiLayerNetworkReconstructionRender;
import org.deeplearning4j.rbm.RBM;
import org.deeplearning4j.transformation.MatrixTransformations;
import org.deeplearning4j.util.MatrixUtil;
import org.deeplearning4j.util.RBMUtil;
import org.jblas.DoubleMatrix;
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
        DataSetIterator iter = new MnistDataSetIterator(10,30,false);

        int codeLayer = 3;

        /*
          Reduction of dimensionality with neural nets Hinton 2006
         */
        Map<Integer,Double> layerLearningRates = new HashMap<>();
        layerLearningRates.put(codeLayer,1e-2);
        RandomGenerator rng = new MersenneTwister(123);


        DBN dbn = new DBN.Builder()
                .learningRateForLayer(layerLearningRates)
                .hiddenLayerSizes(new int[]{1000, 500, 250, 30}).withRng(rng)
                .useRBMPropUpAsActivation(true)
                .activateForLayer(Collections.singletonMap(3, Activations.linear()))
                .withHiddenUnitsByLayer(Collections.singletonMap(codeLayer, RBM.HiddenUnit.GAUSSIAN))
                .numberOfInputs(784)
                .sampleFromHiddenActivations(true)
                .sampleOrActivateByLayer(Collections.singletonMap(3,false))
                .lineSearchBackProp(false).useRegularization(true).withL2(2e-4)
                .withOutputActivationFunction(Activations.sigmoid())
                .numberOfOutPuts(784).withMomentum(0.5)
                .withOutputLossFunction(OutputLayer.LossFunction.RMSE_XENT)
                .build();

        log.info("Training with layers of " + RBMUtil.architecture(dbn));
        log.info("Begin training ");


        for(int i = 0; i < 10; i++) {
            while(iter.hasNext()) {
                DataSet next = iter.next();
                dbn.pretrain(next.getFirst(), new Object[]{1, 1e-1, 100});
            }


            iter.reset();
        }


        DeepAutoEncoder encoder = new DeepAutoEncoder.Builder().withEncoder(dbn).build();
        encoder.setRoundCodeLayerInput(false);
        encoder.setSampleFromHiddenActivations(false);
        encoder.setLineSearchBackProp(false);
        encoder.setOutputLayerLossFunction(OutputLayer.LossFunction.SQUARED_LOSS);
        //log.info("Arch " + RBMUtil.architecture(encoder));


        iter.reset();

        for(int i = 0; i < 10; i++) {
            double error = 0;

            while (iter.hasNext()) {
                DataSet data = iter.next();


                log.info("Fine tune " + data.labelDistribution());
                encoder.finetune(data.getFirst(),1e-2,1000);

                error += encoder.score();
            }
            log.info("Error " + error);

            iter.reset();
        }




        while (iter.hasNext()) {
            DataSet data = iter.next();



            DeepAutoEncoderDataSetReconstructionRender r = new DeepAutoEncoderDataSetReconstructionRender(data.iterator(data.numExamples()),encoder,28,28);
            r.setPicDraw(MatrixTransformations.multiplyScalar(255));
            r.draw();
        }



    }

}
