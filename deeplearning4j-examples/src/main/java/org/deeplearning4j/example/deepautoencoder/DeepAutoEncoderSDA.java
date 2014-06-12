package org.deeplearning4j.example.deepautoencoder;

import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.deeplearning4j.autoencoder.DeepAutoEncoder;
import org.deeplearning4j.datasets.DataSet;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
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
                .hiddenLayerSizes(new int[]{1000, 500, 250,30}).withRng(rng)
                .activateForLayer(Collections.singletonMap(3, Activations.rectifiedLinear()))
                .numberOfInputs(784).sampleFromHiddenActivations(true)
                .lineSearchBackProp(true).useRegularization(true)
                .withL2(2e-4)
                .withOutputActivationFunction(Activations.linear())
                .numberOfOutPuts(784).withMomentum(0.5).withOutputLossFunction(OutputLayer.LossFunction.SQUARED_LOSS)
                .build();

        //log.info("Training with layers of " + RBMUtil.architecture(dbn));
        //log.info("Begin training ");


        while(iter.hasNext()) {
            DataSet next = iter.next();
            dbn.pretrain(next.getFirst(),new Object[]{0.3,1e-1,100});

        }

        iter.reset();

        while(iter.hasNext()) {
            DataSet next = iter.next();
            dbn.finetune(next.getFirst(),1e-1,1000);

        }

        iter.reset();

        while (iter.hasNext()) {
            DataSet data = iter.next();
            MultiLayerNetworkReconstructionRender r = new MultiLayerNetworkReconstructionRender(data.iterator(10),dbn,4);
            //r.setPicDraw(MatrixTransformations.multiplyScalar(255));
            r.draw();
        }



        DeepAutoEncoder encoder = new DeepAutoEncoder(dbn);
        encoder.setRoundCodeLayerInput(true);
        encoder.setSampleFromHiddenActivations(true);
        encoder.setVisibleUnit(RBM.VisibleUnit.BINARY);
        encoder.setHiddenUnit(RBM.HiddenUnit.GAUSSIAN);
        encoder.setCodeLayerActivationFunction(Activations.linear());
        encoder.setOutputLayerActivation(Activations.sigmoid());
        encoder.setOutputLayerLossFunction(OutputLayer.LossFunction.RMSE_XENT);
        log.info("Arch " + RBMUtil.architecture(encoder));


        iter.reset();

        while (iter.hasNext()) {
            DataSet data = iter.next();


            log.info("Fine tune " + data.labelDistribution());
            encoder.finetune(data.getFirst(),1e-3,10);

            DeepAutoEncoderDataSetReconstructionRender r = new DeepAutoEncoderDataSetReconstructionRender(data.iterator(data.numExamples()),encoder,28,28);
            r.setPicDraw(MatrixTransformations.multiplyScalar(255));
            r.draw();
        }




    }


}
