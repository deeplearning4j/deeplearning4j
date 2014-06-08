package org.deeplearning4j.example.deepautoencoder;

import org.deeplearning4j.autoencoder.DeepAutoEncoder;
import org.deeplearning4j.datasets.DataSet;
import org.deeplearning4j.datasets.fetchers.MnistDataFetcher;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.datasets.mnist.draw.DrawReconstruction;
import org.deeplearning4j.dbn.DBN;
import org.deeplearning4j.nn.NeuralNetwork;
import org.deeplearning4j.nn.OutputLayer;
import org.deeplearning4j.nn.activation.Activations;
import org.deeplearning4j.plot.DeepAutoEncoderDataSetReconstructionRender;
import org.deeplearning4j.plot.FilterRenderer;
import org.deeplearning4j.rbm.RBM;
import org.deeplearning4j.util.RBMUtil;
import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

/**
 * Demonstrates a DeepAutoEncoder reconstructions with
 * the MNIST digits
 */
public class DeepAutoEncoderExample {

    private static Logger log = LoggerFactory.getLogger(DeepAutoEncoderExample.class);

    public static void main(String[] args) throws Exception {
        DataSetIterator iter = new MnistDataSetIterator(100,100,false);




        DBN dbn = new DBN.Builder()
                .learningRateForLayer(Collections.singletonMap(3, 1e-1))
                .withLossFunction(NeuralNetwork.LossFunction.RECONSTRUCTION_CROSSENTROPY)
                .withOutputLossFunction(OutputLayer.LossFunction.RMSE_XENT)
                .hiddenLayerSizes(new int[]{1000, 500, 250, 28})
                .withHiddenUnitsByLayer(Collections.singletonMap(3, RBM.HiddenUnit.GAUSSIAN))
                .withOptimizationAlgorithm(NeuralNetwork.OptimizationAlgorithm.GRADIENT_DESCENT)
                .numberOfInputs(784).withMomentum(0.9).withDropOut(0.5).useRegularization(true).withL2(2e-4)
                .numberOfOutPuts(2)
                .build();

        log.info("Training with layers of " + RBMUtil.architecure(dbn));
        for(int i = 0; i < 5; i++) {
            while(iter.hasNext()) {
                DataSet data = iter.next();
                data.scale();
                dbn.pretrain(data.getFirst(), new Object[]{1, 1e-1, 10});



            }

            iter.reset();

        }





        DeepAutoEncoder encoder = new DeepAutoEncoder(dbn);
        encoder.setRoundCodeLayerInput(true);
        //encoder.setNormalizeCodeLayerOutput(false);
        encoder.setUseHiddenActivationsForwardProp(false);
        encoder.setOutputLayerLossFunction(OutputLayer.LossFunction.XENT);

        //encoder.setRoundCodeLayerInput(true);
       for(int i = 0; i < 100 ; i++) {
           while (iter.hasNext()) {
               DataSet next = iter.next();
               next.scale();
               log.info("Fine tune " + next.labelDistribution());
               encoder.finetune(next.getFirst(),1e-2,1000);

           }


           iter.reset();
       }



        while (iter.hasNext()) {
            DataSet data = iter.next();
            FilterRenderer f = new FilterRenderer();
            f.renderFilters(encoder.getOutputLayer().getW(),"outputlayer.png",28,28,data.numExamples());


            DeepAutoEncoderDataSetReconstructionRender r = new DeepAutoEncoderDataSetReconstructionRender(data.iterator(data.numExamples()),encoder);
            r.draw();

        }


    }

}
