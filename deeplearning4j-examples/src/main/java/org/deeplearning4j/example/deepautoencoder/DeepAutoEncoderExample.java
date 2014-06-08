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
import org.deeplearning4j.plot.MultiLayerNetworkReconstructionRender;
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
        DataSetIterator iter = new MnistDataSetIterator(10,10);




        DBN dbn = new DBN.Builder()
                .learningRateForLayer(Collections.singletonMap(3, 1e-1))
                .withLossFunction(NeuralNetwork.LossFunction.RECONSTRUCTION_CROSSENTROPY)
                .withOutputLossFunction(OutputLayer.LossFunction.RMSE_XENT)
                .hiddenLayerSizes(new int[]{1000, 500, 250, 28})
                .withHiddenUnitsByLayer(Collections.singletonMap(3, RBM.HiddenUnit.GAUSSIAN))
                .withOptimizationAlgorithm(NeuralNetwork.OptimizationAlgorithm.GRADIENT_DESCENT)
                .numberOfInputs(784)
                .withMomentum(0.9)
                .withDropOut(0.5)
                .useRegularization(true)
                .withL2(2e-4)
                .numberOfOutPuts(2)
                .build();

        log.info("Training with layers of " + RBMUtil.architecure(dbn));
        while(iter.hasNext()) {
            DataSet data = iter.next();
            //data.scale();
            dbn.pretrain(data.getFirst(), new Object[]{1, 1e-1, 1000});


        }

        iter.reset();







        DeepAutoEncoder encoder = new DeepAutoEncoder(dbn);
        encoder.setRoundCodeLayerInput(false);
        encoder.setVisibleUnit(RBM.VisibleUnit.GAUSSIAN);
        //encoder.setNormalizeCodeLayerOutput(false);
        encoder.setUseHiddenActivationsForwardProp(false);

        //encoder.setRoundCodeLayerInput(true);
        while (iter.hasNext()) {
            DataSet next = iter.next();
            //next.scale();
            log.info("Fine tune " + next.labelDistribution());
            encoder.finetune(next.getFirst(),1e-1,1000);
            DoubleMatrix output = encoder.output(next.getFirst());
            log.info("Output " + output);

        }


        iter.reset();




        while (iter.hasNext()) {
            DataSet data = iter.next();
            FilterRenderer f = new FilterRenderer();
            f.renderFilters(encoder.getOutputLayer().getW(),"outputlayer.png",28,28,data.numExamples());


            DeepAutoEncoderDataSetReconstructionRender r = new DeepAutoEncoderDataSetReconstructionRender(data.iterator(data.numExamples()),encoder);
            r.draw();

        }


    }

}
