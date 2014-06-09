package org.deeplearning4j.example.deepautoencoder;

import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.deeplearning4j.autoencoder.DeepAutoEncoder;
import org.deeplearning4j.datasets.DataSet;
import org.deeplearning4j.datasets.fetchers.MnistDataFetcher;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.SamplingDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.datasets.mnist.draw.DrawReconstruction;
import org.deeplearning4j.dbn.DBN;
import org.deeplearning4j.distributions.Distributions;
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

import java.util.Collection;
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
        DataSetIterator iter = new SamplingDataSetIterator(new MnistDataSetIterator(100,100).next(),100,10000);



        /*
          Reduction of dimensionality with neural nets Hinton 2006
         */
        RandomGenerator rng = new MersenneTwister(123);

        DBN dbn = new DBN.Builder()
                .learningRateForLayer(Collections.singletonMap(3, 1e-1))
                .hiddenLayerSizes(new int[]{600,500,10})
                 .withRng(rng)
                .numberOfInputs(784).withDist(Distributions.normal(rng,0.1))
                 //number of outputs dont matter
                .numberOfOutPuts(2).withMomentum(0.9).withDropOut(0.5)
                .useRegularization(true).withL2(2e-4)
                .build();

        log.info("Training with layers of " + RBMUtil.architecure(dbn));
        log.info("Begin training ");
        int numTimesIter = 0;
        while(iter.hasNext()) {
            DataSet data = iter.next();
            //0 to 1
            //data.scale();
            dbn.pretrain(data.getFirst(), new Object[]{1, 1e-1, 10});
            log.info("Training " + numTimesIter++);

        }

        iter.reset();









        DeepAutoEncoder encoder = new DeepAutoEncoder(dbn);
        encoder.setRoundCodeLayerInput(true);
        encoder.setCodeLayerActivationFunction(Activations.sigmoid());
        encoder.setOutputLayerLossFunction(OutputLayer.LossFunction.RMSE_XENT);
        //encoder.setVisibleUnit(RBM.VisibleUnit.LINEAR);
        //encoder.setNormalizeCodeLayerOutput(false);
        encoder.setUseHiddenActivationsForwardProp(false);



        while (iter.hasNext()) {
            DataSet next = iter.next();
            //next.scale();
            log.info("Fine tune " + next.labelDistribution());
            encoder.finetune(next.getFirst(),1e-1,1000);

        }


        iter.reset();






        while (iter.hasNext()) {
            DataSet data = iter.next();
            // data.scale();
            FilterRenderer f = new FilterRenderer();
            f.renderFilters(encoder.getOutputLayer().getW(),"outputlayer.png",28,28,data.numExamples());


            DeepAutoEncoderDataSetReconstructionRender r = new DeepAutoEncoderDataSetReconstructionRender(data.iterator(data.numExamples()),encoder);
            r.draw();

        }


    }

}
