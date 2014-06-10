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
import org.deeplearning4j.transformation.MatrixTransform;
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
        DataSetIterator iter = new MnistDataSetIterator(100,60000,false);

        int codeLayer = 3;

        /*
          Reduction of dimensionality with neural nets Hinton 2006
         */
        Map<Integer,Double> layerLearningRates = new HashMap<>();
        layerLearningRates.put(codeLayer,1e-3);
        RandomGenerator rng = new MersenneTwister(123);


        DBN dbn = new DBN.Builder()
                .learningRateForLayer(layerLearningRates)
                .hiddenLayerSizes(new int[]{1000, 500, 250,30}).withRng(rng)
                .withHiddenUnitsByLayer(Collections.singletonMap(codeLayer,RBM.HiddenUnit.GAUSSIAN))
                .numberOfInputs(784).useHiddenActivationsForwardProp(true)
                 //number of outputs dont matter
                .numberOfOutPuts(2)
                .useRegularization(true)
                .withL2(2e-4)
                .build();

        log.info("Training with layers of " + RBMUtil.architecure(dbn));
        log.info("Begin training ");


        dbn.pretrain(iter,new Object[]{1,1e-1,1000});



        DeepAutoEncoder encoder = new DeepAutoEncoder(dbn);
        encoder.setUseHiddenActivationsForwardProp(false);
        encoder.setVisibleUnit(RBM.VisibleUnit.GAUSSIAN);
        encoder.setHiddenUnit(RBM.HiddenUnit.BINARY);
        //encoder.setCodeLayerActivationFunction(Activations.linear());
        encoder.setOutputLayerActivation(Activations.sigmoid());
        encoder.setOutputLayerLossFunction(OutputLayer.LossFunction.RMSE_XENT);
        log.info("Arch " + RBMUtil.architecure(encoder));

        for(int i = 0; i < 10; i++) {
            while (iter.hasNext()) {
                DataSet data = iter.next();
              

                log.info("Fine tune " + data.labelDistribution());
                encoder.finetune(data.getFirst(),1e-1,10);
                List<DoubleMatrix> activations = encoder.feedForward();

                DeepAutoEncoderDataSetReconstructionRender r = new DeepAutoEncoderDataSetReconstructionRender(data.iterator(data.numExamples()),encoder,28,28);
                r.setPicDraw(MatrixTransformations.multiplyScalar(255));
                r.draw();
            }

            iter.reset();
        }



    }

}
