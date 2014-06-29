package org.deeplearning4j.example.deepautoencoder;

import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.deeplearning4j.autoencoder.DeepAutoEncoder;
import org.deeplearning4j.datasets.DataSet;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.MultipleEpochsIterator;
import org.deeplearning4j.datasets.iterator.ReconstructionDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.distributions.Distributions;
import org.deeplearning4j.nn.NeuralNetwork;
import org.deeplearning4j.nn.OutputLayer;
import org.deeplearning4j.nn.WeightInit;
import org.deeplearning4j.nn.activation.Activations;
import org.deeplearning4j.plot.DeepAutoEncoderDataSetReconstructionRender;
import org.deeplearning4j.plot.MultiLayerNetworkReconstructionRender;
import org.deeplearning4j.rbm.RBM;
import org.deeplearning4j.sda.StackedDenoisingAutoEncoder;
import org.deeplearning4j.transformation.MatrixTransformations;
import org.deeplearning4j.util.RBMUtil;
import org.deeplearning4j.util.SerializationUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

/**
 * Created by agibsonccc on 6/12/14.
 */
public class DeepAutoEncoderSDA {


    private static Logger log = LoggerFactory.getLogger(DeepAutoEncoderExample.class);

    public static void main(String[] args) throws Exception {
       // DataSet d = SerializationUtils.readObject(new File(args[0]));
        DataSet d = new MnistDataSetIterator(100,100).next();
        //d = new ListDataSetIterator(d.asList(),100).next();
        DataSetIterator iter  = new MultipleEpochsIterator(10,new ReconstructionDataSetIterator(new ListDataSetIterator(d.asList(),100)));

        int codeLayer = 3;

        /*
          Reduction of dimensionality with neural nets Hinton 2006
         */
        Map<Integer,Double> layerLearningRates = new HashMap<>();
        layerLearningRates.put(codeLayer,1e-1);
        RandomGenerator rng = new MersenneTwister(123);


        StackedDenoisingAutoEncoder dbn = new StackedDenoisingAutoEncoder.Builder()
                .learningRateForLayer(layerLearningRates).constrainGradientToUnitNorm(false)
                .hiddenLayerSizes(new int[]{1000, 500, 250, 30}).withRng(rng)
                .activateForLayer(Collections.singletonMap(3, Activations.sigmoid())).useGaussNewtonVectorProductBackProp(true)
                .numberOfInputs(784).sampleFromHiddenActivations(false).withOptimizationAlgorithm(NeuralNetwork.OptimizationAlgorithm.HESSIAN_FREE)
                .lineSearchBackProp(false).useRegularization(false).forceEpochs()
                .withL2(0).lineSearchBackProp(true)
                .withOutputActivationFunction(Activations.sigmoid())
                .numberOfOutPuts(784).withOutputLossFunction(OutputLayer.LossFunction.RMSE_XENT)
                .build();

        //log.info("Training with layers of " + RBMUtil.architecture(dbn));
        //log.info("Begin training ");



        DeepAutoEncoder a = new DeepAutoEncoder.Builder().withEncoder(dbn).build();
        a.setLineSearchBackProp(true);
        while(iter.hasNext()) {
            DataSet next = iter.next();
            a.setInput(next.getFirst());
            a.finetune(next.getFirst(),1e-1,25);



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
