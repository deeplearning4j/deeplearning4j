package org.deeplearning4j.iris;

import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.SamplingDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.distributions.Distributions;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.models.classifiers.dbn.DBN;
import org.deeplearning4j.models.featuredetectors.rbm.RBM;
import org.deeplearning4j.nn.WeightInit;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.nd4j.linalg.api.activation.Activations;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by agibsonccc on 9/12/14.
 */
public class IrisExample {


    private static Logger log = LoggerFactory.getLogger(IrisExample.class);



    public static void main(String[] args) {
        RandomGenerator gen = new MersenneTwister(123);

        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                .momentum(5e-1f)
                .regularization(true)
                .regularizationCoefficient(2e-4f).dist(Distributions.uniform(gen))
                .activationFunction(Activations.tanh()).iterations(100)
                .weightInit(WeightInit.DISTRIBUTION)
                .lossFunction(LossFunctions.LossFunction.RECONSTRUCTION_CROSSENTROPY).rng(gen)
                .learningRate(1e-1f).nIn(4).nOut(3).build();


        List<NeuralNetConfiguration> list = new ArrayList<>();



        DBN d = new DBN.Builder()
                .configure(conf)
                .hiddenLayerSizes(new int[]{3}).forceEpochs()
                .build();

        for(int i = 0; i < d.getnLayers(); i++) {

           //d.getLayers()[i].conf().set
        }


        d.getOutputLayer().conf().setActivationFunction(Activations.softMaxRows());
        d.getOutputLayer().conf().setLossFunction(LossFunctions.LossFunction.MCXENT);
        //note zeros here
        // d.getOutputLayer().setW(Nd4j.zeros(d.getOutputLayer().getW().shape()));

        DataSetIterator iter = new IrisDataSetIterator(150, 150);

        //fetch first
        DataSet next = iter.next(110);
        next.normalizeZeroMeanZeroUnitVariance();



        DataSetIterator iter2 = new SamplingDataSetIterator(next,10,10);
        d.fit(next);




        Evaluation eval = new Evaluation();
        INDArray output = d.output(next.getFeatureMatrix());
        eval.eval(next.getLabels(),output);
        log.info("Score " + eval.stats());
    }



}
