package org.deeplearning4j.example.iris;

import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.datasets.DataSet;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.SamplingDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.dbn.CDBN;
import org.deeplearning4j.dbn.GaussianRectifiedLinearDBN;
import org.deeplearning4j.distributions.Distributions;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.NeuralNetwork;
import org.deeplearning4j.nn.activation.Activations;
import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;

public class IrisExample {

    private static Logger log = LoggerFactory.getLogger(IrisExample.class);

    /**
     * @param args
     */
    public static void main(String[] args) {
        RandomGenerator rng = new MersenneTwister(123);

        double preTrainLr = 0.01;
        int preTrainEpochs = 10000;
        int k = 1;
        int nIns = 4,nOuts = 3;
        int[] hiddenLayerSizes = new int[] {4,3,3};
        double fineTuneLr = 0.01;
        int fineTuneEpochs = 10000;

        GaussianRectifiedLinearDBN dbn = new GaussianRectifiedLinearDBN.Builder()
                .useAdaGrad(true)
                .normalizeByInputRows(true)
                .withOptimizationAlgorithm(NeuralNetwork.OptimizationAlgorithm.GRADIENT_DESCENT)
                .numberOfInputs(nIns).numberOfOutPuts(nOuts)
                .withActivation(Activations.sigmoid())
                .withMomentum(0.5).withDist(Distributions.uniform(rng,5))
                .hiddenLayerSizes(hiddenLayerSizes)
                .useRegularization(false)
                .useHiddenActivationsForwardProp(true)
                .withRng(rng)
                .build();



        DataSetIterator iter = new IrisDataSetIterator(150, 150);

        DataSet next = iter.next(150);
        next.shuffle();
        next.normalizeZeroMeanZeroUnitVariance();

        dbn.pretrain(next.getFirst(),1,1e-3,10000);
        log.info(("\n\nActivations  " + dbn.feedForward(next.getFirst())).replaceAll(";","\n"));

        dbn.finetune(next.getSecond(),1e-3,10000);


        Evaluation eval = new Evaluation();
        DoubleMatrix predict = dbn.predict(next.getFirst());
        eval.eval(predict,next.getSecond());

        log.info(eval.stats());
    }

}
