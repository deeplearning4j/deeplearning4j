package org.deeplearning4j.example.iris;

import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.datasets.DataSet;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.SamplingDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.dbn.GaussianRectifiedLinearDBN;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.NeuralNetwork;
import org.deeplearning4j.nn.activation.Activations;
import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class IrisExample {

    private static Logger log = LoggerFactory.getLogger(IrisExample.class);

    /**
     * @param args
     */
    public static void main(String[] args) {
        DataSetIterator irisData = new IrisDataSetIterator(150,150);
        DataSet next = irisData.next();
        next.normalizeZeroMeanZeroUnitVariance();

        Pair<DataSet,DataSet> trainTest = next.splitTestAndTrain(140);

        DataSet train = trainTest.getFirst();
        DataSet test = trainTest.getSecond();


        int   numExamples = train.numExamples();
        log.info("Training on " + numExamples);
        DataSetIterator sampling = new SamplingDataSetIterator(train,10,1000);


        GaussianRectifiedLinearDBN cdbn1 = new GaussianRectifiedLinearDBN.Builder()
                .hiddenLayerSizes(new int[]{3, 1, 2}).withOptimizationAlgorithm(NeuralNetwork.OptimizationAlgorithm.CONJUGATE_GRADIENT)
                .normalizeByInputRows(true).numberOfInputs(4).numberOfOutPuts(3).withLossFunction(NeuralNetwork.LossFunction.SQUARED_LOSS)
                .useAdaGrad(true).useHiddenActivationsForwardProp(true).withMomentum(0.5)
                .useRegularization(true).withActivation(Activations.tanh())
                .build();

        while(sampling.hasNext()) {
            DataSet sample = sampling.next();
            cdbn1.pretrain(sample.getFirst(), 1, 1e-4, 10000);

        }

        sampling.reset();

        while(sampling.hasNext()) {
            DataSet sample = sampling.next();
            cdbn1.feedForward(sample.getFirst());
            cdbn1.finetune(sample.getSecond(), 1e-4, 10000);

        }



        Evaluation eval = new Evaluation();

        DoubleMatrix predicted = cdbn1.predict(test.getFirst());
        eval.eval(test.getSecond(),predicted);



        log.info(eval.stats());






    }

}
