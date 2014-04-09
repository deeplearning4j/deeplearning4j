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
import org.deeplearning4j.transformation.MatrixTransformations;
import org.deeplearning4j.util.Info;
import org.deeplearning4j.util.SerializationUtils;
import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

public class IrisExample {

    private static Logger log = LoggerFactory.getLogger(IrisExample.class);

    /**
     * @param args
     */
    public static void main(String[] args) {
        RandomGenerator rng = new MersenneTwister(123);
        int nIns = 4,nOuts = 3;
        int[] hiddenLayerSizes = new int[] {4,3};

        DataSetIterator iter = new IrisDataSetIterator(150, 150);

        DataSet next = iter.next();
        next.normalizeZeroMeanZeroUnitVariance();

        for(int i = 0; i < 3000; i++) {
            GaussianRectifiedLinearDBN dbn = new GaussianRectifiedLinearDBN.Builder()
                    .useAdaGrad(true)
                    .normalizeByInputRows(true).withMomentum(0.5)
                    .withOptimizationAlgorithm(NeuralNetwork.OptimizationAlgorithm.CONJUGATE_GRADIENT)
                    .numberOfInputs(nIns).numberOfOutPuts(nOuts)
                    .withLossFunction(NeuralNetwork.LossFunction.RECONSTRUCTION_CROSSENTROPY)
                    .hiddenLayerSizes(hiddenLayerSizes).useRegularization(false).withL2(2e-2)
                    .useHiddenActivationsForwardProp(true).withActivation(Activations.sigmoid())
                    .withRng(rng)
                    .build();



            next.shuffle();

            dbn.pretrain(next.getFirst(),1,1e-2,1000000);



            //log.info(Info.activationsFor(next.getFirst(),dbn));
            dbn.finetune(next.getSecond(),1e-2,1000000);





            Evaluation eval = new Evaluation();
            DoubleMatrix predict = dbn.predict(next.getFirst());
            eval.eval(predict,next.getSecond());

            double f1 = eval.f1();
            if(f1 >= 0.9) {
                log.info("Saving model with high f1 of " + f1);
                File save = new File("iris-model-" + f1 + ".bin");
                log.info("Saving " + save.getAbsolutePath());
                SerializationUtils.saveObject(dbn,save);

            }

            log.info(eval.stats());

        }
       }

}
