package org.deeplearning4j.example.iris;

import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.deeplearning4j.datasets.DataSet;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.dbn.DBN;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.NeuralNetwork;
import org.deeplearning4j.nn.activation.Activations;
import org.deeplearning4j.rbm.RBM;
import org.deeplearning4j.util.SerializationUtils;
import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;

public class IrisExample {

    private static Logger log = LoggerFactory.getLogger(IrisExample.class);

    /**
     * @param args
     */
    public static void main(String[] args) {
        RandomGenerator rng = new MersenneTwister(123);
        int nIns = 4,nOuts = 3;
        int[] hiddenLayerSizes = new int[] {3};

        DataSetIterator iter = new IrisDataSetIterator(150, 150);

        DataSet next = iter.next();

        DBN dbn = new DBN.Builder()
                .numberOfInputs(nIns).numberOfOutPuts(nOuts).withOutputActivationFunction(Activations.softMaxRows())
                .hiddenLayerSizes(hiddenLayerSizes)
                .withRng(rng)
                .build();


        next.shuffle();
        next.scale();

        dbn.setInput(next.getFeatureMatrix());
        dbn.pretrain(next.getFirst(),1,1e-1,100);



        //log.info(Info.activationsFor(next.getFirst(),dbn));
        dbn.finetune(next.getSecond(),1e-1,100);





        Evaluation eval = new Evaluation();
        DoubleMatrix predict = dbn.output(next.getFirst());
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
