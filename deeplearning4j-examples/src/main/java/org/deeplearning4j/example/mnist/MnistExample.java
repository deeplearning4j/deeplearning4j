package org.deeplearning4j.example.mnist;

import java.io.BufferedOutputStream;
import java.io.FileOutputStream;
import java.io.IOException;

import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.deeplearning4j.datasets.DataSet;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.dbn.DBN;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.optimize.OutputLayerTrainingEvaluator;
import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class MnistExample {

    private static Logger log = LoggerFactory.getLogger(MnistExample.class);

    /**
     * @param args
     * @throws IOException
     */
    public static void main(String[] args) throws IOException {
        //batches of 10, 60000 examples total
        DataSetIterator iter = new MnistDataSetIterator(10,20);
        RandomGenerator rng = new MersenneTwister(123);
        //784 input (number of columns in mnist, 10 labels (0-9), no regularization
        DBN dbn = new DBN.Builder()
                .hiddenLayerSizes(new int[]{600, 500, 400}).withRng(rng)
                .numberOfInputs(784).numberOfOutPuts(10)
                .build();
        DataSet test = null;
        while(iter.hasNext()) {
            DataSet next = iter.next();
            //held out test set
            if(!iter.hasNext()) {
                test = next;
            }

            dbn.pretrain(next.getFirst(), 1, 0.001, 10000);
        }


        OutputLayerTrainingEvaluator eval2 = new OutputLayerTrainingEvaluator.Builder()
                .patienceIncrease(2).validationEpochs(100).withNetwork(dbn)
                .testSet(test).improvementThreshold(0.995)
                .bestLoss(Double.POSITIVE_INFINITY).build();



        iter.reset();
        while(iter.hasNext()) {
            DataSet next = iter.next();
            if(!iter.hasNext())
                break;

            dbn.setInput(next.getFirst());
            dbn.finetune(next.getSecond(), 0.1, 10000,eval2);
        }


        BufferedOutputStream bos = new BufferedOutputStream(new FileOutputStream("mnist-dbn.bin"));
        dbn.write(bos);
        bos.flush();
        bos.close();
        log.info("Saved dbn");

        iter.reset();

        Evaluation eval = new Evaluation();

        while(iter.hasNext()) {
            DataSet next = iter.next();
            DoubleMatrix predict = dbn.output(next.getFirst());
            DoubleMatrix labels = next.getSecond();
            eval.eval(labels, predict);
        }

        log.info("Prediction f scores and accuracy");
        log.info(eval.stats());


    }

}
