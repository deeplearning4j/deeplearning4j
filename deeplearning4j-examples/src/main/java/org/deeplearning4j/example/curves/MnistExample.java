package org.deeplearning4j.example.curves;

import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.MultipleEpochsIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.dbn.DBN;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.linalg.dataset.DataSet;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedOutputStream;
import java.io.FileOutputStream;
import java.io.IOException;

public class MnistExample {

    private static Logger log = LoggerFactory.getLogger(MnistExample.class);

    /**
     * @param args
     * @throws java.io.IOException
     */
    public static void main(String[] args) throws IOException {
        //batches of 10, 60000 examples total
        DataSetIterator iter = new MultipleEpochsIterator(50,new MnistDataSetIterator(10,10));
        DataSet ne = iter.next();

        RandomGenerator rng = new MersenneTwister(123);
        //784 input (number of columns in mnist, 10 labels (0-9), no regularization
        DBN dbn = new DBN.Builder()
                .hiddenLayerSizes(new int[]{600, 500, 400}).withRng(rng)
                .useRegularization(true).withL2(2e-5f)
                .numberOfInputs(784).numberOfOutPuts(iter.totalOutcomes())
                .build();

        dbn.pretrain(iter,1,1e-1f,1);


        iter.reset();

        dbn.finetune(iter,1e-1f,1);




        BufferedOutputStream bos = new BufferedOutputStream(new FileOutputStream("mnist-dbn.bin"));
        dbn.write(bos);
        bos.flush();
        bos.close();
        log.info("Saved dbn");

        iter.reset();

        Evaluation eval = new Evaluation();

        while(iter.hasNext()) {
            DataSet next = iter.next();
            INDArray predict = dbn.output(next.getFeatureMatrix());
            INDArray labels = next.getLabels();
            eval.eval(labels, predict);
        }

        log.info("Prediction f scores and accuracy");
        log.info(eval.stats());


    }

}
