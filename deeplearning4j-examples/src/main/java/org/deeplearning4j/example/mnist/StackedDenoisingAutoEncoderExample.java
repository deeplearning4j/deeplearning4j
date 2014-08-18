package org.deeplearning4j.example.mnist;

import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.MultipleEpochsIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.linalg.dataset.DataSet;
import org.deeplearning4j.sda.StackedDenoisingAutoEncoder;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedOutputStream;
import java.io.FileOutputStream;

/**
 * Created by agibsonccc on 6/28/14.
 */
public class StackedDenoisingAutoEncoderExample {
    private static Logger log = LoggerFactory.getLogger(StackedDenoisingAutoEncoderExample.class);

    public static void main(String[] args)  throws Exception {
        //batches of 10, 60000 examples total
        DataSetIterator iter = new MultipleEpochsIterator(50,new MnistDataSetIterator(10,10));
        DataSet ne = iter.next();

        RandomGenerator rng = new MersenneTwister(123);
        //784 input (number of columns in mnist, 10 labels (0-9), no regularization
        StackedDenoisingAutoEncoder sda = new StackedDenoisingAutoEncoder.Builder()
                .hiddenLayerSizes(new int[]{600, 500, 400}).withRng(rng)
                .useRegularization(true).withL2(2e-5)
                .numberOfInputs(784).numberOfOutPuts(iter.totalOutcomes())
                .build();

        sda.pretrain(iter, 1, 1e-1, 1);

        iter.reset();

        sda.finetune(iter, 1e-1, 1);


        BufferedOutputStream bos = new BufferedOutputStream(new FileOutputStream("mnist-dbn.bin"));
        sda.write(bos);
        bos.flush();
        bos.close();
        log.info("Saved dbn");

        iter.reset();

        Evaluation eval = new Evaluation();

        while(iter.hasNext()) {
            DataSet next = iter.next();
            INDArray predict = sda.output(next.getFeatureMatrix());
            INDArray labels = next.getLabels();
            eval.eval(labels, predict);
        }

        log.info("Prediction f scores and accuracy");
        log.info(eval.stats());

    }


}
