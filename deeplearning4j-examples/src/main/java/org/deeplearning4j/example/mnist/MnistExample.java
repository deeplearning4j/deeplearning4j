package org.deeplearning4j.example.mnist;

import java.io.BufferedOutputStream;
import java.io.FileOutputStream;
import java.io.IOException;

import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.MultipleEpochsIterator;
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.dbn.DBN;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.linalg.dataset.DataSet;
import org.deeplearning4j.optimize.OutputLayerTrainingEvaluator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class MnistExample {

    private static Logger log = LoggerFactory.getLogger(MnistExample.class);

    /**
     * @param args
     * @throws java.io.IOException
     */
    public static void main(String[] args) throws IOException {
        //batches of 10, 60000 examples total
        DataSetIterator iter = new MnistDataSetIterator(60000,60000);
        DataSet shuffled = iter.next();
        shuffled.sortByLabel();
        iter = new ListDataSetIterator(shuffled.asList(),100);
        //784 input (number of columns in mnist, 10 labels (0-9), no regularization
        DBN dbn = new DBN.Builder()
                .hiddenLayerSizes(new int[]{600, 500, 400})
                .numberOfInputs(784).numberOfOutPuts(iter.totalOutcomes())
                .build();


        while(iter.hasNext()) {
            DataSet next = iter.next();
            dbn.pretrain(next.getFeatureMatrix(),1,1e-1f,1000);

        }


        iter.reset();
        while(iter.hasNext()) {
            DataSet next = iter.next();
            dbn.finetune(next.getLabels(),1e-1f,1000);

        }


        BufferedOutputStream bos = new BufferedOutputStream(new FileOutputStream("mnist-dbn.bin"));
        dbn.write(bos);
        bos.flush();
        bos.close();

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
