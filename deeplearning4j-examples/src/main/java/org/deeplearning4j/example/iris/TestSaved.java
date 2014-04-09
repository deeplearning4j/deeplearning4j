package org.deeplearning4j.example.iris;

import org.deeplearning4j.datasets.DataSet;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.RawMnistDataSetIterator;
import org.deeplearning4j.dbn.DBN;
import org.deeplearning4j.dbn.GaussianRectifiedLinearDBN;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.util.SerializationUtils;
import org.jblas.DoubleMatrix;

import org.slf4j.LoggerFactory;
import org.slf4j.Logger;

import java.io.File;

/**
 * Created by agibsonccc on 4/7/14.
 */
public class TestSaved {
    private static Logger log = LoggerFactory.getLogger(TestSaved.class);

    public static void main(String[] args) {
        GaussianRectifiedLinearDBN dbn = SerializationUtils.readObject(new File(args[0]));
        //batches of 10, 60000 examples total
        DataSetIterator iter = new IrisDataSetIterator(150, 150);


        Evaluation eval = new Evaluation();

        while(iter.hasNext()) {
            DataSet next = iter.next();
            next.normalizeZeroMeanZeroUnitVariance();

            DoubleMatrix predict = dbn.predict(next.getFirst());
            DoubleMatrix labels = next.getSecond();
            eval.eval(labels, predict);
        }

        log.info("Prediction f scores and accuracy");
        log.info(eval.stats());

    }



}
