package org.deeplearning4j.example.lfw;

import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.LFWDataSetIterator;
import org.deeplearning4j.models.classifiers.dbn.DBN;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.linalg.dataset.DataSet;
import org.deeplearning4j.util.SerializationUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;

/**
 * Created by agibsonccc on 6/11/14.
 */
public class TestLFW {
    private static Logger log = LoggerFactory.getLogger(TestLFW.class);

    public static void main(String[] args) {
        DBN d = SerializationUtils.readObject(new File(args[0]));
        //batches of 10, 60000 examples total
        DataSetIterator iter = new LFWDataSetIterator(100,10000,56,56);
        Evaluation eval = new Evaluation();
        while(iter.hasNext()) {
            DataSet next = iter.next();
            eval.eval(next.getLabels(),d.output(next.getFeatureMatrix()));
            log.info(eval.stats());

        }

        log.info(eval.stats());



    }

}
