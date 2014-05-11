package org.deeplearning4j.example.iris;

import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.deeplearning4j.datasets.DataSet;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.dbn.DBN;
import org.deeplearning4j.eval.Evaluation;
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

        for(int i = 0; i < 3000; i++) {
            DBN dbn = new DBN.Builder().withHiddenUnits(RBM.HiddenUnit.RECTIFIED).withVisibleUnits(RBM.VisibleUnit.GAUSSIAN)
                    .numberOfInputs(nIns).numberOfOutPuts(nOuts)
                    .hiddenLayerSizes(hiddenLayerSizes).useAdaGrad(false)
                    .withRng(rng)
                    .build();



            next.shuffle();

            dbn.pretrain(next.getFirst(),1,1e-4,1000000);



            //log.info(Info.activationsFor(next.getFirst(),dbn));
            dbn.finetune(next.getSecond(),1e-4,1000000);





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

}
