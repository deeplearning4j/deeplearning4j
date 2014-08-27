package org.deeplearning4j.example.cats;

import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.models.classifiers.dbn.DBN;
import org.deeplearning4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.linalg.dataset.DataSet;
import org.deeplearning4j.plot.FilterRenderer;
import org.deeplearning4j.models.featuredetectors.rbm.RBM;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;

/**
 * Created by agibsonccc on 6/3/14.
 */
public class Cats {
    private static Logger log = LoggerFactory.getLogger(Cats.class);




    public static void main(String[] args) throws Exception {
        DataSetIterator iter = new CatsDataSetIterator(new File(args[0]),100,100);
        //784 input (number of columns in mnist, 10 labels (0-9), no regularization
        DBN dbn = new DBN.Builder().withHiddenUnits(RBM.HiddenUnit.RECTIFIED).withVisibleUnits(RBM.VisibleUnit.GAUSSIAN)
                .hiddenLayerSizes(new int[]{2000})
                .numberOfInputs(iter.inputColumns()).numberOfOutPuts(10)
                .build();

        while(iter.hasNext()) {
            DataSet next = iter.next();
            next.normalizeZeroMeanZeroUnitVariance();
            dbn.pretrain(next.getFeatureMatrix(), 1, 1e-1f, 10000);
            FilterRenderer render = new FilterRenderer();
            INDArray w = dbn.getLayers()[0].getW();
            render.renderFilters(w, "currimg.png", (int)Math.sqrt(w.rows()) , (int) Math.sqrt(w.rows()),10);



        }


    }


}
