package org.deeplearning4j.example.cats;

import org.deeplearning4j.datasets.DataSet;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.dbn.DBN;
import org.deeplearning4j.plot.FilterRenderer;
import org.deeplearning4j.plot.MultiLayerNetworkReconstructionRender;
import org.deeplearning4j.rbm.RBM;
import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.Collections;

/**
 * Created by agibsonccc on 6/3/14.
 */
public class Cats {
    private static Logger log = LoggerFactory.getLogger(Cats.class);




    public static void main(String[] args) throws Exception {
        DataSetIterator iter = new CatsDataSetIterator(new File(args[0]),10,10);
        //784 input (number of columns in mnist, 10 labels (0-9), no regularization
        DBN dbn = new DBN.Builder().withHiddenUnits(RBM.HiddenUnit.RECTIFIED).withVisibleUnits(RBM.VisibleUnit.GAUSSIAN)
                .hiddenLayerSizes(new int[]{600})
                .numberOfInputs(iter.inputColumns()).numberOfOutPuts(10)
                .build();

        while(iter.hasNext()) {
            DataSet next = iter.next();
            next.normalizeZeroMeanZeroUnitVariance();
            dbn.pretrain(next.getFirst(), 1, 1e-1, 10000);
            FilterRenderer render = new FilterRenderer();
            DoubleMatrix w = dbn.getLayers()[0].getW();
            render.renderFilters(w, "currimg.png", (int)Math.sqrt(w.rows) , (int) Math.sqrt(w.rows),10);



        }


    }


}
