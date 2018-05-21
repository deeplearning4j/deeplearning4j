package org.nd4j.linalg.dataset;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.KFoldIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import static org.junit.Assert.assertEquals;

/**
 * Created by susaneraly on 11/4/16.
 */
@RunWith(Parameterized.class)
public class KFoldIteratorTest extends BaseNd4jTest {

    public KFoldIteratorTest(Nd4jBackend backend) {
        super(backend);
    }

    @Test
    public void checkFolds() {
        randomDataSet randomDS = new randomDataSet(new int[] {2, 3}, new int[] {3, 3, 3, 2});
        DataSet allData = randomDS.getAllFolds();
        KFoldIterator kiter = new KFoldIterator(4, allData);
        int i = 0;
        while (kiter.hasNext()) {
            DataSet now = kiter.next();
            DataSet test = kiter.testFold();

            assertEquals(now.getFeatures(), randomDS.getFoldbutk(i, true));
            assertEquals(now.getLabels(), randomDS.getFoldbutk(i, false));

            assertEquals(test.getFeatures(), randomDS.getfoldK(i, true));
            assertEquals(test.getLabels(), randomDS.getfoldK(i, false));
            i++;
            System.out.println("Fold " + i + " passed");
        }
        assertEquals(i, 4);
    }

    /*
    //this will throw illegal argument exception
    @Test
    public void checkCornerCaseA() {
        randomDataSet randomDS = new randomDataSet(new int[] {2,3},new int []{3});
        DataSet allData = randomDS.getAllFolds();
        KFoldIterator kiter = new KFoldIterator(1,allData);
        int i = 0;
        while (kiter.hasNext()) {
            DataSet now = kiter.next();
            DataSet test = kiter.testFold();
    
            assertEquals(now.getFeatures(),randomDS.getFoldbutk(i,true));
            assertEquals(now.getLabels(),randomDS.getFoldbutk(i,false));
    
            assertEquals(test.getFeatures(),randomDS.getfoldK(i,true));
            assertEquals(test.getLabels(),randomDS.getfoldK(i,false));
            i++;
            System.out.println("Fold "+i+" passed");
        }
        assertEquals(i,1);
    }
    */

    @Test
    public void checkCornerCaseA() {
        randomDataSet randomDS = new randomDataSet(new int[] {2, 3}, new int[] {2, 1});
        DataSet allData = randomDS.getAllFolds();
        KFoldIterator kiter = new KFoldIterator(2, allData);
        int i = 0;
        while (kiter.hasNext()) {
            DataSet now = kiter.next();
            DataSet test = kiter.testFold();

            assertEquals(now.getFeatures(), randomDS.getFoldbutk(i, true));
            assertEquals(now.getLabels(), randomDS.getFoldbutk(i, false));

            assertEquals(test.getFeatures(), randomDS.getfoldK(i, true));
            assertEquals(test.getLabels(), randomDS.getfoldK(i, false));
            i++;
            System.out.println("Fold " + i + " passed");
        }
        assertEquals(i, 2);
    }

    public class randomDataSet {
        //only one label
        private int[] dataShape;
        private int dataRank;
        private int dataElementCount;
        private int[] ks;
        private DataSet allFolds;
        private INDArray allFeatures;
        private INDArray allLabels;
        private INDArray[] kfoldFeats;
        private INDArray[] kfoldLabels;

        public randomDataSet(int[] dataShape, int[] ks) {
            this.dataShape = dataShape;
            this.dataRank = this.dataShape.length;
            this.ks = ks;
            this.dataElementCount = 1;
            int[] eachFoldSize = new int[dataRank + 1];
            eachFoldSize[0] = 0;
            kfoldFeats = new INDArray[ks.length];
            kfoldLabels = new INDArray[ks.length];
            for (int i = 0; i < dataRank; i++) {
                this.dataElementCount *= dataShape[i];
                eachFoldSize[i + 1] = dataShape[i];
            }
            for (int i = 0; i < ks.length; i++) {
                eachFoldSize[0] = ks[i];
                INDArray currentFoldF = Nd4j.rand(eachFoldSize);
                INDArray currentFoldL = Nd4j.rand(ks[i], 1);
                kfoldFeats[i] = currentFoldF;
                kfoldLabels[i] = currentFoldL;
                if (i == 0) {
                    allFeatures = currentFoldF.dup();
                    allLabels = currentFoldL.dup();
                } else {
                    allFeatures = Nd4j.vstack(allFeatures, currentFoldF).dup();
                    allLabels = Nd4j.vstack(allLabels, currentFoldL).dup();
                }
            }
            allFolds = new DataSet(allFeatures, allLabels.reshape(allFeatures.size(0), 1));
        }

        public DataSet getAllFolds() {
            return allFolds;
        }

        public INDArray getfoldK(int k, boolean feat) {
            return feat == true ? kfoldFeats[k] : kfoldLabels[k];
        }

        public INDArray getFoldbutk(int k, boolean feat) {
            INDArray iFold = null;
            boolean notInit = true;
            for (int i = 0; i < ks.length; i++) {
                if (i == k)
                    continue;
                if (notInit) {
                    iFold = getfoldK(i, feat);
                    notInit = false;
                } else {
                    iFold = Nd4j.vstack(iFold, getfoldK(i, feat)).dup();
                }
            }
            return iFold;
        }
    }

    @Override
    public char ordering() {
        return 'c';
    }
}
