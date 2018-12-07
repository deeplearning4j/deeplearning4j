/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

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
        KBatchRandomDataSet randomDS = new KBatchRandomDataSet(new int[] {2, 3}, new int[] {3, 3, 3, 2});
        DataSet allData = randomDS.getAllBatches();
        KFoldIterator kiter = new KFoldIterator(4, allData);
        int i = 0;
        while (kiter.hasNext()) {
            DataSet now = kiter.next();
            DataSet test = kiter.testFold();

            INDArray fExp = randomDS.getBatchButK(i, true);
            assertEquals(fExp, now.getFeatures());
            INDArray lExp = randomDS.getBatchButK(i, false);
            assertEquals(lExp, now.getLabels());

            assertEquals(test.getFeatures(), randomDS.getBatchK(i, true));
            assertEquals(test.getLabels(), randomDS.getBatchK(i, false));
            i++;
        }
        assertEquals(i, 4);
    }

    @Test(expected = IllegalArgumentException.class)
    public void checkCornerCaseException() {
        DataSet allData = new DataSet(Nd4j.linspace(1,3,3).transpose(), Nd4j.linspace(1,3,3).transpose());
        int k = 1;
        //this will throw illegal argument exception
        KFoldIterator kiter = new KFoldIterator(k, allData);
    }

    @Test
    public void checkCornerCase() {
        KBatchRandomDataSet randomDS = new KBatchRandomDataSet(new int[] {2, 3}, new int[] {2, 1});
        DataSet allData = randomDS.getAllBatches();
        KFoldIterator kiter = new KFoldIterator(2, allData);
        int i = 0;
        while (kiter.hasNext()) {
            DataSet now = kiter.next();
            DataSet test = kiter.testFold();

            assertEquals(now.getFeatures(), randomDS.getBatchButK(i, true));
            assertEquals(now.getLabels(), randomDS.getBatchButK(i, false));

            assertEquals(test.getFeatures(), randomDS.getBatchK(i, true));
            assertEquals(test.getLabels(), randomDS.getBatchK(i, false));
            i++;
            System.out.println("Fold " + i + " passed");
        }
        assertEquals(i, 2);
    }

    /*
     * Dataset built from given sized batches of random data
     */
    public class KBatchRandomDataSet {
        //only one label
        private int[] dataShape;
        private int dataRank;
        private int dataElementCount;
        private int[] batchSizes;
        private DataSet allBatches;
        private INDArray allFeatures;
        private INDArray allLabels;
        private INDArray[] kBatchFeats;
        private INDArray[] kBatchLabels;

        /**
         * Creates a dataset built from given sized batches of random data, with given shape of features and 1D labels
         * @param dataShape shape of features
         * @param batchSizes sizes of consecutive batches
         */
        public KBatchRandomDataSet(int[] dataShape, int[] batchSizes) {
            this.dataShape = dataShape;
            this.dataRank = this.dataShape.length;
            this.batchSizes = batchSizes;
            this.dataElementCount = 1;
            int[] eachBatchSize = new int[dataRank + 1];
            eachBatchSize[0] = 0;
            kBatchFeats = new INDArray[batchSizes.length];
            kBatchLabels = new INDArray[batchSizes.length];
            for (int i = 0; i < dataRank; i++) {
                this.dataElementCount *= dataShape[i];
                eachBatchSize[i + 1] = dataShape[i];
            }
            for (int i = 0; i < batchSizes.length; i++) {
                eachBatchSize[0] = batchSizes[i];
                INDArray currentBatchF = Nd4j.rand(eachBatchSize);
                INDArray currentBatchL = Nd4j.rand(batchSizes[i], 1);
                kBatchFeats[i] = currentBatchF;
                kBatchLabels[i] = currentBatchL;
                if (i == 0) {
                    allFeatures = currentBatchF.dup();
                    allLabels = currentBatchL.dup();
                } else {
                    allFeatures = Nd4j.vstack(allFeatures, currentBatchF).dup();
                    allLabels = Nd4j.vstack(allLabels, currentBatchL).dup();
                }
            }
            allBatches = new DataSet(allFeatures, allLabels.reshape(allFeatures.size(0), 1));
        }

        public DataSet getAllBatches() {
            return allBatches;
        }

        /**
         * Get features or labels for batch k
         * @param k index of batch
         * @param feat true if we want to get features, false if we want to get labels
         */
        public INDArray getBatchK(int k, boolean feat) {
            return feat ? kBatchFeats[k] : kBatchLabels[k];
        }

        /**
         * Get features or labels for all batches except for k
         * @param k index of excluded batch
         * @param feat true if we want to get features, false if we want to get labels
         */
        public INDArray getBatchButK(int k, boolean feat) {
            INDArray iFold = null;
            boolean notInit = true;
            for (int i = 0; i < batchSizes.length; i++) {
                if (i == k)
                    continue;
                if (notInit) {
                    iFold = getBatchK(i, feat);
                    notInit = false;
                } else {
                    iFold = Nd4j.vstack(iFold, getBatchK(i, feat)).dup();
                }
            }
            return iFold;
        }
    }


    @Test
    public void test5974(){
        DataSet ds = new DataSet(Nd4j.linspace(1,99,99).transpose(), Nd4j.linspace(1,99,99).transpose());

        KFoldIterator iter = new KFoldIterator(10, ds);

        int count = 0;
        while(iter.hasNext()){
            DataSet fold = iter.next();
            INDArray testFold;
            int countTrain;
            if(count < 9){
                //Folds 0 to 8: should have 10 examples for test
                testFold = Nd4j.linspace(10*count+1, 10*count+10, 10).transpose();
                countTrain = 99 - 10;
            } else {
                //Fold 9 should have 9 examples for test
                testFold = Nd4j.linspace(10*count+1, 10*count+9, 9).transpose();
                countTrain = 99-9;
            }
            String s = String.valueOf(count);
            DataSet test = iter.testFold();
            assertEquals(s, testFold, test.getFeatures());
            assertEquals(s, testFold, test.getLabels());
            assertEquals(s, countTrain, fold.getFeatures().length());
            assertEquals(s, countTrain, fold.getLabels().length());
            count++;
        }
    }

    @Override
    public char ordering() {
        return 'c';
    }
}
