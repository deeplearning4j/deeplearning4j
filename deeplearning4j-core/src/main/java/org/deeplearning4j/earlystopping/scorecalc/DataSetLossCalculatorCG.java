/*
 *
 *  * Copyright 2016 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */

package org.deeplearning4j.earlystopping.scorecalc;

import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;

/** Given a DataSetIterator: calculate the total loss for the model on that data set.
 * Typically used to calculate the loss on a test set.
 */
public class DataSetLossCalculatorCG implements ScoreCalculator<ComputationGraph>{

    private DataSetIterator dataSetIterator;
    private MultiDataSetIterator multiDataSetIterator;
    private boolean average;

    /**Calculate the score (loss function value) on a given data set (usually a test set)
     *
     * @param dataSetIterator Data set to calculate the score for
     * @param average Whether to return the average (sum of loss / N) or just (sum of loss)
     */
    public DataSetLossCalculatorCG(DataSetIterator dataSetIterator, boolean average) {
        this.dataSetIterator = dataSetIterator;
        this.average = average;
    }

    /**Calculate the score (loss function value) on a given data set (usually a test set)
     *
     * @param dataSetIterator Data set to calculate the score for
     * @param average Whether to return the average (sum of loss / N) or just (sum of loss)
     */
    public DataSetLossCalculatorCG(MultiDataSetIterator dataSetIterator, boolean average) {
        this.multiDataSetIterator = dataSetIterator;
        this.average = average;
    }

    @Override
    public double calculateScore(ComputationGraph network) {
        double lossSum = 0.0;
        int exCount = 0;

        if(dataSetIterator != null){
            dataSetIterator.reset();

            while(dataSetIterator.hasNext()){
                DataSet dataSet = dataSetIterator.next();
                int nEx = dataSet.getFeatureMatrix().size(0);
                lossSum += network.score(dataSet) * nEx;
                exCount += nEx;
            }
        } else {
            multiDataSetIterator.reset();

            while(multiDataSetIterator.hasNext()){
                MultiDataSet dataSet = multiDataSetIterator.next();
                int nEx = dataSet.getFeatures(0).size(0);
                lossSum += network.score(dataSet) * nEx;
                exCount += nEx;
            }
        }

        if(average) return lossSum / exCount;
        else return lossSum;
    }

    @Override
    public String toString(){
        return "DataSetLossCalculatorCG(" + dataSetIterator + ",average="+average+")";
    }
}
