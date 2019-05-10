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

package org.deeplearning4j.earlystopping.scorecalc.base;

import lombok.NonNull;
import org.deeplearning4j.earlystopping.scorecalc.ScoreCalculator;
import org.deeplearning4j.nn.api.Model;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;

public abstract class BaseScoreCalculator<T extends Model> implements ScoreCalculator<T> {

    protected MultiDataSetIterator mdsIterator;
    protected DataSetIterator iterator;
    protected double scoreSum;
    protected int minibatchCount;
    protected int exampleCount;

    protected BaseScoreCalculator(@NonNull DataSetIterator iterator){
        this.iterator = iterator;
    }

    protected BaseScoreCalculator(@NonNull MultiDataSetIterator iterator){
        this.mdsIterator = iterator;
    }

    @Override
    public double calculateScore(T network) {
        reset();

        if(iterator != null) {
            if (!iterator.hasNext())
                iterator.reset();

            while (iterator.hasNext()) {
                DataSet ds = iterator.next();
                INDArray out = output(network, ds.getFeatures(), ds.getFeaturesMaskArray(), ds.getLabelsMaskArray());
                scoreSum += scoreMinibatch(network, ds.getFeatures(), ds.getLabels(), ds.getFeaturesMaskArray(),
                        ds.getLabelsMaskArray(), out);
                minibatchCount++;
                exampleCount += ds.getFeatures().size(0);
            }
        } else {
            if(!mdsIterator.hasNext())
                mdsIterator.reset();

            while(mdsIterator.hasNext()){
                MultiDataSet mds = mdsIterator.next();
                INDArray[] out = output(network, mds.getFeatures(), mds.getFeaturesMaskArrays(), mds.getLabelsMaskArrays() );
                scoreSum += scoreMinibatch(network, mds.getFeatures(), mds.getLabels(), mds.getFeaturesMaskArrays(),
                        mds.getLabelsMaskArrays(), out);
                minibatchCount++;
                exampleCount += mds.getFeatures(0).size(0);
            }
        }

        return finalScore(scoreSum, minibatchCount, exampleCount);
    }

    protected abstract void reset();

    protected abstract INDArray output(T network, INDArray input, INDArray fMask, INDArray lMask);

    protected abstract INDArray[] output(T network, INDArray[] input, INDArray[] fMask, INDArray[] lMask);

    protected double scoreMinibatch(T network, INDArray features, INDArray labels,
                                    INDArray fMask, INDArray lMask, INDArray output){
        return scoreMinibatch(network, arr(features), arr(labels), arr(fMask), arr(lMask), arr(output));
    }

    protected abstract double scoreMinibatch(T network, INDArray[] features, INDArray[] labels,
                                             INDArray[] fMask, INDArray[] lMask, INDArray[] output);

    protected abstract double finalScore(double scoreSum, int minibatchCount, int exampleCount);

    public static INDArray[] arr(INDArray in){
        if(in == null) return null;
        return new INDArray[]{in};
    }

    public static INDArray get0(INDArray[] in){
        if(in == null) return null;
        if(in.length != 1){
            throw new IllegalStateException("Expected length 1 array here: got length " + in.length);
        }
        return in[0];
    }
}
