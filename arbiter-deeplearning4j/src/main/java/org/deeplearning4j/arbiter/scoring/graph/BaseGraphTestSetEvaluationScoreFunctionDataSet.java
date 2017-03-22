/*-
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
package org.deeplearning4j.arbiter.scoring.graph;

import org.deeplearning4j.arbiter.optimize.api.data.DataProvider;
import org.deeplearning4j.arbiter.optimize.api.score.ScoreFunction;

import org.deeplearning4j.arbiter.scoring.util.ScoreUtil;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.util.Map;

/**
 * Base class for accuracy/f1 calculations on a
 * ComputationGraph with a DataSetIterator
 *
 * @author Alex Black
 */
public abstract class BaseGraphTestSetEvaluationScoreFunctionDataSet implements ScoreFunction<ComputationGraph, Object> {

    protected Evaluation getEvaluation(ComputationGraph model, DataProvider<Object> dataProvider, Map<String, Object> dataParameters) {

        if (model.getNumOutputArrays() != 1)
            throw new IllegalStateException("GraphSetSetAccuracyScoreFunctionDataSet cannot be " +
                    "applied to ComputationGraphs with more than one output. NumOutputs = " + model.getNumOutputArrays());

        DataSetIterator testData = ScoreUtil.getIterator(dataProvider.testData(dataParameters));

        Evaluation evaluation = new Evaluation();

        while (testData.hasNext()) {
            DataSet next = testData.next();
            if (next.hasMaskArrays()) {
                INDArray fMask = next.getFeaturesMaskArray();
                INDArray lMask = next.getLabelsMaskArray();

                INDArray[] fMasks = (fMask == null ? null : new INDArray[]{fMask});
                INDArray[] lMasks = (lMask == null ? null : new INDArray[]{lMask});

                model.setLayerMaskArrays(fMasks, lMasks);

                INDArray out = model.output(next.getFeatures())[0];

                //Assume this is time series data. Not much point having a mask array for non TS data
                if (lMask != null) {
                    evaluation.evalTimeSeries(next.getLabels(), out, lMask);
                } else {
                    evaluation.evalTimeSeries(next.getLabels(), out);
                }

                model.clearLayerMaskArrays();
            } else {
                INDArray out = model.output(false, next.getFeatures())[0];
                if (next.getLabels().rank() == 3) evaluation.evalTimeSeries(next.getLabels(), out);
                else evaluation.eval(next.getLabels(), out);
            }
        }

        return evaluation;
    }

    @Override
    public boolean minimize() {
        return false;
    }

    @Override
    public abstract String toString();
}
