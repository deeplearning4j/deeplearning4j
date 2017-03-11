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
package org.deeplearning4j.arbiter.evaluator.graph;

import lombok.AllArgsConstructor;
import org.deeplearning4j.arbiter.optimize.api.data.DataProvider;
import org.deeplearning4j.arbiter.optimize.api.evaluation.ModelEvaluator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIteratorFactory;

import java.util.Map;

/**
 * A model evaluator for doing additional evaluation (classification evaluation)
 * for a {@link ComputationGraph} given a {@link DataSetIterator}
 *
 * In order to construct this evaluator you need to pass in a map
 * containing the parameters for this evaluator. You will likely be using
 * the {@link org.deeplearning4j.arbiter.data.DataSetIteratorFactoryProvider}
 * in which case you need to pass in a map in the constructor containing a key of value:
 * {@link org.deeplearning4j.arbiter.data.DataSetIteratorFactoryProvider#FACTORY_KEY}
 *  with a value of type string which contains the class name of the {@link DataSetIteratorFactory}
 *  to use.
 *
 *
 * @author Adam Gibson
 */
@AllArgsConstructor
public class GraphClassificationDataSetFactoryEvaluator implements ModelEvaluator<ComputationGraph, DataSetIteratorFactory, Evaluation> {
    private Map<String,Object> factoryParams = null;

    @Override
    public Evaluation evaluateModel(ComputationGraph model, DataProvider<DataSetIteratorFactory> dataProvider) {
        DataSetIterator iterator = dataProvider.testData(factoryParams).create();
        Evaluation eval = new Evaluation();
        while (iterator.hasNext()) {
            DataSet next = iterator.next();
            INDArray features = next.getFeatures();
            INDArray labels = next.getLabels();

            if (next.hasMaskArrays()) {
                INDArray fMask = next.getFeaturesMaskArray();
                INDArray lMask = next.getLabelsMaskArray();

                INDArray[] fMasks = (fMask == null ? null : new INDArray[]{fMask});
                INDArray[] lMasks = (lMask == null ? null : new INDArray[]{lMask});
                model.setLayerMaskArrays(fMasks, lMasks);

                INDArray out = model.output(next.getFeatures())[0];

                //Assume this is time series data. Not much point having a mask array for non TS data
                if (lMask != null) {
                    eval.evalTimeSeries(next.getLabels(), out, lMask);
                } else {
                    eval.evalTimeSeries(next.getLabels(), out);
                }

                model.clearLayerMaskArrays();
            } else {
                INDArray out = model.output(features)[0];
                if (out.rank() == 3) {
                    eval.evalTimeSeries(labels, out);
                } else {
                    eval.eval(labels, out);
                }
            }
        }

        return eval;
    }
}
