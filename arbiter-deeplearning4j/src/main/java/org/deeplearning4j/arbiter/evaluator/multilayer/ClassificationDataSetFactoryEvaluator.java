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
package org.deeplearning4j.arbiter.evaluator.multilayer;

import lombok.AllArgsConstructor;
import org.deeplearning4j.arbiter.optimize.api.data.DataProvider;
import org.deeplearning4j.arbiter.optimize.api.evaluation.ModelEvaluator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIteratorFactory;

import java.util.Map;

/**
 * A model evaluator for doing additional evaluation (classification evaluation)
 * for a {@link MultiLayerNetwork} given a {@link DataSetIterator}
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
public class ClassificationDataSetFactoryEvaluator implements ModelEvaluator<MultiLayerNetwork, DataSetIteratorFactory, Evaluation> {
    private Map<String,Object> evalParams = null;


    @Override
    public Evaluation evaluateModel(MultiLayerNetwork model, DataProvider<DataSetIteratorFactory> dataProvider) {
        DataSetIterator iterator = dataProvider.testData(evalParams).create();
        Evaluation eval = new Evaluation();
        while (iterator.hasNext()) {
            DataSet ds = iterator.next();
            INDArray features = ds.getFeatures();
            INDArray labels = ds.getLabels();

            if (ds.hasMaskArrays()) {
                INDArray fMask = ds.getFeaturesMaskArray();
                INDArray lMask = ds.getLabelsMaskArray();

                INDArray out = model.output(ds.getFeatures(), false, fMask, lMask);

                //Assume this is time series data. Not much point having a mask array for non TS data
                if (lMask != null) {
                    eval.evalTimeSeries(labels, out, lMask);
                } else {
                    eval.evalTimeSeries(labels, out);
                }

            } else {
                INDArray out = model.output(features);
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
