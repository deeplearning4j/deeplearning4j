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

import org.deeplearning4j.arbiter.optimize.api.data.DataProvider;
import org.deeplearning4j.arbiter.optimize.api.evaluation.ModelEvaluator;

import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

/**
 * A model evaluator for doing additional evaluation (classification evaluation) for a MultiLayerNetwork given a DataSetIterator
 *
 * @author Alex Black
 */
public class ClassificationEvaluator implements ModelEvaluator<MultiLayerNetwork, DataSetIterator, Evaluation> {
    @Override
    public Evaluation evaluateModel(MultiLayerNetwork model, DataProvider<DataSetIterator> dataProvider) {

        DataSetIterator iterator = dataProvider.testData(null);
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
