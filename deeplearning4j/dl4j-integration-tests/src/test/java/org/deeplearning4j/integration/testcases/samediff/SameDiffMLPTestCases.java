/* ******************************************************************************
 * Copyright (c) 2020 Konduit K.K.
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
package org.deeplearning4j.integration.testcases.samediff;

import org.deeplearning4j.datasets.iterator.EarlyTerminationDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MultiDataSetIteratorAdapter;
import org.deeplearning4j.integration.ModelType;
import org.deeplearning4j.integration.TestCase;
import org.nd4j.autodiff.loss.LossReduce;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.TrainingConfig;
import org.nd4j.evaluation.IEvaluation;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;

import java.util.*;

public class SameDiffMLPTestCases {


    public static TestCase getMLPMnist(){
        return new TestCase() {
            {
                testName = "MLPMnistSD";
                testType = TestType.RANDOM_INIT;
                testPredictions = true;
                testTrainingCurves = true;
                testGradients = true;
                testParamsPostTraining = true;
                testEvaluation = true;
                testOverfitting = true;
                maxRelativeErrorOverfit = 2e-2;
                minAbsErrorOverfit = 1e-2;
            }

            @Override
            public ModelType modelType() {
                return ModelType.SAMEDIFF;
            }

            @Override
            public Object getConfiguration() throws Exception {
                Nd4j.getRandom().setSeed(12345);

                //Define the network structure:
                SameDiff sd = SameDiff.create();
                SDVariable in = sd.placeHolder("in", DataType.FLOAT, -1, 784);
                SDVariable label = sd.placeHolder("label", DataType.FLOAT, -1, 10);

                SDVariable w0 = sd.var("w0", Nd4j.rand(DataType.FLOAT, 784, 256));
                SDVariable b0 = sd.var("b0", Nd4j.rand(DataType.FLOAT, 256));
                SDVariable w1 = sd.var("w1", Nd4j.rand(DataType.FLOAT, 256, 10));
                SDVariable b1 = sd.var("b1", Nd4j.rand(DataType.FLOAT, 10));

                SDVariable a0 = sd.nn.tanh(in.mmul(w0).add(b0));
                SDVariable out = sd.nn.softmax("out", a0.mmul(w1).add(b1));
                SDVariable loss = sd.loss.logLoss("loss", label, out);

                //Also set the training configuration:
                sd.setTrainingConfig(TrainingConfig.builder()
                        .updater(new Adam(0.01))
                        .weightDecay(1e-3, true)
                        .dataSetFeatureMapping("in")            //features[0] -> "in" placeholder
                        .dataSetLabelMapping("label")           //labels[0]   -> "label" placeholder
                        .build());

                return sd;
            }

            @Override
            public List<Map<String, INDArray>> getPredictionsTestDataSameDiff() throws Exception {
                List<Map<String,INDArray>> out = new ArrayList<>();

                DataSetIterator iter = new MnistDataSetIterator(1, true, 12345);
                out.add(Collections.singletonMap("in", iter.next().getFeatures()));

                iter = new MnistDataSetIterator(8, true, 12345);
                out.add(Collections.singletonMap("in", iter.next().getFeatures()));

                return out;
            }

            @Override
            public List<String> getPredictionsNamesSameDiff() throws Exception {
                return Collections.singletonList("out");
            }

            @Override
            public Map<String, INDArray> getGradientsTestDataSameDiff() throws Exception {
                DataSet ds = new MnistDataSetIterator(8, true, 12345).next();
                Map<String,INDArray> map = new HashMap<>();
                map.put("in", ds.getFeatures());
                map.put("label", ds.getLabels());
                return map;
            }

            @Override
            public MultiDataSetIterator getTrainingData() throws Exception {
                DataSetIterator iter = new MnistDataSetIterator(8, true, 12345);
                iter = new EarlyTerminationDataSetIterator(iter, 32);
                return new MultiDataSetIteratorAdapter(iter);
            }

            @Override
            public IEvaluation[] getNewEvaluations() {
                return new IEvaluation[]{new Evaluation()};
            }

            @Override
            public MultiDataSetIterator getEvaluationTestData() throws Exception {
                DataSetIterator iter = new MnistDataSetIterator(8, false, 12345);
                iter = new EarlyTerminationDataSetIterator(iter, 32);
                return new MultiDataSetIteratorAdapter(iter);
            }

            @Override
            public IEvaluation[] doEvaluationSameDiff(SameDiff sd, MultiDataSetIterator iter, IEvaluation[] evaluations) {
                sd.evaluate(iter, "out", 0, evaluations);
                return evaluations;
            }

            @Override
            public MultiDataSet getOverfittingData() throws Exception {
                return new MnistDataSetIterator(1, true, 12345).next().toMultiDataSet();
            }

            @Override
            public int getOverfitNumIterations() {
                return 100;
            }
        };
    }

}
