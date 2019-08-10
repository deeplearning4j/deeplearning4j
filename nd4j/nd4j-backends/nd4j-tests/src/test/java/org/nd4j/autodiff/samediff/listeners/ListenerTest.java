/*
 * Copyright (c) 2015-2019 Skymind, Inc.
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
 */

package org.nd4j.autodiff.samediff.listeners;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.util.Arrays;
import java.util.List;

import org.junit.Test;
import org.nd4j.autodiff.listeners.Operation;
import org.nd4j.autodiff.listeners.impl.ScoreListener;
import org.nd4j.autodiff.listeners.records.History;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.TrainingConfig;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.evaluation.classification.Evaluation.Metric;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.IrisDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.weightinit.impl.XavierInitScheme;

public class ListenerTest extends BaseNd4jTest {

    public ListenerTest(Nd4jBackend backend) {
        super(backend);
    }

    @Test
    public void irisHistoryTest() {

        DataSetIterator iter = new IrisDataSetIterator(150, 150);
        NormalizerStandardize std = new NormalizerStandardize();
        std.fit(iter);
        iter.setPreProcessor(std);

        Nd4j.getRandom().setSeed(12345);
        SameDiff sd = SameDiff.create();

        SDVariable in = sd.placeHolder("input", DataType.FLOAT, -1, 4);
        SDVariable label = sd.placeHolder("label", DataType.FLOAT, -1, 3);

        SDVariable w0 = sd.var("w0", new XavierInitScheme('c', 4, 10), DataType.FLOAT, 4, 10);
        SDVariable b0 = sd.zero("b0", DataType.FLOAT, 1, 10);

        SDVariable w1 = sd.var("w1", new XavierInitScheme('c', 10, 3), DataType.FLOAT, 10, 3);
        SDVariable b1 = sd.zero("b1", DataType.FLOAT, 1, 3);

        SDVariable z0 = in.mmul(w0).add(b0);
        SDVariable a0 = sd.nn().relu(z0, 0);
        SDVariable z1 = a0.mmul(w1).add(b1);
        SDVariable predictions = sd.nn().softmax("predictions", z1, 1);

        SDVariable loss = sd.loss.softmaxCrossEntropy("loss", label, predictions);

        sd.setLossVariables("loss");

        IUpdater updater = new Adam(1e-2);

        Evaluation e = new Evaluation();

        TrainingConfig conf = new TrainingConfig.Builder()
                .l2(1e-4)
                .updater(updater)
                .dataSetFeatureMapping("input")
                .dataSetLabelMapping("label")
                .trainEvaluation(predictions, 0, e)
                .build();

        sd.setTrainingConfig(conf);

        sd.setListeners(new ScoreListener(1));

        History hist = sd.fit(iter, 50);
//        Map<String, List<IEvaluation>> evalMap = new HashMap<>();
//        evalMap.put("prediction", Collections.singletonList(e));
//
//        sd.evaluateMultiple(iter, evalMap);

        e = (Evaluation) hist.finalTrainingEvaluations().evaluation(predictions);

        System.out.println(e.stats());

        float[] losses = hist.lossCurve().meanLoss(loss);

        System.out.println("Losses: " + Arrays.toString(losses));

        double acc = hist.finalTrainingEvaluations().getValue(Metric.ACCURACY);
        assertTrue("Accuracy < 75%, was " + acc, acc >= 0.75);
    }

    @Override
    public char ordering() {
        return 'c';
    }
}
