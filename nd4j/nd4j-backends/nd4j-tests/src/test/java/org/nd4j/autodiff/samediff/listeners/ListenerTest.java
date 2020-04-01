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

import java.util.*;

import org.junit.Test;
import org.nd4j.autodiff.listeners.*;
import org.nd4j.autodiff.listeners.impl.ScoreListener;
import org.nd4j.autodiff.listeners.records.History;
import org.nd4j.autodiff.listeners.records.LossCurve;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.TrainingConfig;
import org.nd4j.autodiff.samediff.internal.SameDiffOp;
import org.nd4j.autodiff.samediff.internal.Variable;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.evaluation.classification.Evaluation.Metric;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.OpContext;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.IrisDataSetIterator;
import org.nd4j.linalg.dataset.adapter.SingletonDataSetIterator;
import org.nd4j.linalg.dataset.api.MultiDataSet;
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

    @Override
    public char ordering() {
        return 'c';
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

        SDVariable loss = sd.loss.softmaxCrossEntropy("loss", label, predictions, null);

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

    @Test
    public void testListenerCalls(){
        SameDiff sd = SameDiff.create();
        SDVariable in = sd.placeHolder("in", DataType.FLOAT, -1, 4);
        SDVariable label = sd.placeHolder("label", DataType.FLOAT, -1, 3);
        SDVariable w = sd.var("w", Nd4j.rand(DataType.FLOAT, 4, 3));
        SDVariable b = sd.var("b", Nd4j.rand(DataType.FLOAT, 3));
        SDVariable z = in.mmul(w).add(b);
        SDVariable softmax = sd.nn.softmax("softmax", z);
        SDVariable loss = sd.loss.logLoss("loss" ,label, softmax);

        TestListener tl = new TestListener(Operation.INFERENCE);
        sd.setListeners(tl);

        //Check listener called during inference
        Map<String,INDArray> phMap = Collections.singletonMap("in", Nd4j.rand(1, 4));

        for( int i=1; i<=5; i++ ) {
            INDArray out = sd.outputSingle(phMap, "softmax");

            assertEquals(0, tl.epochStartCount);
            assertEquals(0, tl.epochEndCount);
            assertEquals(0, tl.validationDoneCount);
            assertEquals(0, tl.iterationStartCount);
            assertEquals(0, tl.iterationDoneCount);
            assertEquals(Collections.singletonMap(Operation.INFERENCE, i), tl.operationStartCount);
            assertEquals(Collections.singletonMap(Operation.INFERENCE, i), tl.operationEndCount);
            assertEquals(3*i, tl.preOpExecutionCount);    //mmul, add, softmax
            assertEquals(3*i, tl.opExecutionCount);
            assertEquals(3*i, tl.activationAvailableCount);   //mmul, add, softmax outputs
            assertEquals(0, tl.preUpdateCount);     //Inference -> no updating
        }

        //Check listener NOT called during inference when set to Operation.TRAINING
        tl = new TestListener(Operation.TRAINING);
        sd.setListeners(tl);
        sd.outputSingle(phMap, "softmax");

        assertEquals(0, tl.epochStartCount);
        assertEquals(0, tl.epochEndCount);
        assertEquals(0, tl.validationDoneCount);
        assertEquals(0, tl.iterationStartCount);
        assertEquals(0, tl.iterationDoneCount);
        assertEquals(Collections.emptyMap(), tl.operationStartCount);
        assertEquals(Collections.emptyMap(), tl.operationEndCount);
        assertEquals(0, tl.preOpExecutionCount);
        assertEquals(0, tl.opExecutionCount);
        assertEquals(0, tl.activationAvailableCount);
        assertEquals(0, tl.preUpdateCount);

        //Check listener called during gradient calculation
        tl = new TestListener(Operation.TRAINING);
        sd.setListeners(tl);
        phMap = new HashMap<>();
        phMap.put("in", Nd4j.rand( DataType.FLOAT, 1, 4));
        phMap.put("label", Nd4j.createFromArray(0f, 1f, 0f).reshape(1, 3));

        for( int i=1; i<=3; i++ ) {
            sd.calculateGradients(phMap, "in", "w", "b");
            assertEquals(0, tl.epochStartCount);
            assertEquals(0, tl.epochEndCount);
            assertEquals(0, tl.validationDoneCount);
            assertEquals(0, tl.iterationStartCount);
            assertEquals(0, tl.iterationDoneCount);
            assertEquals(Collections.singletonMap(Operation.TRAINING, i), tl.operationStartCount);
            assertEquals(Collections.singletonMap(Operation.TRAINING, i), tl.operationEndCount);
            assertEquals(7*i, tl.preOpExecutionCount);    //mmul, add, softmax, loss grad, softmax backward, add backward, mmul backward
            assertEquals(7*i, tl.opExecutionCount);
            assertEquals(11*i, tl.activationAvailableCount); //mmul, add, softmax, loss grad (weight, in, label), softmax bp, add backward (z, b), mmul (in, w)
            assertEquals(0, tl.preUpdateCount);
        }


        //Check listener NOT called during gradient calculation - when listener is still set to INFERENCE mode
        tl = new TestListener(Operation.INFERENCE);
        sd.setListeners(tl);
        for( int i=1; i<=3; i++ ) {
            sd.calculateGradients(phMap, "in", "w", "b");
            assertEquals(0, tl.epochStartCount);
            assertEquals(0, tl.epochEndCount);
            assertEquals(0, tl.validationDoneCount);
            assertEquals(0, tl.iterationStartCount);
            assertEquals(0, tl.iterationDoneCount);
            assertEquals(Collections.emptyMap(), tl.operationStartCount);
            assertEquals(Collections.emptyMap(), tl.operationEndCount);
            assertEquals(0, tl.preOpExecutionCount);
            assertEquals(0, tl.opExecutionCount);
            assertEquals(0, tl.activationAvailableCount);
            assertEquals(0, tl.preUpdateCount);
        }

        //Check fit:
        tl = new TestListener(Operation.TRAINING);
        sd.setListeners(tl);
        sd.setTrainingConfig(TrainingConfig.builder()
                .dataSetFeatureMapping("in")
                .dataSetLabelMapping("label")
                .updater(new Adam(1e-3))
                .build());

        SingletonDataSetIterator dsi = new SingletonDataSetIterator(new DataSet(phMap.get("in"), phMap.get("label")));
        for( int i=1; i<=3; i++ ) {
            sd.fit(dsi, 1);
            assertEquals(i, tl.epochStartCount);
            assertEquals(i, tl.epochEndCount);
            assertEquals(0, tl.validationDoneCount);
            assertEquals(i, tl.iterationStartCount);
            assertEquals(i, tl.iterationDoneCount);
            assertEquals(Collections.singletonMap(Operation.TRAINING, i), tl.operationStartCount);
            assertEquals(Collections.singletonMap(Operation.TRAINING, i), tl.operationEndCount);
            assertEquals(7*i, tl.preOpExecutionCount);    //mmul, add, softmax, loss grad, softmax backward, add backward, mmul backward
            assertEquals(7*i, tl.opExecutionCount);
            assertEquals(11*i, tl.activationAvailableCount); //mmul, add, softmax, loss grad (weight, in, label), softmax bp, add backward (z, b), mmul (in, w)
            assertEquals(2*i, tl.preUpdateCount);   //w, b
        }


        //Check evaluation:
        tl = new TestListener(Operation.EVALUATION);
        sd.setListeners(tl);

        for( int i=1; i<=3; i++ ) {
            sd.evaluate(dsi, "softmax", new Evaluation());
            assertEquals(0, tl.epochStartCount);
            assertEquals(0, tl.epochEndCount);
            assertEquals(0, tl.validationDoneCount);
            assertEquals(0, tl.iterationStartCount);
            assertEquals(0, tl.iterationDoneCount);
            assertEquals(Collections.singletonMap(Operation.EVALUATION, i), tl.operationStartCount);
            assertEquals(Collections.singletonMap(Operation.EVALUATION, i), tl.operationEndCount);
            assertEquals(3*i, tl.preOpExecutionCount);    //mmul, add, softmax
            assertEquals(3*i, tl.opExecutionCount);
            assertEquals(3*i, tl.activationAvailableCount); //mmul, add, softmax
            assertEquals(0, tl.preUpdateCount);   //w, b
        }
    }

    private static class TestListener implements Listener {

        public TestListener(Operation operation){
            this.operation = operation;
        }

        private final Operation operation;

        private int epochStartCount = 0;
        private int epochEndCount = 0;
        private int validationDoneCount = 0;
        private int iterationStartCount = 0;
        private int iterationDoneCount = 0;
        private Map<Operation,Integer> operationStartCount = new HashMap<>();
        private Map<Operation,Integer> operationEndCount = new HashMap<>();
        private int preOpExecutionCount = 0;
        private int opExecutionCount = 0;
        private int activationAvailableCount = 0;
        private int preUpdateCount = 0;


        @Override
        public ListenerVariables requiredVariables(SameDiff sd) {
            return null;
        }

        @Override
        public boolean isActive(Operation operation) {
            return this.operation == null || this.operation == operation;
        }

        @Override
        public void epochStart(SameDiff sd, At at) {
            epochStartCount++;
        }

        @Override
        public ListenerResponse epochEnd(SameDiff sd, At at, LossCurve lossCurve, long epochTimeMillis) {
            epochEndCount++;
            return ListenerResponse.CONTINUE;
        }

        @Override
        public ListenerResponse validationDone(SameDiff sd, At at, long validationTimeMillis) {
            validationDoneCount++;
            return ListenerResponse.CONTINUE;
        }

        @Override
        public void iterationStart(SameDiff sd, At at, MultiDataSet data, long etlTimeMs) {
            iterationStartCount++;
        }

        @Override
        public void iterationDone(SameDiff sd, At at, MultiDataSet dataSet, Loss loss) {
            iterationDoneCount++;
        }

        @Override
        public void operationStart(SameDiff sd, Operation op) {
            if(!operationStartCount.containsKey(op)) {
                operationStartCount.put(op, 1);
            } else {
                operationStartCount.put(op, operationStartCount.get(op) + 1);
            }
        }

        @Override
        public void operationEnd(SameDiff sd, Operation op) {
            if(!operationEndCount.containsKey(op)) {
                operationEndCount.put(op, 1);
            } else {
                operationEndCount.put(op, operationEndCount.get(op) + 1);
            }
        }

        @Override
        public void preOpExecution(SameDiff sd, At at, SameDiffOp op, OpContext opContext) {
            preOpExecutionCount++;
        }

        @Override
        public void opExecution(SameDiff sd, At at, MultiDataSet batch, SameDiffOp op, OpContext opContext, INDArray[] outputs) {
            opExecutionCount++;
        }

        @Override
        public void activationAvailable(SameDiff sd, At at, MultiDataSet batch, SameDiffOp op, String varName, INDArray activation) {
            activationAvailableCount++;
        }

        @Override
        public void preUpdate(SameDiff sd, At at, Variable v, INDArray update) {
            preUpdateCount++;
        }
    }
}
