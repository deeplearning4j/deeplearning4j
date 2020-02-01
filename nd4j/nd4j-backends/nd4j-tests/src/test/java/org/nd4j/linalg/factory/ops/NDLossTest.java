/* *****************************************************************************
 * Copyright (c) 2015-2018 Konduit k.k.
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

package org.nd4j.linalg.factory.ops;

import org.junit.Test;
import org.nd4j.autodiff.loss.LossReduce;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.random.impl.BernoulliDistribution;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

public class NDLossTest extends BaseNd4jTest {
    public NDLossTest(Nd4jBackend backend) {
        super(backend);
    }

    @Override
    public char ordering(){
        return 'c';
    }

    // TODO: We'll remove the new NDBase() at some point.
    @Test
    public void testAbsoluteDifference() {
        SameDiff sd = SameDiff.create();

        int nOut = 4;
        int minibatch = 10;
        SDVariable predictions = sd.var("in", DataType.DOUBLE, minibatch, nOut);
        SDVariable labels = sd.var("labels", DataType.DOUBLE, -1, nOut);

        INDArray wArr = Nd4j.create(new double[][]{
                {0, 0, 0, 0}, {0, 0, 1, 1}, {1, 1, 0, 0}, {1, 1, 1, 1}, {1, 1, 1, 1},
                {2, 2, 2, 2}, {2, 2, 2, 2}, {2, 2, 2, 2}, {2, 2, 2, 2}, {2, 2, 2, 2}});
        SDVariable w = sd.var("weights", wArr);

        LossReduce reduction = LossReduce.MEAN_BY_NONZERO_WEIGHT_COUNT;

        INDArray predictionsArr = Nd4j.randn(DataType.DOUBLE, minibatch, nOut);
        INDArray labelsArr = Nd4j.randn(DataType.DOUBLE, minibatch, nOut);


        SDVariable loss = sd.loss().absoluteDifference("loss", labels, predictions, w, reduction);
        sd.associateArrayWithVariable(predictionsArr, predictions);
        sd.associateArrayWithVariable(labelsArr, labels);

        INDArray y_exp = loss.eval();


        NDLoss ndLoss = new NDLoss();
        INDArray y = ndLoss.absoluteDifference(labelsArr, predictionsArr, wArr, reduction);
        assertEquals(y_exp, y);
    }

    @Test
    public void testCosineDistance() {
        SameDiff sd = SameDiff.create();

        int nOut = 4;
        int minibatch = 10;
        SDVariable predictions = sd.var("in", DataType.DOUBLE, minibatch, nOut);
        SDVariable labels = sd.var("labels", DataType.DOUBLE, -1, nOut);

        INDArray wArr = Nd4j.create(new double[][]{
                {0, 0, 0, 0}, {0, 0, 1, 1}, {1, 1, 0, 0}, {1, 1, 1, 1}, {1, 1, 1, 1},
                {2, 2, 2, 2}, {2, 2, 2, 2}, {2, 2, 2, 2}, {2, 2, 2, 2}, {2, 2, 2, 2}});
        SDVariable w = sd.var("weights", wArr);

        LossReduce reduction = LossReduce.MEAN_BY_NONZERO_WEIGHT_COUNT;

        INDArray predictionsArr = Nd4j.randn(DataType.DOUBLE, minibatch, nOut);
        INDArray labelsArr = Nd4j.randn(DataType.DOUBLE, minibatch, nOut);

        predictionsArr.diviColumnVector(predictionsArr.norm2(1));
        labelsArr.diviColumnVector(labelsArr.norm2(1));

        SDVariable loss = sd.loss().cosineDistance("loss", labels, predictions, w, reduction, 0);
        sd.associateArrayWithVariable(predictionsArr, predictions);
        sd.associateArrayWithVariable(labelsArr, labels);

        INDArray y_exp = loss.eval();
        System.out.println(y_exp);

        NDLoss ndLoss = new NDLoss();
        INDArray y = ndLoss.cosineDistance(labelsArr, predictionsArr, wArr, reduction, 0);
        assertEquals(y_exp, y);
    }

    @Test
    public void testHingeLoss() {
        SameDiff sd = SameDiff.create();

        int nOut = 4;
        int minibatch = 10;
        SDVariable predictions = sd.var("in", DataType.DOUBLE, minibatch, nOut);
        SDVariable labels = sd.var("labels", DataType.DOUBLE, -1, nOut);

        INDArray wArr = Nd4j.create(new double[][]{
                {0, 0, 0, 0}, {0, 0, 1, 1}, {1, 1, 0, 0}, {1, 1, 1, 1}, {1, 1, 1, 1},
                {2, 2, 2, 2}, {2, 2, 2, 2}, {2, 2, 2, 2}, {2, 2, 2, 2}, {2, 2, 2, 2}});
        SDVariable w = sd.var("weights", wArr);

        LossReduce reduction = LossReduce.MEAN_BY_NONZERO_WEIGHT_COUNT;

        INDArray predictionsArr = Nd4j.randn(DataType.DOUBLE, minibatch, nOut);
        INDArray labelsArr = Nd4j.randn(DataType.DOUBLE, minibatch, nOut);

        SDVariable loss = sd.loss().hingeLoss("loss", labels, predictions, w, reduction);
        sd.associateArrayWithVariable(predictionsArr, predictions);
        sd.associateArrayWithVariable(labelsArr, labels);

        INDArray y_exp = loss.eval();

        NDLoss ndLoss = new NDLoss();
        INDArray y = ndLoss.hingeLoss(labelsArr, predictionsArr, wArr, reduction);
        assertEquals(y_exp, y);
    }

    @Test
    public void testHuberLoss() {
        SameDiff sd = SameDiff.create();

        int nOut = 4;
        int minibatch = 10;
        SDVariable predictions = sd.var("in", DataType.DOUBLE, minibatch, nOut);
        SDVariable labels = sd.var("labels", DataType.DOUBLE, -1, nOut);

        INDArray wArr = Nd4j.create(new double[][]{
                {0, 0, 0, 0}, {0, 0, 1, 1}, {1, 1, 0, 0}, {1, 1, 1, 1}, {1, 1, 1, 1},
                {2, 2, 2, 2}, {2, 2, 2, 2}, {2, 2, 2, 2}, {2, 2, 2, 2}, {2, 2, 2, 2}});
        SDVariable w = sd.var("weights", wArr);

        LossReduce reduction = LossReduce.MEAN_BY_NONZERO_WEIGHT_COUNT;

        INDArray predictionsArr = Nd4j.randn(DataType.DOUBLE, minibatch, nOut);
        INDArray labelsArr = Nd4j.randn(DataType.DOUBLE, minibatch, nOut);

        SDVariable loss = sd.loss().huberLoss("loss", labels, predictions, w, reduction, 0.02);
        sd.associateArrayWithVariable(predictionsArr, predictions);
        sd.associateArrayWithVariable(labelsArr, labels);

        INDArray y_exp = loss.eval();

        NDLoss ndLoss = new NDLoss();
        INDArray y = ndLoss.huberLoss(labelsArr, predictionsArr, wArr, reduction, 0.02);
        assertEquals(y_exp, y);
    }

    @Test
    public void testL2Loss() {
        SameDiff sd = SameDiff.create();

        int nOut = 4;
        int minibatch = 10;
        SDVariable predictions = sd.var("in", DataType.DOUBLE, minibatch, nOut);
        INDArray predictionsArr = Nd4j.randn(DataType.DOUBLE, minibatch, nOut);

        SDVariable loss = sd.loss().l2Loss("loss", predictions);
        sd.associateArrayWithVariable(predictionsArr, predictions);

        INDArray y_exp = loss.eval();

        NDLoss ndLoss = new NDLoss();
        INDArray y = ndLoss.l2Loss(predictionsArr);
        assertEquals(y_exp, y);
    }

    @Test
    public void testLogLoss() {
        SameDiff sd = SameDiff.create();

        int nOut = 4;
        int minibatch = 10;
        SDVariable predictions = sd.var("in", DataType.DOUBLE, minibatch, nOut);
        SDVariable labels = sd.var("labels", DataType.DOUBLE, -1, nOut);

        INDArray wArr = Nd4j.create(new double[][]{
                {0, 0, 0, 0}, {0, 0, 1, 1}, {1, 1, 0, 0}, {1, 1, 1, 1}, {1, 1, 1, 1},
                {2, 2, 2, 2}, {2, 2, 2, 2}, {2, 2, 2, 2}, {2, 2, 2, 2}, {2, 2, 2, 2}});
        SDVariable w = sd.var("weights", wArr);

        LossReduce reduction = LossReduce.MEAN_BY_NONZERO_WEIGHT_COUNT;

        INDArray predictionsArr = Nd4j.randn(DataType.DOUBLE, minibatch, nOut);
        INDArray labelsArr = Nd4j.randn(DataType.DOUBLE, minibatch, nOut);
        Nd4j.getExecutioner().exec(new BernoulliDistribution(labelsArr, 0.5));
        predictionsArr = Nd4j.rand(predictionsArr.shape()).muli(0.8).addi(0.1);

        double eps = 1e-7;

        SDVariable loss = sd.loss().logLoss("loss", labels, predictions, w, reduction, eps);
        sd.associateArrayWithVariable(predictionsArr, predictions);
        sd.associateArrayWithVariable(labelsArr, labels);

        INDArray y_exp = loss.eval();

        NDLoss ndLoss = new NDLoss();
        //TODO: Test fails.   "Op [log_loss] execution failed"
        INDArray y = ndLoss.logLoss(labelsArr, predictionsArr, wArr, reduction, eps);
        assertEquals(y_exp, y);
    }

    @Test
    public void testLogPoisson() {
        SameDiff sd = SameDiff.create();

        int nOut = 4;
        int minibatch = 10;
        SDVariable predictions = sd.var("in", DataType.DOUBLE, minibatch, nOut);
        SDVariable labels = sd.var("labels", DataType.DOUBLE, -1, nOut);

        INDArray wArr = Nd4j.create(new double[][]{
                {0, 0, 0, 0}, {0, 0, 1, 1}, {1, 1, 0, 0}, {1, 1, 1, 1}, {1, 1, 1, 1},
                {2, 2, 2, 2}, {2, 2, 2, 2}, {2, 2, 2, 2}, {2, 2, 2, 2}, {2, 2, 2, 2}});
        SDVariable w = sd.var("weights", wArr);

        LossReduce reduction = LossReduce.MEAN_BY_NONZERO_WEIGHT_COUNT;

        INDArray predictionsArr = Nd4j.randn(DataType.DOUBLE, minibatch, nOut);
        INDArray labelsArr = Nd4j.randn(DataType.DOUBLE, minibatch, nOut);

        SDVariable loss = sd.loss().logPoisson("loss", labels, predictions, w, reduction);
        sd.associateArrayWithVariable(predictionsArr, predictions);
        sd.associateArrayWithVariable(labelsArr, labels);

        INDArray y_exp = loss.eval();

        NDLoss ndLoss = new NDLoss();
        INDArray y = ndLoss.logPoisson(labelsArr, predictionsArr, wArr, reduction, false);
        assertEquals(y_exp, y);
    }

    @Test
    public void testMeanPairwiseSquaredError() {
        SameDiff sd = SameDiff.create();

        int nOut = 4;
        int minibatch = 10;
        SDVariable predictions = sd.var("in", DataType.DOUBLE, minibatch, nOut);
        SDVariable labels = sd.var("labels", DataType.DOUBLE, -1, nOut);

        INDArray wArr = Nd4j.create(new double[][]{
                {0, 0, 0, 0}, {0, 0, 1, 1}, {1, 1, 0, 0}, {1, 1, 1, 1}, {1, 1, 1, 1},
                {2, 2, 2, 2}, {2, 2, 2, 2}, {2, 2, 2, 2}, {2, 2, 2, 2}, {2, 2, 2, 2}});
        SDVariable w = sd.var("weights", wArr);

        LossReduce reduction = LossReduce.MEAN_BY_NONZERO_WEIGHT_COUNT;

        INDArray predictionsArr = Nd4j.randn(DataType.DOUBLE, minibatch, nOut);
        INDArray labelsArr = Nd4j.randn(DataType.DOUBLE, minibatch, nOut);

        SDVariable loss = sd.loss().meanPairwiseSquaredError("loss", labels, predictions, w, reduction);
        sd.associateArrayWithVariable(predictionsArr, predictions);
        sd.associateArrayWithVariable(labelsArr, labels);

        INDArray y_exp = loss.eval();

        NDLoss ndLoss = new NDLoss();
        INDArray y = ndLoss.meanPairwiseSquaredError(labelsArr, predictionsArr, wArr, reduction);
        assertEquals(y_exp, y);
    }

    @Test
    public void testMeanSquaredError() {
        SameDiff sd = SameDiff.create();

        int nOut = 4;
        int minibatch = 10;
        SDVariable predictions = sd.var("in", DataType.DOUBLE, minibatch, nOut);
        SDVariable labels = sd.var("labels", DataType.DOUBLE, -1, nOut);

        INDArray wArr = Nd4j.create(new double[][]{
                {0, 0, 0, 0}, {0, 0, 1, 1}, {1, 1, 0, 0}, {1, 1, 1, 1}, {1, 1, 1, 1},
                {2, 2, 2, 2}, {2, 2, 2, 2}, {2, 2, 2, 2}, {2, 2, 2, 2}, {2, 2, 2, 2}});
        SDVariable w = sd.var("weights", wArr);

        LossReduce reduction = LossReduce.MEAN_BY_NONZERO_WEIGHT_COUNT;

        INDArray predictionsArr = Nd4j.randn(DataType.DOUBLE, minibatch, nOut);
        INDArray labelsArr = Nd4j.randn(DataType.DOUBLE, minibatch, nOut);

        SDVariable loss = sd.loss().meanSquaredError("loss", labels, predictions, w, reduction);
        sd.associateArrayWithVariable(predictionsArr, predictions);
        sd.associateArrayWithVariable(labelsArr, labels);

        INDArray y_exp = loss.eval();

        NDLoss ndLoss = new NDLoss();
        INDArray y = ndLoss.meanSquaredError(labelsArr, predictionsArr, wArr, reduction);
        assertEquals(y_exp, y);
    }

    @Test
    public void testSigmoidCrossEntropy() {
        SameDiff sd = SameDiff.create();

        int nOut = 4;
        int minibatch = 10;
        SDVariable predictions = sd.var("in", DataType.DOUBLE, minibatch, nOut);
        SDVariable labels = sd.var("labels", DataType.DOUBLE, -1, nOut);

        INDArray wArr = Nd4j.create(new double[][]{
                {0, 0, 0, 0}, {0, 0, 1, 1}, {1, 1, 0, 0}, {1, 1, 1, 1}, {1, 1, 1, 1},
                {2, 2, 2, 2}, {2, 2, 2, 2}, {2, 2, 2, 2}, {2, 2, 2, 2}, {2, 2, 2, 2}});
        SDVariable w = sd.var("weights", wArr);

        LossReduce reduction = LossReduce.MEAN_BY_NONZERO_WEIGHT_COUNT;

        INDArray predictionsArr = Nd4j.randn(DataType.DOUBLE, minibatch, nOut);
        INDArray labelsArr = Nd4j.randn(DataType.DOUBLE, minibatch, nOut);
        double labelSmoothing = 0.01;

        SDVariable loss = sd.loss().sigmoidCrossEntropy("loss", labels, predictions, w, reduction, labelSmoothing);
        sd.associateArrayWithVariable(predictionsArr, predictions);
        sd.associateArrayWithVariable(labelsArr, labels);

        INDArray y_exp = loss.eval();

        NDLoss ndLoss = new NDLoss();
        INDArray y = ndLoss.sigmoidCrossEntropy(labelsArr, predictionsArr, wArr, reduction, labelSmoothing);
        assertEquals(y_exp, y);
    }

    @Test
    public void testSoftmaxCrossEntropy() {
        SameDiff sd = SameDiff.create();

        int nOut = 4;
        int minibatch = 10;
        SDVariable predictions = sd.var("in", DataType.DOUBLE, minibatch, nOut);
        SDVariable labels = sd.var("labels", DataType.DOUBLE, -1, nOut);

        INDArray wArr = Nd4j.scalar(1.0); //TODO: This test fails with a complex weights array.
        SDVariable w = null;

        LossReduce reduction = LossReduce.MEAN_BY_NONZERO_WEIGHT_COUNT;

        INDArray predictionsArr = Nd4j.randn(DataType.DOUBLE, minibatch, nOut);
        INDArray labelsArr = Nd4j.randn(DataType.DOUBLE, minibatch, nOut);
        labelsArr.assign(0);
        for (int i = 0; i < labelsArr.size(0); i++) {
            labelsArr.putScalar(i, i % labelsArr.size(1), 1.0);
        }

        double labelSmoothing = 0.0;

        //noinspection ConstantConditions
        SDVariable loss = sd.loss().softmaxCrossEntropy("loss", labels, predictions, w, reduction, labelSmoothing);
        sd.associateArrayWithVariable(predictionsArr, predictions);
        sd.associateArrayWithVariable(labelsArr, labels);

        INDArray y_exp = loss.eval();

        NDLoss ndLoss = new NDLoss();
        INDArray y = ndLoss.softmaxCrossEntropy(labelsArr, predictionsArr, wArr, reduction, labelSmoothing);
        assertEquals(y_exp, y);
    }

    @Test
    public void testSparseSoftmaxCrossEntropy() {
        SameDiff sd = SameDiff.create();

        int nOut = 4;
        int minibatch = 10;
        SDVariable predictions = sd.var("in", DataType.DOUBLE, minibatch, nOut);
        SDVariable labels = sd.var("labels", DataType.INT32, -1);


        INDArray predictionsArr = Nd4j.randn(DataType.DOUBLE, minibatch, nOut);
        INDArray labelsArr = Nd4j.create(DataType.INT32, minibatch);
        for( int i=0; i<minibatch; i++ ){
            labelsArr.putScalar(i, i%nOut);
        }

        SDVariable loss = sd.loss().sparseSoftmaxCrossEntropy("loss", predictions, labels);
        sd.associateArrayWithVariable(predictionsArr, predictions);
        sd.associateArrayWithVariable(labelsArr, labels);

        INDArray y_exp = loss.eval();

        NDLoss ndLoss = new NDLoss();
        INDArray y = ndLoss.sparseSoftmaxCrossEntropy(predictionsArr, labelsArr);
        assertEquals(y_exp, y);
    }


    @Test
    public void testWeightedCrossEntropyWithLogits() {
        // This one from SamediffTests.java
        SameDiff sameDiff = SameDiff.create();
        INDArray targets = Nd4j.create(new long[]{1, 5});
        INDArray inputs = Nd4j.create(new long[]{1, 5});
        INDArray weights = Nd4j.create(new long[]{1, 5});

        SDVariable sdInputs = sameDiff.var("inputs", inputs);
        SDVariable sdWeights = sameDiff.var("weights", weights);
        SDVariable sdTargets = sameDiff.var("targets", targets);

        SDVariable res = sameDiff.loss().weightedCrossEntropyWithLogits(sdTargets, sdInputs, sdWeights);

        INDArray resultArray = res.eval();
        assertArrayEquals(new long[]{1, 5}, resultArray.shape());

        // Make sure the INDArray interface produces the same result.
        NDLoss ndLoss = new NDLoss();
        INDArray y = ndLoss.weightedCrossEntropyWithLogits(targets, inputs, weights);
        assertEquals(resultArray , y);
    }
}
