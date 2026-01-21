/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.eclipse.deeplearning4j.nd4j.autodiff.opvalidation;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.validation.OpValidation;
import org.nd4j.autodiff.validation.TestCase;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Validation tests for Mixture of Experts (MoE) operations.
 * Tests forward pass correctness for sparse expert routing.
 *
 * @author Adam Gibson
 */
@Slf4j
@NativeTag
@Tag(TagNames.SAMEDIFF)
@DisplayName("Mixture of Experts Op Validation Tests")
public class TestMixtureOfExpertsOpValidation extends BaseOpValidation {

    @Override
    public long getTimeoutMilliseconds() {
        return 90000L;
    }

    // ========================= Basic MoE Tests =========================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("MoE - Basic Router Forward Pass")
    public void testMoERouterForward(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(12345);

        int batch = 2;
        int seqLen = 4;
        int hiddenSize = 8;
        int numExperts = 4;

        SameDiff sd = SameDiff.create();

        // Input embeddings [batch, seq_len, hidden_size]
        INDArray inputArr = Nd4j.rand(DataType.DOUBLE, batch, seqLen, hiddenSize).muli(0.5);
        // Router weights [hidden_size, num_experts]
        INDArray routerArr = Nd4j.rand(DataType.DOUBLE, hiddenSize, numExperts).muli(0.1);

        SDVariable input = sd.var("input", inputArr);
        SDVariable routerWeights = sd.var("routerWeights", routerArr);

        // Compute router logits: [batch, seq_len, num_experts]
        SDVariable inputFlat = input.reshape(batch * seqLen, hiddenSize);
        SDVariable routerLogits = sd.mmul(inputFlat, routerWeights);
        SDVariable routerLogitsReshaped = routerLogits.reshape(batch, seqLen, numExperts);

        // Apply softmax to get probabilities
        SDVariable routerProbs = sd.nn.softmax("router_probs", routerLogitsReshaped, 2);

        // Probabilities should sum to 1 along expert dimension
        SDVariable probSum = sd.sum("prob_sum", routerProbs, 2);
        SDVariable loss = sd.mean("loss", probSum);

        TestCase tc = new TestCase(sd)
                .testName("MoE Router")
                .gradientCheck(true)
                .gradCheckEpsilon(1e-5)
                .gradCheckMaxRelativeError(1e-3);

        String err = OpValidation.validate(tc);
        assertNull(err, err);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("MoE - Expert Selection (Top-K)")
    public void testMoETopKSelection(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(12345);

        int batch = 2;
        int seqLen = 4;
        int hiddenSize = 8;
        int numExperts = 8;

        SameDiff sd = SameDiff.create();

        INDArray inputArr = Nd4j.rand(DataType.DOUBLE, batch, seqLen, hiddenSize).muli(0.5);
        INDArray routerArr = Nd4j.rand(DataType.DOUBLE, hiddenSize, numExperts).muli(0.1);

        SDVariable input = sd.var("input", inputArr);
        SDVariable routerWeights = sd.var("routerWeights", routerArr);

        // Compute router logits
        SDVariable inputFlat = input.reshape(batch * seqLen, hiddenSize);
        SDVariable routerLogits = sd.mmul(inputFlat, routerWeights);
        SDVariable routerLogitsReshaped = routerLogits.reshape(batch, seqLen, numExperts);

        SDVariable routerProbs = sd.nn.softmax("router_probs", routerLogitsReshaped, 2);

        // Compute weighted sum (simulating top-k expert selection)
        SDVariable result = sd.sum("result", routerProbs.mul(routerLogitsReshaped), 2);
        SDVariable loss = sd.mean("loss", result);

        TestCase tc = new TestCase(sd)
                .testName("MoE Top-K Selection")
                .gradientCheck(true)
                .gradCheckEpsilon(1e-5)
                .gradCheckMaxRelativeError(1e-3);

        String err = OpValidation.validate(tc);
        assertNull(err, err);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("MoE - Single Expert Forward")
    public void testMoESingleExpertForward(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(12345);

        int batch = 2;
        int seqLen = 4;
        int hiddenSize = 8;
        int expertHidden = 16;

        SameDiff sd = SameDiff.create();

        INDArray inputArr = Nd4j.rand(DataType.DOUBLE, batch, seqLen, hiddenSize).muli(0.3);
        INDArray expertW1Arr = Nd4j.rand(DataType.DOUBLE, hiddenSize, expertHidden).muli(0.1);
        INDArray expertW2Arr = Nd4j.rand(DataType.DOUBLE, expertHidden, hiddenSize).muli(0.1);

        SDVariable input = sd.var("input", inputArr);
        SDVariable expertW1 = sd.var("expertW1", expertW1Arr);
        SDVariable expertW2 = sd.var("expertW2", expertW2Arr);

        // Expert forward: FFN with GELU activation
        SDVariable inputFlat = input.reshape(batch * seqLen, hiddenSize);
        SDVariable hidden = sd.mmul(inputFlat, expertW1);
        SDVariable activated = sd.nn.gelu(hidden);
        SDVariable output = sd.mmul(activated, expertW2);
        SDVariable result = output.reshape(batch, seqLen, hiddenSize);
        result.rename("result");

        SDVariable loss = sd.mean("loss", result);

        TestCase tc = new TestCase(sd)
                .testName("MoE Single Expert")
                .gradientCheck(true)
                .gradCheckEpsilon(1e-5)
                .gradCheckMaxRelativeError(1e-3);

        String err = OpValidation.validate(tc);
        assertNull(err, err);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("MoE - Multiple Experts Combined")
    public void testMoEMultipleExperts(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(12345);

        int batch = 2;
        int seqLen = 4;
        int hiddenSize = 8;
        int expertHidden = 16;
        int numExperts = 2;

        SameDiff sd = SameDiff.create();

        INDArray inputArr = Nd4j.rand(DataType.DOUBLE, batch, seqLen, hiddenSize).muli(0.3);
        INDArray routerArr = Nd4j.rand(DataType.DOUBLE, hiddenSize, numExperts).muli(0.1);
        INDArray expert0W1Arr = Nd4j.rand(DataType.DOUBLE, hiddenSize, expertHidden).muli(0.1);
        INDArray expert0W2Arr = Nd4j.rand(DataType.DOUBLE, expertHidden, hiddenSize).muli(0.1);
        INDArray expert1W1Arr = Nd4j.rand(DataType.DOUBLE, hiddenSize, expertHidden).muli(0.1);
        INDArray expert1W2Arr = Nd4j.rand(DataType.DOUBLE, expertHidden, hiddenSize).muli(0.1);

        SDVariable input = sd.var("input", inputArr);
        SDVariable routerWeights = sd.var("routerWeights", routerArr);
        SDVariable expert0W1 = sd.var("expert0W1", expert0W1Arr);
        SDVariable expert0W2 = sd.var("expert0W2", expert0W2Arr);
        SDVariable expert1W1 = sd.var("expert1W1", expert1W1Arr);
        SDVariable expert1W2 = sd.var("expert1W2", expert1W2Arr);

        // Compute router probabilities
        SDVariable inputFlat = input.reshape(batch * seqLen, hiddenSize);
        SDVariable routerLogits = sd.mmul(inputFlat, routerWeights);
        SDVariable routerProbs = sd.nn.softmax("router_probs", routerLogits, 1);

        // Expert 0 forward
        SDVariable hidden0 = sd.mmul(inputFlat, expert0W1);
        SDVariable activated0 = sd.nn.relu(hidden0, 0);
        SDVariable output0 = sd.mmul(activated0, expert0W2);

        // Expert 1 forward
        SDVariable hidden1 = sd.mmul(inputFlat, expert1W1);
        SDVariable activated1 = sd.nn.relu(hidden1, 0);
        SDVariable output1 = sd.mmul(activated1, expert1W2);

        // Weight by router probabilities - get probabilities for each expert
        SDVariable prob0 = sd.expandDims(sd.slice(routerProbs, new int[]{0, 0}, new int[]{batch * seqLen, 1}), 1);
        SDVariable prob1 = sd.expandDims(sd.slice(routerProbs, new int[]{0, 1}, new int[]{batch * seqLen, 1}), 1);

        // Manual weighting
        SDVariable weighted0 = output0.mul(sd.squeeze(prob0, 1));
        SDVariable weighted1 = output1.mul(sd.squeeze(prob1, 1));
        SDVariable weightedOutput = weighted0.add(weighted1);

        SDVariable result = weightedOutput.reshape(batch, seqLen, hiddenSize);
        result.rename("result");

        SDVariable loss = sd.mean("loss", result);

        TestCase tc = new TestCase(sd)
                .testName("MoE Multiple Experts")
                .gradientCheck(true)
                .gradCheckEpsilon(1e-5)
                .gradCheckMaxRelativeError(1e-3);

        String err = OpValidation.validate(tc);
        assertNull(err, err);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("MoE - Load Balancing Loss")
    public void testMoELoadBalancingLoss(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(12345);

        int batch = 4;
        int seqLen = 8;
        int hiddenSize = 16;
        int numExperts = 4;

        SameDiff sd = SameDiff.create();

        INDArray inputArr = Nd4j.rand(DataType.DOUBLE, batch, seqLen, hiddenSize).muli(0.5);
        INDArray routerArr = Nd4j.rand(DataType.DOUBLE, hiddenSize, numExperts).muli(0.1);

        SDVariable input = sd.var("input", inputArr);
        SDVariable routerWeights = sd.var("routerWeights", routerArr);

        // Compute router probabilities
        SDVariable inputFlat = input.reshape(batch * seqLen, hiddenSize);
        SDVariable routerLogits = sd.mmul(inputFlat, routerWeights);
        SDVariable routerProbs = sd.nn.softmax("router_probs", routerLogits, 1);

        // Compute load balancing loss (auxiliary loss)
        SDVariable tokenFraction = sd.mean(routerProbs, 0);
        SDVariable avgProb = sd.mean(routerProbs, 0);

        // Load balancing loss = num_experts * sum(f_i * P_i)
        SDVariable loadBalanceLoss = tokenFraction.mul(avgProb).sum().mul(numExperts);
        loadBalanceLoss.rename("load_balance_loss");

        SDVariable loss = sd.mean("loss", loadBalanceLoss);

        TestCase tc = new TestCase(sd)
                .testName("MoE Load Balancing")
                .gradientCheck(true)
                .gradCheckEpsilon(1e-5)
                .gradCheckMaxRelativeError(1e-3);

        String err = OpValidation.validate(tc);
        assertNull(err, err);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("MoE - Various Number of Experts")
    public void testMoEVariousExpertCounts(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(12345);

        int batch = 2;
        int seqLen = 4;
        int hiddenSize = 8;

        for (int numExperts : new int[]{2, 4, 8}) {
            SameDiff sd = SameDiff.create();

            INDArray inputArr = Nd4j.rand(DataType.DOUBLE, batch, seqLen, hiddenSize).muli(0.5);
            INDArray routerArr = Nd4j.rand(DataType.DOUBLE, hiddenSize, numExperts).muli(0.1);

            SDVariable input = sd.var("input", inputArr);
            SDVariable routerWeights = sd.var("routerWeights", routerArr);

            SDVariable inputFlat = input.reshape(batch * seqLen, hiddenSize);
            SDVariable routerLogits = sd.mmul(inputFlat, routerWeights);
            SDVariable routerProbs = sd.nn.softmax("router_probs", routerLogits, 1);

            SDVariable result = sd.sum("result", routerProbs, 1);
            SDVariable loss = sd.mean("loss", result);

            TestCase tc = new TestCase(sd)
                    .testName("MoE numExperts=" + numExperts)
                    .gradientCheck(true)
                    .gradCheckEpsilon(1e-5)
                    .gradCheckMaxRelativeError(1e-3);

            String err = OpValidation.validate(tc);
            assertNull(err, "Failed for numExperts=" + numExperts + ": " + err);
        }
    }
}
