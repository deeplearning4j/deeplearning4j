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
package org.eclipse.deeplearning4j.dl4jcore.nn.layers.samediff;

import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.preprocessor.CnnToFeedForwardPreProcessor;
import org.deeplearning4j.nn.layers.samediff.SameDiffLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.eclipse.deeplearning4j.dl4jcore.TestUtils;
import org.eclipse.deeplearning4j.dl4jcore.nn.layers.samediff.testlayers.SameDiffConv;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.ArrayHolder;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.VariableType;
import org.nd4j.autodiff.samediff.internal.SameDiffOp;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.CustomOp;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.NoOp;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * SameDiff layers with workspaces, from the first workspace cycle on (the state every trial starts from). TestSameDiffConv's
 * gradient check failed now and then on CPU in its one configuration with workspaces: conv2d_bp's input gradient came
 * from a col2im that added into whatever its recycled output buffer held, and the stale values reached the layer below.
 * A layer also keeps its SameDiff instance and gradient function across iterations, so no array they retain may live in a
 * workspace a later iteration reuses.
 */
@NativeTag
@Tag(TagNames.SAMEDIFF)
@Tag(TagNames.DL4J_OLD_API)
public class SameDiffLayerWorkspaceGradientTest extends BaseDL4JTest {

    private static final int TRIALS = 16;

    private static SameDiffConv conv(int nIn, Activation activation) {
        return new SameDiffConv.Builder()
                .weightInit(WeightInit.XAVIER)
                .nIn(nIn)
                .nOut(4)
                .kernelSize(new int[]{2, 2})
                .stride(new int[]{1, 1})
                .dilation(new int[]{1, 1})
                .convolutionMode(ConvolutionMode.Truncate)
                .activation(activation)
                .hasBias(true)
                .build();
    }

    private static MultiLayerNetwork network(boolean workspaces) {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .dataType(DataType.DOUBLE)
                .seed(12345)
                .updater(new NoOp())
                .trainingWorkspaceMode(workspaces ? WorkspaceMode.ENABLED : WorkspaceMode.NONE)
                .inferenceWorkspaceMode(workspaces ? WorkspaceMode.ENABLED : WorkspaceMode.NONE)
                .list()
                .layer(conv(3, Activation.TANH))
                .layer(conv(4, Activation.SIGMOID))
                .layer(new OutputLayer.Builder().activation(Activation.SOFTMAX)
                        .lossFunction(LossFunctions.LossFunction.MCXENT)
                        .nIn(4 * 6 * 6).nOut(4).build())
                .inputPreProcessor(2, new CnnToFeedForwardPreProcessor(6, 6, 4))
                .build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        return net;
    }

    private static INDArray gradient(MultiLayerNetwork net, INDArray features, INDArray labels) {
        net.setInput(features);
        net.setLabels(labels);
        net.computeGradientAndScore();
        return net.gradient().gradient().dup();
    }

    private static String mismatches(INDArray expected, INDArray actual) {
        double[] e = expected.dup().data().asDouble();
        double[] a = actual.dup().data().asDouble();
        StringBuilder first = new StringBuilder();
        int count = 0;
        for (int i = 0; i < e.length; i++) {
            if (!(Math.abs(e[i] - a[i]) <= 1e-9 * Math.max(1.0, Math.abs(e[i])))) {
                if (count < 6) {
                    first.append(" [").append(i).append("] ").append(a[i]).append(" vs ").append(e[i]);
                }
                count++;
            }
        }
        return count == 0 ? null : count + " of " + e.length + " entries differ:" + first;
    }

    private static INDArray[] trialData(int trial) {
        Nd4j.getRandom().setSeed(1000 + trial);
        INDArray features = Nd4j.rand(DataType.DOUBLE, 1, 3, 8, 8);
        INDArray labels = TestUtils.randomOneHot(DataType.DOUBLE, 1, 4, new Random(1000 + trial));
        return new INDArray[]{features, labels};
    }

    @Test
    public void gradientsWithWorkspacesMatchDetachedGradients() {
        List<String> failures = new ArrayList<>();
        for (int trial = 0; trial < TRIALS; trial++) {
            Nd4j.getWorkspaceManager().destroyAllWorkspacesForCurrentThread();
            INDArray[] data = trialData(trial);
            MultiLayerNetwork detached = network(false);
            MultiLayerNetwork attached = network(true);
            attached.setParams(detached.params());

            INDArray expected = gradient(detached, data[0], data[1]);
            INDArray actual = gradient(attached, data[0], data[1]);
            String mismatch = mismatches(expected, actual);
            if (mismatch != null) {
                failures.add("trial " + trial + ": " + mismatch);
            }
        }
        assertTrue(failures.isEmpty(), String.join("\n", failures));
    }

    private static void collectAttached(List<String> attached, String owner, SameDiff sd) {
        for (String name : sd.getVariables().keySet()) {
            if (sd.getVariable(name).getVariableType() == VariableType.PLACEHOLDER
                    || !sd.arrayAlreadyExistsForVarName(name)) {
                continue;
            }
            INDArray arr = sd.getArrForVarName(name);
            if (arr != null && arr.isAttached()) {
                attached.add(owner + " variable " + name);
            }
        }
        ArrayHolder eager = sd.getEagerArrays();
        for (String name : eager.arrayNames()) {
            INDArray arr = eager.getArray(name);
            if (arr != null && arr.isAttached()) {
                attached.add(owner + " eager array " + name);
            }
        }
        for (SameDiffOp op : sd.getOps().values()) {
            List<INDArray> kept = new ArrayList<>();
            if (op.getOp() instanceof CustomOp) {
                kept.addAll(((CustomOp) op.getOp()).inputArguments());
                kept.addAll(((CustomOp) op.getOp()).outputArguments());
            } else if (op.getOp() instanceof Op) {
                Op o = (Op) op.getOp();
                kept.add(o.x());
                kept.add(o.y());
                kept.add(o.z());
            }
            for (INDArray arr : kept) {
                if (arr != null && arr.isAttached()) {
                    attached.add(owner + " op " + op.getName() + " argument");
                }
            }
        }
    }

    /** After a gradient pass with workspaces, nothing a layer's SameDiff or its gradient function keeps is in a workspace. */
    @Test
    public void layerSameDiffStateIsNotInWorkspaces() throws Exception {
        Nd4j.getWorkspaceManager().destroyAllWorkspacesForCurrentThread();
        INDArray[] data = trialData(0);
        MultiLayerNetwork net = network(true);
        gradient(net, data[0], data[1]);

        Field sameDiffField = SameDiffLayer.class.getDeclaredField("sameDiff");
        sameDiffField.setAccessible(true);
        List<String> attached = new ArrayList<>();
        for (int i = 0; i < 2; i++) {
            SameDiff sd = (SameDiff) sameDiffField.get(net.getLayer(i));
            collectAttached(attached, "layer " + i, sd);
            SameDiff grad = sd.getFunction("grad");
            if (grad != null) {
                collectAttached(attached, "layer " + i + " grad", grad);
            }
        }
        assertTrue(attached.isEmpty(), "arrays kept in workspaces:\n" + String.join("\n", attached));
    }
}
