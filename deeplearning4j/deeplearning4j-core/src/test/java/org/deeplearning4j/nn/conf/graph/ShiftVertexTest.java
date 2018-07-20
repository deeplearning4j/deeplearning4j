/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
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

package org.deeplearning4j.nn.conf.graph;

import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.ActivationLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Assert;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.activations.BaseActivationFunction;
import org.nd4j.linalg.activations.impl.ActivationSigmoid;
import org.nd4j.linalg.activations.impl.ActivationTanH;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.nd4j.linalg.primitives.Pair;

import java.util.Map;
import java.util.TreeMap;

/**
 * Created by binesh on 6/13/2017. 
 */

public class ShiftVertexTest extends BaseDL4JTest {
    @Test
    public void testShiftVertexNumParamsTrue() {
        /*
         * https://github.com/deeplearning4j/deeplearning4j/pull/3514#issuecomment-307754386
         * from @agibsonccc: check for the basics: like 0 numParams
         */

        ShiftVertex sv = new ShiftVertex(0.7); // The 0.7 doesn't really matter.
        Assert.assertEquals(0, sv.numParams(true));
    }

    @Test
    public void testShiftVertexNumParamsFalse() {
        /*
         * https://github.com/deeplearning4j/deeplearning4j/pull/3514#issuecomment-307754386
         * from @agibsonccc: check for the basics: like 0 numParams
         */

        ShiftVertex sv = new ShiftVertex(0.7); // The 0.7 doesn't really matter.
        Assert.assertEquals(0, sv.numParams(false));
    }

    @Test
    public void testGet() {
        ShiftVertex sv = new ShiftVertex(0.7);
        Assert.assertEquals(0.7, sv.getShiftFactor(), this.epsilon);
    }

    @Test
    public void testSimple() {
        /*
         * This function _simply_ tests whether ShiftVertex is _in fact_ adding the shift value to it's inputs.
         */
        // Just first n primes / 10.
        INDArray input = Nd4j
                        .create(new double[][] {{0.2, 0.3, 0.5}, {0.7, 1.1, 1.3}, {1.7, 1.9, 2.3}, {2.9, 3.1, 3.7}});
        double sf = 4.1;
        ComputationGraphConfiguration cgc = new NeuralNetConfiguration.Builder().graphBuilder().addInputs("input")
                        .addLayer("denselayer",
                                        new DenseLayer.Builder().nIn(input.columns()).nOut(1)
                                                        .activation(Activation.IDENTITY).build(),
                                        "input")
                        /* denselayer is not actually used, but it seems that you _need_ to have trainable parameters, otherwise, you get
                         * Invalid shape: Requested INDArray shape [1, 0] contains dimension size values < 1 (all dimensions must be 1 or more)
                         * at org.nd4j.linalg.factory.Nd4j.checkShapeValues(Nd4j.java:4877)
                         * at org.nd4j.linalg.factory.Nd4j.create(Nd4j.java:4867)
                         * at org.nd4j.linalg.factory.Nd4j.create(Nd4j.java:4820)
                         * at org.nd4j.linalg.factory.Nd4j.create(Nd4j.java:3948)
                         * at org.deeplearning4j.nn.graph.ComputationGraph.init(ComputationGraph.java:409)
                         * at org.deeplearning4j.nn.graph.ComputationGraph.init(ComputationGraph.java:341)
                         */
                        .addLayer("identityinputactivation",
                                        new ActivationLayer.Builder().activation(Activation.IDENTITY).build(), "input")
                        .addVertex("shiftvertex", new ShiftVertex(sf), "identityinputactivation")
                        .addLayer("identityshiftvertex",
                                        new ActivationLayer.Builder().activation(Activation.IDENTITY).build(),
                                        "shiftvertex")
                        .setOutputs("identityshiftvertex", "denselayer").build();

        ComputationGraph cg = new ComputationGraph(cgc);
        cg.init();

        // We can call outputSingle, because we only have a single output layer. It has nothing to do with minibatches.
        INDArray output = cg.output(true, input)[0];
        INDArray target = Nd4j.zeros(input.shape());
        target.addi(input);
        target.addi(sf);

        INDArray squared = output.sub(target);
        double rms = squared.mul(squared).sumNumber().doubleValue();
        Assert.assertEquals(0.0, rms, this.epsilon);
    }

    @Test
    public void testComprehensive() {
        /*
         * This function tests ShiftVertex more comprehensively. Specifically, it verifies that the lossfunction works as
         * expected on a ComputationGraph _with_ a ShiftVertex and it verifies that the derivatives produced by 
         * back propagating work as expected.
         */
        BaseActivationFunction a1 = new ActivationTanH();
        BaseActivationFunction a2 = new ActivationSigmoid();
        // Just first n primes / 10.
        INDArray input = Nd4j
                        .create(new double[][] {{0.2, 0.3, 0.5}, {0.7, 1.1, 1.3}, {1.7, 1.9, 2.3}, {2.9, 3.1, 3.7}});
        double sf = 4.1;
        // Actually, given that I'm using a sigmoid on the output,
        // these should really be between 0 and 1
        INDArray target = Nd4j.create(new double[][] {{0.05, 0.10, 0.15, 0.20, 0.25}, {0.30, 0.35, 0.40, 0.45, 0.50},
                        {0.55, 0.60, 0.65, 0.70, 0.75}, {0.80, 0.85, 0.90, 0.95, 0.99}});

        ComputationGraphConfiguration cgc = new NeuralNetConfiguration.Builder().weightInit(WeightInit.XAVIER)
                        .updater(new Sgd(0.01))
                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).graphBuilder()
                        .addInputs("input")
                        .addLayer("denselayer",
                                        new DenseLayer.Builder().nIn(input.columns()).nOut(input.columns())
                                                        .activation(a1).build(),
                                        "input")
                        .addVertex("shiftvertex", new ShiftVertex(sf), "denselayer")
                        .addLayer("output",
                                        new OutputLayer.Builder().nIn(input.columns()).nOut(target.columns())
                                                        .activation(a2).lossFunction(LossFunction.MSE).build(),
                                        "shiftvertex")
                        .setOutputs("output").backprop(true).build();
        ComputationGraph cg = new ComputationGraph(cgc);
        cg.init();
        cg.setInput(0, input);
        cg.setLabel(0, target);
        cg.computeGradientAndScore();
        double score_dl4j = cg.score();
        Map<String, INDArray> weights = cg.paramTable();
        Gradient g = cg.gradient();
        Map<String, INDArray> gradients = g.gradientForVariable();
        Map<String, INDArray> manual_gradients = new TreeMap<String, INDArray>();

        INDArray W = nullsafe(weights.get("denselayer_W"));
        INDArray b = nullsafe(weights.get("denselayer_b"));
        INDArray V = nullsafe(weights.get("output_W"));
        INDArray c = nullsafe(weights.get("output_b"));

        Map<String, INDArray> manual_weights = new TreeMap<String, INDArray>();
        manual_weights.put("denselayer_W", W);
        manual_weights.put("denselayer_b", b);
        manual_weights.put("output_W", V);
        manual_weights.put("output_b", c);

        // First things first, let's calculate the score.
        // FIXME: int cast
        int batchsz = (int) input.shape()[0];
        INDArray z = input.mmul(W).add(b.repmat(batchsz, 1));
        INDArray a = a1.getActivation(z.dup(), true).add(sf); // activation modifies it's input!!
        INDArray q = a.mmul(V).add(c.repmat(batchsz, 1));
        INDArray o = nullsafe(a2.getActivation(q.dup(), true));
        double score_manual = sum_errors(o, target) / (o.columns() * o.rows());

        /*
         * So. We have
         * z5 = input1 * W15 + input2 * W25 + input3 * W35 + b5
         * a5 = activation(z5) + sr
         * q9 = a1 * V19 + a2 * V29 + a3 * V39 + c9
         * o9 = activation(q9)
         *  
         * dE/do = 2(o-t)
         * doj/dqj = activation'(qj)
         * dqj/dVij = ai dqj/dai = Vij dqj/dbj = 1
         * 
         * dq1/dv11 = a1 dq2/dV12 = a1 dq3/dV13 = a1 ...
         * dq1/dv21 = a2 dq2...
         */
        INDArray dEdo = Nd4j.zeros(target.shape());
        dEdo.addi(o).subi(target).muli(2); // This should be of size batchsz x outputsz
        dEdo.divi(target.shape()[1]); // Why? Because the LossFunction divides by the _element size_ of the output.

        Pair<INDArray, INDArray> derivs2 = a2.backprop(q, dEdo);
        INDArray dEdq = derivs2.getFirst(); // This should be of size batchsz x outputsz (dE/do * do/dq) this _should_ be o * (1-o) * dE/do for Sigmoid.
        // Should be o = q^3 do/dq = 3 q^2 for Cube.
        /*
        INDArray dodq = q.mul(q).mul(3);
        INDArray tbv = dodq.mul(dEdo);
        System.err.println("----");
        System.err.println(q);
        System.err.println(o);
        System.err.println(tbv);
        System.err.println(dEdq);
        */

        INDArray dqdc = Nd4j.ones(1, batchsz);
        INDArray dEdc = dqdc.mmul(dEdq); // This should be of size 1 x outputsz
        INDArray dEdV = a.transpose().mmul(dEdq);
        INDArray dEda = dEdq.mmul(V.transpose()); // This should be dEdo * dodq * dqda

        Pair<INDArray, INDArray> derivs1 = a1.backprop(z, dEda);
        INDArray dEdz = derivs1.getFirst();
        INDArray dzdb = Nd4j.ones(1, batchsz);
        INDArray dEdb = dzdb.mmul(dEdz);
        INDArray dEdW = input.transpose().mmul(dEdz);

        manual_gradients.put("output_b", dEdc);
        manual_gradients.put("output_W", dEdV);
        manual_gradients.put("denselayer_b", dEdb);
        manual_gradients.put("denselayer_W", dEdW);

        double summse = Math.pow((score_manual - score_dl4j), 2);
        int denominator = 1;

        for (Map.Entry<String, INDArray> mesi : gradients.entrySet()) {
            String name = mesi.getKey();
            INDArray dl4j_gradient = nullsafe(mesi.getValue());
            INDArray manual_gradient = nullsafe(manual_gradients.get(name));
            double se = sum_errors(dl4j_gradient, manual_gradient);
            summse += se;
            denominator += dl4j_gradient.columns() * dl4j_gradient.rows();
        }

        Assert.assertEquals(0.0, summse / denominator, this.epsilon);

    }

    private static double sum_errors(INDArray a, INDArray b) {
        INDArray o = a.sub(b);
        return o.mul(o).sumNumber().doubleValue();
    }

    private static <T> T nullsafe(T obj) {
        if (obj == null)
            throw new NullPointerException();
        T clean = obj;
        return clean;
    }

    private double epsilon = 1e-10;
}
