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
import org.nd4j.linalg.activations.impl.ActivationSigmoid;
import org.nd4j.linalg.activations.impl.ActivationTanH;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.distribution.impl.UniformDistribution;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.primitives.Pair;

import java.util.Map;

/**
 * Created by binesh on 6/14/2017.
 */

public class ElementWiseVertexTest extends BaseDL4JTest {
    @Test
    public void testElementWiseVertexNumParams() {
        /*
         * https://github.com/deeplearning4j/deeplearning4j/pull/3514#issuecomment-307754386
         * from @agibsonccc: check for the basics: like 0 numParams
         */

        ElementWiseVertex.Op ops[] = new ElementWiseVertex.Op[] {ElementWiseVertex.Op.Add,
                        ElementWiseVertex.Op.Subtract, ElementWiseVertex.Op.Product};

        for (ElementWiseVertex.Op op : ops) {
            ElementWiseVertex ewv = new ElementWiseVertex(op);
            Assert.assertEquals(0, ewv.numParams(true));
            Assert.assertEquals(0, ewv.numParams(false));
        }
    }

    @Test
    public void testElementWiseVertexForwardAdd() {
        int batchsz = 24;
        int featuresz = 17;
        ComputationGraphConfiguration cgc = new NeuralNetConfiguration.Builder().graphBuilder()
                        .addInputs("input1", "input2", "input3")
                        .addLayer("denselayer",
                                        new DenseLayer.Builder().nIn(featuresz).nOut(1).activation(Activation.IDENTITY)
                                                        .build(),
                                        "input1")
                        /* denselayer is not actually used, but it seems that you _need_ to have trainable parameters, otherwise, you get
                         * Invalid shape: Requested INDArray shape [1, 0] contains dimension size values < 1 (all dimensions must be 1 or more)
                         * at org.nd4j.linalg.factory.Nd4j.checkShapeValues(Nd4j.java:4877)
                         * at org.nd4j.linalg.factory.Nd4j.create(Nd4j.java:4867)
                         * at org.nd4j.linalg.factory.Nd4j.create(Nd4j.java:4820)
                         * at org.nd4j.linalg.factory.Nd4j.create(Nd4j.java:3948)
                         * at org.deeplearning4j.nn.graph.ComputationGraph.init(ComputationGraph.java:409)
                         * at org.deeplearning4j.nn.graph.ComputationGraph.init(ComputationGraph.java:341)
                         */
                        .addVertex("elementwiseAdd", new ElementWiseVertex(ElementWiseVertex.Op.Add), "input1",
                                        "input2", "input3")
                        .addLayer("Add", new ActivationLayer.Builder().activation(Activation.IDENTITY).build(),
                                        "elementwiseAdd")
                        .setOutputs("Add", "denselayer").build();

        ComputationGraph cg = new ComputationGraph(cgc);
        cg.init();


        INDArray input1 = Nd4j.rand(batchsz, featuresz);
        INDArray input2 = Nd4j.rand(batchsz, featuresz);
        INDArray input3 = Nd4j.rand(batchsz, featuresz);

        INDArray target = input1.dup().addi(input2).addi(input3);

        INDArray output = cg.output(input1, input2, input3)[0];
        INDArray squared = output.sub(target);
        double rms = squared.mul(squared).sumNumber().doubleValue();
        Assert.assertEquals(0.0, rms, this.epsilon);
    }

    @Test
    public void testElementWiseVertexForwardProduct() {
        int batchsz = 24;
        int featuresz = 17;
        ComputationGraphConfiguration cgc = new NeuralNetConfiguration.Builder().graphBuilder()
                        .addInputs("input1", "input2", "input3")
                        .addLayer("denselayer",
                                        new DenseLayer.Builder().nIn(featuresz).nOut(1).activation(Activation.IDENTITY)
                                                        .build(),
                                        "input1")
                        /* denselayer is not actually used, but it seems that you _need_ to have trainable parameters, otherwise, you get
                         * Invalid shape: Requested INDArray shape [1, 0] contains dimension size values < 1 (all dimensions must be 1 or more)
                         * at org.nd4j.linalg.factory.Nd4j.checkShapeValues(Nd4j.java:4877)
                         * at org.nd4j.linalg.factory.Nd4j.create(Nd4j.java:4867)
                         * at org.nd4j.linalg.factory.Nd4j.create(Nd4j.java:4820)
                         * at org.nd4j.linalg.factory.Nd4j.create(Nd4j.java:3948)
                         * at org.deeplearning4j.nn.graph.ComputationGraph.init(ComputationGraph.java:409)
                         * at org.deeplearning4j.nn.graph.ComputationGraph.init(ComputationGraph.java:341)
                         */
                        .addVertex("elementwiseProduct", new ElementWiseVertex(ElementWiseVertex.Op.Product), "input1",
                                        "input2", "input3")
                        .addLayer("Product", new ActivationLayer.Builder().activation(Activation.IDENTITY).build(),
                                        "elementwiseProduct")
                        .setOutputs("Product", "denselayer").build();

        ComputationGraph cg = new ComputationGraph(cgc);
        cg.init();


        INDArray input1 = Nd4j.rand(batchsz, featuresz);
        INDArray input2 = Nd4j.rand(batchsz, featuresz);
        INDArray input3 = Nd4j.rand(batchsz, featuresz);

        INDArray target = input1.dup().muli(input2).muli(input3);

        INDArray output = cg.output(input1, input2, input3)[0];
        INDArray squared = output.sub(target);
        double rms = squared.mul(squared).sumNumber().doubleValue();
        Assert.assertEquals(0.0, rms, this.epsilon);
    }

    @Test
    public void testElementWiseVertexForwardSubtract() {
        int batchsz = 24;
        int featuresz = 17;
        ComputationGraphConfiguration cgc = new NeuralNetConfiguration.Builder().graphBuilder()
                        .addInputs("input1", "input2")
                        .addLayer("denselayer",
                                        new DenseLayer.Builder().nIn(featuresz).nOut(1).activation(Activation.IDENTITY)
                                                        .build(),
                                        "input1")
                        /* denselayer is not actually used, but it seems that you _need_ to have trainable parameters, otherwise, you get
                         * Invalid shape: Requested INDArray shape [1, 0] contains dimension size values < 1 (all dimensions must be 1 or more)
                         * at org.nd4j.linalg.factory.Nd4j.checkShapeValues(Nd4j.java:4877)
                         * at org.nd4j.linalg.factory.Nd4j.create(Nd4j.java:4867)
                         * at org.nd4j.linalg.factory.Nd4j.create(Nd4j.java:4820)
                         * at org.nd4j.linalg.factory.Nd4j.create(Nd4j.java:3948)
                         * at org.deeplearning4j.nn.graph.ComputationGraph.init(ComputationGraph.java:409)
                         * at org.deeplearning4j.nn.graph.ComputationGraph.init(ComputationGraph.java:341)
                         */
                        .addVertex("elementwiseSubtract", new ElementWiseVertex(ElementWiseVertex.Op.Subtract),
                                        "input1", "input2")
                        .addLayer("Subtract", new ActivationLayer.Builder().activation(Activation.IDENTITY).build(),
                                        "elementwiseSubtract")
                        .setOutputs("Subtract", "denselayer").build();

        ComputationGraph cg = new ComputationGraph(cgc);
        cg.init();


        INDArray input1 = Nd4j.rand(batchsz, featuresz);
        INDArray input2 = Nd4j.rand(batchsz, featuresz);

        INDArray target = input1.dup().subi(input2);

        INDArray output = cg.output(input1, input2)[0];
        INDArray squared = output.sub(target);
        double rms = Math.sqrt(squared.mul(squared).sumNumber().doubleValue());
        Assert.assertEquals(0.0, rms, this.epsilon);
    }

    @Test
    public void testElementWiseVertexFullAdd() {
        int batchsz = 24;
        int featuresz = 17;
        int midsz = 13;
        int outputsz = 11;
        ComputationGraphConfiguration cgc = new NeuralNetConfiguration.Builder().weightInit(WeightInit.XAVIER)
                        .biasInit(0.0).updater(new Sgd())
                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).graphBuilder()
                        .addInputs("input1", "input2", "input3")
                        .addLayer("dense1",
                                        new DenseLayer.Builder().nIn(featuresz).nOut(midsz)
                                                        .activation(new ActivationTanH()).build(),
                                        "input1")
                        .addLayer("dense2",
                                        new DenseLayer.Builder().nIn(featuresz).nOut(midsz)
                                                        .activation(new ActivationTanH()).build(),
                                        "input2")
                        .addLayer("dense3",
                                        new DenseLayer.Builder().nIn(featuresz).nOut(midsz)
                                                        .activation(new ActivationTanH()).build(),
                                        "input3")
                        .addVertex("elementwiseAdd", new ElementWiseVertex(ElementWiseVertex.Op.Add), "dense1",
                                        "dense2", "dense3")
                        .addLayer("output",
                                        new OutputLayer.Builder().nIn(midsz).nOut(outputsz)
                                                        .activation(new ActivationSigmoid())
                                                        .lossFunction(LossFunction.MSE).build(),
                                        "elementwiseAdd")
                        .setOutputs("output").backprop(true).build();

        ComputationGraph cg = new ComputationGraph(cgc);
        cg.init();
        INDArray input1 = Nd4j.rand(new int[] {batchsz, featuresz}, new UniformDistribution(-1, 1));
        INDArray input2 = Nd4j.rand(new int[] {batchsz, featuresz}, new UniformDistribution(-1, 1));
        INDArray input3 = Nd4j.rand(new int[] {batchsz, featuresz}, new UniformDistribution(-1, 1));
        INDArray target = nullsafe(Nd4j.rand(new int[] {batchsz, outputsz}, new UniformDistribution(0, 1)));
        cg.setInputs(input1, input2, input3);
        cg.setLabels(target);

        cg.computeGradientAndScore();

        // Let's figure out what our params are now.
        Map<String, INDArray> params = cg.paramTable();
        INDArray dense1_W = nullsafe(params.get("dense1_W"));
        INDArray dense1_b = nullsafe(params.get("dense1_b"));
        INDArray dense2_W = nullsafe(params.get("dense2_W"));
        INDArray dense2_b = nullsafe(params.get("dense2_b"));
        INDArray dense3_W = nullsafe(params.get("dense3_W"));
        INDArray dense3_b = nullsafe(params.get("dense3_b"));
        INDArray output_W = nullsafe(params.get("output_W"));
        INDArray output_b = nullsafe(params.get("output_b"));

        // Now, let's calculate what we expect the output to be.

        INDArray mh = input1.mmul(dense1_W).addi(dense1_b.repmat(batchsz, 1));
        INDArray m = (Transforms.tanh(mh));

        INDArray nh = input2.mmul(dense2_W).addi(dense2_b.repmat(batchsz, 1));
        INDArray n = (Transforms.tanh(nh));

        INDArray oh = input3.mmul(dense3_W).addi(dense3_b.repmat(batchsz, 1));
        INDArray o = (Transforms.tanh(oh));

        INDArray middle = Nd4j.zeros(batchsz, midsz);
        middle.addi(m).addi(n).addi(o);


        INDArray expect = Nd4j.zeros(batchsz, outputsz);
        expect.addi(Transforms.sigmoid(middle.mmul(output_W).addi(output_b.repmat(batchsz, 1))));


        INDArray output = nullsafe(cg.output(input1, input2, input3)[0]);

        Assert.assertEquals(0.0, mse(output, expect), this.epsilon);

        Pair<Gradient, Double> pgd = cg.gradientAndScore();

        double score = pgd.getSecond();
        Assert.assertEquals(score, mse(output, target), this.epsilon);

        Map<String, INDArray> gradients = pgd.getFirst().gradientForVariable();
        /*
         * So. Let's say we have inputs a, b, c
         * mh = a W1 + b1
         * m = tanh(mh)
         *
         * nh = b W2 + b2
         * n = tanh(nh)
         *
         * oh = c W3 + b3
         * o = tanh(oh)
         *
         * s = m+n+o
         *
         * yh = s W4 + b4
         * y = sigmoid(yh)
         *
         * E = (y-t)^2
         * dE/dy = 2 (y-t)
         *
         * dy/dyh = y * (1-y)
         * dE/dyh = 2 * y * (1-y) * (y-t)
         *
         * dyh/dW4 = s.transpose()
         * dyh/db4 = Nd4j.ones(1, batchsz)
         * dyh/ds = W4.tranpose()
         *
         * ds/dm = Nd4j.ones(1, midsz)
         *
         * dm/dmh = 1-(m^2)
         *
         * dmh/dW1 = a.transpose()
         * dmh/db1 = Nd4j.ones(1, batchsz)
         *
         */

        INDArray y = output;
        INDArray s = middle;
        INDArray W4 = output_W;

        INDArray dEdy = Nd4j.zeros(target.shape());
        dEdy.addi(y).subi(target).muli(2); // This should be of size batchsz x outputsz
        dEdy.divi(target.shape()[1]); // Why? Because the LossFunction divides by the _element size_ of the output.

        INDArray dydyh = y.mul(y.mul(-1).add(1)); // This is of size batchsz x outputsz
        INDArray dEdyh = dydyh.mul(dEdy);

        INDArray dyhdW4 = s.transpose();
        INDArray dEdW4 = nullsafe(dyhdW4.mmul(dEdyh));

        INDArray dyhdb4 = Nd4j.ones(1, batchsz);
        INDArray dEdb4 = nullsafe(dyhdb4.mmul(dEdyh));

        INDArray dyhds = W4.transpose();
        INDArray dEds = dEdyh.mmul(dyhds);

        INDArray dsdm = Nd4j.ones(batchsz, midsz);
        INDArray dEdm = dsdm.mul(dEds);
        INDArray dmdmh = (m.mul(m)).mul(-1).add(1);
        INDArray dEdmh = dmdmh.mul(dEdm);
        INDArray dmhdW1 = input1.transpose();
        INDArray dEdW1 = nullsafe(dmhdW1.mmul(dEdmh));
        INDArray dmhdb1 = Nd4j.ones(1, batchsz);
        INDArray dEdb1 = nullsafe(dmhdb1.mmul(dEdmh));

        INDArray dsdn = Nd4j.ones(batchsz, midsz);
        INDArray dEdn = dsdn.mul(dEds);
        INDArray dndnh = (n.mul(n)).mul(-1).add(1);
        INDArray dEdnh = dndnh.mul(dEdn);
        INDArray dnhdW2 = input2.transpose();
        INDArray dEdW2 = nullsafe(dnhdW2.mmul(dEdnh));
        INDArray dnhdb2 = Nd4j.ones(1, batchsz);
        INDArray dEdb2 = nullsafe(dnhdb2.mmul(dEdnh));

        INDArray dsdo = Nd4j.ones(batchsz, midsz);
        INDArray dEdo = dsdo.mul(dEds);
        INDArray dodoh = (o.mul(o)).mul(-1).add(1);
        INDArray dEdoh = dodoh.mul(dEdo);
        INDArray dohdW3 = input3.transpose();
        INDArray dEdW3 = nullsafe(dohdW3.mmul(dEdoh));
        INDArray dohdb3 = Nd4j.ones(1, batchsz);
        INDArray dEdb3 = nullsafe(dohdb3.mmul(dEdoh));


        Assert.assertEquals(0, mse(nullsafe(gradients.get("output_W")), dEdW4), this.epsilon);
        Assert.assertEquals(0, mse(nullsafe(gradients.get("output_b")), dEdb4), this.epsilon);
        Assert.assertEquals(0, mse(nullsafe(gradients.get("dense1_W")), dEdW1), this.epsilon);
        Assert.assertEquals(0, mse(nullsafe(gradients.get("dense1_b")), dEdb1), this.epsilon);
        Assert.assertEquals(0, mse(nullsafe(gradients.get("dense2_W")), dEdW2), this.epsilon);
        Assert.assertEquals(0, mse(nullsafe(gradients.get("dense2_b")), dEdb2), this.epsilon);
        Assert.assertEquals(0, mse(nullsafe(gradients.get("dense3_W")), dEdW3), this.epsilon);
        Assert.assertEquals(0, mse(nullsafe(gradients.get("dense3_b")), dEdb3), this.epsilon);
    }

    @Test
    public void testElementWiseVertexFullProduct() {
        int batchsz = 24;
        int featuresz = 17;
        int midsz = 13;
        int outputsz = 11;
        ComputationGraphConfiguration cgc = new NeuralNetConfiguration.Builder().weightInit(WeightInit.XAVIER)
                        .biasInit(0.0).updater(new Sgd())
                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).graphBuilder()
                        .addInputs("input1", "input2", "input3")
                        .addLayer("dense1",
                                        new DenseLayer.Builder().nIn(featuresz).nOut(midsz)
                                                        .activation(new ActivationTanH()).build(),
                                        "input1")
                        .addLayer("dense2",
                                        new DenseLayer.Builder().nIn(featuresz).nOut(midsz)
                                                        .activation(new ActivationTanH()).build(),
                                        "input2")
                        .addLayer("dense3",
                                        new DenseLayer.Builder().nIn(featuresz).nOut(midsz)
                                                        .activation(new ActivationTanH()).build(),
                                        "input3")
                        .addVertex("elementwiseProduct", new ElementWiseVertex(ElementWiseVertex.Op.Product), "dense1",
                                        "dense2", "dense3")
                        .addLayer("output",
                                        new OutputLayer.Builder().nIn(midsz).nOut(outputsz)
                                                        .activation(new ActivationSigmoid())
                                                        .lossFunction(LossFunction.MSE).build(),
                                        "elementwiseProduct")
                        .setOutputs("output").backprop(true).build();

        ComputationGraph cg = new ComputationGraph(cgc);
        cg.init();
        INDArray input1 = Nd4j.rand(new int[] {batchsz, featuresz}, new UniformDistribution(-1, 1));
        INDArray input2 = Nd4j.rand(new int[] {batchsz, featuresz}, new UniformDistribution(-1, 1));
        INDArray input3 = Nd4j.rand(new int[] {batchsz, featuresz}, new UniformDistribution(-1, 1));
        INDArray target = nullsafe(Nd4j.rand(new int[] {batchsz, outputsz}, new UniformDistribution(0, 1)));
        cg.setInputs(input1, input2, input3);
        cg.setLabels(target);

        cg.computeGradientAndScore();

        // Let's figure out what our params are now.
        Map<String, INDArray> params = cg.paramTable();
        INDArray dense1_W = nullsafe(params.get("dense1_W"));
        INDArray dense1_b = nullsafe(params.get("dense1_b"));
        INDArray dense2_W = nullsafe(params.get("dense2_W"));
        INDArray dense2_b = nullsafe(params.get("dense2_b"));
        INDArray dense3_W = nullsafe(params.get("dense3_W"));
        INDArray dense3_b = nullsafe(params.get("dense3_b"));
        INDArray output_W = nullsafe(params.get("output_W"));
        INDArray output_b = nullsafe(params.get("output_b"));

        // Now, let's calculate what we expect the output to be.

        INDArray mh = input1.mmul(dense1_W).addi(dense1_b.repmat(batchsz, 1));
        INDArray m = (Transforms.tanh(mh));

        INDArray nh = input2.mmul(dense2_W).addi(dense2_b.repmat(batchsz, 1));
        INDArray n = (Transforms.tanh(nh));

        INDArray oh = input3.mmul(dense3_W).addi(dense3_b.repmat(batchsz, 1));
        INDArray o = (Transforms.tanh(oh));

        INDArray middle = Nd4j.ones(batchsz, midsz);
        middle.muli(m).muli(n).muli(o);


        INDArray expect = Nd4j.zeros(batchsz, outputsz);
        expect.addi(Transforms.sigmoid(middle.mmul(output_W).addi(output_b.repmat(batchsz, 1))));


        INDArray output = nullsafe(cg.output(input1, input2, input3)[0]);

        Assert.assertEquals(0.0, mse(output, expect), this.epsilon);

        Pair<Gradient, Double> pgd = cg.gradientAndScore();

        double score = pgd.getSecond();
        Assert.assertEquals(score, mse(output, target), this.epsilon);

        Map<String, INDArray> gradients = pgd.getFirst().gradientForVariable();
        /*
         * So. Let's say we have inputs a, b, c
         * mh = a W1 + b1
         * m = tanh(mh)
         *
         * nh = b W2 + b2
         * n = tanh(nh)
         *
         * oh = c W3 + b3
         * o = tanh(oh)
         *
         * s = m*n*o
         *
         * yh = s W4 + b4
         * y = sigmoid(yh)
         *
         * E = (y-t)^2
         * dE/dy = 2 (y-t)
         *
         * dy/dyh = y * (1-y)
         * dE/dyh = 2 * y * (1-y) * (y-t)
         *
         * dyh/dW4 = s.transpose()
         * dyh/db4 = Nd4j.ones(1, batchsz)
         * dyh/ds = W4.tranpose()
         *
         * ds/dm = Nd4j.ones(1, midsz).mul(o).mul(n) // Basically the _rest_ of the middle layers
         *
         * dm/dmh = 1-(m^2)
         *
         * dmh/dW1 = a.transpose()
         * dmh/db1 = Nd4j.ones(1, batchsz)
         *
         */

        INDArray y = output;
        INDArray s = middle;
        INDArray W4 = output_W;

        INDArray dEdy = Nd4j.zeros(target.shape());
        dEdy.addi(y).subi(target).muli(2); // This should be of size batchsz x outputsz
        dEdy.divi(target.shape()[1]); // Why? Because the LossFunction divides by the _element size_ of the output.

        INDArray dydyh = y.mul(y.mul(-1).add(1)); // This is of size batchsz x outputsz
        INDArray dEdyh = dydyh.mul(dEdy);

        INDArray dyhdW4 = s.transpose();
        INDArray dEdW4 = nullsafe(dyhdW4.mmul(dEdyh));

        INDArray dyhdb4 = Nd4j.ones(1, batchsz);
        INDArray dEdb4 = nullsafe(dyhdb4.mmul(dEdyh));

        INDArray dyhds = W4.transpose();
        INDArray dEds = dEdyh.mmul(dyhds);

        INDArray dsdm = Nd4j.ones(batchsz, midsz).muli(n).muli(o);
        INDArray dEdm = dsdm.mul(dEds);
        INDArray dmdmh = (m.mul(m)).mul(-1).add(1);
        INDArray dEdmh = dmdmh.mul(dEdm);
        INDArray dmhdW1 = input1.transpose();
        INDArray dEdW1 = nullsafe(dmhdW1.mmul(dEdmh));
        INDArray dmhdb1 = Nd4j.ones(1, batchsz);
        INDArray dEdb1 = nullsafe(dmhdb1.mmul(dEdmh));

        INDArray dsdn = Nd4j.ones(batchsz, midsz).muli(m).muli(o);
        INDArray dEdn = dsdn.mul(dEds);
        INDArray dndnh = (n.mul(n)).mul(-1).add(1);
        INDArray dEdnh = dndnh.mul(dEdn);
        INDArray dnhdW2 = input2.transpose();
        INDArray dEdW2 = nullsafe(dnhdW2.mmul(dEdnh));
        INDArray dnhdb2 = Nd4j.ones(1, batchsz);
        INDArray dEdb2 = nullsafe(dnhdb2.mmul(dEdnh));

        INDArray dsdo = Nd4j.ones(batchsz, midsz).muli(m).muli(n);
        INDArray dEdo = dsdo.mul(dEds);
        INDArray dodoh = (o.mul(o)).mul(-1).add(1);
        INDArray dEdoh = dodoh.mul(dEdo);
        INDArray dohdW3 = input3.transpose();
        INDArray dEdW3 = nullsafe(dohdW3.mmul(dEdoh));
        INDArray dohdb3 = Nd4j.ones(1, batchsz);
        INDArray dEdb3 = nullsafe(dohdb3.mmul(dEdoh));

        Assert.assertEquals(0, mse(nullsafe(gradients.get("output_W")), dEdW4), this.epsilon);
        Assert.assertEquals(0, mse(nullsafe(gradients.get("output_b")), dEdb4), this.epsilon);
        Assert.assertEquals(0, mse(nullsafe(gradients.get("dense1_W")), dEdW1), this.epsilon);
        Assert.assertEquals(0, mse(nullsafe(gradients.get("dense1_b")), dEdb1), this.epsilon);
        Assert.assertEquals(0, mse(nullsafe(gradients.get("dense2_W")), dEdW2), this.epsilon);
        Assert.assertEquals(0, mse(nullsafe(gradients.get("dense2_b")), dEdb2), this.epsilon);
        Assert.assertEquals(0, mse(nullsafe(gradients.get("dense3_W")), dEdW3), this.epsilon);
        Assert.assertEquals(0, mse(nullsafe(gradients.get("dense3_b")), dEdb3), this.epsilon);
    }

    @Test
    public void testElementWiseVertexFullSubtract() {
        int batchsz = 24;
        int featuresz = 17;
        int midsz = 13;
        int outputsz = 11;
        ComputationGraphConfiguration cgc = new NeuralNetConfiguration.Builder().weightInit(WeightInit.XAVIER)
                        .biasInit(0.0).updater(new Sgd())
                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).graphBuilder()
                        .addInputs("input1", "input2")
                        .addLayer("dense1",
                                        new DenseLayer.Builder().nIn(featuresz).nOut(midsz)
                                                        .activation(new ActivationTanH()).build(),
                                        "input1")
                        .addLayer("dense2",
                                        new DenseLayer.Builder().nIn(featuresz).nOut(midsz)
                                                        .activation(new ActivationTanH()).build(),
                                        "input2")
                        .addVertex("elementwiseSubtract", new ElementWiseVertex(ElementWiseVertex.Op.Subtract),
                                        "dense1", "dense2")
                        .addLayer("output",
                                        new OutputLayer.Builder().nIn(midsz).nOut(outputsz)
                                                        .activation(new ActivationSigmoid())
                                                        .lossFunction(LossFunction.MSE).build(),
                                        "elementwiseSubtract")
                        .setOutputs("output").backprop(true).build();

        ComputationGraph cg = new ComputationGraph(cgc);
        cg.init();
        INDArray input1 = Nd4j.rand(new int[] {batchsz, featuresz}, new UniformDistribution(-1, 1));
        INDArray input2 = Nd4j.rand(new int[] {batchsz, featuresz}, new UniformDistribution(-1, 1));
        INDArray target = nullsafe(Nd4j.rand(new int[] {batchsz, outputsz}, new UniformDistribution(0, 1)));
        cg.setInputs(input1, input2);
        cg.setLabels(target);

        cg.computeGradientAndScore();

        // Let's figure out what our params are now.
        Map<String, INDArray> params = cg.paramTable();
        INDArray dense1_W = nullsafe(params.get("dense1_W"));
        INDArray dense1_b = nullsafe(params.get("dense1_b"));
        INDArray dense2_W = nullsafe(params.get("dense2_W"));
        INDArray dense2_b = nullsafe(params.get("dense2_b"));
        INDArray output_W = nullsafe(params.get("output_W"));
        INDArray output_b = nullsafe(params.get("output_b"));

        // Now, let's calculate what we expect the output to be.

        INDArray mh = input1.mmul(dense1_W).addi(dense1_b.repmat(batchsz, 1));
        INDArray m = (Transforms.tanh(mh));

        INDArray nh = input2.mmul(dense2_W).addi(dense2_b.repmat(batchsz, 1));
        INDArray n = (Transforms.tanh(nh));

        INDArray middle = Nd4j.zeros(batchsz, midsz);
        middle.addi(m).subi(n);


        INDArray expect = Nd4j.zeros(batchsz, outputsz);
        expect.addi(Transforms.sigmoid(middle.mmul(output_W).addi(output_b.repmat(batchsz, 1))));


        INDArray output = nullsafe(cg.output(input1, input2)[0]);

        Assert.assertEquals(0.0, mse(output, expect), this.epsilon);

        Pair<Gradient, Double> pgd = cg.gradientAndScore();

        double score = pgd.getSecond();
        Assert.assertEquals(score, mse(output, target), this.epsilon);

        Map<String, INDArray> gradients = pgd.getFirst().gradientForVariable();
        /*
         * So. Let's say we have inputs a, b, c
         * mh = a W1 + b1
         * m = tanh(mh)
         *
         * nh = b W2 + b2
         * n = tanh(nh)
         *
         * s = m-n
         *
         * yh = s W4 + b4
         * y = sigmoid(yh)
         *
         * E = (y-t)^2
         * dE/dy = 2 (y-t)
         *
         * dy/dyh = y * (1-y)
         * dE/dyh = 2 * y * (1-y) * (y-t)
         *
         * dyh/dW4 = s.transpose()
         * dyh/db4 = Nd4j.ones(1, batchsz)
         * dyh/ds = W4.tranpose()
         *
         * ds/dm = Nd4j.ones(1, midsz)
         * ds/dn = Nd4j.ones(1, midsz).muli(-1)
         *
         * dm/dmh = 1-(m^2)
         *
         * dmh/dW1 = a.transpose()
         * dmh/db1 = Nd4j.ones(1, batchsz)
         *
         */

        INDArray y = output;
        INDArray s = middle;
        INDArray W4 = output_W;

        INDArray dEdy = Nd4j.zeros(target.shape());
        dEdy.addi(y).subi(target).muli(2); // This should be of size batchsz x outputsz
        dEdy.divi(target.shape()[1]); // Why? Because the LossFunction divides by the _element size_ of the output.

        INDArray dydyh = y.mul(y.mul(-1).add(1)); // This is of size batchsz x outputsz
        INDArray dEdyh = dydyh.mul(dEdy);

        INDArray dyhdW4 = s.transpose();
        INDArray dEdW4 = nullsafe(dyhdW4.mmul(dEdyh));

        INDArray dyhdb4 = Nd4j.ones(1, batchsz);
        INDArray dEdb4 = nullsafe(dyhdb4.mmul(dEdyh));

        INDArray dyhds = W4.transpose();
        INDArray dEds = dEdyh.mmul(dyhds);

        INDArray dsdm = Nd4j.ones(batchsz, midsz);
        INDArray dEdm = dsdm.mul(dEds);
        INDArray dmdmh = (m.mul(m)).mul(-1).add(1);
        INDArray dEdmh = dmdmh.mul(dEdm);
        INDArray dmhdW1 = input1.transpose();
        INDArray dEdW1 = nullsafe(dmhdW1.mmul(dEdmh));
        INDArray dmhdb1 = Nd4j.ones(1, batchsz);
        INDArray dEdb1 = nullsafe(dmhdb1.mmul(dEdmh));

        INDArray dsdn = Nd4j.ones(batchsz, midsz).muli(-1);
        INDArray dEdn = dsdn.mul(dEds);
        INDArray dndnh = (n.mul(n)).mul(-1).add(1);
        INDArray dEdnh = dndnh.mul(dEdn);
        INDArray dnhdW2 = input2.transpose();
        INDArray dEdW2 = nullsafe(dnhdW2.mmul(dEdnh));
        INDArray dnhdb2 = Nd4j.ones(1, batchsz);
        INDArray dEdb2 = nullsafe(dnhdb2.mmul(dEdnh));


        Assert.assertEquals(0, mse(nullsafe(gradients.get("output_W")), dEdW4), this.epsilon);
        Assert.assertEquals(0, mse(nullsafe(gradients.get("output_b")), dEdb4), this.epsilon);
        Assert.assertEquals(0, mse(nullsafe(gradients.get("dense1_W")), dEdW1), this.epsilon);
        Assert.assertEquals(0, mse(nullsafe(gradients.get("dense1_b")), dEdb1), this.epsilon);
        Assert.assertEquals(0, mse(nullsafe(gradients.get("dense2_W")), dEdW2), this.epsilon);
        Assert.assertEquals(0, mse(nullsafe(gradients.get("dense2_b")), dEdb2), this.epsilon);
    }

    private static double mse(INDArray output, INDArray target) {
        double mse_expect = Transforms.pow(output.sub(target), 2.0).sumNumber().doubleValue()
                        / (output.columns() * output.rows());
        return mse_expect;
    }

    private static <T> T nullsafe(T obj) {
        if (obj == null)
            throw new NullPointerException();
        T clean = obj;
        return clean;
    }

    private double epsilon = 1e-10;
}
