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
package org.eclipse.deeplearning4j.dl4jcore.nn.conf.graph;

import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.gradientcheck.GradientCheckUtil;
import org.deeplearning4j.gradientcheck.GraphConfig;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.graph.ElementWiseVertex;
import org.deeplearning4j.nn.conf.layers.ActivationLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.activations.impl.ActivationSigmoid;
import org.nd4j.linalg.activations.impl.ActivationTanH;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.distribution.impl.UniformDistribution;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.NoOp;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.common.primitives.Pair;
import java.util.Map;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import org.junit.jupiter.api.DisplayName;

@DisplayName("Element Wise Vertex Test")
@NativeTag
@Tag(TagNames.DL4J_OLD_API)
class ElementWiseVertexTest extends BaseDL4JTest {

    @Test
    @DisplayName("Test Element Wise Vertex Num Params")
    void testElementWiseVertexNumParams() {
        /*
         * https://github.com/eclipse/deeplearning4j/pull/3514#issuecomment-307754386
         * from @agibsonccc: check for the basics: like 0 numParams
         */
        ElementWiseVertex.Op[] ops = new ElementWiseVertex.Op[] { ElementWiseVertex.Op.Add, ElementWiseVertex.Op.Subtract, ElementWiseVertex.Op.Product };
        for (ElementWiseVertex.Op op : ops) {
            ElementWiseVertex ewv = new ElementWiseVertex(op);
            Assertions.assertEquals(0, ewv.numParams(true));
            Assertions.assertEquals(0, ewv.numParams(false));
        }
    }

    @Test
    @DisplayName("Test Element Wise Vertex Forward Add")
    void testElementWiseVertexForwardAdd() {
        int batchsz = 24;
        int featuresz = 17;
        ComputationGraphConfiguration cgc = new NeuralNetConfiguration.Builder().graphBuilder().addInputs("input1", "input2", "input3").addLayer("denselayer", new DenseLayer.Builder().nIn(featuresz).nOut(1).activation(Activation.IDENTITY).build(), "input1").addVertex("elementwiseAdd", new ElementWiseVertex(ElementWiseVertex.Op.Add), "input1", "input2", "input3").addLayer("Add", new ActivationLayer.Builder().activation(Activation.IDENTITY).build(), "elementwiseAdd").setOutputs("Add", "denselayer").build();
        ComputationGraph cg = new ComputationGraph(cgc);
        cg.init();
        INDArray input1 = Nd4j.rand(batchsz, featuresz);
        INDArray input2 = Nd4j.rand(batchsz, featuresz);
        INDArray input3 = Nd4j.rand(batchsz, featuresz);
        INDArray target = input1.dup().addi(input2).addi(input3);
        INDArray output = cg.output(input1, input2, input3)[0];
        INDArray squared = output.sub(target.castTo(output.dataType()));
        double rms = squared.mul(squared).sumNumber().doubleValue();
        Assertions.assertEquals(0.0, rms, this.epsilon);
    }

    @Test
    @DisplayName("Test Element Wise Vertex Forward Product")
    void testElementWiseVertexForwardProduct() {
        int batchsz = 24;
        int featuresz = 17;
        ComputationGraphConfiguration cgc = new NeuralNetConfiguration.Builder().graphBuilder().addInputs("input1", "input2", "input3").addLayer("denselayer", new DenseLayer.Builder().nIn(featuresz).nOut(1).activation(Activation.IDENTITY).build(), "input1").addVertex("elementwiseProduct", new ElementWiseVertex(ElementWiseVertex.Op.Product), "input1", "input2", "input3").addLayer("Product", new ActivationLayer.Builder().activation(Activation.IDENTITY).build(), "elementwiseProduct").setOutputs("Product", "denselayer").build();
        ComputationGraph cg = new ComputationGraph(cgc);
        cg.init();
        INDArray input1 = Nd4j.rand(batchsz, featuresz);
        INDArray input2 = Nd4j.rand(batchsz, featuresz);
        INDArray input3 = Nd4j.rand(batchsz, featuresz);
        INDArray target = input1.dup().muli(input2).muli(input3);
        INDArray output = cg.output(input1, input2, input3)[0];
        INDArray squared = output.sub(target.castTo(output.dataType()));
        double rms = squared.mul(squared).sumNumber().doubleValue();
        Assertions.assertEquals(0.0, rms, this.epsilon);
    }

    @Test
    @DisplayName("Test Element Wise Vertex Forward Subtract")
    void testElementWiseVertexForwardSubtract() {
        int batchsz = 24;
        int featuresz = 17;
        ComputationGraphConfiguration cgc = new NeuralNetConfiguration.Builder().graphBuilder().addInputs("input1", "input2").addLayer("denselayer", new DenseLayer.Builder().nIn(featuresz).nOut(1).activation(Activation.IDENTITY).build(), "input1").addVertex("elementwiseSubtract", new ElementWiseVertex(ElementWiseVertex.Op.Subtract), "input1", "input2").addLayer("Subtract", new ActivationLayer.Builder().activation(Activation.IDENTITY).build(), "elementwiseSubtract").setOutputs("Subtract", "denselayer").build();
        ComputationGraph cg = new ComputationGraph(cgc);
        cg.init();
        INDArray input1 = Nd4j.rand(batchsz, featuresz);
        INDArray input2 = Nd4j.rand(batchsz, featuresz);
        INDArray target = input1.dup().subi(input2);
        INDArray output = cg.output(input1, input2)[0];
        INDArray squared = output.sub(target);
        double rms = Math.sqrt(squared.mul(squared).sumNumber().doubleValue());
        Assertions.assertEquals(0.0, rms, this.epsilon);
    }

    @Test
    @DisplayName("Test Element Wise Vertex Full Add")
    void testElementWiseVertexFullAdd() {
        int batchsz = 24;
        int featuresz = 17;
        int midsz = 13;
        int outputsz = 11;
        ComputationGraphConfiguration cgc = new NeuralNetConfiguration.Builder().weightInit(WeightInit.XAVIER).dataType(DataType.DOUBLE).biasInit(0.0).updater(new NoOp()).optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).graphBuilder().addInputs("input1", "input2", "input3").addLayer("dense1", new DenseLayer.Builder().nIn(featuresz).nOut(midsz).activation(new ActivationTanH()).build(), "input1").addLayer("dense2", new DenseLayer.Builder().nIn(featuresz).nOut(midsz).activation(new ActivationTanH()).build(), "input2").addLayer("dense3", new DenseLayer.Builder().nIn(featuresz).nOut(midsz).activation(new ActivationTanH()).build(), "input3").addVertex("elementwiseAdd", new ElementWiseVertex(ElementWiseVertex.Op.Add), "dense1", "dense2", "dense3").addLayer("output", new OutputLayer.Builder().nIn(midsz).nOut(outputsz).activation(new ActivationSigmoid()).lossFunction(LossFunction.MSE).build(), "elementwiseAdd").setOutputs("output").build();
        ComputationGraph cg = new ComputationGraph(cgc);
        cg.init();
        INDArray input1 = Nd4j.rand(new int[] { batchsz, featuresz }, new UniformDistribution(-1, 1)).castTo(DataType.DOUBLE);
        INDArray input2 = Nd4j.rand(new int[] { batchsz, featuresz }, new UniformDistribution(-1, 1)).castTo(DataType.DOUBLE);
        INDArray input3 = Nd4j.rand(new int[] { batchsz, featuresz }, new UniformDistribution(-1, 1)).castTo(DataType.DOUBLE);
        INDArray target = nullsafe(Nd4j.rand(new int[] { batchsz, outputsz }, new UniformDistribution(0, 1)).castTo(DataType.DOUBLE));
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
        INDArray middle = Nd4j.zeros(DataType.DOUBLE, batchsz, midsz);
        middle.addi(m).addi(n).addi(o);
        INDArray expect = Nd4j.zeros(DataType.DOUBLE, batchsz, outputsz);
        expect.addi(Transforms.sigmoid(middle.mmul(output_W).addi(output_b.repmat(batchsz, 1))));
        INDArray output = nullsafe(cg.output(input1, input2, input3)[0]);
        Assertions.assertEquals(0.0, mse(output, expect), this.epsilon);
        Pair<Gradient, Double> pgd = cg.gradientAndScore();
        double score = pgd.getSecond();
        Assertions.assertEquals(score, mse(output, target), this.epsilon);
        Map<String, INDArray> gradients = pgd.getFirst().gradientForVariable();
        assertGradientCheckPasses(cg, new INDArray[] { input1, input2, input3 }, target);
    }

    @Test
    @DisplayName("Test Element Wise Vertex Full Product")
    void testElementWiseVertexFullProduct() {
        int batchsz = 24;
        int featuresz = 17;
        int midsz = 13;
        int outputsz = 11;
        ComputationGraphConfiguration cgc = new NeuralNetConfiguration.Builder().weightInit(WeightInit.XAVIER).dataType(DataType.DOUBLE).biasInit(0.0).updater(new NoOp()).optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).graphBuilder().addInputs("input1", "input2", "input3").addLayer("dense1", new DenseLayer.Builder().nIn(featuresz).nOut(midsz).activation(new ActivationTanH()).build(), "input1").addLayer("dense2", new DenseLayer.Builder().nIn(featuresz).nOut(midsz).activation(new ActivationTanH()).build(), "input2").addLayer("dense3", new DenseLayer.Builder().nIn(featuresz).nOut(midsz).activation(new ActivationTanH()).build(), "input3").addVertex("elementwiseProduct", new ElementWiseVertex(ElementWiseVertex.Op.Product), "dense1", "dense2", "dense3").addLayer("output", new OutputLayer.Builder().nIn(midsz).nOut(outputsz).activation(new ActivationSigmoid()).lossFunction(LossFunction.MSE).build(), "elementwiseProduct").setOutputs("output").build();
        ComputationGraph cg = new ComputationGraph(cgc);
        cg.init();
        INDArray input1 = Nd4j.rand(new int[] { batchsz, featuresz }, new UniformDistribution(-1, 1)).castTo(DataType.DOUBLE);
        INDArray input2 = Nd4j.rand(new int[] { batchsz, featuresz }, new UniformDistribution(-1, 1)).castTo(DataType.DOUBLE);
        INDArray input3 = Nd4j.rand(new int[] { batchsz, featuresz }, new UniformDistribution(-1, 1)).castTo(DataType.DOUBLE);
        INDArray target = nullsafe(Nd4j.rand(new int[] { batchsz, outputsz }, new UniformDistribution(0, 1)).castTo(DataType.DOUBLE));
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
        INDArray middle = Nd4j.ones(DataType.DOUBLE, batchsz, midsz);
        middle.muli(m).muli(n).muli(o);
        INDArray expect = Nd4j.zeros(DataType.DOUBLE, batchsz, outputsz);
        expect.addi(Transforms.sigmoid(middle.mmul(output_W).addi(output_b.repmat(batchsz, 1))));
        INDArray output = nullsafe(cg.output(input1, input2, input3)[0]);
        Assertions.assertEquals(0.0, mse(output, expect), this.epsilon);
        Pair<Gradient, Double> pgd = cg.gradientAndScore();
        double score = pgd.getSecond();
        Assertions.assertEquals(score, mse(output, target), this.epsilon);
        Map<String, INDArray> gradients = pgd.getFirst().gradientForVariable();
        assertGradientCheckPasses(cg, new INDArray[] { input1, input2, input3 }, target);
    }

    @Test
    @DisplayName("Test Element Wise Vertex Full Subtract")
    void testElementWiseVertexFullSubtract() {
        int batchsz = 24;
        int featuresz = 17;
        int midsz = 13;
        int outputsz = 11;
        ComputationGraphConfiguration cgc = new NeuralNetConfiguration.Builder().weightInit(WeightInit.XAVIER).dataType(DataType.DOUBLE).biasInit(0.0).updater(new NoOp()).optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).graphBuilder().addInputs("input1", "input2").addLayer("dense1", new DenseLayer.Builder().nIn(featuresz).nOut(midsz).activation(new ActivationTanH()).build(), "input1").addLayer("dense2", new DenseLayer.Builder().nIn(featuresz).nOut(midsz).activation(new ActivationTanH()).build(), "input2").addVertex("elementwiseSubtract", new ElementWiseVertex(ElementWiseVertex.Op.Subtract), "dense1", "dense2").addLayer("output", new OutputLayer.Builder().nIn(midsz).nOut(outputsz).activation(new ActivationSigmoid()).lossFunction(LossFunction.MSE).build(), "elementwiseSubtract").setOutputs("output").build();
        ComputationGraph cg = new ComputationGraph(cgc);
        cg.init();
        INDArray input1 = Nd4j.rand(new int[] { batchsz, featuresz }, new UniformDistribution(-1, 1)).castTo(DataType.DOUBLE);
        INDArray input2 = Nd4j.rand(new int[] { batchsz, featuresz }, new UniformDistribution(-1, 1)).castTo(DataType.DOUBLE);
        INDArray target = nullsafe(Nd4j.rand(new int[] { batchsz, outputsz }, new UniformDistribution(0, 1)).castTo(DataType.DOUBLE));
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
        INDArray middle = Nd4j.zeros(DataType.DOUBLE, batchsz, midsz);
        middle.addi(m).subi(n);
        INDArray expect = Nd4j.zeros(DataType.DOUBLE, batchsz, outputsz);
        expect.addi(Transforms.sigmoid(middle.mmul(output_W).addi(output_b.repmat(batchsz, 1))));
        INDArray output = nullsafe(cg.output(input1, input2)[0]);
        Assertions.assertEquals(0.0, mse(output, expect), this.epsilon);
        Pair<Gradient, Double> pgd = cg.gradientAndScore();
        double score = pgd.getSecond();
        Assertions.assertEquals(score, mse(output, target), this.epsilon);
        Map<String, INDArray> gradients = pgd.getFirst().gradientForVariable();
        assertGradientCheckPasses(cg, new INDArray[] { input1, input2 }, target);
    }

    private void assertGradientCheckPasses(ComputationGraph cg, INDArray[] inputs, INDArray target) {
        boolean gradOK = GradientCheckUtil.checkGradients(new GraphConfig().net(cg).inputs(inputs)
                        .labels(new INDArray[] { target }).subset(true).maxPerParam(100));
        Assertions.assertTrue(gradOK);
    }

    private static double mse(INDArray output, INDArray target) {
        double mse_expect = Transforms.pow(output.sub(target), 2.0).sumNumber().doubleValue() / (output.columns() * output.rows());
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
