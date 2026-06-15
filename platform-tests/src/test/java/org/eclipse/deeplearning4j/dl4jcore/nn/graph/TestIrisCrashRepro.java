/*
 *  ******************************************************************************
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.eclipse.deeplearning4j.dl4jcore.nn.graph;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.executioner.OpExecutioner;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import static org.junit.jupiter.api.Assertions.assertNotNull;

/**
 * Minimal reproducer for the "free(): invalid pointer" / "double free or corruption" crash
 * that happens in TestComputationGraphNetwork.testIrisFit.
 *
 * Investigation strategy:
 * 1. Test MLN fit alone (no CG)
 * 2. Test CG fit alone (no MLN)
 * 3. Test with SCOPE_PANIC disabled
 * 4. Test with WorkspaceMode.NONE
 */
@Slf4j
public class TestIrisCrashRepro extends BaseDL4JTest {

    private static ComputationGraphConfiguration getIrisGraphConfiguration() {
        return new NeuralNetConfiguration.Builder().seed(12345)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .graphBuilder()
                .addInputs("input")
                .addLayer("firstLayer", new DenseLayer.Builder().nIn(4).nOut(5).build(), "input")
                .addLayer("outputLayer", new OutputLayer.Builder()
                        .nIn(5).nOut(3).activation(Activation.SOFTMAX)
                        .lossFunction(LossFunctions.LossFunction.MCXENT).build(), "firstLayer")
                .setOutputs("outputLayer").build();
    }

    private static MultiLayerConfiguration getIrisMLNConfiguration() {
        return new NeuralNetConfiguration.Builder().seed(12345)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).list()
                .layer(0, new DenseLayer.Builder().nIn(4).nOut(5).build())
                .layer(1, new OutputLayer.Builder()
                        .nIn(5).nOut(3).activation(Activation.SOFTMAX)
                        .lossFunction(LossFunctions.LossFunction.MCXENT).build())
                .build();
    }

    /**
     * Step 1: Test only MLN fit with SCOPE_PANIC enabled (as in the original test).
     * If this crashes, the issue is in MLN training path.
     */
    @Test
    public void testMLNFitAloneWithScopePanic() {
        log.info("=== testMLNFitAloneWithScopePanic ===");
        Nd4j.getExecutioner().setProfilingMode(OpExecutioner.ProfilingMode.SCOPE_PANIC);
        Nd4j.getEnvironment().setDebug(true);
        Nd4j.getEnvironment().setVerbose(true);
        try {
            MultiLayerConfiguration mlnConfig = getIrisMLNConfiguration();
            MultiLayerNetwork net = new MultiLayerNetwork(mlnConfig);
            net.init();

            Nd4j.getRandom().setSeed(12345);
            INDArray params = Nd4j.rand(DataType.FLOAT, 1, (4 * 5 + 5) + (5 * 3 + 3));
            net.setParams(params.dup());

            DataSetIterator iris = new IrisDataSetIterator(75, 150);
            log.info("Starting MLN fit...");
            net.fit(iris);
            log.info("MLN fit PASSED");
            assertNotNull(net.params());
        } finally {
            Nd4j.getEnvironment().setDebug(false);
            Nd4j.getEnvironment().setVerbose(false);
            Nd4j.getExecutioner().setProfilingMode(OpExecutioner.ProfilingMode.DISABLED);
        }
    }

    /**
     * Step 2: Test only CG fit with SCOPE_PANIC enabled.
     * If this crashes, the issue is in CG training path.
     */
    @Test
    public void testCGFitAloneWithScopePanic() {
        log.info("=== testCGFitAloneWithScopePanic ===");
        Nd4j.getExecutioner().setProfilingMode(OpExecutioner.ProfilingMode.SCOPE_PANIC);
        Nd4j.getEnvironment().setDebug(true);
        Nd4j.getEnvironment().setVerbose(true);
        try {
            ComputationGraphConfiguration configuration = getIrisGraphConfiguration();
            ComputationGraph graph = new ComputationGraph(configuration);
            graph.init();

            Nd4j.getRandom().setSeed(12345);
            INDArray params = Nd4j.rand(DataType.FLOAT, 1, (4 * 5 + 5) + (5 * 3 + 3));
            graph.setParams(params.dup());

            DataSetIterator iris = new IrisDataSetIterator(75, 150);
            log.info("Starting CG fit...");
            graph.fit(iris);
            log.info("CG fit PASSED");
            assertNotNull(graph.params());
        } finally {
            Nd4j.getEnvironment().setDebug(false);
            Nd4j.getEnvironment().setVerbose(false);
            Nd4j.getExecutioner().setProfilingMode(OpExecutioner.ProfilingMode.DISABLED);
        }
    }

    /**
     * Step 3: Test MLN + CG fit with SCOPE_PANIC DISABLED.
     * If this passes but the above crash, SCOPE_PANIC is triggering wrong detection.
     */
    @Test
    public void testBothFitNoScopePanic() {
        log.info("=== testBothFitNoScopePanic ===");
        // SCOPE_PANIC deliberately disabled
        Nd4j.getExecutioner().setProfilingMode(OpExecutioner.ProfilingMode.DISABLED);
        Nd4j.getEnvironment().setDebug(true);
        Nd4j.getEnvironment().setVerbose(true);
        try {
            ComputationGraphConfiguration configuration = getIrisGraphConfiguration();
            ComputationGraph graph = new ComputationGraph(configuration);
            graph.init();

            MultiLayerConfiguration mlnConfig = getIrisMLNConfiguration();
            MultiLayerNetwork net = new MultiLayerNetwork(mlnConfig);
            net.init();

            Nd4j.getRandom().setSeed(12345);
            INDArray params = Nd4j.rand(DataType.FLOAT, 1, (4 * 5 + 5) + (5 * 3 + 3));
            graph.setParams(params.dup());
            net.setParams(params.dup());

            DataSetIterator iris = new IrisDataSetIterator(75, 150);
            log.info("Starting MLN fit (no SCOPE_PANIC)...");
            net.fit(iris);
            log.info("MLN fit PASSED");

            iris.reset();
            log.info("Starting CG fit (no SCOPE_PANIC)...");
            graph.fit(iris);
            log.info("CG fit PASSED");

            assertNotNull(net.params());
            assertNotNull(graph.params());
        } finally {
            Nd4j.getEnvironment().setDebug(false);
            Nd4j.getEnvironment().setVerbose(false);
        }
    }

    /**
     * Step 4: Test with WorkspaceMode.NONE to isolate workspace-related crashes.
     * If this passes, the crash is workspace-related.
     */
    @Test
    public void testBothFitWorkspaceNone() {
        log.info("=== testBothFitWorkspaceNone ===");
        Nd4j.getExecutioner().setProfilingMode(OpExecutioner.ProfilingMode.SCOPE_PANIC);
        Nd4j.getEnvironment().setDebug(true);
        Nd4j.getEnvironment().setVerbose(true);
        try {
            ComputationGraphConfiguration configuration = new NeuralNetConfiguration.Builder().seed(12345)
                    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                    .trainingWorkspaceMode(WorkspaceMode.NONE)
                    .inferenceWorkspaceMode(WorkspaceMode.NONE)
                    .graphBuilder()
                    .addInputs("input")
                    .addLayer("firstLayer", new DenseLayer.Builder().nIn(4).nOut(5).build(), "input")
                    .addLayer("outputLayer", new OutputLayer.Builder()
                            .nIn(5).nOut(3).activation(Activation.SOFTMAX)
                            .lossFunction(LossFunctions.LossFunction.MCXENT).build(), "firstLayer")
                    .setOutputs("outputLayer").build();

            MultiLayerConfiguration mlnConfig = new NeuralNetConfiguration.Builder().seed(12345)
                    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                    .trainingWorkspaceMode(WorkspaceMode.NONE)
                    .inferenceWorkspaceMode(WorkspaceMode.NONE)
                    .list()
                    .layer(0, new DenseLayer.Builder().nIn(4).nOut(5).build())
                    .layer(1, new OutputLayer.Builder()
                            .nIn(5).nOut(3).activation(Activation.SOFTMAX)
                            .lossFunction(LossFunctions.LossFunction.MCXENT).build())
                    .build();

            ComputationGraph graph = new ComputationGraph(configuration);
            graph.init();

            MultiLayerNetwork net = new MultiLayerNetwork(mlnConfig);
            net.init();

            Nd4j.getRandom().setSeed(12345);
            INDArray params = Nd4j.rand(DataType.FLOAT, 1, (4 * 5 + 5) + (5 * 3 + 3));
            graph.setParams(params.dup());
            net.setParams(params.dup());

            DataSetIterator iris = new IrisDataSetIterator(75, 150);
            log.info("Starting MLN fit (WorkspaceMode.NONE)...");
            net.fit(iris);
            log.info("MLN fit PASSED");

            iris.reset();
            log.info("Starting CG fit (WorkspaceMode.NONE)...");
            graph.fit(iris);
            log.info("CG fit PASSED");

            assertNotNull(net.params());
            assertNotNull(graph.params());
        } finally {
            Nd4j.getEnvironment().setDebug(false);
            Nd4j.getEnvironment().setVerbose(false);
            Nd4j.getExecutioner().setProfilingMode(OpExecutioner.ProfilingMode.DISABLED);
        }
    }

    /**
     * Step 5: Single minibatch CG forward pass to find which op crashes.
     * This replicates just the first computeGradientAndScore call.
     */
    @Test
    public void testCGSingleMinibatch() {
        log.info("=== testCGSingleMinibatch ===");
        Nd4j.getExecutioner().setProfilingMode(OpExecutioner.ProfilingMode.SCOPE_PANIC);
        Nd4j.getEnvironment().setDebug(true);
        Nd4j.getEnvironment().setVerbose(true);
        try {
            ComputationGraphConfiguration configuration = getIrisGraphConfiguration();
            ComputationGraph graph = new ComputationGraph(configuration);
            graph.init();

            Nd4j.getRandom().setSeed(12345);
            INDArray params = Nd4j.rand(DataType.FLOAT, 1, (4 * 5 + 5) + (5 * 3 + 3));
            graph.setParams(params.dup());

            // Get a single batch
            DataSetIterator iris = new IrisDataSetIterator(75, 150);
            org.nd4j.linalg.dataset.DataSet ds = (org.nd4j.linalg.dataset.DataSet) iris.next();

            log.info("Input shape: {}", ds.getFeatures().shapeInfoToString());
            log.info("Label shape: {}", ds.getLabels().shapeInfoToString());

            log.info("Calling graph.fit(ds)...");
            graph.fit(ds);
            log.info("graph.fit(ds) PASSED");

            assertNotNull(graph.params());
        } finally {
            Nd4j.getEnvironment().setDebug(false);
            Nd4j.getEnvironment().setVerbose(false);
            Nd4j.getExecutioner().setProfilingMode(OpExecutioner.ProfilingMode.DISABLED);
        }
    }

    /**
     * Step 6: Minimal fit test without any debug/verbose/SCOPE_PANIC.
     * Isolates whether the crash is debug-mode-specific or a real training bug.
     */
    @Test
    public void testBothFitNoDebug() {
        log.info("=== testBothFitNoDebug ===");
        // Ensure all diagnostics are off
        Nd4j.getExecutioner().setProfilingMode(OpExecutioner.ProfilingMode.DISABLED);
        Nd4j.getEnvironment().setDebug(false);
        Nd4j.getEnvironment().setVerbose(false);
        try {
            ComputationGraphConfiguration configuration = getIrisGraphConfiguration();
            ComputationGraph graph = new ComputationGraph(configuration);
            graph.init();

            MultiLayerConfiguration mlnConfig = getIrisMLNConfiguration();
            MultiLayerNetwork net = new MultiLayerNetwork(mlnConfig);
            net.init();

            Nd4j.getRandom().setSeed(12345);
            INDArray params = Nd4j.rand(DataType.FLOAT, 1, (4 * 5 + 5) + (5 * 3 + 3));
            graph.setParams(params.dup());
            net.setParams(params.dup());

            DataSetIterator iris = new IrisDataSetIterator(75, 150);
            log.info("Starting MLN fit (no debug)...");
            net.fit(iris);
            log.info("MLN fit PASSED");

            iris.reset();
            log.info("Starting CG fit (no debug)...");
            graph.fit(iris);
            log.info("CG fit PASSED");

            assertNotNull(net.params());
            assertNotNull(graph.params());
        } finally {
            // ensure clean state
            Nd4j.getEnvironment().setDebug(false);
            Nd4j.getEnvironment().setVerbose(false);
        }
    }
}
