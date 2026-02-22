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

package org.eclipse.deeplearning4j.nd4j.linalg.ops;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIf;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.CollectiveCommunicator;
import org.nd4j.autodiff.samediff.execution.CollectiveCommunicatorFactory;
import org.nd4j.autodiff.samediff.execution.LocalCollectiveCommunicator;
import org.nd4j.autodiff.samediff.execution.PipelineParallelRunner;
import org.nd4j.autodiff.samediff.execution.TensorParallelConfig;
import org.nd4j.autodiff.samediff.execution.TensorParallelRunner;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.custom.ColumnParallelLinear;
import org.nd4j.linalg.api.ops.impl.transforms.custom.RowParallelLinear;
import org.nd4j.linalg.factory.Nd4j;

import java.util.HashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Integration tests for tensor and pipeline parallelism.
 * <p>
 * All tests use the backend-agnostic {@link CollectiveCommunicator} interface.
 * No backend-specific classes (e.g., NCCL) are referenced directly.
 * <p>
 * Tests are split into:
 * - Single-GPU tests (always run): verify API contracts, weight sharding, etc.
 * - Multi-GPU tests (conditional): require 2+ GPUs and a distributed communicator
 */
public class TensorParallelIntegrationTest {

    private static int numGpus;

    @BeforeAll
    static void setup() {
        numGpus = Nd4j.getAffinityManager().getNumberOfDevices();
    }

    static boolean hasMultipleGpus() {
        return Nd4j.getAffinityManager().getNumberOfDevices() >= 2;
    }

    static boolean hasDistributedComm() {
        return CollectiveCommunicatorFactory.isDistributedAvailable();
    }

    static boolean hasMultiGpuAndDistributedComm() {
        return hasMultipleGpus() && hasDistributedComm();
    }

    // =========================================================================
    // Single-GPU Tests (always run)
    // =========================================================================

    @Test
    public void testTensorParallelConfigCreation() {
        TensorParallelConfig config = TensorParallelConfig.create(4, 2);
        assertEquals(4, config.getTensorParallelSize());
        assertEquals(2, config.getRank());
        assertTrue(config.isEnabled());
        assertEquals(2, config.myDeviceId());
    }

    @Test
    public void testTensorParallelConfigSingleGpu() {
        TensorParallelConfig config = TensorParallelConfig.create(1, 0);
        assertFalse(config.isEnabled());
    }

    @Test
    public void testTensorParallelConfigWithDeviceIds() {
        TensorParallelConfig config = TensorParallelConfig.create(2, 0)
                .withDeviceIds(0, 2);
        assertEquals(0, config.deviceIdForRank(0));
        assertEquals(2, config.deviceIdForRank(1));
    }

    @Test
    public void testLocalCommunicatorAllReduceIsNoop() {
        // Local communicator should be a no-op for single rank
        try (CollectiveCommunicator comm = new LocalCollectiveCommunicator()) {
            assertEquals(1, comm.getNumRanks());
            assertEquals(0, comm.getRank());
            assertEquals("Local", comm.backendName());

            INDArray tensor = Nd4j.ones(DataType.FLOAT, 4, 8).muli(3.0);
            INDArray expected = tensor.dup();
            comm.allReduceSum(tensor);
            assertEquals(expected, tensor, "Local allReduceSum should be a no-op");
        }
    }

    @Test
    public void testLocalCommunicatorAllGather() {
        try (CollectiveCommunicator comm = new LocalCollectiveCommunicator()) {
            INDArray data = Nd4j.rand(DataType.FLOAT, 4, 8);
            INDArray gathered = comm.allGather(data);
            assertEquals(data, gathered, "Local allGather should return a copy");
            assertNotSame(data, gathered, "Should be a different object");
        }
    }

    @Test
    public void testCollectiveCommunicatorFactorySingleGpu() {
        TensorParallelConfig config = TensorParallelConfig.create(1, 0);
        try (CollectiveCommunicator comm = CollectiveCommunicatorFactory.create(config)) {
            assertEquals("Local", comm.backendName(),
                    "Single-GPU config should produce local communicator");
        }
    }

    @Test
    public void testWeightShardingColumnParallel() {
        SameDiff sd = SameDiff.create();

        int inFeatures = 32, outFeatures = 16;
        SDVariable input = sd.var("input", Nd4j.rand(DataType.FLOAT, 2, inFeatures));
        SDVariable weight = sd.var("gate_proj.weight", Nd4j.rand(DataType.FLOAT, inFeatures, outFeatures));
        SDVariable output = sd.mmul(input, weight);

        TensorParallelConfig config = TensorParallelConfig.create(2, 0);
        try (TensorParallelRunner runner = new TensorParallelRunner(sd, config)) {
            runner.autoDetectSharding();
            runner.shardWeights();

            INDArray shardedWeight = sd.getVariable("gate_proj.weight").getArr();
            assertNotNull(shardedWeight);
            assertArrayEquals(new long[]{inFeatures, outFeatures / 2}, shardedWeight.shape(),
                    "Column-parallel shard should split output dim by tp_size");
        }
    }

    @Test
    public void testWeightShardingRowParallel() {
        SameDiff sd = SameDiff.create();

        int inFeatures = 32, outFeatures = 16;
        SDVariable input = sd.var("input", Nd4j.rand(DataType.FLOAT, 2, inFeatures));
        SDVariable weight = sd.var("down_proj.weight", Nd4j.rand(DataType.FLOAT, inFeatures, outFeatures));
        SDVariable output = sd.mmul(input, weight);

        TensorParallelConfig config = TensorParallelConfig.create(2, 1);
        try (TensorParallelRunner runner = new TensorParallelRunner(sd, config)) {
            runner.autoDetectSharding();
            runner.shardWeights();

            INDArray shardedWeight = sd.getVariable("down_proj.weight").getArr();
            assertNotNull(shardedWeight);
            assertArrayEquals(new long[]{inFeatures / 2, outFeatures}, shardedWeight.shape(),
                    "Row-parallel shard should split input dim by tp_size");
        }
    }

    @Test
    public void testWeightRestoration() {
        SameDiff sd = SameDiff.create();

        int inFeatures = 32, outFeatures = 16;
        INDArray originalWeight = Nd4j.rand(DataType.FLOAT, inFeatures, outFeatures);
        SDVariable weight = sd.var("gate_proj.weight", originalWeight.dup());

        TensorParallelConfig config = TensorParallelConfig.create(2, 0);
        try (TensorParallelRunner runner = new TensorParallelRunner(sd, config)) {
            runner.autoDetectSharding();
            runner.shardWeights();

            assertArrayEquals(new long[]{inFeatures, outFeatures / 2},
                    sd.getVariable("gate_proj.weight").getArr().shape());

            runner.restoreWeights();
            assertArrayEquals(new long[]{inFeatures, outFeatures},
                    sd.getVariable("gate_proj.weight").getArr().shape());
        }
    }

    @Test
    public void testAutoDetectSharding() {
        SameDiff sd = SameDiff.create();

        sd.var("model.layers.0.self_attn.q_proj.weight", Nd4j.rand(DataType.FLOAT, 32, 32));
        sd.var("model.layers.0.self_attn.k_proj.weight", Nd4j.rand(DataType.FLOAT, 32, 8));
        sd.var("model.layers.0.self_attn.v_proj.weight", Nd4j.rand(DataType.FLOAT, 32, 8));
        sd.var("model.layers.0.self_attn.o_proj.weight", Nd4j.rand(DataType.FLOAT, 32, 32));
        sd.var("model.layers.0.mlp.gate_proj.weight", Nd4j.rand(DataType.FLOAT, 32, 64));
        sd.var("model.layers.0.mlp.up_proj.weight", Nd4j.rand(DataType.FLOAT, 32, 64));
        sd.var("model.layers.0.mlp.down_proj.weight", Nd4j.rand(DataType.FLOAT, 64, 32));
        sd.var("model.norm.weight", Nd4j.rand(DataType.FLOAT, 32));

        TensorParallelConfig config = TensorParallelConfig.create(2, 0);
        try (TensorParallelRunner runner = new TensorParallelRunner(sd, config)) {
            runner.autoDetectSharding();
            runner.shardWeights();

            // Column-parallel: q, k, v, gate, up should be split on last dim
            assertArrayEquals(new long[]{32, 16},
                    sd.getVariable("model.layers.0.self_attn.q_proj.weight").getArr().shape());
            assertArrayEquals(new long[]{32, 4},
                    sd.getVariable("model.layers.0.self_attn.k_proj.weight").getArr().shape());
            assertArrayEquals(new long[]{32, 32},
                    sd.getVariable("model.layers.0.mlp.gate_proj.weight").getArr().shape());

            // Row-parallel: o_proj, down_proj should be split on first dim
            assertArrayEquals(new long[]{16, 32},
                    sd.getVariable("model.layers.0.self_attn.o_proj.weight").getArr().shape());
            assertArrayEquals(new long[]{32, 32},
                    sd.getVariable("model.layers.0.mlp.down_proj.weight").getArr().shape());

            // Norm weights should be unchanged (replicated)
            assertArrayEquals(new long[]{32},
                    sd.getVariable("model.norm.weight").getArr().shape());
        }
    }

    @Test
    public void testSingleGpuExecutionUsesLocalComm() {
        SameDiff sd = SameDiff.create();

        int B = 2, I = 16, O = 8;
        SDVariable input = sd.var("input", Nd4j.rand(DataType.FLOAT, B, I));
        SDVariable weight = sd.var("weight", Nd4j.rand(DataType.FLOAT, I, O));
        SDVariable output = sd.mmul("output", input, weight);

        TensorParallelConfig config = TensorParallelConfig.create(1, 0);
        try (TensorParallelRunner runner = new TensorParallelRunner(sd, config)) {
            Map<String, INDArray> inputs = new HashMap<>();
            inputs.put("input", Nd4j.rand(DataType.FLOAT, B, I));

            Map<String, INDArray> results = runner.execute(inputs);
            assertNotNull(results);
            assertFalse(results.isEmpty());
        }
    }

    @Test
    public void testRunnerAcceptsCustomCommunicator() {
        SameDiff sd = SameDiff.create();

        int B = 2, I = 16, O = 8;
        SDVariable input = sd.var("input", Nd4j.rand(DataType.FLOAT, B, I));
        SDVariable weight = sd.var("weight", Nd4j.rand(DataType.FLOAT, I, O));
        SDVariable output = sd.mmul("output", input, weight);

        TensorParallelConfig config = TensorParallelConfig.create(2, 0);
        try (TensorParallelRunner runner = new TensorParallelRunner(sd, config)) {
            // Inject a local communicator (simulating a test double)
            CollectiveCommunicator localComm = new LocalCollectiveCommunicator(2, 0);
            runner.setCommunicator(localComm);

            assertEquals("Local", runner.getCommunicator().backendName());
        }
    }

    @Test
    public void testColumnParallelLinearSingleGpu() {
        SameDiff sd = SameDiff.create();

        int B = 4, I = 32, O = 16;
        SDVariable input = sd.var("input", Nd4j.rand(DataType.FLOAT, B, I));
        SDVariable weight = sd.var("weight", Nd4j.rand(DataType.FLOAT, I, O));
        SDVariable bias = sd.var("bias", Nd4j.rand(DataType.FLOAT, O));

        // tpSize=1, tpRank=0, gatherOutput=1
        SDVariable output = new ColumnParallelLinear(sd, input, weight, bias, 1, 0, 1)
                .outputVariable();

        Map<String, INDArray> results = sd.outputAll(null);
        INDArray result = results.get(output.name());

        assertNotNull(result);
        assertArrayEquals(new long[]{B, O}, result.shape());

        // Verify it matches standard matmul + bias
        INDArray expected = input.getArr().mmul(weight.getArr()).addi(bias.getArr());
        assertTrue(result.equalsWithEps(expected, 1e-4),
                "Column-parallel (tpSize=1) should match matmul+bias");
    }

    @Test
    public void testRowParallelLinearSingleGpu() {
        SameDiff sd = SameDiff.create();

        int B = 4, I = 32, O = 16;
        SDVariable input = sd.var("input", Nd4j.rand(DataType.FLOAT, B, I));
        SDVariable weight = sd.var("weight", Nd4j.rand(DataType.FLOAT, I, O));
        SDVariable bias = sd.var("bias", Nd4j.rand(DataType.FLOAT, O));

        // tpSize=1, tpRank=0, reduceOutput=1
        SDVariable output = new RowParallelLinear(sd, input, weight, bias, 1, 0, 1)
                .outputVariable();

        Map<String, INDArray> results = sd.outputAll(null);
        INDArray result = results.get(output.name());

        assertNotNull(result);
        assertArrayEquals(new long[]{B, O}, result.shape());

        // With tpSize=1, bias is added directly in the op
        INDArray expected = input.getArr().mmul(weight.getArr()).addi(bias.getArr());
        assertTrue(result.equalsWithEps(expected, 1e-4),
                "Row-parallel (tpSize=1) should match matmul+bias");
    }

    // =========================================================================
    // Pipeline Parallel Tests (always run, single-stage)
    // =========================================================================

    @Test
    public void testPipelineParallelSingleStage() {
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.var("input", Nd4j.rand(DataType.FLOAT, 2, 16));
        SDVariable weight = sd.var("weight", Nd4j.rand(DataType.FLOAT, 16, 8));
        SDVariable output = sd.mmul("output", input, weight);

        PipelineParallelRunner.PipelineConfig config = new PipelineParallelRunner.PipelineConfig(1);
        try (PipelineParallelRunner runner = new PipelineParallelRunner(sd, config)) {
            Map<String, INDArray> inputs = new HashMap<>();
            inputs.put("input", Nd4j.rand(DataType.FLOAT, 2, 16));

            Map<String, INDArray> results = runner.executeInference(inputs);
            assertNotNull(results);
            assertFalse(results.isEmpty());
        }
    }

    // =========================================================================
    // Multi-GPU Tests (require 2+ GPUs + distributed communicator)
    // =========================================================================

    @Test
    @EnabledIf("hasDistributedComm")
    public void testDistributedCommunicatorAvailable() {
        assertTrue(CollectiveCommunicatorFactory.isDistributedAvailable(),
                "Distributed communicator should be available");
    }

    @Test
    @EnabledIf("hasMultiGpuAndDistributedComm")
    public void testMultiGpuAllReduceSum() {
        // Test AllReduce sum across 2 GPUs using the factory
        TensorParallelConfig config0 = TensorParallelConfig.create(2, 0);
        TensorParallelConfig config1 = TensorParallelConfig.create(2, 1);

        try (CollectiveCommunicator comm0 = CollectiveCommunicatorFactory.create(config0);
             CollectiveCommunicator comm1 = CollectiveCommunicatorFactory.create(config1)) {

            INDArray tensor0 = Nd4j.ones(DataType.FLOAT, 4, 8);
            INDArray tensor1 = Nd4j.ones(DataType.FLOAT, 4, 8);

            comm0.allReduceSum(tensor0);
            comm1.allReduceSum(tensor1);

            INDArray expected = Nd4j.ones(DataType.FLOAT, 4, 8).muli(2.0);
            assertTrue(tensor0.equalsWithEps(expected, 1e-5),
                    "AllReduce sum should double the values");
            assertTrue(tensor1.equalsWithEps(expected, 1e-5),
                    "AllReduce sum should produce same result on all ranks");
        }
    }

    @Test
    @EnabledIf("hasMultiGpuAndDistributedComm")
    public void testTensorParallelMlpLayer() {
        // Simulate a transformer MLP with tensor parallelism across 2 GPUs:
        //   gate = x @ W_gate_shard  (column-parallel, each GPU has half)
        //   up   = x @ W_up_shard    (column-parallel)
        //   h    = silu(gate) * up
        //   out  = h_shard @ W_down_shard  (row-parallel, AllReduce)

        int B = 2, hiddenDim = 64, ffnDim = 128;
        INDArray input = Nd4j.rand(DataType.FLOAT, B, hiddenDim);

        // Full weights for reference
        INDArray W_gate = Nd4j.rand(DataType.FLOAT, hiddenDim, ffnDim);
        INDArray W_up = Nd4j.rand(DataType.FLOAT, hiddenDim, ffnDim);
        INDArray W_down = Nd4j.rand(DataType.FLOAT, ffnDim, hiddenDim);

        // Reference single-GPU computation
        INDArray gate = input.mmul(W_gate);
        INDArray up = input.mmul(W_up);
        INDArray silu_gate = Nd4j.nn().sigmoid(gate.dup()).muli(gate);
        INDArray h = silu_gate.muli(up);
        INDArray referenceOutput = h.mmul(W_down);

        // Sharded computation (simulating 2 GPUs)
        int shardSize = ffnDim / 2;

        INDArray W_gate_shard0 = W_gate.get(
                org.nd4j.linalg.indexing.NDArrayIndex.all(),
                org.nd4j.linalg.indexing.NDArrayIndex.interval(0, shardSize)).dup();
        INDArray W_gate_shard1 = W_gate.get(
                org.nd4j.linalg.indexing.NDArrayIndex.all(),
                org.nd4j.linalg.indexing.NDArrayIndex.interval(shardSize, ffnDim)).dup();

        INDArray W_up_shard0 = W_up.get(
                org.nd4j.linalg.indexing.NDArrayIndex.all(),
                org.nd4j.linalg.indexing.NDArrayIndex.interval(0, shardSize)).dup();
        INDArray W_up_shard1 = W_up.get(
                org.nd4j.linalg.indexing.NDArrayIndex.all(),
                org.nd4j.linalg.indexing.NDArrayIndex.interval(shardSize, ffnDim)).dup();

        INDArray W_down_shard0 = W_down.get(
                org.nd4j.linalg.indexing.NDArrayIndex.interval(0, shardSize),
                org.nd4j.linalg.indexing.NDArrayIndex.all()).dup();
        INDArray W_down_shard1 = W_down.get(
                org.nd4j.linalg.indexing.NDArrayIndex.interval(shardSize, ffnDim),
                org.nd4j.linalg.indexing.NDArrayIndex.all()).dup();

        // Each rank computes its shard independently
        INDArray gate0 = input.mmul(W_gate_shard0);
        INDArray gate1 = input.mmul(W_gate_shard1);
        INDArray up0 = input.mmul(W_up_shard0);
        INDArray up1 = input.mmul(W_up_shard1);

        INDArray silu_gate0 = Nd4j.nn().sigmoid(gate0.dup()).muli(gate0);
        INDArray silu_gate1 = Nd4j.nn().sigmoid(gate1.dup()).muli(gate1);
        INDArray h0 = silu_gate0.muli(up0);
        INDArray h1 = silu_gate1.muli(up1);

        // Row-parallel: each rank does partial matmul, then AllReduce sum
        INDArray partial0 = h0.mmul(W_down_shard0);
        INDArray partial1 = h1.mmul(W_down_shard1);

        // Simulate AllReduce by adding partials
        INDArray shardedOutput = partial0.add(partial1);

        // Verify sharded computation matches reference
        assertTrue(shardedOutput.equalsWithEps(referenceOutput, 1e-3),
                "Tensor-parallel MLP output should match single-GPU reference.\n" +
                "Max diff: " + referenceOutput.sub(shardedOutput).amaxNumber().doubleValue());
    }
}
