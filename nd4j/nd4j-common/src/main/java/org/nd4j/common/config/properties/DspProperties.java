/*
 *  ******************************************************************************
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
package org.nd4j.common.config.properties;

/**
 * System property constants for DSP (DynamicShapePlan) configuration.
 * These correspond to the C++ DspConfig subsystem and ND4J_DSP_* env vars.
 *
 * <p>New properties should be added here instead of in ND4JSystemProperties.</p>
 */
public final class DspProperties {
    private DspProperties() {}

    // Execution modes
    public static final String ENABLED = "org.nd4j.inference.dynamicShapePlan";
    public static final String NATIVE_EXECUTOR_ENABLED = "nd4j.dsp.nativeExecutor.enabled";
    public static final String CUDA_GRAPHS_ENABLED = "nd4j.dsp.cudaGraphs.enabled";
    public static final String JIT_MODE = "nd4j.dsp.jitMode";
    public static final String GRAPH_EXECUTION_MODE = "nd4j.dsp.graphExecutionMode";

    // Batch operations
    public static final String BATCH_ZERO = "nd4j.dsp.batchZero";
    public static final String BATCH_ZERO_VERBOSE = "nd4j.dsp.batchZero.verbose";
    public static final String BATCH_ZERO_GAP_ONLY = "nd4j.dsp.batchZero.gapOnly";
    public static final String BATCH_ZERO_KERNEL = "nd4j.dsp.batchZero.kernel";
    public static final String BATCHED_GEMM = "nd4j.dsp.batchedGemm";

    // Optimization flags
    public static final String CAST_ELIMINATION = "nd4j.dsp.castElimination";
    public static final String MATMUL_SEGMENTATION = "nd4j.dsp.matmulSegmentation";
    public static final String FP16_COMPUTE = "nd4j.dsp.fp16Compute";
    public static final String CAST_SINK_MATMUL = "nd4j.dsp.castSinkMatmul";

    // Symbolic shapes
    public static final String SYMBOLIC_SHAPES = "nd4j.dsp.symbolicShapes";
    public static final String SYMBOLIC_SHAPE_WARMUP = "nd4j.dsp.symbolicShapeWarmup";

    // Capture pool
    public static final String CAPTURE_POOL_ENABLED = "nd4j.dsp.capturePoolEnabled";
    public static final String CAPTURE_POOL_MAX_BYTES = "nd4j.dsp.capturePoolMaxBytes";
    public static final String CAPTURE_HOST_WORKSPACE_MB = "nd4j.dsp.captureHostWorkspaceMb";
    public static final String CAPTURE_WORKSPACE_MB = "nd4j.dsp.captureWorkspaceMb";

    // Memory management
    public static final String NO_FREEZE = "nd4j.dsp.noFreeze";
    public static final String TRIM_INTERVAL = "nd4j.dsp.trimInterval";
    public static final String SLOT_CACHE_GROWTH_FACTOR = "org.nd4j.dsp.slotCacheGrowthFactor";
    public static final String POOL_MAX_BYTES = "org.nd4j.dsp.pool.maxBytes";

    // Frozen-shape transition
    public static final String FREEZE_MERGE_SEGMENTS = "nd4j.dsp.freezeMergeSegments";
    public static final String FREEZE_RECOMPILE = "nd4j.dsp.freezeRecompile";

    // Replay graph cache
    public static final String REPLAY_CACHE_DIR = "nd4j.dsp.replayCacheDir";
    public static final String REPLAY_CACHE_ENABLED = "nd4j.dsp.replayCacheEnabled";
    public static final String TRACE_SLOT = "nd4j.dsp.traceSlot";

    // Diagnostics
    public static final String TRACE = "nd4j.dsp.trace";
    public static final String DIAGNOSTICS = "nd4j.dsp.diagnostics";
    public static final String DIAGNOSTICS_LEVEL = "nd4j.dsp.diagnostics.level";
    public static final String DIAGNOSTICS_FILE = "nd4j.dsp.diagnostics.file";
    public static final String EXECUTION_TIMING = "nd4j.dsp.executionTiming";
    public static final String JAVA_DUMP_OUTPUTS = "nd4j.dsp.java.dumpOutputs";
    public static final String NATIVE_DUMP_OUTPUTS = "nd4j.dsp.native.dumpOutputs";
}
