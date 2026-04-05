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
package org.nd4j.linalg.factory.config;

/**
 * DSP (DynamicShapePlan) configuration: batch-zero, batched GEMM,
 * cast elimination, matmul segmentation, FP16, cuBLAS, symbolic shapes,
 * capture pool, and OOM retry.
 */
public interface DspEnvironmentConfig {

    default boolean dspBatchZero() { return false; }
    default void setDspBatchZero(boolean v) {}
    default boolean dspBatchZeroVerbose() { return false; }
    default void setDspBatchZeroVerbose(boolean v) {}
    default boolean dspBatchZeroGapOnly() { return true; }
    default void setDspBatchZeroGapOnly(boolean v) {}
    default boolean dspBatchZeroKernel() { return false; }
    default void setDspBatchZeroKernel(boolean v) {}
    default boolean dspBatchedGemm() { return false; }
    default void setDspBatchedGemm(boolean v) {}

    default boolean dspCastElimination() { return true; }
    default void setDspCastElimination(boolean enabled) {}
    default boolean dspMatmulSegmentation() { return false; }
    default void setDspMatmulSegmentation(boolean enabled) {}
    default boolean dspFp16Compute() { return false; }
    default void setDspFp16Compute(boolean enabled) {}
    default boolean cublasTf32Enabled() { return false; }
    default void setCublasTf32Enabled(boolean enabled) {}
    default boolean cublasCaptureWorkspace() { return true; }
    default void setCublasCaptureWorkspace(boolean enabled) {}
    default boolean dspCastSinkMatmul() { return false; }
    default void setDspCastSinkMatmul(boolean enabled) {}

    default boolean dspSymbolicShapes() { return true; }
    default void setDspSymbolicShapes(boolean enabled) {}
    default int dspSymbolicShapeWarmup() { return 2; }
    default void setDspSymbolicShapeWarmup(int steps) {}

    default boolean dspCapturePoolEnabled() { return true; }
    default void setDspCapturePoolEnabled(boolean enabled) {}
    default long dspCapturePoolMaxBytes() { return 1073741824L; }
    default void setDspCapturePoolMaxBytes(long bytes) {}

    default int dspCaptureOomMaxRetries() { return 3; }
    default void setDspCaptureOomMaxRetries(int retries) {}
    default int dspCaptureOomRetryInterval() { return 4; }
    default void setDspCaptureOomRetryInterval(int interval) {}
    default int dspCublasWorkspaceMb() { return 256; }
    default void setDspCublasWorkspaceMb(int mb) {}
    default int dspGraphMetadataSafetyMb() { return 16; }
    default void setDspGraphMetadataSafetyMb(int mb) {}
    default boolean dspProactiveEvictBeforeCapture() { return true; }
    default void setDspProactiveEvictBeforeCapture(boolean enabled) {}
    default boolean dspLruEviction() { return true; }
    default void setDspLruEviction(boolean enabled) {}
}
