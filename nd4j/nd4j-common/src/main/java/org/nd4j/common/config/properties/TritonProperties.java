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
 * System property constants for Triton GPU compiler configuration.
 * These correspond to the C++ TritonConfig subsystem and ND4J_TRITON_* env vars.
 *
 * <p>New properties should be added here instead of in ND4JSystemProperties.</p>
 */
public final class TritonProperties {
    private TritonProperties() {}

    public static final String ALLOW_FALLBACK_CAPTURE = "nd4j.triton.allowFallbackCapture";
    public static final String GRAPH_CAPTURE = "nd4j.triton.graphCapture";
    public static final String DUMP_GRAPH_DOT = "nd4j.triton.dumpGraphDot";
    public static final String SKIP_KERNELS = "nd4j.triton.skipKernels";
    public static final String VERIFY_KERNELS = "nd4j.triton.verifyKernels";
    public static final String COMPILE_ALL = "nd4j.triton.compileAll";
    public static final String EXCLUDE_OPS = "nd4j.triton.excludeOps";
    public static final String FUSE_IDENTITY_SHAPES = "nd4j.triton.fuseIdentityShapes";
    public static final String FUSE_CAST_CHAINS = "nd4j.triton.fuseCastChains";
    public static final String SPECIALIZE_PERMUTE_SEQ1 = "nd4j.triton.specializePermuteSeq1";
    public static final String FUSED_MATMUL = "nd4j.triton.fusedMatmul";
    public static final String TF32 = "nd4j.triton.tf32";
    public static final String CONSOLIDATED_ARG_TABLE = "nd4j.triton.consolidatedArgTable";
    public static final String ARG_DIRTY_TRACKING = "nd4j.triton.argDirtyTracking";
    public static final String SECTION_FUSION = "nd4j.triton.sectionFusion";
    public static final String FUSION_SCORING = "nd4j.triton.fusionScoring";
    public static final String FUSION_MIN_SCORE = "nd4j.triton.fusionMinScore";
}
