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

package org.nd4j.autodiff.samediff.execution;

/**
 * Preset DSP/Triton compilation profiles.
 */
public enum DspCompilationMode {
    /**
     * Minimize startup compile cost. Prioritizes faster JIT path selection.
     */
    REDUCE_OVERHEAD,

    /**
     * Balanced split-and-stitch Triton compilation for faster compile with strong runtime throughput.
     * Intended as an explicit opt-in profile.
     */
    SPLIT_STITCH,

    /**
     * Maximize steady-state performance. Prioritizes Triton compilation quality.
     */
    MAX_AUTOTUNE
}
