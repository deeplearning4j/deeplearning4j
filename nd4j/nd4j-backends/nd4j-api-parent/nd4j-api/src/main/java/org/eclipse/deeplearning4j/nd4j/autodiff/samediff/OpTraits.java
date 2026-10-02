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
package org.eclipse.deeplearning4j.nd4j.autodiff.samediff;

import org.nd4j.nativeblas.NativeOpsHolder;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * The traits a native op declares on its own descriptor ({@code libnd4j/include/ops/declarable/OpDescriptor.h}),
 * by op name. Each name is queried over JNI once.
 */
public final class OpTraits {

    public static final long TERNARY_ELEMENTWISE = 1L << 2;
    public static final long REDUCTION = 1L << 3;
    public static final long VIEW_PRODUCING = 1L << 7;
    public static final long VALUE_DEPENDENT_SHAPE = 1L << 8;
    public static final long DATA_DEPENDENT = 1L << 9;
    public static final long CONCAT = 1L << 20;
    public static final long DYNAMIC_OUTPUT_SIZE = 1L << 31;
    /** Observes or mutates implicit execution state, such as the random generator: never constant-fold. */
    public static final long STATEFUL = 1L << 32;

    private static final Map<String, Long> TRAITS = new ConcurrentHashMap<>();

    private OpTraits() {
    }

    /** The op's complete trait mask; 0 for an op the native registry does not know. */
    public static long of(String opName) {
        if (opName == null) {
            return 0;
        }
        return TRAITS.computeIfAbsent(opName,
                name -> NativeOpsHolder.getInstance().getDeviceNativeOps().getOpTraitMask(name));
    }

    /** Whether the op declares every trait in {@code traits}. */
    public static boolean has(String opName, long traits) {
        return (of(opName) & traits) == traits;
    }
}
