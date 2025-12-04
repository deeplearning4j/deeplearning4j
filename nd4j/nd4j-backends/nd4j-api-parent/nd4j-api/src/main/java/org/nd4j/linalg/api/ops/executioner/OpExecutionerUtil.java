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

package org.nd4j.linalg.api.ops.executioner;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.CustomOp;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.api.ops.OpContext;
import org.nd4j.linalg.api.ops.impl.reduce.longer.MatchCondition;
import org.nd4j.linalg.api.ops.impl.transforms.comparison.CompareAndReplace;
import org.nd4j.linalg.api.ops.impl.transforms.comparison.CompareAndSet;
import org.nd4j.linalg.exception.ND4JOpProfilerException;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.conditions.Conditions;

import java.util.List;

@Slf4j
public class OpExecutionerUtil {

    private OpExecutionerUtil() {}

    // ThreadLocal flags to prevent infinite recursion when checking for NaN/Inf
    // The exec() call inside checkForNaN/checkForInf would trigger profilingHookOut(),
    // which would call checkForNaN/checkForInf again, causing infinite recursion.
    private static final ThreadLocal<Boolean> checkingNaN = ThreadLocal.withInitial(() -> false);
    private static final ThreadLocal<Boolean> checkingInf = ThreadLocal.withInitial(() -> false);

    // ThreadLocal to track diagnostic checks to prevent recursion in diagnostic info gathering
    private static final ThreadLocal<Boolean> gatheringDiagnostics = ThreadLocal.withInitial(() -> false);

    public static void checkForNaN(INDArray z) {
        if(z == null || z.isEmpty() || !z.dataType().isFPType() || z.wasClosed())
            return;

        // Prevent infinite recursion: if we're already checking, just return
        if (checkingNaN.get()) {
            return;
        }

        int match = 0;
        if (!z.isScalar()) {
            MatchCondition condition = new MatchCondition(z, Conditions.isNan());
            INDArray result = null;

            checkingNaN.set(true);
            try {
                result = Nd4j.getExecutioner().exec(condition);
                match = result.getInt(0);
            } finally {
                checkingNaN.set(false);
                // Clean up the result array to prevent OpaqueNDArray leak
                if (result != null && result.closeable()) {
                    result.close();
                }
                // Clean up the condition's internal arrays including data buffers
                condition.clearArrays();
            }
        } else {
            if (z.data().dataType() == DataType.DOUBLE) {
                if (Double.isNaN(z.getDouble(0)))
                    match = 1;
            } else {
                if (Float.isNaN(z.getFloat(0)))
                    match = 1;
            }
        }

        if (match > 0)
            throw new ND4JOpProfilerException("P.A.N.I.C.! Op.Z() contains " + match + " NaN value(s)");
    }

    public static void checkForAny(INDArray z) {
        checkForNaN(z);
        checkForInf(z);
    }

    public static void checkForInf(INDArray z) {
        if(z == null || z.isEmpty() || !z.dataType().isFPType() || z.wasClosed())
            return;

        // Prevent infinite recursion: if we're already checking, just return
        if (checkingInf.get()) {
            return;
        }

        int match = 0;
        if (!z.isScalar()) {
            MatchCondition condition = new MatchCondition(z, Conditions.isInfinite());
            INDArray result = null;

            checkingInf.set(true);
            try {
                result = Nd4j.getExecutioner().exec(condition);
                match = result.getInt(0);
            } finally {
                checkingInf.set(false);
                // Clean up the result array to prevent OpaqueNDArray leak
                if (result != null && result.closeable()) {
                    result.close();
                }
                // Clean up the condition's internal arrays including data buffers
                condition.clearArrays();
            }
        } else {
            if (z.data().dataType() == DataType.DOUBLE) {
                if (Double.isInfinite(z.getDouble(0)))
                    match = 1;
            } else {
                if (Float.isInfinite(z.getFloat(0)))
                    match = 1;
            }
        }

        if (match > 0)
            throw new ND4JOpProfilerException("P.A.N.I.C.! Op.Z() contains " + match + " Inf value(s)");

    }

    public static void checkForNaN(Op op, OpContext oc) {
        INDArray z = oc != null ? oc.getOutputArray(0) : op.z();
        if (z != null && !(op instanceof MatchCondition)) {
            int match = countNaN(z);
            if (match > 0) {
                String opName = op.opName() != null ? op.opName() : op.getClass().getSimpleName();
                throw new ND4JOpProfilerException("P.A.N.I.C.! Op [" + opName + "] output contains " + match + " NaN value(s)");
            }
        }
    }

    public static void checkForInf(Op op, OpContext oc) {
        INDArray z = oc != null ? oc.getOutputArray(0) : op.z();
        if (z != null && !(op instanceof MatchCondition) && !(op instanceof CompareAndSet) && !(op instanceof CompareAndReplace)) {
            int match = countInf(z);
            if (match > 0) {
                String opName = op.opName() != null ? op.opName() : op.getClass().getSimpleName();
                throw new ND4JOpProfilerException("P.A.N.I.C.! Op [" + opName + "] output contains " + match + " Inf value(s)");
            }
        }
    }

    public static void checkForInf(CustomOp op, OpContext oc) {
        List<INDArray> inArgs = oc != null ? oc.getInputArrays() : op.inputArguments();
        List<INDArray> outArgs = oc != null ? oc.getOutputArrays() : op.outputArguments();
        String opName = op.opName() != null ? op.opName() : op.getClass().getSimpleName();

        // Check inputs first (to provide context about whether issue originated earlier)
        for (int i = 0; i < inArgs.size(); i++) {
            INDArray input = inArgs.get(i);
            int match = countInf(input);
            if (match > 0) {
                throw new ND4JOpProfilerException("P.A.N.I.C.! Op [" + opName + "] input[" + i + "] contains " + match + " Inf value(s)");
            }
        }

        // Check outputs
        for (int i = 0; i < outArgs.size(); i++) {
            INDArray output = outArgs.get(i);
            int match = countInf(output);
            if (match > 0) {
                throw new ND4JOpProfilerException("P.A.N.I.C.! Op [" + opName + "] output[" + i + "] contains " + match + " Inf value(s)");
            }
        }
    }


    public static void checkForNaN(CustomOp op, OpContext oc) {
        List<INDArray> inArgs = oc != null ? oc.getInputArrays() : op.inputArguments();
        List<INDArray> outArgs = oc != null ? oc.getOutputArrays() : op.outputArguments();
        String opName = op.opName() != null ? op.opName() : op.getClass().getSimpleName();

        // Check inputs first (to provide context about whether issue originated earlier)
        for (int i = 0; i < inArgs.size(); i++) {
            INDArray input = inArgs.get(i);
            int match = countNaN(input);
            if (match > 0) {
                throw new ND4JOpProfilerException("P.A.N.I.C.! Op [" + opName + "] input[" + i + "] contains " + match + " NaN value(s)");
            }
        }

        // Check outputs
        for (int i = 0; i < outArgs.size(); i++) {
            INDArray output = outArgs.get(i);
            int match = countNaN(output);
            if (match > 0) {
                throw new ND4JOpProfilerException("P.A.N.I.C.! Op [" + opName + "] output[" + i + "] contains " + match + " NaN value(s)");
            }
        }
    }

    /**
     * Count the number of NaN values in an array.
     * This method is safe to call during diagnostics and won't trigger recursive checks.
     */
    private static int countNaN(INDArray arr) {
        if (arr == null || arr.isEmpty() || !arr.dataType().isFPType() || arr.wasClosed()) {
            return 0;
        }

        // Prevent recursion if we're already gathering diagnostics or checking
        if (gatheringDiagnostics.get() || checkingNaN.get()) {
            return countNaNSimple(arr);
        }

        try {
            gatheringDiagnostics.set(true);

            if (arr.isScalar()) {
                if (arr.data().dataType() == DataType.DOUBLE) {
                    return Double.isNaN(arr.getDouble(0)) ? 1 : 0;
                } else {
                    return Float.isNaN(arr.getFloat(0)) ? 1 : 0;
                }
            }

            // For small arrays, use simple counting to avoid overhead
            if (arr.length() <= 1000) {
                return countNaNSimple(arr);
            }

            // For larger arrays, use MatchCondition with recursion prevention
            MatchCondition condition = new MatchCondition(arr, Conditions.isNan());
            INDArray result = null;
            checkingNaN.set(true);
            try {
                result = Nd4j.getExecutioner().exec(condition);
                return result.getInt(0);
            } finally {
                checkingNaN.set(false);
                if (result != null && result.closeable()) {
                    result.close();
                }
                condition.clearArrays();
            }
        } catch (Exception e) {
            // Fall back to simple counting
            return countNaNSimple(arr);
        } finally {
            gatheringDiagnostics.set(false);
        }
    }

    /**
     * Simple NaN counting without using ops (for recursion prevention)
     */
    private static int countNaNSimple(INDArray arr) {
        if (arr == null || arr.isEmpty() || !arr.dataType().isFPType() || arr.wasClosed()) {
            return 0;
        }

        int count = 0;
        long len = Math.min(arr.length(), 10000); // Limit for performance
        for (long i = 0; i < len; i++) {
            if (arr.data().dataType() == DataType.DOUBLE) {
                if (Double.isNaN(arr.getDouble(i))) count++;
            } else {
                if (Float.isNaN(arr.getFloat(i))) count++;
            }
        }
        return count;
    }

    /**
     * Count the number of Inf values in an array.
     * This method is safe to call during diagnostics and won't trigger recursive checks.
     */
    private static int countInf(INDArray arr) {
        if (arr == null || arr.isEmpty() || !arr.dataType().isFPType() || arr.wasClosed()) {
            return 0;
        }

        // Prevent recursion if we're already gathering diagnostics or checking
        if (gatheringDiagnostics.get() || checkingInf.get()) {
            return countInfSimple(arr);
        }

        try {
            gatheringDiagnostics.set(true);

            if (arr.isScalar()) {
                if (arr.data().dataType() == DataType.DOUBLE) {
                    return Double.isInfinite(arr.getDouble(0)) ? 1 : 0;
                } else {
                    return Float.isInfinite(arr.getFloat(0)) ? 1 : 0;
                }
            }

            // For small arrays, use simple counting to avoid overhead
            if (arr.length() <= 1000) {
                return countInfSimple(arr);
            }

            // For larger arrays, use MatchCondition with recursion prevention
            MatchCondition condition = new MatchCondition(arr, Conditions.isInfinite());
            INDArray result = null;
            checkingInf.set(true);
            try {
                result = Nd4j.getExecutioner().exec(condition);
                return result.getInt(0);
            } finally {
                checkingInf.set(false);
                if (result != null && result.closeable()) {
                    result.close();
                }
                condition.clearArrays();
            }
        } catch (Exception e) {
            // Fall back to simple counting
            return countInfSimple(arr);
        } finally {
            gatheringDiagnostics.set(false);
        }
    }

    /**
     * Simple Inf counting without using ops (for recursion prevention)
     */
    private static int countInfSimple(INDArray arr) {
        if (arr == null || arr.isEmpty() || !arr.dataType().isFPType() || arr.wasClosed()) {
            return 0;
        }

        int count = 0;
        long len = Math.min(arr.length(), 10000); // Limit for performance
        for (long i = 0; i < len; i++) {
            if (arr.data().dataType() == DataType.DOUBLE) {
                if (Double.isInfinite(arr.getDouble(i))) count++;
            } else {
                if (Float.isInfinite(arr.getFloat(i))) count++;
            }
        }
        return count;
    }
}
