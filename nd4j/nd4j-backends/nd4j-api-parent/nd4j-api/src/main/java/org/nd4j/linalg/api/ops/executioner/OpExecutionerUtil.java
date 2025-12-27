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
        if (op instanceof MatchCondition) {
            return; // Skip MatchCondition ops to prevent infinite recursion
        }

        String opName = op.opName() != null ? op.opName() : op.getClass().getSimpleName();

        // Get input arrays
        INDArray x = oc != null ? oc.getInputArray(0) : op.x();
        INDArray y = oc != null ? (oc.getInputArrays().size() > 1 ? oc.getInputArray(1) : null) : op.y();
        INDArray z = oc != null ? oc.getOutputArray(0) : op.z();

        // Check input X first - if NaN is in input, the problem originated earlier
        if (x != null) {
            int inputNaN = countNaN(x);
            if (inputNaN > 0) {
                StringBuilder diagnostics = new StringBuilder();
                diagnostics.append("P.A.N.I.C.! Op [").append(opName).append("] input X contains ").append(inputNaN).append(" NaN value(s)\n");
                diagnostics.append("\n=== INPUT DIAGNOSTICS ===\n");
                diagnostics.append("NaN in INPUT indicates the problem originated EARLIER in the computation.\n");
                diagnostics.append("The SameDiff inference or previous operations produced corrupted output.\n");
                appendArrayDiagnostics(diagnostics, "Input X", x);
                if (y != null) {
                    appendArrayDiagnostics(diagnostics, "Input Y", y);
                }
                diagnostics.append("\n=== ACTION REQUIRED ===\n");
                diagnostics.append("Check if the input array's data buffer was freed prematurely.\n");
                diagnostics.append("Verify that memory management in SameDiff is not releasing arrays too early.\n");
                throw new ND4JOpProfilerException(diagnostics.toString());
            }
        }

        // Check input Y if present
        if (y != null) {
            int inputNaN = countNaN(y);
            if (inputNaN > 0) {
                StringBuilder diagnostics = new StringBuilder();
                diagnostics.append("P.A.N.I.C.! Op [").append(opName).append("] input Y contains ").append(inputNaN).append(" NaN value(s)\n");
                diagnostics.append("\n=== INPUT DIAGNOSTICS ===\n");
                diagnostics.append("NaN in INPUT indicates the problem originated EARLIER in the computation.\n");
                appendArrayDiagnostics(diagnostics, "Input X", x);
                appendArrayDiagnostics(diagnostics, "Input Y", y);
                diagnostics.append("\n=== ACTION REQUIRED ===\n");
                diagnostics.append("Trace back to find which earlier operation produced the NaN values.\n");
                throw new ND4JOpProfilerException(diagnostics.toString());
            }
        }

        // Now check output - if inputs were valid but output has NaN, the operation itself generated it
        if (z != null) {
            int match = countNaN(z);
            if (match > 0) {
                StringBuilder diagnostics = new StringBuilder();
                diagnostics.append("P.A.N.I.C.! Op [").append(opName).append("] output contains ").append(match).append(" NaN value(s)\n");
                diagnostics.append("\n=== INPUT DIAGNOSTICS (inputs were valid, issue is in THIS op) ===\n");

                appendArrayDiagnostics(diagnostics, "Input X", x);
                if (y != null) {
                    appendArrayDiagnostics(diagnostics, "Input Y", y);
                }
                appendArrayDiagnostics(diagnostics, "Output Z", z);

                diagnostics.append("\n=== POTENTIAL ROOT CAUSES ===\n");
                diagnostics.append("1. Divide by zero: Check if any input contains zeros that are used as divisors\n");
                diagnostics.append("2. Invalid math: sqrt of negative, log of zero/negative, etc.\n");
                diagnostics.append("3. 0 * Inf or Inf - Inf operations\n");
                diagnostics.append("4. Numerical overflow: Values too large for floating point representation\n");

                throw new ND4JOpProfilerException(diagnostics.toString());
            }
        }
    }

    public static void checkForInf(Op op, OpContext oc) {
        if (op instanceof MatchCondition || op instanceof CompareAndSet || op instanceof CompareAndReplace) {
            return; // Skip these ops to prevent infinite recursion and false positives
        }

        String opName = op.opName() != null ? op.opName() : op.getClass().getSimpleName();

        // Get input arrays
        INDArray x = oc != null ? oc.getInputArray(0) : op.x();
        INDArray y = oc != null ? (oc.getInputArrays().size() > 1 ? oc.getInputArray(1) : null) : op.y();
        INDArray z = oc != null ? oc.getOutputArray(0) : op.z();

        // Check input X first - if Inf is in input, the problem originated earlier
        if (x != null) {
            int inputInf = countInf(x);
            if (inputInf > 0) {
                StringBuilder diagnostics = new StringBuilder();
                diagnostics.append("P.A.N.I.C.! Op [").append(opName).append("] input X contains ").append(inputInf).append(" Inf value(s)\n");
                diagnostics.append("\n=== INPUT DIAGNOSTICS ===\n");
                diagnostics.append("Inf in INPUT indicates the problem originated EARLIER in the computation.\n");
                appendArrayDiagnostics(diagnostics, "Input X", x);
                if (y != null) {
                    appendArrayDiagnostics(diagnostics, "Input Y", y);
                }
                diagnostics.append("\n=== ACTION REQUIRED ===\n");
                diagnostics.append("Trace back to find which earlier operation produced the Inf values.\n");
                throw new ND4JOpProfilerException(diagnostics.toString());
            }
        }

        // Check input Y if present
        if (y != null) {
            int inputInf = countInf(y);
            if (inputInf > 0) {
                StringBuilder diagnostics = new StringBuilder();
                diagnostics.append("P.A.N.I.C.! Op [").append(opName).append("] input Y contains ").append(inputInf).append(" Inf value(s)\n");
                diagnostics.append("\n=== INPUT DIAGNOSTICS ===\n");
                diagnostics.append("Inf in INPUT indicates the problem originated EARLIER in the computation.\n");
                appendArrayDiagnostics(diagnostics, "Input X", x);
                appendArrayDiagnostics(diagnostics, "Input Y", y);
                diagnostics.append("\n=== ACTION REQUIRED ===\n");
                diagnostics.append("Trace back to find which earlier operation produced the Inf values.\n");
                throw new ND4JOpProfilerException(diagnostics.toString());
            }
        }

        // Now check output - if inputs were valid but output has Inf, the operation itself generated it
        if (z != null) {
            int match = countInf(z);
            if (match > 0) {
                StringBuilder diagnostics = new StringBuilder();
                diagnostics.append("P.A.N.I.C.! Op [").append(opName).append("] output contains ").append(match).append(" Inf value(s)\n");
                diagnostics.append("\n=== INPUT DIAGNOSTICS (inputs were valid, issue is in THIS op) ===\n");

                appendArrayDiagnostics(diagnostics, "Input X", x);
                if (y != null) {
                    appendArrayDiagnostics(diagnostics, "Input Y", y);
                }
                appendArrayDiagnostics(diagnostics, "Output Z", z);

                diagnostics.append("\n=== POTENTIAL ROOT CAUSES ===\n");
                diagnostics.append("1. Divide by zero: Division by zero produces Inf\n");
                diagnostics.append("2. Numerical overflow: Values exceeded floating point range\n");
                diagnostics.append("3. Exponential overflow: exp() of large values\n");

                throw new ND4JOpProfilerException(diagnostics.toString());
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
                StringBuilder diagnostics = new StringBuilder();
                diagnostics.append("P.A.N.I.C.! Op [").append(opName).append("] input[").append(i).append("] contains ").append(match).append(" Inf value(s)\n");
                diagnostics.append("\n=== INPUT DIAGNOSTICS ===\n");
                diagnostics.append("Inf in INPUT indicates the problem originated EARLIER in the computation graph.\n");
                for (int j = 0; j < inArgs.size(); j++) {
                    appendArrayDiagnostics(diagnostics, "Input[" + j + "]", inArgs.get(j));
                }
                diagnostics.append("\n=== ACTION REQUIRED ===\n");
                diagnostics.append("Trace back to find which earlier operation produced the Inf values.\n");
                throw new ND4JOpProfilerException(diagnostics.toString());
            }
        }

        // Check outputs
        for (int i = 0; i < outArgs.size(); i++) {
            INDArray output = outArgs.get(i);
            int match = countInf(output);
            if (match > 0) {
                StringBuilder diagnostics = new StringBuilder();
                diagnostics.append("P.A.N.I.C.! Op [").append(opName).append("] output[").append(i).append("] contains ").append(match).append(" Inf value(s)\n");
                diagnostics.append("\n=== INPUT DIAGNOSTICS (inputs were valid, issue is in THIS op) ===\n");
                for (int j = 0; j < inArgs.size(); j++) {
                    appendArrayDiagnostics(diagnostics, "Input[" + j + "]", inArgs.get(j));
                }
                appendArrayDiagnostics(diagnostics, "Output[" + i + "]", output);
                diagnostics.append("\n=== POTENTIAL ROOT CAUSES ===\n");
                diagnostics.append("1. Divide by zero: Check if any input contains zeros used as divisors\n");
                diagnostics.append("2. Numerical overflow: Large values exceeded float range\n");
                diagnostics.append("3. Exponential overflow: exp() of large positive values\n");
                throw new ND4JOpProfilerException(diagnostics.toString());
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
                StringBuilder diagnostics = new StringBuilder();
                diagnostics.append("P.A.N.I.C.! Op [").append(opName).append("] input[").append(i).append("] contains ").append(match).append(" NaN value(s)\n");
                diagnostics.append("\n=== INPUT DIAGNOSTICS ===\n");
                diagnostics.append("NaN in INPUT indicates the problem originated EARLIER in the computation graph.\n");
                for (int j = 0; j < inArgs.size(); j++) {
                    appendArrayDiagnostics(diagnostics, "Input[" + j + "]", inArgs.get(j));
                }
                diagnostics.append("\n=== ACTION REQUIRED ===\n");
                diagnostics.append("Trace back to find which earlier operation produced the NaN values.\n");
                throw new ND4JOpProfilerException(diagnostics.toString());
            }
        }

        // Check outputs
        for (int i = 0; i < outArgs.size(); i++) {
            INDArray output = outArgs.get(i);
            int match = countNaN(output);
            if (match > 0) {
                StringBuilder diagnostics = new StringBuilder();
                diagnostics.append("P.A.N.I.C.! Op [").append(opName).append("] output[").append(i).append("] contains ").append(match).append(" NaN value(s)\n");
                diagnostics.append("\n=== INPUT DIAGNOSTICS (inputs were valid, issue is in THIS op) ===\n");
                for (int j = 0; j < inArgs.size(); j++) {
                    appendArrayDiagnostics(diagnostics, "Input[" + j + "]", inArgs.get(j));
                }
                appendArrayDiagnostics(diagnostics, "Output[" + i + "]", output);
                diagnostics.append("\n=== POTENTIAL ROOT CAUSES ===\n");
                diagnostics.append("1. Divide by zero: 0/0 produces NaN\n");
                diagnostics.append("2. Invalid math: sqrt of negative, log of negative, acos/asin out of range\n");
                diagnostics.append("3. Inf - Inf or 0 * Inf operations\n");
                diagnostics.append("4. Numerical instability in normalization (e.g., norm of all-zero vector)\n");
                throw new ND4JOpProfilerException(diagnostics.toString());
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

    /**
     * Append detailed diagnostics for an array to help identify root cause of NaN/Inf.
     * Includes shape, dtype, statistics, and sample values.
     */
    private static void appendArrayDiagnostics(StringBuilder sb, String name, INDArray arr) {
        if (arr == null) {
            sb.append(name).append(": null\n");
            return;
        }
        if (arr.wasClosed()) {
            sb.append(name).append(": [CLOSED]\n");
            return;
        }
        if (arr.isEmpty()) {
            sb.append(name).append(": [EMPTY] shape=").append(java.util.Arrays.toString(arr.shape())).append("\n");
            return;
        }

        sb.append(name).append(":\n");
        sb.append("  Shape: ").append(java.util.Arrays.toString(arr.shape())).append("\n");
        sb.append("  DataType: ").append(arr.dataType()).append("\n");
        sb.append("  Length: ").append(arr.length()).append("\n");

        // Compute statistics (safely)
        try {
            if (arr.dataType().isFPType()) {
                // Sample values (first few and last few)
                int sampleSize = (int) Math.min(5, arr.length());
                sb.append("  First ").append(sampleSize).append(" values: [");
                for (int i = 0; i < sampleSize; i++) {
                    if (i > 0) sb.append(", ");
                    sb.append(arr.getDouble(i));
                }
                sb.append("]\n");

                if (arr.length() > 10) {
                    sb.append("  Last ").append(sampleSize).append(" values: [");
                    for (int i = 0; i < sampleSize; i++) {
                        if (i > 0) sb.append(", ");
                        sb.append(arr.getDouble(arr.length() - sampleSize + i));
                    }
                    sb.append("]\n");
                }

                // Check for zeros (important for divide-by-zero)
                int zeroCount = countZerosSimple(arr);
                if (zeroCount > 0) {
                    sb.append("  *** CONTAINS ").append(zeroCount).append(" ZERO VALUE(S) - potential divide-by-zero! ***\n");
                }

                // Check for very small values that might cause numerical instability
                int tinyCount = countTinyValuesSimple(arr);
                if (tinyCount > 0) {
                    sb.append("  *** CONTAINS ").append(tinyCount).append(" TINY VALUE(S) (< 1e-30) - potential numerical instability! ***\n");
                }

                // Count existing NaN/Inf in this array
                int nanCount = countNaNSimple(arr);
                int infCount = countInfSimple(arr);
                if (nanCount > 0) {
                    sb.append("  *** CONTAINS ").append(nanCount).append(" NaN VALUE(S) ***\n");
                }
                if (infCount > 0) {
                    sb.append("  *** CONTAINS ").append(infCount).append(" Inf VALUE(S) ***\n");
                }

                // Min/Max (useful for detecting overflow-prone values)
                double min = findMinSimple(arr);
                double max = findMaxSimple(arr);
                sb.append("  Min: ").append(min).append(", Max: ").append(max).append("\n");
            }
        } catch (Exception e) {
            sb.append("  [Error computing statistics: ").append(e.getMessage()).append("]\n");
        }
    }

    /**
     * Count zeros in array (for divide-by-zero detection)
     */
    private static int countZerosSimple(INDArray arr) {
        if (arr == null || arr.isEmpty() || arr.wasClosed()) return 0;
        int count = 0;
        long len = Math.min(arr.length(), 10000);
        for (long i = 0; i < len; i++) {
            double val = arr.getDouble(i);
            if (val == 0.0) count++;
        }
        return count;
    }

    /**
     * Count tiny values that might cause numerical instability
     */
    private static int countTinyValuesSimple(INDArray arr) {
        if (arr == null || arr.isEmpty() || arr.wasClosed()) return 0;
        int count = 0;
        long len = Math.min(arr.length(), 10000);
        for (long i = 0; i < len; i++) {
            double val = Math.abs(arr.getDouble(i));
            if (val > 0 && val < 1e-30) count++;
        }
        return count;
    }

    /**
     * Find minimum value in array
     */
    private static double findMinSimple(INDArray arr) {
        if (arr == null || arr.isEmpty() || arr.wasClosed()) return Double.NaN;
        double min = Double.MAX_VALUE;
        long len = Math.min(arr.length(), 10000);
        for (long i = 0; i < len; i++) {
            double val = arr.getDouble(i);
            if (!Double.isNaN(val) && val < min) min = val;
        }
        return min == Double.MAX_VALUE ? Double.NaN : min;
    }

    /**
     * Find maximum value in array
     */
    private static double findMaxSimple(INDArray arr) {
        if (arr == null || arr.isEmpty() || arr.wasClosed()) return Double.NaN;
        double max = -Double.MAX_VALUE;
        long len = Math.min(arr.length(), 10000);
        for (long i = 0; i < len; i++) {
            double val = arr.getDouble(i);
            if (!Double.isNaN(val) && val > max) max = val;
        }
        return max == -Double.MAX_VALUE ? Double.NaN : max;
    }
}
