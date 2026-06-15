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

package org.nd4j.linalg.exception;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.CustomOp;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.api.ops.OpContext;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;
import java.util.List;

/**
 * Utility class for building rich exception messages with array provenance information.
 * <p>
 * When an op execution fails, this class helps gather detailed diagnostic information including:
 * <ul>
 *   <li>Array shapes, data types, and strides</li>
 *   <li>Java allocation stack traces (if funcTrace is enabled)</li>
 *   <li>Variable names from SameDiff context</li>
 *   <li>Memory state (constant, attached to workspace, etc.)</li>
 *   <li>Data buffer pointers for debugging native crashes</li>
 * </ul>
 * <p>
 * This information is invaluable for debugging use-after-free and other memory corruption issues.
 *
 * Adam Gibson
 */
@Slf4j
public class ND4JOpExceptionUtils {

    private static final int MAX_STACK_FRAMES = 15;
    private static final int MAX_DATA_PREVIEW = 10;

    /**
     * Creates a detailed exception for an op execution failure.
     *
     * @param opName The name of the operation that failed
     * @param nativeErrorMessage The error message from native code (may be null)
     * @param context The op context with input/output arrays
     * @param cause The underlying exception (may be null)
     * @return A RuntimeException with detailed diagnostic information
     */
    public static RuntimeException opExecutionException(String opName, String nativeErrorMessage,
                                                        OpContext context, Throwable cause) {
        StringBuilder sb = new StringBuilder();
        sb.append("Op [").append(opName).append("] execution failed");

        if (nativeErrorMessage != null && !nativeErrorMessage.isEmpty()) {
            sb.append("\n\n=== Native Error ===\n");
            sb.append(nativeErrorMessage);
        }

        if (context != null) {
            appendContextInfo(sb, context);
        }

        String message = sb.toString();

        if (cause instanceof ND4JIllegalStateException) {
            return (ND4JIllegalStateException) cause;
        }

        if (cause != null) {
            return new RuntimeException(message, cause);
        }
        return new RuntimeException(message);
    }

    /**
     * Creates a detailed exception for an op execution failure with a CustomOp.
     *
     * @param op The operation that failed
     * @param nativeErrorMessage The error message from native code
     * @param cause The underlying exception
     * @return A RuntimeException with detailed diagnostic information
     */
    public static RuntimeException opExecutionException(CustomOp op, String nativeErrorMessage, Throwable cause) {
        StringBuilder sb = new StringBuilder();
        sb.append("Op [").append(op.opName()).append("] execution failed");

        if (nativeErrorMessage != null && !nativeErrorMessage.isEmpty()) {
            sb.append("\n\n=== Native Error ===\n");
            sb.append(nativeErrorMessage);
        }

        sb.append("\n\n=== Op Info ===\n");
        sb.append("Op class: ").append(op.getClass().getName()).append("\n");
        sb.append("Op hash: ").append(op.opHash()).append("\n");

        // If this is a DifferentialFunction, include SameDiff context
        if (op instanceof DifferentialFunction) {
            appendDifferentialFunctionInfo(sb, (DifferentialFunction) op);
        }

        // Input arrays
        List<INDArray> inputs = op.inputArguments();
        if (inputs != null && !inputs.isEmpty()) {
            sb.append("\n=== Input Arrays (").append(inputs.size()).append(") ===\n");
            for (int i = 0; i < inputs.size(); i++) {
                appendArrayInfo(sb, "Input[" + i + "]", inputs.get(i), i);
            }
        }

        // Output arrays
        List<INDArray> outputs = op.outputArguments();
        if (outputs != null && !outputs.isEmpty()) {
            sb.append("\n=== Output Arrays (").append(outputs.size()).append(") ===\n");
            for (int i = 0; i < outputs.size(); i++) {
                appendArrayInfo(sb, "Output[" + i + "]", outputs.get(i), i);
            }
        }

        // Integer arguments
        long[] iArgs = op.iArgs();
        if (iArgs != null && iArgs.length > 0) {
            sb.append("\n=== Integer Args ===\n");
            sb.append(Arrays.toString(iArgs)).append("\n");
        }

        // Float arguments
        double[] tArgs = op.tArgs();
        if (tArgs != null && tArgs.length > 0) {
            sb.append("\n=== Float Args ===\n");
            sb.append(Arrays.toString(tArgs)).append("\n");
        }

        String message = sb.toString();

        if (cause instanceof ND4JIllegalStateException) {
            return (ND4JIllegalStateException) cause;
        }

        if (cause != null) {
            return new RuntimeException(message, cause);
        }
        return new RuntimeException(message);
    }

    /**
     * Creates a detailed exception for a legacy op execution failure.
     *
     * @param op The operation that failed
     * @param nativeErrorMessage The error message from native code
     * @return A RuntimeException with detailed diagnostic information
     */
    public static RuntimeException opExecutionException(Op op, String nativeErrorMessage) {
        StringBuilder sb = new StringBuilder();
        sb.append("Op [").append(op.opName()).append("] execution failed");

        if (nativeErrorMessage != null && !nativeErrorMessage.isEmpty()) {
            sb.append("\n\n=== Native Error ===\n");
            sb.append(nativeErrorMessage);
        }

        sb.append("\n\n=== Op Info ===\n");
        sb.append("Op class: ").append(op.getClass().getName()).append("\n");

        // X array
        if (op.x() != null) {
            sb.append("\n=== X Array ===\n");
            appendArrayInfo(sb, "X", op.x(), 0);
        }

        // Y array
        if (op.y() != null) {
            sb.append("\n=== Y Array ===\n");
            appendArrayInfo(sb, "Y", op.y(), 1);
        }

        // Z array
        if (op.z() != null) {
            sb.append("\n=== Z Array ===\n");
            appendArrayInfo(sb, "Z", op.z(), 2);
        }

        return new RuntimeException(sb.toString());
    }

    /**
     * Appends OpContext information to the message.
     */
    private static void appendContextInfo(StringBuilder sb, OpContext context) {
        // Input arrays
        List<INDArray> inputs = context.getInputArrays();
        if (inputs != null && !inputs.isEmpty()) {
            sb.append("\n\n=== Input Arrays (").append(inputs.size()).append(") ===\n");
            for (int i = 0; i < inputs.size(); i++) {
                appendArrayInfo(sb, "Input[" + i + "]", inputs.get(i), i);
            }
        }

        // Output arrays
        List<INDArray> outputs = context.getOutputArrays();
        if (outputs != null && !outputs.isEmpty()) {
            sb.append("\n=== Output Arrays (").append(outputs.size()).append(") ===\n");
            for (int i = 0; i < outputs.size(); i++) {
                appendArrayInfo(sb, "Output[" + i + "]", outputs.get(i), i);
            }
        }

        // Arguments
        List<Long> iArgs = context.getIArguments();
        if (iArgs != null && !iArgs.isEmpty()) {
            sb.append("\n=== Integer Args ===\n").append(iArgs).append("\n");
        }

        List<Double> tArgs = context.getTArguments();
        if (tArgs != null && !tArgs.isEmpty()) {
            sb.append("\n=== Float Args ===\n").append(tArgs).append("\n");
        }

        List<Boolean> bArgs = context.getBArguments();
        if (bArgs != null && !bArgs.isEmpty()) {
            sb.append("\n=== Boolean Args ===\n").append(bArgs).append("\n");
        }
    }

    /**
     * Appends SameDiff/DifferentialFunction context information.
     * This includes variable names, op location in graph, and creation stack traces.
     */
    private static void appendDifferentialFunctionInfo(StringBuilder sb, DifferentialFunction func) {
        try {
            sb.append("\n=== SameDiff Context ===\n");

            // Op own name
            String ownName = func.getOwnName();
            if (ownName != null) {
                sb.append("  Op Name (in graph): ").append(ownName).append("\n");
            }

            // SameDiff reference
            SameDiff sd = func.getSameDiff();
            if (sd != null) {
                // Input variable names
                String[] inputNames = sd.getInputsForOp(func);
                if (inputNames != null && inputNames.length > 0) {
                    sb.append("  Input Variables: ").append(Arrays.toString(inputNames)).append("\n");
                    // Show details for each input variable
                    for (String varName : inputNames) {
                        try {
                            SDVariable sdVar = sd.getVariable(varName);
                            if (sdVar != null) {
                                sb.append("    ").append(varName).append(": ");
                                sb.append("type=").append(sdVar.getVariableType());
                                sb.append(", dtype=").append(sdVar.dataType());
                                long[] shape = sdVar.getShape();
                                if (shape != null) {
                                    sb.append(", shape=").append(Arrays.toString(shape));
                                }
                                // Check if array exists and show its state
                                INDArray arr = sdVar.getArr();
                                if (arr != null) {
                                    sb.append(", arrayId=").append(arr.getId());
                                    if (arr.data() != null) {
                                        sb.append(", isConstant=").append(arr.data().isConstant());
                                    }
                                } else {
                                    sb.append(", arr=null");
                                }
                                sb.append("\n");
                            }
                        } catch (Exception e) {
                            sb.append("    ").append(varName).append(": [error: ").append(e.getMessage()).append("]\n");
                        }
                    }
                }

                // Output variable names
                String[] outputNames = sd.getOutputsForOp(func);
                if (outputNames != null && outputNames.length > 0) {
                    sb.append("  Output Variables: ").append(Arrays.toString(outputNames)).append("\n");
                }
            } else {
                sb.append("  SameDiff: [not available - standalone op]\n");
            }

            // Creation location (where the op was defined)
            StackTraceElement creationLoc = func.getCreationLocation();
            if (creationLoc != null) {
                sb.append("  Creation Location: ").append(creationLoc).append("\n");
            }

            // SameDiff calls stack (relevant calls from SameDiff class)
            StackTraceElement[] sdCalls = func.getSameDiffCalls();
            if (sdCalls != null && sdCalls.length > 0) {
                sb.append("  SameDiff Call Stack:\n");
                int framesToShow = Math.min(sdCalls.length, 5);
                for (int i = 0; i < framesToShow; i++) {
                    sb.append("    at ").append(sdCalls[i]).append("\n");
                }
                if (sdCalls.length > 5) {
                    sb.append("    ... ").append(sdCalls.length - 5).append(" more frames\n");
                }
            }

            // Full creation call stack (if funcTrace enabled or in debug mode)
            if (Nd4j.getEnvironment().isFuncTracePrintAllocate()) {
                StackTraceElement[] creationStack = func.getCreationCallStack();
                if (creationStack != null && creationStack.length > 0) {
                    sb.append("  Full Creation Stack:\n");
                    int framesToShow = Math.min(creationStack.length, MAX_STACK_FRAMES);
                    for (int i = 0; i < framesToShow; i++) {
                        sb.append("    at ").append(creationStack[i]).append("\n");
                    }
                    if (creationStack.length > MAX_STACK_FRAMES) {
                        sb.append("    ... ").append(creationStack.length - MAX_STACK_FRAMES).append(" more frames\n");
                    }
                }
            }

        } catch (Exception e) {
            sb.append("  [Error getting SameDiff context: ").append(e.getMessage()).append("]\n");
        }
    }

    /**
     * Appends detailed information about an array to the message.
     */
    private static void appendArrayInfo(StringBuilder sb, String label, INDArray array, int index) {
        sb.append("\n--- ").append(label).append(" ---\n");

        if (array == null) {
            sb.append("  [NULL ARRAY]\n");
            return;
        }

        try {
            // Basic info
            sb.append("  Shape: ").append(Arrays.toString(array.shape())).append("\n");
            sb.append("  DataType: ").append(array.dataType()).append("\n");
            sb.append("  Rank: ").append(array.rank()).append("\n");
            sb.append("  Length: ").append(array.length()).append("\n");
            sb.append("  Stride: ").append(Arrays.toString(array.stride())).append("\n");
            sb.append("  Order: ").append(array.ordering()).append("\n");
            sb.append("  ArrayId: ").append(array.getId()).append("\n");

            // Memory state
            sb.append("  IsView: ").append(array.isView()).append("\n");
            sb.append("  IsAttached: ").append(array.isAttached()).append("\n");
            sb.append("  IsEmpty: ").append(array.isEmpty()).append("\n");

            // Raw shape info buffer - critical for debugging shape corruption
            DataBuffer shapeInfoBuf = array.shapeInfoDataBuffer();
            if (shapeInfoBuf != null) {
                try {
                    long[] rawShapeInfo = shapeInfoBuf.asLong();
                    sb.append("  RawShapeInfo: ").append(Arrays.toString(rawShapeInfo)).append("\n");
                    // Decode the raw shape info for clarity
                    if (rawShapeInfo.length > 0) {
                        sb.append("    [0]=rank: ").append(rawShapeInfo[0]).append("\n");
                        int rank = (int) rawShapeInfo[0];
                        if (rank > 0 && rawShapeInfo.length > rank) {
                            sb.append("    [1..").append(rank).append("]=shape: ");
                            for (int j = 1; j <= rank && j < rawShapeInfo.length; j++) {
                                sb.append(rawShapeInfo[j]).append(j < rank ? "," : "");
                            }
                            sb.append("\n");
                        }
                        if (rank >= 0 && rawShapeInfo.length > 2*rank + 1) {
                            int optionsIdx = rank == 0 ? 3 : (rank * 2 + 1);
                            if (optionsIdx < rawShapeInfo.length) {
                                long options = rawShapeInfo[optionsIdx];
                                sb.append("    [").append(optionsIdx).append("]=options: 0x")
                                  .append(Long.toHexString(options)).append("\n");
                                // Check for EMPTY flag (bit 3 = 8)
                                boolean isEmpty = (options & 8) == 8;
                                sb.append("      isEmpty flag: ").append(isEmpty).append("\n");
                            }
                        }
                    }
                    sb.append("  ShapeInfo.isConstant: ").append(shapeInfoBuf.isConstant()).append("\n");
                    sb.append("  ShapeInfo.length: ").append(shapeInfoBuf.length()).append("\n");
                    // Get pointer address for debugging memory issues
                    try {
                        if (shapeInfoBuf.pointer() != null) {
                            sb.append("  ShapeInfo.pointer: 0x")
                              .append(Long.toHexString(shapeInfoBuf.pointer().address())).append("\n");
                        }
                    } catch (Exception pe) {
                        sb.append("  ShapeInfo.pointer: [error: ").append(pe.getMessage()).append("]\n");
                    }
                } catch (Exception e) {
                    sb.append("  RawShapeInfo: [error reading: ").append(e.getMessage()).append("]\n");
                }
            } else {
                sb.append("  ShapeInfoBuffer: [NULL - possible use-after-free!]\n");
            }

            // Check JvmShapeInfo cache vs native buffer for discrepancy
            try {
                long[] jvmShape = array.shape();
                int jvmRank = array.rank();
                sb.append("  JvmShapeInfo.rank: ").append(jvmRank).append("\n");
                sb.append("  JvmShapeInfo.shape: ").append(Arrays.toString(jvmShape)).append("\n");
                // Note: if JvmShapeInfo differs from RawShapeInfo, it means the shape info was
                // corrupted AFTER the array was created (JvmShapeInfo is a snapshot at creation time)
            } catch (Exception e) {
                sb.append("  JvmShapeInfo: [error: ").append(e.getMessage()).append("]\n");
            }

            // Data buffer info
            DataBuffer data = array.data();
            if (data != null) {
                sb.append("  DataBuffer.isConstant: ").append(data.isConstant()).append("\n");
                sb.append("  DataBuffer.length: ").append(data.length()).append("\n");

                // Try to get native pointer addresses for debugging
                try {
                    if (data.pointer() != null) {
                        sb.append("  DataBuffer.pointer: 0x")
                          .append(Long.toHexString(data.pointer().address())).append("\n");
                    }
                } catch (Exception e) {
                    sb.append("  DataBuffer.pointer: [error: ").append(e.getMessage()).append("]\n");
                }

                // Data preview (only for small arrays or if funcTrace is enabled)
                if (array.length() <= MAX_DATA_PREVIEW && !array.isEmpty()) {
                    try {
                        sb.append("  DataPreview: ").append(Arrays.toString(array.toDoubleVector())).append("\n");
                    } catch (Exception e) {
                        sb.append("  DataPreview: [error reading data: ").append(e.getMessage()).append("]\n");
                    }
                } else if (Nd4j.getEnvironment().isFuncTracePrintAllocate() && array.length() > 0) {
                    // For larger arrays in debug mode, show first few values
                    try {
                        double[] preview = new double[Math.min((int)array.length(), MAX_DATA_PREVIEW)];
                        for (int i = 0; i < preview.length; i++) {
                            preview[i] = array.getDouble(i);
                        }
                        sb.append("  DataPreview (first ").append(preview.length).append("): ")
                          .append(Arrays.toString(preview)).append("...\n");
                    } catch (Exception e) {
                        sb.append("  DataPreview: [error: ").append(e.getMessage()).append("]\n");
                    }
                }
            } else {
                sb.append("  DataBuffer: [NULL - possible use-after-free!]\n");
            }

            // Allocation trace (if available)
            StackTraceElement[] allocationTrace = array.allocationTrace();
            if (allocationTrace != null && allocationTrace.length > 0) {
                sb.append("  AllocationTrace:\n");
                int framesToShow = Math.min(allocationTrace.length, MAX_STACK_FRAMES);
                for (int i = 0; i < framesToShow; i++) {
                    sb.append("    at ").append(allocationTrace[i]).append("\n");
                }
                if (allocationTrace.length > MAX_STACK_FRAMES) {
                    sb.append("    ... ").append(allocationTrace.length - MAX_STACK_FRAMES)
                      .append(" more frames\n");
                }
            } else {
                sb.append("  AllocationTrace: [not captured - enable with functrace build]\n");
            }

        } catch (Exception e) {
            sb.append("  [Error getting array info: ").append(e.getMessage()).append("]\n");
            sb.append("  This may indicate the array has been deallocated (use-after-free)\n");
        }
    }

    /**
     * Helper method to check native error code and throw enhanced exception if error.
     *
     * @param opName The name of the operation
     * @param context The op context (may be null)
     */
    public static void checkNativeError(String opName, OpContext context) {
        int errorCode = Nd4j.getNativeOps().lastErrorCode();
        if (errorCode != 0) {
            String errorMessage = Nd4j.getNativeOps().lastErrorMessage();
            throw opExecutionException(opName, errorMessage, context, null);
        }
    }

    /**
     * Helper method to check native error code and throw enhanced exception if error.
     *
     * @param op The CustomOp that was executed
     */
    public static void checkNativeError(CustomOp op) {
        int errorCode = Nd4j.getNativeOps().lastErrorCode();
        if (errorCode != 0) {
            String errorMessage = Nd4j.getNativeOps().lastErrorMessage();
            throw opExecutionException(op, errorMessage, null);
        }
    }

    /**
     * Helper method to check native error code and throw enhanced exception if error.
     *
     * @param op The legacy Op that was executed
     */
    public static void checkNativeError(Op op) {
        int errorCode = Nd4j.getNativeOps().lastErrorCode();
        if (errorCode != 0) {
            String errorMessage = Nd4j.getNativeOps().lastErrorMessage();
            throw opExecutionException(op, errorMessage);
        }
    }

    /**
     * Builds a string representation of an array for debug logging.
     * This is a lightweight version that doesn't include allocation traces.
     *
     * @param array The array to describe
     * @return A string with basic array info
     */
    public static String arrayToDebugString(INDArray array) {
        if (array == null) {
            return "[NULL]";
        }
        try {
            return String.format("INDArray[id=%d, shape=%s, dtype=%s, constant=%s]",
                    array.getId(),
                    Arrays.toString(array.shape()),
                    array.dataType(),
                    array.data() != null && array.data().isConstant());
        } catch (Exception e) {
            return "[Error: " + e.getMessage() + "]";
        }
    }
}
