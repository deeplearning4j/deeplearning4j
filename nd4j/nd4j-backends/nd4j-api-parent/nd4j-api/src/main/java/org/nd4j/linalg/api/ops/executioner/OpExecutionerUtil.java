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

    public static void checkForNaN(INDArray z) {
        if(z == null || z.isEmpty() || !z.dataType().isFPType() || z.wasClosed())
            return;

        int match = 0;
        if (!z.isScalar()) {
            MatchCondition condition = new MatchCondition(z, Conditions.isNan());
            INDArray result = null;

            // The exec(condition) call below will trigger profilingConfigurableHookOut(),
            // which would call checkForNaN() again, creating infinite recursion.
            // Save current profiler state, disable checking, execute, then restore.
            OpProfiler profiler = Nd4j.getExecutioner().getProfiler();
            boolean wasCheckingNaN = profiler != null && profiler.getConfig().isCheckForNAN();
            boolean wasCheckingInf = profiler != null && profiler.getConfig().isCheckForINF();
            if (profiler != null) {
                profiler.getConfig().setCheckForNAN(false);
                profiler.getConfig().setCheckForINF(false);
            }

            try {
                result = Nd4j.getExecutioner().exec(condition);
                match = result.getInt(0);
            } finally {
                // Restore profiler state FIRST, before cleanup
                if (profiler != null) {
                    profiler.getConfig().setCheckForNAN(wasCheckingNaN);
                    profiler.getConfig().setCheckForINF(wasCheckingInf);
                }
                // Clean up result array
                if (result != null) {
                    try {
                        if (result.data() != null) {
                            result.data().close();
                        }
                        result.close();
                    } catch (Exception e) {
                        // Ignore close errors
                    }
                }
                // Clean up MatchCondition's internal dimensions array
                // clearArrays() alone is not sufficient - causes DataBuffer leaks
                if (condition.dimensions() != null) {
                    try {
                        if (condition.dimensions().data() != null) {
                            condition.dimensions().data().close();
                        }
                        condition.dimensions().close();
                    } catch (Exception e) {
                        // Ignore close errors
                    }
                }
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

        int match = 0;
        if (!z.isScalar()) {
            MatchCondition condition = new MatchCondition(z, Conditions.isInfinite());
            INDArray result = null;

            // The exec(condition) call below will trigger profilingConfigurableHookOut(),
            // which would call checkForInf() again, creating infinite recursion.
            // Save current profiler state, disable checking, execute, then restore.
            OpProfiler profiler = Nd4j.getExecutioner().getProfiler();
            boolean wasCheckingNaN = profiler != null && profiler.getConfig().isCheckForNAN();
            boolean wasCheckingInf = profiler != null && profiler.getConfig().isCheckForINF();
            if (profiler != null) {
                profiler.getConfig().setCheckForNAN(false);
                profiler.getConfig().setCheckForINF(false);
            }

            try {
                result = Nd4j.getExecutioner().exec(condition);
                match = result.getInt(0);
            } finally {
                // Restore profiler state FIRST, before cleanup
                if (profiler != null) {
                    profiler.getConfig().setCheckForNAN(wasCheckingNaN);
                    profiler.getConfig().setCheckForINF(wasCheckingInf);
                }
                // Clean up result array AND its data buffer
                if (result != null) {
                    try {
                        // Close data buffer first
                        if (result.data() != null) {
                            result.data().close();
                        }
                        result.close();
                    } catch (Exception e) {
                        // Ignore close errors
                    }
                }
                // Clean up MatchCondition's internal dimensionz array
                if (condition.dimensions() != null) {
                    try {
                        if (condition.dimensions().data() != null) {
                            condition.dimensions().data().close();
                        }
                        condition.dimensions().close();
                    } catch (Exception e) {
                        // Ignore close errors
                    }
                }
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
            checkForNaN(z);
        }
    }

    public static void checkForInf(Op op, OpContext oc) {
        INDArray z = oc != null ? oc.getOutputArray(0) : op.z();
        if (z != null && !(op instanceof MatchCondition) && !(op instanceof CompareAndSet) && !(op instanceof CompareAndReplace)) {
            checkForInf(z);
        }
    }

    public static void checkForInf(CustomOp op, OpContext oc) {
        List<INDArray> inArgs = oc != null ? oc.getInputArrays() : op.inputArguments();
        List<INDArray> outArgs = oc != null ? oc.getOutputArrays() : op.outputArguments();

        for (val input: inArgs)
            checkForInf(input);

        for (val output: outArgs)
            checkForInf(output);
    }


    public static void checkForNaN(CustomOp op, OpContext oc) {
        List<INDArray> inArgs = oc != null ? oc.getInputArrays() : op.inputArguments();
        List<INDArray> outArgs = oc != null ? oc.getOutputArrays() : op.outputArguments();

        for (val input: inArgs)
            checkForNaN(input);

        for (val output: outArgs)
            checkForNaN(output);
    }
}
