/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.nd4j.linalg.api.ops.executioner;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.CustomOp;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.api.ops.impl.accum.MatchCondition;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.nd4j.linalg.util.ArrayUtil;

import java.util.Arrays;

/**Utility functions for the DefaultOpExecutioner
 * @author Alex Black
 */
@Slf4j
public class OpExecutionerUtil {

    private OpExecutionerUtil() {}

    /** Can we do the op (X = Op(X)) directly on the arrays without breaking X up into 1d tensors first?
     * In general, this is possible if the elements of X are contiguous in the buffer, OR if every element
     * of X is at position offset+i*elementWiseStride in the buffer
     * */
    public static boolean canDoOpDirectly(INDArray x) {
        if (x.elementWiseStride() < 1)
            return false;
        if (x.isVector())
            return true;

        //For a single NDArray all we require is that the elements are contiguous in the buffer or every nth element

        //Full buffer -> implies all elements are contiguous (and match)
        long l1 = x.lengthLong();
        long dl1 = x.data().length();
        if (l1 == dl1)
            return true;

        //Strides are same as a zero offset NDArray -> all elements are contiguous (even if not offset 0)
        long[] shape1 = x.shape();
        long[] stridesAsInit =
                        (x.ordering() == 'c' ? ArrayUtil.calcStrides(shape1) : ArrayUtil.calcStridesFortran(shape1));
        boolean stridesSameAsInit = Arrays.equals(x.stride(), stridesAsInit);
        return stridesSameAsInit;
    }

    public static void checkForNaN(INDArray z) {
        if (Nd4j.getExecutioner().getProfilingMode() != OpExecutioner.ProfilingMode.NAN_PANIC
                        && Nd4j.getExecutioner().getProfilingMode() != OpExecutioner.ProfilingMode.ANY_PANIC)
            return;

        if(z.isEmpty())
            return;

        int match = 0;
        if (!z.isScalar()) {
            MatchCondition condition = new MatchCondition(z, Conditions.isNan());
            match = Nd4j.getExecutioner().exec(condition, Integer.MAX_VALUE).getInt(0);
        } else {
            if (z.data().dataType() == DataBuffer.Type.DOUBLE) {
                if (Double.isNaN(z.getDouble(0)))
                    match = 1;
            } else {
                if (Float.isNaN(z.getFloat(0)))
                    match = 1;
            }
        }

        if (match > 0)
            throw new ND4JIllegalStateException("P.A.N.I.C.! Op.Z() contains " + match + " NaN value(s): ");
    }

    public static void checkForAny(INDArray z) {
        checkForNaN(z);
        checkForInf(z);
    }

    public static void checkForInf(INDArray z) {
        if (Nd4j.getExecutioner().getProfilingMode() != OpExecutioner.ProfilingMode.INF_PANIC
                        && Nd4j.getExecutioner().getProfilingMode() != OpExecutioner.ProfilingMode.ANY_PANIC)
            return;

        if(z.isEmpty())
            return;

        int match = 0;
        if (!z.isScalar()) {
            MatchCondition condition = new MatchCondition(z, Conditions.isInfinite());
            match = Nd4j.getExecutioner().exec(condition, Integer.MAX_VALUE).getInt(0);
        } else {
            if (z.data().dataType() == DataBuffer.Type.DOUBLE) {
                if (Double.isInfinite(z.getDouble(0)))
                    match = 1;
            } else {
                if (Float.isInfinite(z.getFloat(0)))
                    match = 1;
            }
        }

        if (match > 0)
            throw new ND4JIllegalStateException("P.A.N.I.C.! Op.Z() contains " + match + " Inf value(s)");

    }

    public static void checkForNaN(Op op) {
        if (Nd4j.getExecutioner().getProfilingMode() != OpExecutioner.ProfilingMode.NAN_PANIC
                        && Nd4j.getExecutioner().getProfilingMode() != OpExecutioner.ProfilingMode.ANY_PANIC)
            return;

        if (op.z() != null && !(op instanceof MatchCondition)) {
            checkForNaN(op.z());
        }
    }

    public static void checkForInf(Op op) {
        if (Nd4j.getExecutioner().getProfilingMode() != OpExecutioner.ProfilingMode.INF_PANIC
                        && Nd4j.getExecutioner().getProfilingMode() != OpExecutioner.ProfilingMode.ANY_PANIC)
            return;

        if (op.z() != null && !(op instanceof MatchCondition)) {
            checkForInf(op.z());
        }
    }

    public static void checkForInf(CustomOp op) {
        if (Nd4j.getExecutioner().getProfilingMode() != OpExecutioner.ProfilingMode.INF_PANIC
                && Nd4j.getExecutioner().getProfilingMode() != OpExecutioner.ProfilingMode.ANY_PANIC)
            return;

        for (val input: op.inputArguments())
            checkForInf(input);

        for (val output: op.outputArguments())
            checkForInf(output);
    }


    public static void checkForNaN(CustomOp op) {
        if (Nd4j.getExecutioner().getProfilingMode() != OpExecutioner.ProfilingMode.NAN_PANIC
                && Nd4j.getExecutioner().getProfilingMode() != OpExecutioner.ProfilingMode.ANY_PANIC)
            return;

        for (val input: op.inputArguments())
            checkForNaN(input);

        for (val output: op.outputArguments())
            checkForNaN(output);
    }
}
