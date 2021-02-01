/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.linalg.indexing;

import lombok.NonNull;
import lombok.val;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.indexaccum.FirstIndex;
import org.nd4j.linalg.api.ops.impl.indexaccum.LastIndex;
import org.nd4j.linalg.api.ops.impl.reduce.longer.MatchCondition;
import org.nd4j.linalg.api.ops.impl.transforms.comparison.CompareAndReplace;
import org.nd4j.linalg.api.ops.impl.transforms.comparison.CompareAndSet;
import org.nd4j.linalg.api.ops.impl.transforms.custom.Choose;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.conditions.BaseCondition;
import org.nd4j.linalg.indexing.conditions.Condition;

import java.util.List;

/**
 * Boolean indexing
 *
 * @author Adam Gibson
 */
public class BooleanIndexing {

    /**
     * And over the whole ndarray given some condition
     *
     * @param n    the ndarray to test
     * @param cond the condition to test against
     * @return true if all of the elements meet the specified
     * condition false otherwise
     */
    public static boolean and(final INDArray n, final Condition cond) {
        if (cond instanceof BaseCondition) {
            long val = (long) Nd4j.getExecutioner().exec(new MatchCondition(n, cond)).getDouble(0);

            if (val == n.length())
                return true;
            else
                return false;

        } else {
            throw new RuntimeException("Can only execute BaseCondition conditions using this method");
        }
    }

    /**
     * And over the whole ndarray given some condition, with respect to dimensions
     *
     * @param n    the ndarray to test
     * @param condition the condition to test against
     * @return true if all of the elements meet the specified
     * condition false otherwise
     */
    public static boolean[] and(final INDArray n, final Condition condition, int... dimension) {
        if (!(condition instanceof BaseCondition))
            throw new UnsupportedOperationException("Only static Conditions are supported");

        MatchCondition op = new MatchCondition(n, condition, dimension);
        INDArray arr = Nd4j.getExecutioner().exec(op);
        boolean[] result = new boolean[(int) arr.length()];

        long tadLength = Shape.getTADLength(n.shape(), dimension);

        for (int i = 0; i < arr.length(); i++) {
            if (arr.getDouble(i) == tadLength)
                result[i] = true;
            else
                result[i] = false;
        }

        return result;
    }


    /**
     * Or over the whole ndarray given some condition, with respect to dimensions
     *
     * @param n    the ndarray to test
     * @param condition the condition to test against
     * @return true if all of the elements meet the specified
     * condition false otherwise
     */
    public static boolean[] or(final INDArray n, final Condition condition, int... dimension) {
        if (!(condition instanceof BaseCondition))
            throw new UnsupportedOperationException("Only static Conditions are supported");

        MatchCondition op = new MatchCondition(n, condition, dimension);
        INDArray arr = Nd4j.getExecutioner().exec(op);

        boolean[] result = new boolean[(int) arr.length()];

        for (int i = 0; i < arr.length(); i++) {
            if (arr.getDouble(i) > 0)
                result[i] = true;
            else
                result[i] = false;
        }

        return result;
    }

    /**
     * Or over the whole ndarray given some condition
     *
     * @param n
     * @param cond
     * @return
     */
    public static boolean or(final INDArray n, final Condition cond) {
        if (cond instanceof BaseCondition) {
            long val = (long) Nd4j.getExecutioner().exec(new MatchCondition(n, cond)).getDouble(0);

            if (val > 0)
                return true;
            else
                return false;

        } else {
            throw new RuntimeException("Can only execute BaseCondition conditions using this method");
        }
    }

    /**
     * This method does element-wise comparison
     * for 2 equal-sized matrices, for each element that matches Condition.
     * To is the array to apply the indexing to
     * from is a condition mask array (0 or 1).
     * This would come from the output of a bit masking method like:
     * {@link INDArray#gt(Number)},{@link INDArray#gte(Number)},
     * {@link INDArray#lt(Number)},..
     *
     * @param to the array to apply the condition to
     * @param from the mask array
     * @param condition the condition to apply
     */
    public static void assignIf(@NonNull INDArray to, @NonNull INDArray from, @NonNull Condition condition) {
        if (!(condition instanceof BaseCondition))
            throw new UnsupportedOperationException("Only static Conditions are supported");

        if (to.length() != from.length())
            throw new IllegalStateException("Mis matched length for to and from");

        Nd4j.getExecutioner().exec(new CompareAndSet(to, from, to, condition));
    }


    /**
     * This method does element-wise comparison for 2 equal-sized matrices, for each element that matches Condition
     *
     * @param to
     * @param from
     * @param condition
     */
    public static void replaceWhere(@NonNull INDArray to, @NonNull INDArray from, @NonNull Condition condition) {
        if (!(condition instanceof BaseCondition))
            throw new UnsupportedOperationException("Only static Conditions are supported");

        if (to.length() != from.length())
            throw new IllegalStateException("Mis matched length for to and from");

        Nd4j.getExecutioner().exec(new CompareAndReplace(to, from, to, condition));
    }

    /**
     * Choose from the inputs based on the given condition.
     * This returns a row vector of all elements fulfilling the
     * condition listed within the array for input
     * @param input the input to filter
     * @param condition the condition to filter based on
     * @return a row vector of the input elements that are true
     * ffor the given conditions
     */
    public static INDArray chooseFrom(@NonNull  INDArray[] input,@NonNull  Condition condition) {
        val choose = new Choose(input,condition);
        val outputs = Nd4j.exec(choose);
        int secondOutput = outputs[1].getInt(0);
        if(secondOutput < 1) {
            return null;
        }

        return choose.getOutputArgument(0);
    }

    /**
     * A minor shortcut for applying a bitmask to
     * a matrix
     * @param arr The array to apply the mask to
     * @param mask the mask to apply
     * @return the array with the mask applied
     */
    public static INDArray applyMask(INDArray arr,INDArray mask)  {
        return arr.mul(mask);
    }

    /**
     * A minor shortcut for applying a bitmask to
     * a matrix
     * @param arr The array to apply the mask to
     * @param mask the mask to apply
     * @return the array with the mask applied
     */
    public static INDArray applyMaskInPlace(INDArray arr,INDArray mask)  {
        return arr.muli(mask);
    }



    /**
     * Choose from the inputs based on the given condition.
     * This returns a row vector of all elements fulfilling the
     * condition listed within the array for input.
     * The double and integer arguments are only relevant
     * for scalar operations (like when you have a scalar
     * you are trying to compare each element in your input against)
     *
     * @param input the input to filter
     * @param tArgs the double args
     * @param iArgs the integer args
     * @param condition the condition to filter based on
     * @return a row vector of the input elements that are true
     * ffor the given conditions
     */
    public static INDArray chooseFrom(@NonNull  INDArray[] input, @NonNull  List<Double> tArgs, @NonNull List<Integer> iArgs, @NonNull Condition condition) {
        Choose choose = new Choose(input,iArgs,tArgs,condition);
        Nd4j.getExecutioner().execAndReturn(choose);
        int secondOutput = choose.getOutputArgument(1).getInt(0);
        if(secondOutput < 1) {
            return null;
        }

        INDArray ret =  choose.getOutputArgument(0).get(NDArrayIndex.interval(0,secondOutput));
        ret = ret.reshape(ret.length());
        return ret;
    }

    /**
     * This method does element-wise assessing for 2 equal-sized matrices, for each element that matches Condition
     *
     * @param to
     * @param set
     * @param condition
     */
    public static void replaceWhere(@NonNull INDArray to, @NonNull Number set, @NonNull Condition condition) {
        if (!(condition instanceof BaseCondition))
            throw new UnsupportedOperationException("Only static Conditions are supported");

        Nd4j.getExecutioner().exec(new CompareAndSet(to, set.doubleValue(), condition));
    }

    /**
     * This method returns first index matching given condition
     *
     * PLEASE NOTE: This method will return -1 value if condition wasn't met
     *
     * @param array
     * @param condition
     * @return
     */
    public static INDArray firstIndex(INDArray array, Condition condition) {
        if (!(condition instanceof BaseCondition))
            throw new UnsupportedOperationException("Only static Conditions are supported");

        FirstIndex idx = new FirstIndex(array, condition);
        Nd4j.getExecutioner().exec(idx);
        return Nd4j.scalar(DataType.LONG, idx.getFinalResult().longValue());
    }

    /**
     * This method returns first index matching given condition along given dimensions
     *
     * PLEASE NOTE: This method will return -1 values for missing conditions
     *
     * @param array
     * @param condition
     * @param dimension
     * @return
     */
    public static INDArray firstIndex(INDArray array, Condition condition, int... dimension) {
        if (!(condition instanceof BaseCondition))
            throw new UnsupportedOperationException("Only static Conditions are supported");

        return Nd4j.getExecutioner().exec(new FirstIndex(array, condition, dimension));
    }


    /**
     * This method returns last index matching given condition
     *
     * PLEASE NOTE: This method will return -1 value if condition wasn't met
     *
     * @param array
     * @param condition
     * @return
     */
    public static INDArray lastIndex(INDArray array, Condition condition) {
        if (!(condition instanceof BaseCondition))
            throw new UnsupportedOperationException("Only static Conditions are supported");

        LastIndex idx = new LastIndex(array, condition);
        Nd4j.getExecutioner().exec(idx);
        return Nd4j.scalar(DataType.LONG, idx.getFinalResult().longValue());
    }

    /**
     * This method returns first index matching given condition along given dimensions
     *
     * PLEASE NOTE: This method will return -1 values for missing conditions
     *
     * @param array
     * @param condition
     * @param dimension
     * @return
     */
    public static INDArray lastIndex(INDArray array, Condition condition, int... dimension) {
        if (!(condition instanceof BaseCondition))
            throw new UnsupportedOperationException("Only static Conditions are supported");

        return Nd4j.getExecutioner().exec(new LastIndex(array, condition, dimension));
    }
}
