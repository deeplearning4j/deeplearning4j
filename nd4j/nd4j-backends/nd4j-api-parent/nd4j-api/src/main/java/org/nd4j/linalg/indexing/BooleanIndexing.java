/*-
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 *
 */

package org.nd4j.linalg.indexing;

import com.google.common.base.Function;
import lombok.NonNull;
import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.accum.MatchCondition;
import org.nd4j.linalg.api.ops.impl.indexaccum.FirstIndex;
import org.nd4j.linalg.api.ops.impl.indexaccum.LastIndex;
import org.nd4j.linalg.api.ops.impl.transforms.comparison.Choose;
import org.nd4j.linalg.api.ops.impl.transforms.comparison.CompareAndReplace;
import org.nd4j.linalg.api.ops.impl.transforms.comparison.CompareAndSet;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.api.shape.loop.coordinatefunction.CoordinateFunction;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.conditions.BaseCondition;
import org.nd4j.linalg.indexing.conditions.Condition;

import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Boolean indexing
 *
 * @author Adam Gibson
 */
public class BooleanIndexing {
    /**
     * And
     *
     * @param n
     * @param cond
     * @return
     */
    public static boolean and(IComplexNDArray n, Condition cond) {
        boolean ret = true;
        IComplexNDArray linear = n.linearView();
        for (int i = 0; i < linear.length(); i++) {
            ret = ret && cond.apply(linear.getComplex(i));
        }

        return ret;
    }

    /**
     * Or over the whole ndarray given some condition
     *
     * @param n
     * @param cond
     * @return
     */
    public static boolean or(IComplexNDArray n, Condition cond) {
        boolean ret = false;
        IComplexNDArray linear = n.linearView();
        for (int i = 0; i < linear.length(); i++) {
            ret = ret || cond.apply(linear.getComplex(i));
        }

        return ret;
    }


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
            long val = (long) Nd4j.getExecutioner().exec(new MatchCondition(n, cond), Integer.MAX_VALUE).getDouble(0);

            if (val == n.lengthLong())
                return true;
            else
                return false;

        } else {
            boolean ret = true;
            final AtomicBoolean a = new AtomicBoolean(ret);
            Shape.iterate(n, new CoordinateFunction() {
                @Override
                public void process(long[]... coord) {
                    if (a.get())
                        a.compareAndSet(true, a.get() && cond.apply(n.getDouble(coord[0])));
                }
            });

            return a.get();
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

        MatchCondition op = new MatchCondition(n, condition);
        INDArray arr = Nd4j.getExecutioner().exec(op, dimension);
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

        MatchCondition op = new MatchCondition(n, condition);
        INDArray arr = Nd4j.getExecutioner().exec(op, dimension);

        // FIXME: int cast

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
            long val = (long) Nd4j.getExecutioner().exec(new MatchCondition(n, cond), Integer.MAX_VALUE).getDouble(0);

            if (val > 0)
                return true;
            else
                return false;

        } else {
            boolean ret = false;
            final AtomicBoolean a = new AtomicBoolean(ret);
            Shape.iterate(n, new CoordinateFunction() {
                @Override
                public void process(long[]... coord) {
                    if (!a.get())
                        a.compareAndSet(false, a.get() || cond.apply(n.getDouble(coord[0])));
                }
            });

            return a.get();
        }
    }

    /**
     * Based on the matching elements
     * op to based on condition to with function function
     *
     * @param to        the ndarray to op
     * @param condition the condition on op
     * @param function  the function to apply the op to
     */
    public static void applyWhere(final INDArray to, final Condition condition,
                                  final Function<Number, Number> function) {
        // keep original java implementation for dynamic

        Shape.iterate(to, new CoordinateFunction() {
            @Override
            public void process(long[]... coord) {
                if (condition.apply(to.getDouble(coord[0])))
                    to.putScalar(coord[0], function.apply(to.getDouble(coord[0])).doubleValue());

            }
        });
    }

    /**
     * This method sets provided number to all elements which match specified condition
     *
     * @param to
     * @param condition
     * @param number
     */
    public static void applyWhere(final INDArray to, final Condition condition, final Number number) {
        if (condition instanceof BaseCondition) {
            // for all static conditions we go native

            Nd4j.getExecutioner().exec(new CompareAndSet(to, number.doubleValue(), condition));

        } else {
            final double value = number.doubleValue();

            final Function<Number, Number> dynamic = new Function<Number, Number>() {
                @Override
                public Number apply(Number number) {
                    return value;
                }
            };

            Shape.iterate(to, new CoordinateFunction() {
                @Override
                public void process(long[]... coord) {
                    if (condition.apply(to.getDouble(coord[0])))
                        to.putScalar(coord[0], dynamic.apply(to.getDouble(coord[0])).doubleValue());

                }
            });
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

        if (to.lengthLong() != from.lengthLong())
            throw new IllegalStateException("Mis matched length for to and from");

        Nd4j.getExecutioner().exec(new CompareAndSet(to, from, condition));
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

        if (to.lengthLong() != from.lengthLong())
            throw new IllegalStateException("Mis matched length for to and from");

        Nd4j.getExecutioner().exec(new CompareAndReplace(to, from, condition));
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
        Choose choose = new Choose(input,condition);
        Nd4j.getExecutioner().exec(choose);
        int secondOutput = choose.getOutputArgument(1).getInt(0);
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
        Nd4j.getExecutioner().exec(choose);
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
     * Based on the matching elements
     * op to based on condition to with function function
     *
     * @param to        the ndarray to op
     * @param condition the condition on op
     * @param function  the function to apply the op to
     */
    public static void applyWhere(final INDArray to, final Condition condition, final Function<Number, Number> function,
                                  final Function<Number, Number> alternativeFunction) {
        Shape.iterate(to, new CoordinateFunction() {
            @Override
            public void process(long[]... coord) {
                if (condition.apply(to.getDouble(coord[0]))) {
                    to.putScalar(coord[0], function.apply(to.getDouble(coord[0])).doubleValue());
                } else {
                    to.putScalar(coord[0], alternativeFunction.apply(to.getDouble(coord[0])).doubleValue());
                }
            }
        });

    }

    /**
     * Based on the matching elements
     * op to based on condition to with function function
     *
     * @param to        the ndarray to op
     * @param condition the condition on op
     * @param function  the function to apply the op to
     */
    public static void applyWhere(IComplexNDArray to, Condition condition,
                                  Function<IComplexNumber, IComplexNumber> function) {
        IComplexNDArray linear = to.linearView();
        for (int i = 0; i < linear.linearView().length(); i++) {
            if (condition.apply(linear.getDouble(i))) {
                linear.putScalar(i, function.apply(linear.getComplex(i)));
            }
        }
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
        return Nd4j.scalar((double) idx.getFinalResult());
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

        return Nd4j.getExecutioner().exec(new FirstIndex(array, condition), dimension);
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
        return Nd4j.scalar((double) idx.getFinalResult());
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

        return Nd4j.getExecutioner().exec(new LastIndex(array, condition), dimension);
    }
}
