/*
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
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.api.shape.loop.coordinatefunction.CoordinateFunction;
import org.nd4j.linalg.indexing.conditions.Condition;

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
        boolean ret = true;
        final AtomicBoolean a = new AtomicBoolean(ret);
        Shape.iterate(n, new CoordinateFunction() {
            @Override
            public void process(int[]... coord) {
                if (a.get())
                    a.compareAndSet(true, a.get() && cond.apply(n.getFloat(coord[0])));
            }
        });

        return a.get();
    }

    /**
     * Or over the whole ndarray given some condition
     *
     * @param n
     * @param cond
     * @return
     */
    public static boolean or(final INDArray n, final Condition cond) {
        boolean ret = false;
        final AtomicBoolean a = new AtomicBoolean(ret);
        Shape.iterate(n, new CoordinateFunction() {
            @Override
            public void process(int[]... coord) {
                if (!a.get())
                    a.compareAndSet(false, a.get() || cond.apply(n.getFloat(coord[0])));
            }
        });

        return a.get();
    }

    /**
     * Based on the matching elements
     * op to based on condition to with function function
     *
     * @param to        the ndarray to op
     * @param condition the condition on op
     * @param function  the function to apply the op to
     */
    public static void applyWhere(final INDArray to, final Condition condition, final Function<Number, Number> function) {
        Shape.iterate(to, new CoordinateFunction() {
            @Override
            public void process(int[]... coord) {
                if(condition.apply(to.getDouble(coord[0])))
                    to.putScalar(coord[0], function.apply(to.getDouble(coord[0])).floatValue());

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
    public static void applyWhere(final INDArray to, final Condition condition, final Function<Number, Number> function,final Function<Number, Number> alternativeFunction) {
        Shape.iterate(to, new CoordinateFunction() {
            @Override
            public void process(int[]... coord) {
                if (condition.apply(to.getFloat(coord[0]))) {
                    to.putScalar(coord[0], function.apply(to.getDouble(coord[0])).floatValue());
                } else {
                    to.putScalar(coord[0], alternativeFunction.apply(to.getDouble(coord[0])).floatValue());
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
    public static void applyWhere(IComplexNDArray to, Condition condition, Function<IComplexNumber, IComplexNumber> function) {
        IComplexNDArray linear = to.linearView();
        for (int i = 0; i < linear.linearView().length(); i++) {
            if (condition.apply(linear.getFloat(i))) {
                linear.putScalar(i, function.apply(linear.getComplex(i)));
            }
        }
    }


}
