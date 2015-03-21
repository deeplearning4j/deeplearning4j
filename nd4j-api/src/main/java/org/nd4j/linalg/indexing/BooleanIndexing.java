/*
 * Copyright 2015 Skymind,Inc.
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 */

package org.nd4j.linalg.indexing;

import com.google.common.base.Function;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.indexing.conditions.Condition;

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
        boolean ret = true;
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
    public static boolean and(INDArray n, Condition cond) {
        boolean ret = true;
        INDArray linear = n.linearView();
        for (int i = 0; i < linear.length(); i++) {
            ret = ret && cond.apply(linear.getFloat(i));
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
    public static boolean or(INDArray n, Condition cond) {
        boolean ret = true;
        INDArray linear = n.linearView();
        for (int i = 0; i < linear.length(); i++) {
            ret = ret || cond.apply(linear.getFloat(i));
        }

        return ret;
    }

    /**
     * Based on the matching elements
     * op to based on condition to with function function
     *
     * @param to        the ndarray to op
     * @param condition the condition on op
     * @param function  the function to apply the op to
     */
    public static void applyWhere(INDArray to, Condition condition, Function<Number, Number> function) {
        INDArray linear = to.linearView();
        for (int i = 0; i < linear.linearView().length(); i++) {
            if (linear.data().dataType() == (DataBuffer.FLOAT)) {
                if (condition.apply(linear.getFloat(i))) {
                    linear.putScalar(i, function.apply(linear.getFloat(i)).floatValue());
                }
            } else if (condition.apply(linear.getDouble(i)))
                linear.putScalar(i, function.apply(linear.getDouble(i)).doubleValue());


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
    public static void applyWhere(IComplexNDArray to, Condition condition, Function<IComplexNumber, IComplexNumber> function) {
        IComplexNDArray linear = to.linearView();
        for (int i = 0; i < linear.linearView().length(); i++) {
            if (condition.apply(linear.getFloat(i))) {
                linear.putScalar(i, function.apply(linear.getComplex(i)));
            }
        }
    }


}
