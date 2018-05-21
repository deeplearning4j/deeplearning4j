/*-
 *  * Copyright 2017 Skymind, Inc.
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
 */

package org.datavec.api.transform.transform.doubletransform;

import lombok.Data;
import org.datavec.api.transform.MathFunction;
import org.datavec.api.writable.DoubleWritable;
import org.datavec.api.writable.Writable;
import org.nd4j.shade.jackson.annotation.JsonProperty;

/**
 * A simple transform to do common mathematical operations, such as sin(x), ceil(x), etc.<br>
 * Operations are specified by {@link MathFunction}
 *
 * @author Alex Black
 */
@Data
public class DoubleMathFunctionTransform extends BaseDoubleTransform {

    private MathFunction mathFunction;

    public DoubleMathFunctionTransform(@JsonProperty("columnName") String columnName,
                                       @JsonProperty("mathFunction") MathFunction mathFunction) {
        super(columnName);
        this.mathFunction = mathFunction;
    }

    @Override
    public Writable map(Writable w) {
        switch (mathFunction) {
            case ABS:
                return new DoubleWritable(Math.abs(w.toDouble()));
            case ACOS:
                return new DoubleWritable(Math.acos(w.toDouble()));
            case ASIN:
                return new DoubleWritable(Math.asin(w.toDouble()));
            case ATAN:
                return new DoubleWritable(Math.atan(w.toDouble()));
            case CEIL:
                return new DoubleWritable(Math.ceil(w.toDouble()));
            case COS:
                return new DoubleWritable(Math.cos(w.toDouble()));
            case COSH:
                return new DoubleWritable(Math.cosh(w.toDouble()));
            case EXP:
                return new DoubleWritable(Math.exp(w.toDouble()));
            case FLOOR:
                return new DoubleWritable(Math.floor(w.toDouble()));
            case LOG:
                return new DoubleWritable(Math.log(w.toDouble()));
            case LOG10:
                return new DoubleWritable(Math.log10(w.toDouble()));
            case SIGNUM:
                return new DoubleWritable(Math.signum(w.toDouble()));
            case SIN:
                return new DoubleWritable(Math.sin(w.toDouble()));
            case SINH:
                return new DoubleWritable(Math.sinh(w.toDouble()));
            case SQRT:
                return new DoubleWritable(Math.sqrt(w.toDouble()));
            case TAN:
                return new DoubleWritable(Math.tan(w.toDouble()));
            case TANH:
                return new DoubleWritable(Math.tanh(w.toDouble()));
            default:
                throw new RuntimeException("Unknown function: " + mathFunction);
        }
    }

    @Override
    public Object map(Object input) {
        double d = ((Number) input).doubleValue();
        switch (mathFunction) {
            case ABS:
                return Math.abs(d);
            case ACOS:
                return Math.acos(d);
            case ASIN:
                return Math.asin(d);
            case ATAN:
                return Math.atan(d);
            case CEIL:
                return Math.ceil(d);
            case COS:
                return Math.cos(d);
            case COSH:
                return Math.cosh(d);
            case EXP:
                return Math.exp(d);
            case FLOOR:
                return Math.floor(d);
            case LOG:
                return Math.log(d);
            case LOG10:
                return Math.log10(d);
            case SIGNUM:
                return Math.signum(d);
            case SIN:
                return Math.sin(d);
            case SINH:
                return Math.sinh(d);
            case SQRT:
                return Math.sqrt(d);
            case TAN:
                return Math.tan(d);
            case TANH:
                return Math.tanh(d);
            default:
                throw new RuntimeException("Unknown function: " + mathFunction);
        }
    }
}
