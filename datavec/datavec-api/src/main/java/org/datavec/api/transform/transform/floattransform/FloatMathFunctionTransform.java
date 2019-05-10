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

package org.datavec.api.transform.transform.floattransform;

import lombok.Data;
import org.datavec.api.transform.MathFunction;
import org.datavec.api.writable.FloatWritable;
import org.datavec.api.writable.Writable;
import org.nd4j.shade.jackson.annotation.JsonProperty;

/**
 * A simple transform to do common mathematical operations, such as sin(x), ceil(x), etc.<br>
 * Operations are specified by {@link MathFunction}
 *
 * @author Alex Black
 */
@Data
public class FloatMathFunctionTransform extends BaseFloatTransform {

    private MathFunction mathFunction;

    public FloatMathFunctionTransform(@JsonProperty("columnName") String columnName,
                                       @JsonProperty("mathFunction") MathFunction mathFunction) {
        super(columnName);
        this.mathFunction = mathFunction;
    }

    @Override
    public Writable map(Writable w) {
        switch (mathFunction) {
            case ABS:
                return new FloatWritable(Math.abs(w.toFloat()));
            case ACOS:
                return new FloatWritable((float)Math.acos(w.toFloat()));
            case ASIN:
                return new FloatWritable((float)Math.asin(w.toFloat()));
            case ATAN:
                return new FloatWritable((float)Math.atan(w.toFloat()));
            case CEIL:
                return new FloatWritable((float)Math.ceil(w.toFloat()));
            case COS:
                return new FloatWritable((float)Math.cos(w.toFloat()));
            case COSH:
                return new FloatWritable((float)Math.cosh(w.toFloat()));
            case EXP:
                return new FloatWritable((float)Math.exp(w.toFloat()));
            case FLOOR:
                return new FloatWritable((float)Math.floor(w.toFloat()));
            case LOG:
                return new FloatWritable((float)Math.log(w.toFloat()));
            case LOG10:
                return new FloatWritable((float)Math.log10(w.toFloat()));
            case SIGNUM:
                return new FloatWritable(Math.signum(w.toFloat()));
            case SIN:
                return new FloatWritable((float)Math.sin(w.toFloat()));
            case SINH:
                return new FloatWritable((float)Math.sinh(w.toFloat()));
            case SQRT:
                return new FloatWritable((float)Math.sqrt(w.toFloat()));
            case TAN:
                return new FloatWritable((float)Math.tan(w.toFloat()));
            case TANH:
                return new FloatWritable((float)Math.tanh(w.toFloat()));
            default:
                throw new RuntimeException("Unknown function: " + mathFunction);
        }
    }

    @Override
    public Object map(Object input) {
        Float d = ((Number) input).floatValue();
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
