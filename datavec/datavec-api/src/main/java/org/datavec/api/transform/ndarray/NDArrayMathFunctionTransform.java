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

package org.datavec.api.transform.ndarray;

import lombok.Data;
import org.datavec.api.transform.MathFunction;
import org.datavec.api.transform.metadata.ColumnMetaData;
import org.datavec.api.transform.transform.BaseColumnTransform;
import org.datavec.api.writable.NDArrayWritable;
import org.datavec.api.writable.Writable;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.shade.jackson.annotation.JsonProperty;

/**
 * A simple transform to do common mathematical operations, such as sin(x), ceil(x), etc.<br>
 * Operations are performed element-wise on each value in the INDArray; operations are specified by {@link MathFunction}
 *
 * @author Alex Black
 */
@Data
public class NDArrayMathFunctionTransform extends BaseColumnTransform {

    //Can't guarantee that the writable won't be re-used, for example in different Spark ops on the same RDD
    private static final boolean DUP = true;

    private final MathFunction mathFunction;

    public NDArrayMathFunctionTransform(@JsonProperty("columnName") String columnName,
                    @JsonProperty("mathFunction") MathFunction mathFunction) {
        super(columnName);
        this.mathFunction = mathFunction;
    }

    @Override
    public ColumnMetaData getNewColumnMetaData(String newName, ColumnMetaData oldColumnType) {
        ColumnMetaData m = oldColumnType.clone();
        m.setName(newName);
        return m;
    }

    @Override
    public NDArrayWritable map(Writable w) {
        NDArrayWritable n = (NDArrayWritable) w;
        INDArray i = n.get();
        if (i == null) {
            return n;
        }

        NDArrayWritable o;
        switch (mathFunction) {
            case ABS:
                o = new NDArrayWritable(Transforms.abs(i, DUP));
                break;
            case ACOS:
                o = new NDArrayWritable(Transforms.acos(i, DUP));
                break;
            case ASIN:
                o = new NDArrayWritable(Transforms.asin(i, DUP));
                break;
            case ATAN:
                o = new NDArrayWritable(Transforms.atan(i, DUP));
                break;
            case CEIL:
                o = new NDArrayWritable(Transforms.ceil(i, DUP));
                break;
            case COS:
                o = new NDArrayWritable(Transforms.cos(i, DUP));
                break;
            case COSH:
                //No cosh operation in ND4J
                throw new UnsupportedOperationException("sinh operation not yet supported for NDArray columns");
            case EXP:
                o = new NDArrayWritable(Transforms.exp(i, DUP));
                break;
            case FLOOR:
                o = new NDArrayWritable(Transforms.floor(i, DUP));
                break;
            case LOG:
                o = new NDArrayWritable(Transforms.log(i, DUP));
                break;
            case LOG10:
                o = new NDArrayWritable(Transforms.log(i, 10.0, DUP));
                break;
            case SIGNUM:
                o = new NDArrayWritable(Transforms.sign(i, DUP));
                break;
            case SIN:
                o = new NDArrayWritable(Transforms.sin(i, DUP));
                break;
            case SINH:
                //No sinh op in ND4J
                throw new UnsupportedOperationException("sinh operation not yet supported for NDArray columns");
            case SQRT:
                o = new NDArrayWritable(Transforms.sqrt(i, DUP));
                break;
            case TAN:
                //No tan op in ND4J yet - but tan(x) = sin(x)/cos(x)
                INDArray sinx = Transforms.sin(i, true);
                INDArray cosx = Transforms.cos(i, true);
                o = new NDArrayWritable(sinx.divi(cosx));
                break;
            case TANH:
                o = new NDArrayWritable(Transforms.tanh(i, DUP));
                break;
            default:
                throw new RuntimeException("Unknown function: " + mathFunction);
        }

        //To avoid threading issues...
        Nd4j.getExecutioner().commit();

        return o;

    }

    @Override
    public String toString() {
        return "NDArrayMathFunctionTransform(column=" + columnName + ",function=" + mathFunction + ")";
    }

    @Override
    public Object map(Object input) {
        if (input instanceof NDArrayWritable) {
            return map((NDArrayWritable) input);
        } else if (input instanceof INDArray) {
            return map(new NDArrayWritable((INDArray) input)).get();
        } else {
            throw new UnsupportedOperationException(
                            "Unknown object type: " + (input == null ? null : input.getClass()));
        }
    }
}
