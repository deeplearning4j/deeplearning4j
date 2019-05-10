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
import org.datavec.api.transform.MathOp;
import org.datavec.api.transform.metadata.ColumnMetaData;
import org.datavec.api.transform.metadata.NDArrayMetaData;
import org.datavec.api.transform.transform.BaseColumnTransform;
import org.datavec.api.writable.NDArrayWritable;
import org.datavec.api.writable.Writable;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.shade.jackson.annotation.JsonProperty;

/**
 * Perform an NDArray/scalar element wise operation, such as X.addi(scalar).
 * Element wise operations are performed in place on each value of the underlying INDArray
 *
 * @author Alex Black
 */
@Data
public class NDArrayScalarOpTransform extends BaseColumnTransform {

    private final MathOp mathOp;
    private final double scalar;

    /**
     *
     * @param columnName Name of the column to perform the operation on
     * @param mathOp     Operation to perform
     * @param scalar     Scalar value for the operation
     */
    public NDArrayScalarOpTransform(@JsonProperty("columnName") String columnName,
                    @JsonProperty("mathOp") MathOp mathOp, @JsonProperty("scalar") double scalar) {
        super(columnName);
        this.mathOp = mathOp;
        this.scalar = scalar;
    }

    @Override
    public ColumnMetaData getNewColumnMetaData(String newName, ColumnMetaData oldColumnType) {
        if (!(oldColumnType instanceof NDArrayMetaData)) {
            throw new IllegalStateException("Column " + newName + " is not a NDArray column");
        }

        NDArrayMetaData oldMeta = (NDArrayMetaData) oldColumnType;
        NDArrayMetaData newMeta = oldMeta.clone();
        newMeta.setName(newName);

        return newMeta;
    }

    @Override
    public NDArrayWritable map(Writable w) {
        if (!(w instanceof NDArrayWritable)) {
            throw new IllegalArgumentException("Input writable is not an NDArrayWritable: is " + w.getClass());
        }

        //Make a copy - can't always assume that the original INDArray won't be used again in the future
        NDArrayWritable n = ((NDArrayWritable) w);
        INDArray a = n.get().dup();
        switch (mathOp) {
            case Add:
                a.addi(scalar);
                break;
            case Subtract:
                a.subi(scalar);
                break;
            case Multiply:
                a.muli(scalar);
                break;
            case Divide:
                a.divi(scalar);
                break;
            case Modulus:
                throw new UnsupportedOperationException(mathOp + " is not supported for NDArrayWritable");
            case ReverseSubtract:
                a.rsubi(scalar);
                break;
            case ReverseDivide:
                a.rdivi(scalar);
                break;
            case ScalarMin:
                Transforms.min(a, scalar, false);
                break;
            case ScalarMax:
                Transforms.max(a, scalar, false);
                break;
            default:
                throw new UnsupportedOperationException("Unknown or not supported op: " + mathOp);
        }

        //To avoid threading issues...
        Nd4j.getExecutioner().commit();

        return new NDArrayWritable(a);
    }

    @Override
    public String toString() {
        return "NDArrayScalarOpTransform(mathOp=" + mathOp + ",scalar=" + scalar + ")";
    }

    @Override
    public Object map(Object input) {
        if (input instanceof INDArray) {
            return map(new NDArrayWritable((INDArray) input)).get();
        }
        throw new RuntimeException("Unsupported class: " + input.getClass());
    }
}
