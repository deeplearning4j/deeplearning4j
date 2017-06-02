/*
 *  * Copyright 2016 Skymind, Inc.
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

package org.datavec.api.transform.ndarray;

import org.datavec.api.transform.MathOp;
import org.datavec.api.transform.metadata.ColumnMetaData;
import org.datavec.api.transform.metadata.NDArrayMetaData;
import org.datavec.api.transform.transform.BaseColumnTransform;
import org.datavec.api.writable.Writable;
import org.datavec.common.data.NDArrayWritable;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.shade.jackson.annotation.JsonProperty;

/**
 * Created by Alex on 02/06/2017.
 */
public class NDArrayMathOpTransform extends BaseColumnTransform {

    private final MathOp mathOp;
    private final double scalar;

    public NDArrayMathOpTransform(@JsonProperty("columnName") String columnName, @JsonProperty("mathOp") MathOp mathOp,
                                 @JsonProperty("scalar") double scalar) {
        super(columnName);
        this.mathOp = mathOp;
        this.scalar = scalar;
    }

    @Override
    public ColumnMetaData getNewColumnMetaData(String newName, ColumnMetaData oldColumnType) {
        if(!(oldColumnType instanceof NDArrayMetaData)){
            throw new IllegalStateException("Column " + newName + " is not a NDArray column");
        }

        NDArrayMetaData oldMeta = (NDArrayMetaData)oldColumnType;
        NDArrayMetaData newMeta = oldMeta.clone();
        newMeta.setName(newName);

        return newMeta;
    }

    @Override
    public NDArrayWritable map(Writable w) {
        if(!(w instanceof NDArrayWritable)){
            throw new IllegalArgumentException("Input writable is not an NDArrayWritable: is " + w.getClass());
        }

        //TODO is in-place always safe?
        NDArrayWritable n = ((NDArrayWritable) w);
        INDArray a = n.get();
        switch (mathOp){
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

        return n;
    }

    @Override
    public String toString() {
        return "NDArrayMathOpTransform(mathOp=" + mathOp + ",scalar=" + scalar + ")";
    }

    @Override
    public Object map(Object input) {
        if(input instanceof INDArray){
            return map(new NDArrayWritable((INDArray) input)).get();
        }
        throw new RuntimeException("Unsupported class: " + input.getClass());
    }
}
