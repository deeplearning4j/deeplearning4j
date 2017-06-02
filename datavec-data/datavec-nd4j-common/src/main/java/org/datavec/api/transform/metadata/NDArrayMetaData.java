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

package org.datavec.api.transform.metadata;

import org.datavec.api.transform.ColumnType;
import org.datavec.api.writable.Writable;
import org.datavec.common.data.NDArrayWritable;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Arrays;

/**
 * Created by Alex on 02/06/2017.
 */
public class NDArrayMetaData extends BaseColumnMetaData {

    private int[] shape;
    private boolean allowVarLength;

    /**
     *
     * @param name
     * @param shape
     */
    public NDArrayMetaData(String name, int[] shape) {
        super(name);
        this.shape = shape;
        for( int i : shape ){
            if(i < 0){
                allowVarLength = true;
                break;
            }
        }
    }

    @Override
    public ColumnType getColumnType() {
        return ColumnType.NDArray;
    }

    @Override
    public boolean isValid(Writable writable) {
        if(!(writable instanceof NDArrayWritable)){
            return false;
        }
        INDArray arr = ((NDArrayWritable) writable).get();
        if(arr == null){
            return false;
        }
        if(allowVarLength){
            for( int i=0; i<shape.length; i++ ){
                if (shape[i] < 0) {
                    continue;
                }
                if(shape[i] != arr.size(i)){
                    return false;
                }
            }
            return true;
        } else {
            return Arrays.equals(shape, arr.shape());
        }
    }

    @Override
    public boolean isValid(Object input) {
        return isValid((Writable)input);
    }

    @Override
    public NDArrayMetaData clone() {
        return new NDArrayMetaData(name, shape.clone());
    }
}
