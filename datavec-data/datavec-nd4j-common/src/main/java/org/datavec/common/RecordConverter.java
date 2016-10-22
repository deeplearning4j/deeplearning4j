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

package org.datavec.common;

import org.datavec.api.writable.Writable;
import org.datavec.common.data.NDArrayWritable;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Iterator;
import java.util.List;

/**
 * @author Adam Gibson
 */
public class RecordConverter {
    private RecordConverter() {}
    /**
     * Convert a record to an ndarray
     * @param record the record to convert
     *
     * @return the array
     */
    public static INDArray toArray(Collection<Writable> record,int size) {
        Iterator<Writable> writables = record.iterator();
        Writable firstWritable = writables.next();
        if(firstWritable instanceof NDArrayWritable) {
            NDArrayWritable ret = (NDArrayWritable) firstWritable;
            return ret.get();
        }
        else {
            INDArray vector = Nd4j.create(size);
            vector.putScalar(0,firstWritable.toDouble());
            int count = 1;
            while(writables.hasNext()) {
                Writable w = writables.next();
                vector.putScalar(count++,w.toDouble());
            }

            return vector;
        }
    }

    /**
     * Convert a set of records in to a matrix
     * @param records the records ot convert
     * @return the matrix for the records
     */
    public static INDArray toMatrix(List<List<Writable>> records) {
        INDArray arr = Nd4j.create(records.size(),records.get(0).size());
        for(int i = 0; i < arr.rows(); i++) {
            arr.putRow(i,toArray(records.get(i)));
        }

        return arr;
    }

    /**
     * Convert a record to an ndarray
     * @param record the record to convert
     * @return the array
     */
    public static INDArray toArray(Collection<Writable> record) {
       return toArray(record,record.size());
    }



    /**
     * Convert an ndarray to a record
     * @param array the array to convert
     * @return the record
     */
    public static List<Writable> toRecord(INDArray array) {
        List<Writable> writables = new ArrayList<>();
        writables.add(new NDArrayWritable(array));
        return writables;
    }

}
