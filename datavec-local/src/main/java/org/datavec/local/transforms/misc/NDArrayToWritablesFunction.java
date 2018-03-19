/*-
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

package org.datavec.local.transforms.misc;

import lombok.AllArgsConstructor;
import org.datavec.api.writable.DoubleWritable;
import org.datavec.api.writable.NDArrayWritable;
import org.datavec.api.writable.Writable;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.function.Function;

import java.util.ArrayList;
import java.util.List;

/**
 * Function for converting NDArrays to lists of writables.
 *
 * @author dave@skymind.io
 */
@AllArgsConstructor
public class NDArrayToWritablesFunction implements Function<INDArray, List<Writable>> {
    private boolean useNdarrayWritable = false;

    public NDArrayToWritablesFunction() {
        useNdarrayWritable = false;
    }

    @Override
    public List<Writable> apply(INDArray arr) {
        if (arr.rows() != 1)
            throw new UnsupportedOperationException("Only NDArray row vectors can be converted to list"
                                                + " of Writables (found " + arr.rows() + " rows)");
        List<Writable> record = new ArrayList<>();
        if (useNdarrayWritable) {
            record.add(new NDArrayWritable(arr));
        } else {
            for (int i = 0; i < arr.columns(); i++)
                record.add(new DoubleWritable(arr.getDouble(i)));
        }
        return record;
    }
}
