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

package org.datavec.spark.transform.misc;

import lombok.AllArgsConstructor;
import org.apache.spark.api.java.function.Function;
import org.datavec.api.writable.DoubleWritable;
import org.datavec.api.writable.NDArrayWritable;
import org.datavec.api.writable.Writable;
import org.nd4j.linalg.api.ndarray.INDArray;

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
    public List<Writable> call(INDArray arr) throws Exception {
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
