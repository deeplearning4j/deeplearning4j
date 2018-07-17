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

package org.datavec.api.writable;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Created by huitseeker on 5/13/17.
 */
public class UnsafeWritableInjector {
    public static <T> Writable inject(T x) {
        if (x == null)
            return NullWritable.INSTANCE;
        else if (x instanceof Writable)
            return (Writable) x;
        else if (x instanceof INDArray) {
            throw new IllegalArgumentException("Wrong argument of type INDArray (" + x.getClass().getName()
                            + ") please use org.datavec.common.data.NDArrayWritable manually to convert.");
        } else if (x.getClass() == Integer.class) {
            return new IntWritable((Integer) x);
        } else if (x.getClass() == Long.class) {
            return new LongWritable((Long) x);
        } else if (x.getClass() == Float.class) {
            return new FloatWritable((Float) x);
        } else if (x.getClass() == Double.class) {
            return new DoubleWritable((Double) x);
        } else if (x instanceof String) {
            return new Text((String) x);
        } else if (x instanceof Byte) {
            return new ByteWritable((Byte) x);
        } else
            throw new IllegalArgumentException("Wrong argument type for writable conversion " + x.getClass().getName());
    }
}
