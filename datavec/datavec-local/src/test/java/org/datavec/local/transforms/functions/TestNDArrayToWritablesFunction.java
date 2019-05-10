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

package org.datavec.local.transforms.functions;

import org.datavec.api.writable.DoubleWritable;
import org.datavec.api.writable.NDArrayWritable;
import org.datavec.api.writable.Writable;

import org.datavec.local.transforms.misc.NDArrayToWritablesFunction;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertEquals;

/**
 * Unit tests for NDArrayToWritablesFunction.
 *
 * @author dave@skymind.io
 */
public class TestNDArrayToWritablesFunction {

    @Test
    public void testNDArrayToWritablesScalars() throws Exception {
        INDArray arr = Nd4j.arange(5);
        List<Writable> expected = new ArrayList<>();
        for (int i = 0; i < 5; i++)
            expected.add(new DoubleWritable(i));
        List<Writable> actual = new NDArrayToWritablesFunction().apply(arr);
        assertEquals(expected, actual);
    }

    @Test
    public void testNDArrayToWritablesArray() throws Exception {
        INDArray arr = Nd4j.arange(5);
        List<Writable> expected = Arrays.asList((Writable) new NDArrayWritable(arr));
        List<Writable> actual = new NDArrayToWritablesFunction(true).apply(arr);
        assertEquals(expected, actual);
    }
}
