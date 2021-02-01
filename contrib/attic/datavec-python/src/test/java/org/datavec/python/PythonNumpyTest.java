/*
 *  ******************************************************************************
 *  * Copyright (c) 2021 Deeplearning4j Contributors
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.datavec.python;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static junit.framework.TestCase.assertEquals;

@RunWith(Parameterized.class)
public class PythonNumpyTest {

    @Parameterized.Parameters(name = "{index}: Testing with DataType={0}")
    public static DataType[] data() {
        return new DataType[] {
                DataType.BOOL,
                DataType.FLOAT16,
                DataType.BFLOAT16,
                DataType.FLOAT,
                DataType.DOUBLE,
                DataType.INT8,
                DataType.INT16,
                DataType.INT32,
                DataType.INT64,
                DataType.UINT8,
                DataType.UINT16,
                DataType.UINT32,
                DataType.UINT64
        };
    }

    private DataType dataType;

    public PythonNumpyTest(DataType dataType) {
        this.dataType = dataType;
    }

    @Test
    public void numpyAndNd4jConversions() throws Exception {
        INDArray input = Nd4j.ones(dataType, 2, 2, 2);

        PythonVariables inputs = new PythonVariables();
        inputs.addNDArray("x", input);

        PythonVariables outputs = new PythonVariables();
        outputs.addNDArray("y");

        PythonJob pythonJob = new PythonJob(String.format("job_%s", dataType.name()) + dataType.name(), "y = x", false);

        pythonJob.exec(inputs, outputs);

        INDArray output = outputs.getNDArrayValue("y");

        // As numpy doesn't support BFLOAT16 we'll convert it to FLOAT
        assertEquals(dataType == DataType.BFLOAT16 ? input.castTo(DataType.FLOAT) : input,
                output);
    }
}
