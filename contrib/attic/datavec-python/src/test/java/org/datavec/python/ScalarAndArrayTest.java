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
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static junit.framework.TestCase.assertEquals;

@RunWith(Parameterized.class)
public class ScalarAndArrayTest {

    @Parameterized.Parameters(name = "{index}: Testing with INDArray={0}")
    public static INDArray[] data() {
        return new INDArray[]{
                Nd4j.scalar(10),
                Nd4j.ones(10, 10, 10, 10)
        };
    }

    private INDArray indArray;

    public ScalarAndArrayTest(INDArray indArray) {
        this.indArray = indArray;
    }

    @Test
    public void testINDArray() throws PythonException {
        assertEquals(indArray, new PythonObject(indArray).toNumpy().getNd4jArray());
    }
}
