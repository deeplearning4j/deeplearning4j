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

package org.nd4j.linalg.util;

import org.junit.Test;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.Arrays;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

public class PreconditionsTest extends BaseNd4jTest {

    public PreconditionsTest(Nd4jBackend backend) {
        super(backend);
    }

    @Test
    public void test(){

        INDArray arr = Nd4j.linspace(1,60,60).reshape('c',3,4,5);

        try{
            Preconditions.checkArgument(false, "Shape is: %ndShape with rank %ndRank", arr, arr);
            fail("Expected exception");
        } catch (Throwable t){
            assertEquals("Shape is: [3, 4, 5] with rank 3", t.getMessage());
        }

        try{

            Preconditions.checkArgument(false, "Stride is: %ndStride with shape info %ndSInfo", arr, arr);
            fail("Expected exception");
        } catch (Throwable t){
            String si = arr.shapeInfoToString().replaceAll("\n","");
            assertEquals("Stride is: " + Arrays.toString(arr.stride()) + " with shape info " + si, t.getMessage());
        }

        INDArray asVector = arr.reshape(arr.length());
        try{
            Preconditions.checkArgument(false, "First 10: %nd10", arr);
            fail("Expected exception");
        } catch (Throwable t){
            INDArray get10 = asVector.get(NDArrayIndex.interval(0, 10));
            assertEquals("First 10: " + get10, t.getMessage());
        }

    }


    public char ordering() {
        return 'c';
    }

}
