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

package org.nd4j.serde.base64;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.IOException;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

/**
 * Created by agibsonccc on 6/17/16.
 */
public class Nd4jBase64Test {
    @Test
    public void testBase64Several() throws IOException {
        INDArray[] arrs = new INDArray[2];
        arrs[0] = Nd4j.linspace(1, 4, 4);
        arrs[1] = arrs[0].dup();
        assertArrayEquals(arrs, Nd4jBase64.arraysFromBase64(Nd4jBase64.arraysToBase64(arrs)));
    }

    @Test
    public void testBase64() throws Exception {
        INDArray arr = Nd4j.linspace(1, 4, 4);
        String base64 = Nd4jBase64.base64String(arr);
        //        assertTrue(Nd4jBase64.isMultiple(base64));
        INDArray from = Nd4jBase64.fromBase64(base64);
        assertEquals(arr, from);
    }

    @Test
    public void testBase64Npy() throws Exception {
        INDArray arr = Nd4j.linspace(1, 4, 4);
        String base64Npy = Nd4jBase64.base64StringNumpy(arr);
        INDArray fromBase64 = Nd4jBase64.fromNpyBase64(base64Npy);
        assertEquals(arr,fromBase64);
    }

}
