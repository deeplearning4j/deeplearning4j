/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.eclipse.deeplearning4j.nd4j.linalg.api.buffer;

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.nativeblas.OpaqueDataBuffer;

import java.lang.reflect.Method;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class CpuDataBufferCloseCoordinationIsolationTest extends BaseNd4jTestWithBackends {

    private static OpaqueDataBuffer getOpaqueDataBuffer(DataBuffer buffer) {
        try {
            Method method = buffer.getClass().getMethod("getOpaqueDataBuffer");
            return (OpaqueDataBuffer) method.invoke(buffer);
        } catch (Exception e) {
            throw new RuntimeException("Unable to access OpaqueDataBuffer for " + buffer.getClass().getName(), e);
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testDataBufferCloseMarksOpaqueBufferForCoordinatedDeallocation(Nd4jBackend backend) {
        INDArray array = Nd4j.linspace(1, 16, 16, DataType.DOUBLE).reshape(4, 4);
        OpaqueDataBuffer opaqueDataBuffer = getOpaqueDataBuffer(array.data());

        assertFalse(opaqueDataBuffer.isMarkedForDeallocation());

        array.data().close();

        assertTrue(opaqueDataBuffer.isMarkedForDeallocation());
        assertTrue(opaqueDataBuffer.isNull());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testRepeatedDataBufferCloseUnderGcPressure(Nd4jBackend backend) {
        for (int i = 0; i < 200; i++) {
            INDArray array = Nd4j.randn(DataType.DOUBLE, 32, 32);
            OpaqueDataBuffer opaqueDataBuffer = getOpaqueDataBuffer(array.data());

            array.data().close();

            assertTrue(opaqueDataBuffer.isMarkedForDeallocation(), "Opaque buffer was not marked on iteration " + i);
            assertTrue(opaqueDataBuffer.isNull(), "Opaque buffer was not closed on iteration " + i);

            array = null;
            if (i % 10 == 0) {
                System.gc();
                System.runFinalization();
            }
        }
    }
}
