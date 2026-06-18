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

package org.eclipse.deeplearning4j.llm.generation;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import static org.junit.jupiter.api.Assertions.*;

@NativeTag
@Tag(TagNames.SAMEDIFF)
public class SameDiffMemoryUtilsTest extends BaseNd4jTestWithBackends {

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSafeCloseNullArray(Nd4jBackend backend) {
        // Should not throw
        SameDiffMemoryUtils.safeClose(null);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSafeCloseAlreadyClosed(Nd4jBackend backend) {
        INDArray arr = Nd4j.zeros(DataType.FLOAT, 2, 3);
        arr.close();
        assertTrue(arr.wasClosed());

        // Should not throw on already-closed array
        SameDiffMemoryUtils.safeClose(arr);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSafeClosePoisonedArray(Nd4jBackend backend) {
        INDArray arr = Nd4j.zeros(DataType.FLOAT, 2, 3);

        // Simulate the poisoning that directExecHelper does
        arr.setCloseable(false);

        // Normal close() would be a no-op on poisoned array
        // safeClose should undo the poisoning and actually close
        SameDiffMemoryUtils.safeClose(arr);
        assertTrue(arr.wasClosed());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSafeCloseNormalArray(Nd4jBackend backend) {
        INDArray arr = Nd4j.zeros(DataType.FLOAT, 4, 5);
        assertFalse(arr.wasClosed());

        SameDiffMemoryUtils.safeClose(arr);
        assertTrue(arr.wasClosed());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testFreeModelArrays(Nd4jBackend backend) {
        SameDiff sd = SameDiff.create();

        // Add some constants
        sd.constant("const1", Nd4j.ones(DataType.FLOAT, 3, 4));
        sd.constant("const2", Nd4j.zeros(DataType.FLOAT, 2, 2));

        // Verify they exist
        assertNotNull(sd.getConstantArrays().getArray("const1"));
        assertNotNull(sd.getConstantArrays().getArray("const2"));

        int freed = SameDiffMemoryUtils.freeModelArrays(sd);
        assertTrue(freed >= 2, "Expected at least 2 arrays freed, got " + freed);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testFreeModelArraysNull(Nd4jBackend backend) {
        assertEquals(0, SameDiffMemoryUtils.freeModelArrays(null));
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testFreeModelArraysEmpty(Nd4jBackend backend) {
        SameDiff sd = SameDiff.create();
        // No constants or variables added
        int freed = SameDiffMemoryUtils.freeModelArrays(sd);
        assertEquals(0, freed);
    }
}
