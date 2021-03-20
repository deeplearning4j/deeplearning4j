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

package org.nd4j.linalg.api.indexing.resolve;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.indexing.PointIndex;

import static org.junit.jupiter.api.Assertions.*;

/**
 * @author Adam Gibson
 */
@Tag(TagNames.NDARRAY_INDEXING)
@NativeTag
public class NDArrayIndexResolveTests extends BaseNd4jTestWithBackends {


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testResolvePoint(Nd4jBackend backend) {
        INDArray arr = Nd4j.linspace(1, 4, 4).reshape(2, 2);
        INDArrayIndex[] test = NDArrayIndex.resolve(arr.shape(), NDArrayIndex.point(1));
        INDArrayIndex[] assertion = {NDArrayIndex.point(1), NDArrayIndex.all()};
        assertArrayEquals(assertion, test);

        INDArrayIndex[] allAssertion = {NDArrayIndex.all(), NDArrayIndex.all()};
        assertArrayEquals(allAssertion, NDArrayIndex.resolve(arr.shape(), NDArrayIndex.all()));

        INDArrayIndex[] allAndOne = new INDArrayIndex[] {NDArrayIndex.all(), NDArrayIndex.point(1)};
        assertArrayEquals(allAndOne, NDArrayIndex.resolve(arr.shape(), allAndOne));
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testResolvePointVector() {
        INDArray arr = Nd4j.linspace(1, 4, 4);
        INDArrayIndex[] getPoint = {NDArrayIndex.point(1)};
        INDArrayIndex[] resolved = NDArrayIndex.resolve(arr.shape(), getPoint);
        if (getPoint.length == resolved.length)
            assertArrayEquals(getPoint, resolved);
        else {
            assertEquals(2, resolved.length);
            assertTrue(resolved[0] instanceof PointIndex);
            assertEquals(0, resolved[0].offset());
            assertTrue(resolved[1] instanceof PointIndex);
            assertEquals(1, resolved[1].offset());
        }

    }

    @Override
    public char ordering() {
        return 'f';
    }
}
