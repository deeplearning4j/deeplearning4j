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

package org.eclipse.deeplearning4j.nd4j.linalg.factory;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ndarray.LazyINDArray;
import org.nd4j.linalg.factory.CompiledGraphFunction;
import org.nd4j.linalg.factory.GraphScope;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import static org.junit.jupiter.api.Assertions.*;

@NativeTag
public class GraphScopeTest extends BaseNd4jTestWithBackends {

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testLazyINDArrayMetadata(Nd4jBackend backend) {
        // Test LazyINDArray shape/dtype/rank before materialization
        LazyINDArray lazy = new LazyINDArray("test_var", new long[]{3, 4}, DataType.FLOAT);

        assertArrayEquals(new long[]{3, 4}, lazy.shape());
        assertEquals(DataType.FLOAT, lazy.dataType());
        assertEquals(2, lazy.rank());
        assertEquals(12, lazy.length());
        assertFalse(lazy.isMaterialized());

        // Data access should throw before materialization
        assertThrows(IllegalStateException.class, () -> lazy.getDouble(0));
        assertThrows(IllegalStateException.class, () -> lazy.data());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testLazyINDArrayMaterialization(Nd4jBackend backend) {
        LazyINDArray lazy = new LazyINDArray("test_var", new long[]{2, 3}, DataType.FLOAT);
        INDArray real = Nd4j.ones(2, 3);
        lazy.materialize(real);

        assertTrue(lazy.isMaterialized());
        assertEquals(real, lazy.getMaterialized());
        assertArrayEquals(new long[]{2, 3}, lazy.shape());
        assertEquals(1.0, lazy.getDouble(0), 1e-5);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testGraphScopeTracingState(Nd4jBackend backend) {
        // Test basic state transitions
        assertFalse(GraphScope.isTracing());
        assertNull(GraphScope.active());

        try (GraphScope scope = new GraphScope()) {
            scope.begin();
            assertTrue(GraphScope.isTracing());
            assertNotNull(GraphScope.active());

            scope.end();
            assertFalse(GraphScope.isTracing());
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testNestedScopeThrows(Nd4jBackend backend) {
        try (GraphScope scope1 = new GraphScope()) {
            scope1.begin();
            try (GraphScope scope2 = new GraphScope()) {
                assertThrows(IllegalStateException.class, () -> scope2.begin());
            }
            scope1.end();
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testEmptyScope(Nd4jBackend backend) {
        try (GraphScope scope = new GraphScope()) {
            scope.begin();
            // No ops traced
            scope.end();
            // Should complete without error
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testCompiledFunctionWrongInputCountThrows(Nd4jBackend backend) {
        CompiledGraphFunction fn = Nd4j.compile(2, inputs -> {
            return new INDArray[]{inputs[0]};
        });
        assertThrows(IllegalArgumentException.class, () -> fn.execute(Nd4j.ones(2, 2)));
        fn.close();
    }
}
