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
package org.eclipse.deeplearning4j.nd4j.linalg.ops;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.custom.Polygamma;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertThrows;

/**
 * polygamma(n, x) needs every n >= 0 and every x > 0. The op checked x with a reduction that is true when ANY element
 * is positive, so an array with one positive element passed and the others came out as NaN or garbage; the Java
 * constructor compared the two shape arrays with != (their identity), so it neither refused different shapes nor
 * accepted an array as both arguments.
 */
@NativeTag
@Tag(TagNames.CUSTOM_FUNCTIONALITY)
public class PolygammaTest extends BaseNd4jTestWithBackends {

    private static INDArray polygamma(INDArray n, INDArray x) {
        return Nd4j.exec(new Polygamma(n, x))[0];
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void valuesAtTheClosedFormPoints(Nd4jBackend backend) {
        double gamma = 0.5772156649015329;
        double zeta3 = 1.2020569031595942;
        double pi2 = Math.PI * Math.PI;
        INDArray n = Nd4j.createFromArray(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0);
        INDArray x = Nd4j.createFromArray(1.0, 2.0, 0.5, 1.0, 2.0, 0.5, 1.0, 2.0);
        double[] expected = {-gamma, 1 - gamma, -gamma - 2 * Math.log(2), pi2 / 6, pi2 / 6 - 1, pi2 / 2, -2 * zeta3,
                -2 * zeta3 + 2};
        assertArrayEquals(expected, polygamma(n, x).data().asDouble(), 1e-10);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void everyElementOfXMustBePositive(Nd4jBackend backend) {
        INDArray n = Nd4j.createFromArray(1.0, 1.0, 1.0);
        // one positive element used to be enough
        assertThrows(RuntimeException.class, () -> polygamma(n, Nd4j.createFromArray(1.0, -1.0, 2.0)));
        assertThrows(RuntimeException.class, () -> polygamma(n, Nd4j.createFromArray(2.0, 3.0, 0.0)));
        assertThrows(RuntimeException.class, () -> polygamma(n, Nd4j.createFromArray(2.0, Double.NaN, 3.0)));
        assertThrows(RuntimeException.class, () -> polygamma(n, Nd4j.createFromArray(-1.0, -2.0, -3.0)));
        // and the orders must not be negative
        assertThrows(RuntimeException.class,
                () -> polygamma(Nd4j.createFromArray(1.0, -1.0, 1.0), Nd4j.createFromArray(1.0, 2.0, 3.0)));
        // an array of positive elements is accepted, in each floating type
        for (DataType type : new DataType[]{DataType.FLOAT, DataType.DOUBLE}) {
            assertDoesNotThrow(() -> polygamma(n.castTo(type), Nd4j.createFromArray(0.1, 2.0, 30.0).castTo(type)));
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void constructorComparesShapesNotArrays(Nd4jBackend backend) {
        INDArray a = Nd4j.createFromArray(1.0, 2.0, 3.0);
        // differently shaped arguments are refused where the op is built
        assertThrows(IllegalArgumentException.class, () -> new Polygamma(a, Nd4j.createFromArray(1.0, 2.0, 3.0, 4.0)));
        assertThrows(IllegalArgumentException.class,
                () -> new Polygamma(a, Nd4j.createFromArray(new double[][]{{1.0, 2.0, 3.0}})));
        // equally shaped ones are accepted, the same array as both of them too
        assertDoesNotThrow(() -> new Polygamma(a, a.dup()));
        assertDoesNotThrow(() -> new Polygamma(a, a));
    }
}
