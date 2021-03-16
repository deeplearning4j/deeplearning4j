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
package org.nd4j.linalg.generated;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

public class SDLinalgTest extends BaseNd4jTestWithBackends {

    @Override
    public char ordering(){
        return 'c';
    }

    private SameDiff sameDiff;

    @BeforeEach
    public void setup() {
        sameDiff = SameDiff.create();
    }

    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testCholesky(Nd4jBackend backend) {
        INDArray input = Nd4j.createFromArray(
                new float[]{
                        10.f,  14.f,
                        14.f,  20.f,
                        74.f,  86.f,
                        86.f, 100.f
                }
        ).reshape(2,2,2);

        INDArray expected = Nd4j.createFromArray(
                new float[]{
                        3.1622777f, 0.f,  4.427189f,  0.6324552f,
                        8.602325f,  0.f,  9.997296f, 0.23252854f
                }
        ).reshape(2,2,2);

        SDVariable sdinput = sameDiff.var(input);
        SDVariable out = sameDiff.linalg().cholesky(sdinput);
        assertEquals(expected, out.eval());
    }

    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testLstsq() {
        INDArray a = Nd4j.createFromArray(new float[]{
                1.f,    2.f,    3.f, 4.f,
                5.f,    6.f,    7.f, 8.f
        }).reshape(2,2,2);

        INDArray b = Nd4j.createFromArray(new float[]{
                3.f,    7.f,    11.f, 15.f
        }).reshape(2,2,1);

        INDArray expected = Nd4j.createFromArray(new float[]{
                0.831169367f,           1.090908766f,           0.920544624f,            1.063016534f
        }).reshape(2,2,1);

        SDVariable sda = sameDiff.var(a);
        SDVariable sdb = sameDiff.var(b);

        SDVariable res = sameDiff.linalg().lstsq(sda,sdb,0.5,true);
        assertEquals(expected, res.eval());
    }

    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testLu() {
        SDVariable sdInput = sameDiff.var(Nd4j.createFromArray(new double[]{
                1., 2., 3., 0., 2., 3., 0., 0., 7.
        }).reshape(3,3));

        INDArray expected = Nd4j.createFromArray(new double[]{
                1., 2., 3., 0., 2., 3., 0., 0., 7
        }).reshape(3,3);

        SDVariable out = sameDiff.linalg().lu("lu", sdInput);
        assertEquals(expected, out.eval());
    }

    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testMatrixBandPart() {
        INDArray x = Nd4j.linspace(DataType.DOUBLE, 1.0, 1.0, 2*3*3).reshape(2,3,3);
        INDArray expected = Nd4j.linspace(DataType.DOUBLE, 1.0, 1.0, 2*3*3).reshape(2,3,3);

        SDVariable sdx = sameDiff.var(x);
        SDVariable[] res = sameDiff.linalg().matrixBandPart(sdx, 1, 1);
        assertArrayEquals(x.shape(), res[0].eval().shape());
    }

    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testQr() {
        INDArray input = Nd4j.createFromArray(new double[]{
                12.,  -51.,    4.,
                6.,   167.,  -68.,
                -4.,    24.,  -41.,
                -1.,     1.,    0.,
                2.,     0.,    3.
        }).reshape(5,3);

        INDArray expectedQ = Nd4j.createFromArray(new double[]{
                0.8464147390303179,    -0.3912908119746455,    0.34312406418022884,
                0.42320736951515897,     0.9040872694197354,   -0.02927016186366648,
                -0.2821382463434393,    0.17042054976392634,     0.9328559865183932,
                -0.07053456158585983,    0.01404065236547358,   -0.00109937201747271,
                0.14106912317171966,   -0.01665551070074392,   -0.10577161246232346
        }).reshape(5,3);

        INDArray expectedR = Nd4j.createFromArray(new double[]{
                14.177446878757824,     20.666626544656932,    -13.401566701313369,
                -0.0000000000000006,     175.04253925050244,      -70.0803066408638,
                0.00000000000000017,   -0.00000000000000881,     -35.20154302119086
        }).reshape(3,3);

        SDVariable sdInput = sameDiff.var(input);
        SDVariable[] res = sameDiff.linalg().qr(sdInput);

        SDVariable mmulResult = sameDiff.mmul(res[0], res[1]);

        assertEquals(input, mmulResult.eval());
    }

    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testSolve() {
        INDArray a = Nd4j.createFromArray(new float[] {
                2.f, -1.f, -2.f, -4.f, 6.f, 3.f, -4.f, -2.f, 8.f
        }).reshape(3,3);

        INDArray b = Nd4j.createFromArray(new float[] {
                2.f, 4.f, 3.f
        }).reshape(3,1);

        INDArray expected = Nd4j.createFromArray(new float[] {
                7.625f, 3.25f, 5.f
        }).reshape(3,1);

        SDVariable sda = sameDiff.var(a);
        SDVariable sdb = sameDiff.var(b);

        SDVariable res = sameDiff.linalg().solve(sda, sdb);
        assertEquals(expected, res.eval());
    }

    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testTriangularSolve() {
        INDArray a = Nd4j.createFromArray(new float[] {
                0.7788f,    0.8012f,    0.7244f,
                0.2309f,    0.7271f,    0.1804f,
                0.5056f,    0.8925f,    0.5461f
        }).reshape(3,3);

        INDArray b = Nd4j.createFromArray(new float[] {
                0.7717f,    0.9281f,    0.9846f,
                0.4838f,    0.6433f,    0.6041f,
                0.6501f,    0.7612f,    0.7605f
        }).reshape(3,3);

        INDArray expected = Nd4j.createFromArray(new float[] {
                0.99088347f,  1.1917052f,    1.2642528f,
                0.35071516f,  0.50630623f,  0.42935497f,
                -0.30013534f, -0.53690606f, -0.47959247f
        }).reshape(3,3);

        SDVariable sda = sameDiff.var(a);
        SDVariable sdb = sameDiff.var(b);

        SDVariable res = sameDiff.linalg().triangularSolve(sda, sdb, true, false);
        assertEquals(expected, res.eval());
    }

    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testCross() {
        INDArray a = Nd4j.createFromArray(new double[]{1, 2, 3});
        INDArray b = Nd4j.createFromArray(new double[]{6, 7, 8});
        INDArray expected = Nd4j.createFromArray(new double[]{-5, 10, -5});

        SDVariable sda = sameDiff.var(a);
        SDVariable sdb = sameDiff.var(b);

        SDVariable res = sameDiff.linalg().cross(sda, sdb);
        assertEquals(expected, res.eval());
    }

    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testDiag() {
        INDArray x = Nd4j.createFromArray(new double[]{1,2});
        INDArray expected = Nd4j.createFromArray(new double[]{1,0,0,2}).reshape(2,2);

        SDVariable sdx = sameDiff.var(x);

        SDVariable res = sameDiff.linalg().diag(sdx);
        assertEquals(expected, res.eval());
    }

    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testDiagPart() {
        INDArray x = Nd4j.linspace(DataType.DOUBLE, 1.0, 1.0, 4).reshape(2,2);
        INDArray expected = Nd4j.createFromArray(new double[]{1,4});

        SDVariable sdx = sameDiff.var(x);

        SDVariable res = sameDiff.linalg().diag_part(sdx);
        assertEquals(expected, res.eval());
    }

    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testLogdet() {
        INDArray x = Nd4j.createFromArray(new double[]{
                4,12,-16,12,37,-43,-16,-43,98, 4,1.2,-1.6,1.2,3.7,-4.3,-1.6,-4.3,9.8
        }).reshape(2,3,3);
        INDArray expected = Nd4j.createFromArray(new double[]{3.5835189, 4.159008});

        SDVariable sdx = sameDiff.var(x);

        SDVariable res = sameDiff.linalg().logdet(sdx);
        assertEquals(expected, res.eval());
    }

    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testSvd() {
        INDArray x = Nd4j.createFromArray(new double[]{
                0.7787856f, 0.80119777f, 0.72437465f, 0.23089433f, 0.72714126f, 0.18039072f,0.50563407f, 0.89252293f, 0.5461209f
        }).reshape(3,3);
        INDArray expected = Nd4j.createFromArray(new double[]{1.8967269987492157,     0.3709665595850617,    0.05524869852188223});

        SDVariable sdx = sameDiff.var(x);
        SDVariable res = sameDiff.linalg().svd(sdx, false, false);
        assertEquals(expected, res.eval());
    }

    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testLogdetName() {
        INDArray x = Nd4j.createFromArray(new double[]{
                4,12,-16,12,37,-43,-16,-43,98, 4,1.2,-1.6,1.2,3.7,-4.3,-1.6,-4.3,9.8
        }).reshape(2,3,3);

        SDVariable sdx = sameDiff.var(x);

        SDVariable res = sameDiff.linalg().logdet("logdet", sdx);
        assertEquals("logdet", res.name());
    }

    @Test
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
    public void testQrNames() {
        INDArray input = Nd4j.createFromArray(new double[]{
                12.,  -51.,    4.,
                6.,   167.,  -68.,
                -4.,    24.,  -41.,
                -1.,     1.,    0.,
                2.,     0.,    3.
        }).reshape(5,3);

        SDVariable sdInput = sameDiff.var(input);
        SDVariable[] res = sameDiff.linalg().qr(new String[]{"ret0", "ret1"}, sdInput);

        assertEquals("ret0", res[0].name());
        assertEquals("ret1", res[1].name());
    }
}
