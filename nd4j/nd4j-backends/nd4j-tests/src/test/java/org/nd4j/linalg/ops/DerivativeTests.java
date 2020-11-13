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

package org.nd4j.linalg.ops;

import org.apache.commons.math3.util.FastMath;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.scalar.Step;
import org.nd4j.linalg.api.ops.impl.transforms.gradient.CubeDerivative;
import org.nd4j.linalg.api.ops.impl.transforms.gradient.HardSigmoidDerivative;
import org.nd4j.linalg.api.ops.impl.transforms.gradient.HardTanhDerivative;
import org.nd4j.linalg.api.ops.impl.transforms.gradient.LeakyReLUDerivative;
import org.nd4j.linalg.api.ops.impl.transforms.gradient.SoftSignDerivative;
import org.nd4j.linalg.api.ops.impl.transforms.strict.Sigmoid;
import org.nd4j.linalg.api.ops.impl.transforms.strict.SigmoidDerivative;
import org.nd4j.linalg.api.ops.impl.transforms.strict.TanhDerivative;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.ops.transforms.Transforms;

import static org.junit.Assert.*;


@RunWith(Parameterized.class)
public class DerivativeTests extends BaseNd4jTest {

    public static final double REL_ERROR_TOLERANCE = 1e-3;


    DataType initialType;

    public DerivativeTests(Nd4jBackend backend) {
        super(backend);
        this.initialType = Nd4j.dataType();
    }

    @Before
    public void before() {
        Nd4j.setDataType(DataType.DOUBLE);
    }

    @After
    public void after() {
        Nd4j.setDataType(this.initialType);
    }

    @Test
    public void testHardTanhDerivative() {
            //HardTanh:
        //f(x) = 1 if x > 1
        //f(x) = -1 if x < -1
        //f(x) = x otherwise
        //This is piecewise differentiable.
        //f'(x) = 0 if |x|>1
        //f'(x) = 1 otherwise
        //Note for x= +/- 1, HardTanh is not differentiable. Choose f'(+/- 1) = 1

        INDArray z = Nd4j.zeros(100);
        double[] expOut = new double[100];
        for (int i = 0; i < 100; i++) {
            double x = 0.1 * (i - 50);
            z.putScalar(i, x);
            expOut[i] = (Math.abs(x) <= 1.0 ? 1 : 0);
        }

        INDArray zPrime = Nd4j.getExecutioner().exec(new HardTanhDerivative(z));

        for (int i = 0; i < 100; i++) {
            assertEquals(expOut[i], zPrime.getDouble(i), 1e-1);
        }
    }

    @Test
    public void testRectifiedLinearDerivative() {
        //ReLU:
        //f(x) = max(0,x)
        //Piecewise differentiable; choose f'(0) = 0
        //f'(x) = 1 if x > 0
        //f'(x) = 0 if x <= 0

        INDArray z = Nd4j.zeros(100).castTo(DataType.DOUBLE);
        double[] expOut = new double[100];
        for (int i = 0; i < 100; i++) {
            double x = 0.1 * (i - 50);
            z.putScalar(i, x);
            expOut[i] = (x > 0 ? 1 : 0);
        }

        INDArray zPrime = Nd4j.getExecutioner().exec(new Step(z));

        for (int i = 0; i < 100; i++) {
            assertTrue(expOut[i] == zPrime.getDouble(i));
        }
    }

    @Test
    public void testSigmoidDerivative() {
        //Derivative of sigmoid: ds(x)/dx = s(x)*(1-s(x))
        //s(x) = 1 / (exp(-x) + 1)
        INDArray z = Nd4j.zeros(100);
        double[] expOut = new double[100];
        for (int i = 0; i < 100; i++) {
            double x = 0.1 * (i - 50);
            z.putScalar(i, x);
            double sigmoid = 1.0 / (FastMath.exp(-x) + 1);
            expOut[i] = sigmoid * (1 - sigmoid);
        }

        INDArray zPrime = Nd4j.getExecutioner().exec(new SigmoidDerivative(z));

        for (int i = 0; i < 100; i++) {
            double relError = Math.abs(expOut[i] - zPrime.getDouble(i))
                            / (Math.abs(expOut[i]) + Math.abs(zPrime.getDouble(i)));
            assertTrue(relError < REL_ERROR_TOLERANCE);
        }
    }


    @Test
    public void testHardSigmoidDerivative() {
        /*
        f(x) = min(1, max(0, 0.2*x + 0.5))
        or equivalently: clip 0.2*x+0.5 to range 0 to 1
        where clipping bounds are -2.5 and 2.5
        
        Hard sigmoid derivative:
        f'(x) =
        0 if x < -2.5 or x > 2.5
        0.2 otherwise
         */

        double[] expHSOut = new double[300];
        double[] expDerivOut = new double[300];
        INDArray xArr = Nd4j.linspace(-3, 3, 300, Nd4j.dataType());
        for (int i = 0; i < xArr.length(); i++) {
            double x = xArr.getDouble(i);
            double hs = 0.2 * x + 0.5;
            if (hs < 0)
                hs = 0;
            if (hs > 1)
                hs = 1;
            expHSOut[i] = hs;

            double hsDeriv;
            if (x < -2.5 || x > 2.5)
                hsDeriv = 0;
            else
                hsDeriv = 0.2;

            expDerivOut[i] = hsDeriv;
        }

        INDArray z = Transforms.hardSigmoid(xArr, true);
        INDArray zPrime = Nd4j.getExecutioner().exec(new HardSigmoidDerivative(xArr.dup()));;

        for (int i = 0; i < expHSOut.length; i++) {
            double relErrorHS =
                            Math.abs(expHSOut[i] - z.getDouble(i)) / (Math.abs(expHSOut[i]) + Math.abs(z.getDouble(i)));
            if (!(expHSOut[i] == 0 && z.getDouble(i) == 0)) {
                assertTrue(relErrorHS < REL_ERROR_TOLERANCE);
            }
            double relErrorDeriv = Math.abs(expDerivOut[i] - zPrime.getDouble(i))
                            / (Math.abs(expDerivOut[i]) + Math.abs(zPrime.getDouble(i)));
            if (!(expDerivOut[i] == 0 && zPrime.getDouble(i) == 0)) {
                assertTrue(relErrorDeriv < REL_ERROR_TOLERANCE);
            }
        }

    }


    @Test
    public void testSoftPlusDerivative() {
        //s(x) = 1 / (exp(-x) + 1)
        INDArray z = Nd4j.zeros(100);
        double[] expOut = new double[100];
        for (int i = 0; i < 100; i++) {
            double x = 0.1 * (i - 50);
            z.putScalar(i, x);
            expOut[i] = 1.0 / (1.0 + FastMath.exp(-x));
        }

        INDArray zPrime = Nd4j.getExecutioner().exec(new Sigmoid( z));

        for (int i = 0; i < 100; i++) {
            double relError = Math.abs(expOut[i] - zPrime.getDouble(i))
                            / (Math.abs(expOut[i]) + Math.abs(zPrime.getDouble(i)));
            assertTrue(relError < REL_ERROR_TOLERANCE);
        }
    }

    @Test
    public void testTanhDerivative() {

        //Derivative of sigmoid: ds(x)/dx = s(x)*(1-s(x))
        //s(x) = 1 / (exp(-x) + 1)
        INDArray z = Nd4j.zeros(100);
        double[] expOut = new double[100];
        for (int i = 0; i < 100; i++) {
            double x = 0.1 * (i - 50);
            z.putScalar(i, x);
            double tanh = FastMath.tanh(x);
            expOut[i] = 1.0 - tanh * tanh;
        }

        INDArray zPrime = Nd4j.getExecutioner().exec(new TanhDerivative(z));

        for (int i = 0; i < 100; i++) {
            double relError = Math.abs(expOut[i] - zPrime.getDouble(i))
                            / (Math.abs(expOut[i]) + Math.abs(zPrime.getDouble(i)));
            assertTrue(relError < REL_ERROR_TOLERANCE);
        }
    }

    @Test
    public void testCubeDerivative() {

        //Derivative of cube: 3*x^2
        INDArray z = Nd4j.zeros(100);
        double[] expOut = new double[100];
        for (int i = 0; i < 100; i++) {
            double x = 0.1 * (i - 50);
            z.putScalar(i, x);
            expOut[i] = 3 * x * x;
        }

        INDArray zPrime = Nd4j.getExecutioner().exec(new CubeDerivative(z));

        for (int i = 0; i < 100; i++) {
            double d1 = expOut[i];
            double d2 = zPrime.getDouble(i);
            double relError = Math.abs(d1 - d1) / (Math.abs(d1) + Math.abs(d2));
            if (d1 == 0.0 && d2 == 0.0)
                relError = 0.0;
            String str = "exp=" + expOut[i] + ", act=" + zPrime.getDouble(i) + "; relError = " + relError;
            assertTrue(str, relError < REL_ERROR_TOLERANCE);
        }
    }

    @Test
    public void testLeakyReLUDerivative() {
        //Derivative: 0.01 if x<0, 1 otherwise
        INDArray z = Nd4j.zeros(100);
        double[] expOut = new double[100];
        for (int i = 0; i < 100; i++) {
            double x = 0.1 * (i - 50);
            z.putScalar(i, x);
            expOut[i] = (x >= 0 ? 1 : 0.25);
        }

        INDArray zPrime = Nd4j.getExecutioner().exec(new LeakyReLUDerivative(z, 0.25));

        for (int i = 0; i < 100; i++) {
            double relError = Math.abs(expOut[i] - zPrime.getDouble(i))
                            / (Math.abs(expOut[i]) + Math.abs(zPrime.getDouble(i)));
            assertTrue(relError < REL_ERROR_TOLERANCE);
        }
    }

    @Test
    public void testSoftSignDerivative() {
        //Derivative: 1 / (1+abs(x))^2
        INDArray z = Nd4j.zeros(100).castTo(DataType.DOUBLE);
        double[] expOut = new double[100];
        for (int i = 0; i < 100; i++) {
            double x = 0.1 * (i - 50);
            z.putScalar(i, x);
            double temp = 1 + Math.abs(x);
            expOut[i] = 1.0 / (temp * temp);
        }

        INDArray zPrime = Nd4j.getExecutioner().exec(new SoftSignDerivative(z));

        for (int i = 0; i < 100; i++) {
            double relError = Math.abs(expOut[i] - zPrime.getDouble(i))
                            / (Math.abs(expOut[i]) + Math.abs(zPrime.getDouble(i)));
            assertTrue(relError < REL_ERROR_TOLERANCE);
        }
    }

    @Override
    public char ordering() {
        return 'f';
    }
}
