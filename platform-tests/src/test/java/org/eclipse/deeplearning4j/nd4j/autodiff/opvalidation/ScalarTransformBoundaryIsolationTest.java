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

package org.eclipse.deeplearning4j.nd4j.autodiff.opvalidation;

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.validation.OpValidation;
import org.nd4j.autodiff.validation.TestCase;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import static org.junit.jupiter.api.Assertions.assertNull;

public class ScalarTransformBoundaryIsolationTest extends BaseOpValidation {

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testFirstScalarTransformsWithOpValidationHarness(Nd4jBackend backend) {
        runScalarBoundarySequence(false);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testFirstScalarTransformsWithForcedGcBetweenIterations(Nd4jBackend backend) throws Exception {
        runScalarBoundarySequence(true);
    }

    private static void runScalarBoundarySequence(boolean forceGcBetweenIterations) {
        Nd4j.create(1);
        Nd4j.setDataType(DataType.DOUBLE);
        Nd4j.getRandom().setSeed(12345);

        for (int i = 0; i < 6; i++) {
            SameDiff sd = SameDiff.create();

            int nOut = 4;
            int minibatch = 5;
            SDVariable in = sd.var("in", minibatch, nOut);

            INDArray ia = Nd4j.randn(DataType.DOUBLE, minibatch, nOut);
            INDArray inputForGraph = null;

            SDVariable t;
            TestCase tc = new TestCase(sd);

            switch (i) {
                case 0:
                    t = in.add(5.0);
                    tc.expectedOutput(t.name(), ia.add(5.0));
                    break;
                case 1:
                    t = in.sub(5.0);
                    tc.expectedOutput(t.name(), ia.sub(5.0));
                    break;
                case 2:
                    t = in.mul(2.5);
                    tc.expectedOutput(t.name(), ia.mul(2.5));
                    break;
                case 3:
                    t = in.div(4.0);
                    tc.expectedOutput(t.name(), ia.div(4.0));
                    break;
                case 4:
                    t = in.rsub(5.0);
                    tc.expectedOutput(t.name(), ia.rsub(5.0));
                    break;
                case 5:
                    t = in.rdiv(1.0);
                    tc.expectedOutput(t.name(), ia.rdiv(1.0));
                    break;
                default:
                    throw new IllegalStateException();
            }

            DifferentialFunction[] funcs = sd.ops();
            String name = funcs[0].opName();
            sd.mean("loss", t);
            if (inputForGraph == null) {
                inputForGraph = ia.dup(ia.ordering());
            }
            sd.associateArrayWithVariable(inputForGraph, in);
            tc.testName(name);

            assertNull(OpValidation.validate(tc, true), "Failed on scalar transform index " + i);

            if (forceGcBetweenIterations) {
                System.gc();
                System.runFinalization();
                try {
                    Thread.sleep(25);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    throw new RuntimeException(e);
                }
            }
        }
    }
}
