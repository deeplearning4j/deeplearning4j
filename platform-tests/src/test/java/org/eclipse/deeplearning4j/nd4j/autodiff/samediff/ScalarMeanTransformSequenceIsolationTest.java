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

package org.eclipse.deeplearning4j.nd4j.autodiff.samediff;

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.validation.OpValidation;
import org.nd4j.autodiff.validation.TestCase;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import java.nio.ByteBuffer;

import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertNotNull;

public class ScalarMeanTransformSequenceIsolationTest extends BaseNd4jTestWithBackends {

    private static final int MINIBATCH = 5;
    private static final int N_OUT = 4;

    private static TestCase addScalarTransformCase(SameDiff sd, SDVariable in, INDArray ia, int transformIdx) {
        TestCase tc = new TestCase(sd);
        SDVariable t;
        switch (transformIdx) {
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

        sd.mean("loss", t);
        return tc;
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testFirstScalarMeanTransformSequence(Nd4jBackend backend) {
        DataType initialType = Nd4j.dataType();
        try {
            Nd4j.setDataType(DataType.DOUBLE);
            Nd4j.getRandom().setSeed(12345);

            for (int i = 0; i < 5; i++) {
                SameDiff sd = SameDiff.create();
                SDVariable in = sd.var("in", DataType.DOUBLE, MINIBATCH, N_OUT);
                INDArray input = Nd4j.randn(DataType.DOUBLE, MINIBATCH, N_OUT);

                TestCase tc = addScalarTransformCase(sd, in, input, i);
                sd.associateArrayWithVariable(input, in);
                assertNull(OpValidation.validate(tc, true), "Failed on scalar transform index " + i);
            }
        } finally {
            Nd4j.setDataType(initialType);
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testFirstScalarMeanTransformSequenceExactTransformsSetup(Nd4jBackend backend) {
        DataType initialType = Nd4j.dataType();
        try {
            Nd4j.create(1);
            Nd4j.setDataType(DataType.DOUBLE);
            Nd4j.getRandom().setSeed(12345);

            for (int i = 0; i < 6; i++) {
                SameDiff sd = SameDiff.create();
                SDVariable in = sd.var("in", MINIBATCH, N_OUT);
                INDArray ia = Nd4j.randn(DataType.DOUBLE, MINIBATCH, N_OUT);
                INDArray inputForGraph = ia.dup(ia.ordering());
                TestCase tc = addScalarTransformCase(sd, in, ia, i);
                String name = sd.ops()[0].opName();
                sd.associateArrayWithVariable(inputForGraph, in);
                tc.testName(name);

                assertNull(OpValidation.validate(tc, true), "Failed on scalar transform index " + i);
            }
        } finally {
            Nd4j.setDataType(initialType);
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testScalarTransformFlatBufferRoundTripSequence(Nd4jBackend backend) throws Exception {
        DataType initialType = Nd4j.dataType();
        try {
            Nd4j.create(1);
            Nd4j.setDataType(DataType.DOUBLE);
            Nd4j.getRandom().setSeed(12345);

            for (int cycle = 0; cycle < 20; cycle++) {
                for (int i = 0; i < 6; i++) {
                    SameDiff sd = SameDiff.create();
                    SDVariable in = sd.var("in", MINIBATCH, N_OUT);
                    INDArray ia = Nd4j.randn(DataType.DOUBLE, MINIBATCH, N_OUT);
                    addScalarTransformCase(sd, in, ia, i);
                    sd.associateArrayWithVariable(ia.dup(ia.ordering()), in);

                    ByteBuffer serialized = sd.asFlatBuffers(true);
                    assertNotNull(serialized, "Serialized graph was null for cycle " + cycle + ", transform " + i);
                    assertNotNull(SameDiff.fromFlatBuffers(serialized),
                            "Round-trip deserialization failed for cycle " + cycle + ", transform " + i);
                }
            }
        } finally {
            Nd4j.setDataType(initialType);
        }
    }
}
