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

package org.nd4j.linalg.profiling;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.executioner.OpExecutioner;
import org.nd4j.linalg.api.ops.executioner.OpExecutionerUtil;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import static org.junit.jupiter.api.Assertions.assertThrows;


public class InfNanTests extends BaseNd4jTestWithBackends {


    @BeforeEach
    public void setUp() {

    }

    @AfterEach
    public void cleanUp() {
        Nd4j.getExecutioner().setProfilingMode(OpExecutioner.ProfilingMode.DISABLED);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testInf1(Nd4jBackend backend) {
        assertThrows(ND4JIllegalStateException.class,() -> {
            Nd4j.getExecutioner().setProfilingMode(OpExecutioner.ProfilingMode.INF_PANIC);

            INDArray x = Nd4j.create(100);

            x.putScalar(2, Float.NEGATIVE_INFINITY);

            OpExecutionerUtil.checkForAny(x);
        });

    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testInf2(Nd4jBackend backend) {
        assertThrows(ND4JIllegalStateException.class,() -> {
            Nd4j.getExecutioner().setProfilingMode(OpExecutioner.ProfilingMode.ANY_PANIC);

            INDArray x = Nd4j.create(100);

            x.putScalar(2, Float.NEGATIVE_INFINITY);

            OpExecutionerUtil.checkForAny(x);
        });

    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testInf3(Nd4jBackend backend) {
        Nd4j.getExecutioner().setProfilingMode(OpExecutioner.ProfilingMode.ANY_PANIC);

        INDArray x = Nd4j.create(100);

        OpExecutionerUtil.checkForAny(x);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testInf4(Nd4jBackend backend) {
        Nd4j.getExecutioner().setProfilingMode(OpExecutioner.ProfilingMode.DISABLED);

        INDArray x = Nd4j.create(100);

        OpExecutionerUtil.checkForAny(x);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testNaN1(Nd4jBackend backend) {
        assertThrows(ND4JIllegalStateException.class,() -> {
            Nd4j.getExecutioner().setProfilingMode(OpExecutioner.ProfilingMode.NAN_PANIC);

            INDArray x = Nd4j.create(100);

            x.putScalar(2, Float.NaN);

            OpExecutionerUtil.checkForAny(x);
        });

    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testNaN2(Nd4jBackend backend) {
        assertThrows(ND4JIllegalStateException.class,() -> {
            Nd4j.getExecutioner().setProfilingMode(OpExecutioner.ProfilingMode.ANY_PANIC);

            INDArray x = Nd4j.create(100);

            x.putScalar(2, Float.NaN);

            OpExecutionerUtil.checkForAny(x);
        });

    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testNaN3(Nd4jBackend backend) {
        Nd4j.getExecutioner().setProfilingMode(OpExecutioner.ProfilingMode.ANY_PANIC);

        INDArray x = Nd4j.create(100);

        OpExecutionerUtil.checkForAny(x);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testNaN4(Nd4jBackend backend) {
        Nd4j.getExecutioner().setProfilingMode(OpExecutioner.ProfilingMode.DISABLED);

        INDArray x = Nd4j.create(100);

        OpExecutionerUtil.checkForAny(x);
    }

    @Override
    public char ordering() {
        return 'c';
    }
}
