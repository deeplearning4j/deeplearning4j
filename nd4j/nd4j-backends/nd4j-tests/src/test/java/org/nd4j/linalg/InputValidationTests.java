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

package org.nd4j.linalg;

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import static org.junit.jupiter.api.Assertions.fail;

@NativeTag
public class InputValidationTests extends BaseNd4jTestWithBackends {

    @Override
    public char ordering() {
        return 'c';
    }

    /////////////////////////////////////////////////////////////
    ///////////////////// Broadcast Tests ///////////////////////

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testInvalidColVectorOp1(Nd4jBackend backend) {
        INDArray first = Nd4j.create(10, 10);
        INDArray col = Nd4j.create(5, 1);
        try {
            first.muliColumnVector(col);
            fail("Should have thrown IllegalStateException");
        } catch (IllegalStateException e) {
            //OK
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testInvalidColVectorOp2(Nd4jBackend backend) {
        INDArray first = Nd4j.create(10, 10);
        INDArray col = Nd4j.create(5, 1);
        try {
            first.addColumnVector(col);
            fail("Should have thrown IllegalStateException");
        } catch (IllegalStateException e) {
            //OK
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testInvalidRowVectorOp1(Nd4jBackend backend) {
        INDArray first = Nd4j.create(10, 10);
        INDArray row = Nd4j.create(1, 5);
        try {
            first.addiRowVector(row);
            fail("Should have thrown IllegalStateException");
        } catch (IllegalStateException e) {
            //OK
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testInvalidRowVectorOp2(Nd4jBackend backend) {
        INDArray first = Nd4j.create(10, 10);
        INDArray row = Nd4j.create(1, 5);
        try {
            first.subRowVector(row);
            fail("Should have thrown IllegalStateException");
        } catch (IllegalStateException e) {
            //OK
        }
    }



}
