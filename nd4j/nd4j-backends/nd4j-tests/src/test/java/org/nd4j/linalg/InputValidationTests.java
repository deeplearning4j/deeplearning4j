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

package org.nd4j.linalg;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import static org.junit.Assert.fail;

/**
 * Created by Alex on 05/08/2016.
 */
@RunWith(Parameterized.class)
public class InputValidationTests extends BaseNd4jTest {

    public InputValidationTests(Nd4jBackend backend) {
        super(backend);
    }

    @Override
    public char ordering() {
        return 'c';
    }

    /////////////////////////////////////////////////////////////
    ///////////////////// Broadcast Tests ///////////////////////

    @Test
    public void testInvalidColVectorOp1() {
        INDArray first = Nd4j.create(10, 10);
        INDArray col = Nd4j.create(5, 1);
        try {
            first.muliColumnVector(col);
            fail("Should have thrown IllegalStateException");
        } catch (IllegalStateException e) {
            //OK
        }
    }

    @Test
    public void testInvalidColVectorOp2() {
        INDArray first = Nd4j.create(10, 10);
        INDArray col = Nd4j.create(5, 1);
        try {
            first.addColumnVector(col);
            fail("Should have thrown IllegalStateException");
        } catch (IllegalStateException e) {
            //OK
        }
    }

    @Test
    public void testInvalidRowVectorOp1() {
        INDArray first = Nd4j.create(10, 10);
        INDArray row = Nd4j.create(1, 5);
        try {
            first.addiRowVector(row);
            fail("Should have thrown IllegalStateException");
        } catch (IllegalStateException e) {
            //OK
        }
    }

    @Test
    public void testInvalidRowVectorOp2() {
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
