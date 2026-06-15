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
import org.nd4j.autodiff.validation.GradCheckUtil;
import org.nd4j.autodiff.validation.OpValidation;
import org.nd4j.autodiff.validation.TestCase;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class ScalarReverseSubtractionGradientIsolationTest extends BaseNd4jTestWithBackends {

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testScalarRSubAnalyticalGradientMatchesExpected(Nd4jBackend backend) {
        SameDiff sd = SameDiff.create();
        SDVariable in = sd.var("in", DataType.DOUBLE, 5, 4);
        SDVariable out = in.rsub(5.0);
        SDVariable loss = sd.mean("loss", out);

        INDArray input = Nd4j.linspace(1, 20, 20, DataType.DOUBLE).reshape('c', 5, 4);
        sd.associateArrayWithVariable(input, in);

        INDArray grad = sd.calculateGradients(null, "in").get("in");
        INDArray expected = Nd4j.valueArrayOf(input.shape(), -1.0 / input.length());

        assertEquals(expected, grad);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testScalarRSubGradCheckPasses(Nd4jBackend backend) {
        SameDiff sd = SameDiff.create();
        SDVariable in = sd.var("in", DataType.DOUBLE, 5, 4);
        SDVariable out = in.rsub(5.0);
        sd.mean("loss", out);

        INDArray input = Nd4j.linspace(1, 20, 20, DataType.DOUBLE).reshape('c', 5, 4).divi(10.0);
        sd.associateArrayWithVariable(input, in);

        assertTrue(GradCheckUtil.checkGradients(sd, null), "Grad check failed for scalar rsub");
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testScalarRSubOpValidationMatchesTransformSetup(Nd4jBackend backend) {
        SameDiff sd = SameDiff.create();
        SDVariable in = sd.var("in", DataType.DOUBLE, 5, 4);
        SDVariable out = in.rsub(5.0);
        INDArray input = Nd4j.randn(DataType.DOUBLE, 5, 4);

        TestCase tc = new TestCase(sd);
        tc.expectedOutput(out.name(), input.rsub(5.0));
        sd.mean("loss", out);
        sd.associateArrayWithVariable(input, in);

        assertNull(OpValidation.validate(tc, true));
    }
}
