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
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class ScalarPowFlatBufferIsolationTest extends BaseNd4jTestWithBackends {

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testScalarPowRoundTripOutputAll(Nd4jBackend backend) throws IOException {
        for (char order : new char[]{'c', 'f'}) {
            SameDiff sd = SameDiff.create();
            INDArray input = Nd4j.linspace(1, 24, 24, DataType.DOUBLE).reshape('c', 2, 3, 4).dup(order);
            SDVariable in = sd.var("in", input);
            SDVariable out = sd.math().pow("out", in, 2.0);
            sd.standardDeviation("loss", out, true);

            ByteBuffer serialized = sd.asFlatBuffers(true);
            SameDiff restored = SameDiff.fromFlatBuffers(serialized);

            Map<String, INDArray> originalOutputs = sd.outputAll(null);
            Map<String, INDArray> restoredOutputs = restored.outputAll(null);

            assertEquals(originalOutputs.get("out"), restoredOutputs.get("out"), "Scalar pow output mismatch for order " + order);
            assertEquals(originalOutputs.get("loss"), restoredOutputs.get("loss"), "Scalar pow loss mismatch for order " + order);
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testScalarPowRoundTripGradientsRetainExponent(Nd4jBackend backend) throws IOException {
        SameDiff sd = SameDiff.create();
        INDArray input = Nd4j.linspace(1, 24, 24, DataType.DOUBLE).reshape('c', 2, 3, 4);
        SDVariable in = sd.var("in", input.dup());
        SDVariable out = sd.math().pow("out", in, 2.5);
        SDVariable loss = sd.sum("loss", out);
        sd.setLossVariables(loss.name());

        SameDiff restored = SameDiff.fromFlatBuffers(sd.asFlatBuffers(true));
        INDArray originalGrad = sd.calculateGradients(null, "in").get("in");
        INDArray grad = restored.calculateGradients(null, "in").get("in");
        INDArray expected = Transforms.pow(input.dup(), 1.5, true).muli(2.5);

        assertTrue(expected.equalsWithEps(originalGrad, 1e-6), "Scalar pow gradient mismatch before FlatBuffers round-trip");
        assertTrue(expected.equalsWithEps(grad, 1e-6), "Scalar pow gradient mismatch after FlatBuffers round-trip");
    }
}
