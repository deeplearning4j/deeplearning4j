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

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import java.io.File;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Validates that DOUBLE scalar constants in SameDiff graphs do not cascade
 * through matmul dtype promotion to produce DOUBLE output tensors.
 *
 * Root cause: FlatBuffersMapper and SameDiffSerializer deserialize scalar
 * constants via Nd4j.constantScalar(bb.getDouble()) which creates DataType.DOUBLE.
 * matmul.cpp:209 does dtypeZ = max(dtypeX, dtypeY) where DOUBLE(6) > FLOAT(5),
 * cascading DOUBLE through every matmul in the graph. The fix losslessly
 * downcasts DOUBLE scalars to FLOAT when the value survives float32 round-trip.
 */
@NativeTag
@Tag(TagNames.SAMEDIFF)
@DisplayName("DOUBLE scalar constant promotion tests")
public class SameDiffDoubleScalarPromotionTest extends BaseNd4jTestWithBackends {

    @Override
    public char ordering() {
        return 'c';
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("DOUBLE scalar should not promote matmul output dtype after serialize/deserialize")
    public void testDoubleScalarDoesNotPromoteMatmulOutput(Nd4jBackend backend) throws Exception {
        SameDiff sd = SameDiff.create();

        SDVariable input = sd.placeHolder("input", DataType.FLOAT, 2, 64);
        SDVariable weight = sd.var("weight", Nd4j.rand(DataType.FLOAT, 64, 64));
        // 0.125 = 1/sqrt(64) — survives float32 round-trip
        SDVariable scalingFactor = sd.constant("scale", Nd4j.scalar(DataType.DOUBLE, 0.125));

        SDVariable mm = sd.mmul("mm", input, weight);
        SDVariable scaled = mm.mul("scaled", scalingFactor);

        // Serialize and reload — the downcast happens at deserialization
        File tmp = File.createTempFile("double_matmul_test", ".fb");
        tmp.deleteOnExit();
        sd.save(tmp, true);
        SameDiff loaded = SameDiff.load(tmp, false);

        INDArray inputArr = Nd4j.rand(DataType.FLOAT, 2, 64);
        Map<String, INDArray> out = loaded.output(Map.of("input", inputArr), "scaled");

        INDArray result = out.get("scaled");
        assertNotNull(result);
        assertEquals(DataType.FLOAT, result.dataType(),
                "Output should be FLOAT — DOUBLE scalar 0.125 downcast to FLOAT at deserialization");
        assertEquals(2, result.shape()[0]);
        assertEquals(64, result.shape()[1]);

        double maxVal = result.maxNumber().doubleValue();
        assertTrue(maxVal > 1e-6, "Output should have non-zero values, got max=" + maxVal);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Serialized DOUBLE scalars should deserialize as FLOAT when lossless")
    public void testSerializedDoubleScalarDeserializesAsFloat(Nd4jBackend backend) throws Exception {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        // Use values that survive float32 round-trip: powers of 2 and exact fractions
        SDVariable scale = sd.constant("scale", Nd4j.scalar(DataType.DOUBLE, 0.125));
        SDVariable half = sd.constant("half", Nd4j.scalar(DataType.DOUBLE, 0.5));
        SDVariable result = x.add("out", scale).mul(half);

        File tmp = File.createTempFile("double_scalar_test", ".fb");
        tmp.deleteOnExit();
        sd.save(tmp, true);
        SameDiff loaded = SameDiff.load(tmp, false);

        SDVariable loadedScale = loaded.getVariable("scale");
        SDVariable loadedHalf = loaded.getVariable("half");
        assertEquals(DataType.FLOAT, loadedScale.getArr().dataType(),
                "0.125 round-trips losslessly to float32, should be FLOAT after load");
        assertEquals(DataType.FLOAT, loadedHalf.getArr().dataType(),
                "0.5 round-trips losslessly to float32, should be FLOAT after load");

        INDArray input = Nd4j.rand(DataType.FLOAT, 3, 4);
        Map<String, INDArray> out = loaded.output(Map.of("x", input), "out");
        assertEquals(DataType.FLOAT, out.get("out").dataType());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Non-round-trippable DOUBLE values must be preserved")
    public void testNonRoundTrippableDoublePreserved(Nd4jBackend backend) throws Exception {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        // Math.PI does NOT survive float32 round-trip
        SDVariable pi = sd.constant("pi", Nd4j.scalar(DataType.DOUBLE, Math.PI));
        SDVariable result = x.mul("out", pi);

        File tmp = File.createTempFile("double_pi_test", ".fb");
        tmp.deleteOnExit();
        sd.save(tmp, true);
        SameDiff loaded = SameDiff.load(tmp, false);

        SDVariable loadedPi = loaded.getVariable("pi");
        assertEquals(DataType.DOUBLE, loadedPi.getArr().dataType(),
                "Math.PI does not round-trip to float32, must remain DOUBLE");
    }
}
