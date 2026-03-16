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

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for {@link SameDiff#findVariablesByOpPattern(String, String)}.
 * This method performs structural graph queries — finding variables produced
 * by one op type and consumed by another, without any name matching.
 *
 * Note: scalar ops use different opNames than element-wise ops:
 *   - x.mul(scalar) → "mul_scalar", x.mul(SDVariable) → "multiply"
 *   - x.add(scalar) → "add_scalar", x.add(SDVariable) → "add"
 */
@Slf4j
@DisplayName("GraphQueryTest")
public class GraphQueryTest {

    /**
     * Build a simple graph with a producer→consumer pattern using element-wise ops.
     * Graph: x -> mul(y) -> mulResult -> add(z) -> output
     * Query: find "multiply" → "add" pattern.
     */
    @Test
    @DisplayName("testFindVariablesByOpPattern")
    public void testFindVariablesByOpPattern() {
        SameDiff sd = SameDiff.create();

        // Use element-wise ops (SDVariable operands) to get "multiply" and "add" opNames
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable y = sd.constant("y", Nd4j.ones(DataType.FLOAT, 1, 4).mul(2));
        SDVariable z = sd.constant("z", Nd4j.ones(DataType.FLOAT, 1, 4));
        SDVariable mulResult = x.mul("mulResult", y);
        SDVariable output = mulResult.add("output", z);

        // Query: find multiply→add pattern
        List<String> matches = sd.findVariablesByOpPattern("multiply", "add");
        log.info("Found matches: {}", matches);
        assertFalse(matches.isEmpty(), "Should find multiply→add pattern");
        assertTrue(matches.contains("mulResult"),
                "Matched variable should be 'mulResult', got: " + matches);
    }

    /**
     * Build a graph without the target pattern → verify empty list.
     */
    @Test
    @DisplayName("testNoMatchReturnsEmpty")
    public void testNoMatchReturnsEmpty() {
        SameDiff sd = SameDiff.create();

        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable output = x.add("output", 1.0);

        // Query for a pattern that doesn't exist
        List<String> matches = sd.findVariablesByOpPattern("Tile", "onnx_multi_head_attention");
        assertTrue(matches.isEmpty(), "Should return empty list when no pattern matches");
    }

    /**
     * Build a graph with multiple producer→consumer matches → verify all found.
     * Uses element-wise ops (SDVariable operands) for correct opName matching.
     */
    @Test
    @DisplayName("testMultipleMatches")
    public void testMultipleMatches() {
        SameDiff sd = SameDiff.create();

        // Create multiple multiply→add chains using element-wise ops
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable scale1 = sd.constant("scale1", Nd4j.ones(DataType.FLOAT, 1, 4).mul(2));
        SDVariable scale2 = sd.constant("scale2", Nd4j.ones(DataType.FLOAT, 1, 4).mul(3));
        SDVariable scale3 = sd.constant("scale3", Nd4j.ones(DataType.FLOAT, 1, 4).mul(4));
        SDVariable offset = sd.constant("offset", Nd4j.ones(DataType.FLOAT, 1, 4));

        SDVariable mul1 = x.mul("mul1", scale1);       // opName = "multiply"
        SDVariable out1 = mul1.add("out1", offset);     // opName = "add"
        SDVariable mul2 = x.mul("mul2", scale2);
        SDVariable out2 = mul2.add("out2", offset);
        SDVariable mul3 = x.mul("mul3", scale3);
        SDVariable out3 = mul3.add("out3", offset);

        // Query: find multiply→add pattern
        List<String> matches = sd.findVariablesByOpPattern("multiply", "add");
        log.info("Found {} matches: {}", matches.size(), matches);
        assertEquals(3, matches.size(), "Should find 3 multiply→add patterns");
        assertTrue(matches.contains("mul1"), "Should contain mul1");
        assertTrue(matches.contains("mul2"), "Should contain mul2");
        assertTrue(matches.contains("mul3"), "Should contain mul3");
    }

    /**
     * Verify the query doesn't match indirect connections.
     * Graph: x -> mul(y) -> mulResult -> add(z) -> intermediate -> add(w) -> output
     * Only mulResult should match "multiply"→"add", not intermediate.
     */
    @Test
    @DisplayName("testNoIndirectMatches")
    public void testNoIndirectMatches() {
        SameDiff sd = SameDiff.create();

        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable y = sd.constant("y", Nd4j.ones(DataType.FLOAT, 1, 4).mul(2));
        SDVariable z = sd.constant("z", Nd4j.zeros(DataType.FLOAT, 1, 4));
        SDVariable w = sd.constant("w", Nd4j.ones(DataType.FLOAT, 1, 4));

        SDVariable mulResult = x.mul("mulResult", y);       // multiply op
        SDVariable intermediate = mulResult.add("intermediate", z); // add op
        SDVariable output = intermediate.add("output", w);         // add op

        // Query: find multiply→add
        List<String> matches = sd.findVariablesByOpPattern("multiply", "add");
        log.info("Matches: {}", matches);
        // mulResult is produced by multiply, consumed by add(intermediate) → should match
        assertTrue(matches.contains("mulResult"), "mulResult→add(intermediate) should match");
        // intermediate is produced by add, not multiply → should NOT match
        assertFalse(matches.contains("intermediate"),
                "intermediate is produced by add, not multiply — should not match");
    }
}
