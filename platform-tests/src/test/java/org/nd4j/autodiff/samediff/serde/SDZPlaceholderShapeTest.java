/*
 * Copyright (c) Eclipse Deeplearning4j
 * SPDX-License-Identifier: Apache-2.0
 */
package org.nd4j.autodiff.samediff.serde;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.VariableType;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.factory.Nd4j;

import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;

class SDZPlaceholderShapeTest {

    @TempDir
    Path directory;

    @BeforeAll
    static void initializeBackend() {
        // Metadata-only graphs allocate no arrays, but SameDiff.close() uses native memory cleanup.
        Nd4j.getBackend();
    }

    @Test
    void scalarInt64PlaceholderRetainsEmptyShapeVector() throws Exception {
        try (SameDiff original = SameDiff.create()) {
            original.placeHolder("actual_sequence_length", DataType.INT64, new long[0]);
            assertArrayEquals(new long[0], original.getVariable("actual_sequence_length").getShape());
            Path archive = directory.resolve("scalar.sdz");
            SDZSerializer.save(original, archive.toFile(), false, null);
            try (SameDiff restored = SDZSerializer.load(archive.toFile(), false)) {
                assertEquals(VariableType.PLACEHOLDER,
                        restored.getVariable("actual_sequence_length").getVariableType());
                assertEquals(DataType.INT64, restored.getVariable("actual_sequence_length").dataType());
                assertArrayEquals(new long[0], restored.getVariable("actual_sequence_length").getShape());
            }
        }
    }

    @Test
    void unknownRankPlaceholderRetainsAbsentShapeVector() throws Exception {
        try (SameDiff original = SameDiff.create()) {
            original.placeHolder("unknown", DataType.FLOAT, (long[]) null);
            assertNull(original.getVariable("unknown").getShape());
            Path archive = directory.resolve("unknown.sdz");
            SDZSerializer.save(original, archive.toFile(), false, null);
            try (SameDiff restored = SDZSerializer.load(archive.toFile(), false)) {
                assertEquals(VariableType.PLACEHOLDER, restored.getVariable("unknown").getVariableType());
                assertEquals(DataType.FLOAT, restored.getVariable("unknown").dataType());
                assertNull(restored.getVariable("unknown").getShape());
            }
        }
    }
}
