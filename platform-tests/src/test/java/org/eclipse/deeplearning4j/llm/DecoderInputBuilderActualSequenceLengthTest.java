/*
 *  ******************************************************************************
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  ******************************************************************************
 */
package org.eclipse.deeplearning4j.llm;

import org.eclipse.deeplearning4j.llm.generation.DecoderInputBuilder;
import org.eclipse.deeplearning4j.llm.generation.ModelIOConfig;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.Map;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;

class DecoderInputBuilderActualSequenceLengthTest {

    @Test
    void scoringMapFeedsDiscoveredActualSequenceLengthUsingDeclaredType() {
        SameDiff decoder = controlGraph(DataType.INT32);
        INDArray inputIds = Nd4j.zeros(DataType.INT64, 1, 3);
        Map<String, INDArray> inputs = DecoderInputBuilder.buildScoringInputMap(decoder, inputIds);
        try {
            ModelIOConfig ioConfig = ModelIOConfig.discover(decoder);
            assertEquals("actual_sequence_length", ioConfig.getActualSequenceLengthName());

            INDArray actualLength = inputs.get("actual_sequence_length");
            assertNotNull(actualLength);
            assertEquals(DataType.INT32, actualLength.dataType());
            assertEquals(3L, actualLength.getLong(0));
            assertEquals(0L, inputs.get("position_offset").getLong(0));
            assertEquals(0L, inputs.get("cache_position").getLong(0));
        } finally {
            closeDistinct(inputs);
        }
    }

    @Test
    void fixedBufferPrefillFeedsRealLengthInsteadOfMaterializedWidth() {
        SameDiff decoder = controlGraph(DataType.INT64);
        ModelIOConfig ioConfig = ModelIOConfig.discover(decoder);
        INDArray paddedIds = Nd4j.zeros(DataType.INT64, 1, 8);
        Map<String, INDArray> inputs = DecoderInputBuilder.buildDecoderInputMap(
                ioConfig, decoder.inputs(), decoder,
                null, paddedIds,
                0L, 8L,
                null, 12L, 0L,
                false, 0L,
                null, false,
                null, null,
                3L);
        try {
            assertEquals(3L, inputs.get("actual_sequence_length").getLong(0));
            assertEquals(0L, inputs.get("position_offset").getLong(0));
            assertEquals(0L, inputs.get("cache_position").getLong(0));
        } finally {
            closeDistinct(inputs);
        }
    }

    private static SameDiff controlGraph(DataType actualLengthType) {
        SameDiff decoder = SameDiff.create();
        SDVariable inputIds = decoder.placeHolder("input_ids", DataType.INT64, -1, -1);
        SDVariable positionOffset = decoder.placeHolder("position_offset", DataType.INT64);
        SDVariable cachePosition = decoder.placeHolder("cache_position", DataType.INT64);
        SDVariable actualLength = decoder.placeHolder("actual_sequence_length", actualLengthType);
        SDVariable controlSum = actualLength.castTo(DataType.INT64)
                .add(positionOffset)
                .add(cachePosition)
                .add(inputIds.sum());
        decoder.setOutputs(controlSum.name());
        return decoder;
    }

    private static void closeDistinct(Map<String, INDArray> arrays) {
        Set<INDArray> closed = Collections.newSetFromMap(new IdentityHashMap<>());
        for (INDArray array : arrays.values()) {
            if (array != null && closed.add(array)) {
                array.close();
            }
        }
    }
}
