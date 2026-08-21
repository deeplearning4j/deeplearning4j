/*
 *  ******************************************************************************
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License version 2.0.
 *  *
 *  ******************************************************************************
 */
package org.nd4j.ggml.convert;

import org.junit.jupiter.api.Test;
import org.nd4j.common.config.ND4JInferenceWeightDataType;
import org.nd4j.ggml.format.GGMLDataType;
import org.nd4j.ggml.format.GGMLTensorInfo;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;

class GGMLToSameDiffConverterMemoryEstimateTest {

    @Test
    void dequantizedAdmissionUsesDestinationArrayBytes() {
        GGMLTensorInfo tensor = GGMLTensorInfo.builder()
                .name("blk.0.attn_q.weight")
                .shape(new long[]{2048, 2048})
                .numDimensions(2)
                .dataType(GGMLDataType.GGML_TYPE_Q4_K)
                .build();
        GGMLToSameDiffConverter converter = new GGMLToSameDiffConverter(
                ConversionOptions.forInference(ND4JInferenceWeightDataType.FLOAT32));

        long estimated = converter.estimateDestinationBytes(tensor, false);

        assertEquals(Math.multiplyExact(tensor.getNumElements(), 4L), estimated);
        assertNotEquals(tensor.getDataSize(), estimated,
                "Compressed GGUF bytes must not drive destination-memory admission");
    }

    @Test
    void runtimePackedAdmissionPreservesCompressedStorageBytes() {
        GGMLTensorInfo tensor = GGMLTensorInfo.builder()
                .name("blk.0.attn_q.weight")
                .shape(new long[]{2048, 2048})
                .numDimensions(2)
                .dataType(GGMLDataType.GGML_TYPE_Q4_K)
                .build();
        GGMLToSameDiffConverter converter = new GGMLToSameDiffConverter(
                ConversionOptions.forInference(ND4JInferenceWeightDataType.INT4));

        assertEquals(tensor.getDataSize(), converter.estimateDestinationBytes(tensor, false));
    }
}
