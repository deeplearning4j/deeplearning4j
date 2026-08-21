/*
 *  ******************************************************************************
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */
package org.nd4j.autodiff.samediff.dsp;

import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Throwaway isolation test for the SmolDocling vision-mask failure reported as
 * {@code [992,1,2] -> [1024,2]} inside a one-input {@code Where} DSP slot.
 */
class SmolDoclingWhereShapeThrowawayTest {

    @Test
    void oneInputWhereRefreshesShapeThroughCoordinateReshape() {
        try (SameDiff sd = SameDiff.create()) {
            sd.setDspAutoCompileEnabled(true);
            sd.setDspNativeAutoCompileEnabled(true);

            SDVariable patchMask = sd.placeHolder("pixel_attention_mask", DataType.BOOL, 32, 32);
            SDVariable coordinates = sd.where("Where_1", patchMask);
            sd.reshape("coordinates_with_group_axis", coordinates, -1, 1, 2);

            int[] validPatchCounts = {1024, 992, 1024, 992};
            for (int execution = 0; execution < validPatchCounts.length; execution++) {
                int validPatchCount = validPatchCounts[execution];
                INDArray mask = Nd4j.ones(DataType.BOOL, 32, 32);
                if (validPatchCount == 992) {
                    mask.get(NDArrayIndex.point(31), NDArrayIndex.all()).assign(0);
                }

                INDArray output = sd.outputSingle(
                        Map.of("pixel_attention_mask", mask),
                        "coordinates_with_group_axis");

                assertArrayEquals(new long[]{validPatchCount, 1, 2}, output.shape(),
                        "Execution " + execution + " reused a stale Where extent");
                assertEquals((long) validPatchCount * 2, output.length(),
                        "Execution " + execution + " allocated the wrong coordinate buffer length");

                output.close();
                mask.close();
            }
        }
    }
}
