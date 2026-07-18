/*
 *  ******************************************************************************
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * See the NOTICE file distributed with this work for additional information
 *  * regarding copyright ownership.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */
package org.nd4j.autodiff.samediff.dsp;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.GraphExecutionMode;
import org.nd4j.autodiff.samediff.internal.InferenceSession;
import org.nd4j.common.config.ND4JSystemProperties;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Collections;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Regression for requested DSP outputs that are themselves views.
 *
 * <p>The output boundary materializes a requested view so callers can retain it.
 * Frozen replay must not treat that boundary copy as the replay-stable internal
 * buffer: the next execution legitimately restores the reshape's parent-backed
 * view before lifecycle validation.
 */
@Tag("dsp")
public class DspRequestedViewOutputReplayTest {

    private static final String INPUT = "x";
    private static final String OUTPUT = "requested_view";
    private static final int REPLAYS = 6;

    private SameDiff sameDiff;

    @BeforeEach
    public void setUp() {
        System.setProperty(ND4JSystemProperties.DYNAMIC_SHAPE_PLAN_ENABLED, "true");
        InferenceSession.setDynamicShapePlanEnabled(true);
    }

    @AfterEach
    public void tearDown() {
        if (sameDiff != null) {
            sameDiff.close();
            sameDiff = null;
        }
        Nd4j.getExecutioner().commit();
    }

    @Test
    public void testRequestedReshapeViewSurvivesFrozenReplay() {
        INDArray reference = captureReference();
        try {
            sameDiff = buildGraph();
            sameDiff.setDspAutoCompileEnabled(true);
            sameDiff.setDspNativeAutoCompileEnabled(true);
            sameDiff.setGraphExecutionMode(GraphExecutionMode.CUDA_GRAPHS);

            for (int replay = 0; replay < REPLAYS; replay++) {
                INDArray input = input();
                try {
                    Map<String, INDArray> outputs =
                            sameDiff.output(Collections.singletonMap(INPUT, input), OUTPUT);
                    INDArray actual = outputs.get(OUTPUT);
                    assertNotNull(actual, "replay " + replay + " returned a null requested view");
                    assertArrayEquals(reference.shape(), actual.shape(),
                            "replay " + replay + " changed requested-view shape");
                    assertTrue(reference.equalsWithEps((Object) actual, 1e-4),
                            "replay " + replay + " changed requested-view values");
                } finally {
                    input.close();
                }
            }
        } finally {
            reference.close();
        }
    }

    private static INDArray captureReference() {
        SameDiff referenceGraph = buildGraph();
        INDArray input = input();
        try {
            referenceGraph.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);
            INDArray output = referenceGraph
                    .output(Collections.singletonMap(INPUT, input), OUTPUT)
                    .get(OUTPUT);
            assertNotNull(output, "slot-by-slot reference returned null");
            return output.dup();
        } finally {
            input.close();
            referenceGraph.close();
        }
    }

    private static SameDiff buildGraph() {
        SameDiff graph = SameDiff.create();
        SDVariable input = graph.placeHolder(INPUT, DataType.FLOAT, 4, 16);
        SDVariable weights = graph.var("weights",
                Nd4j.linspace(DataType.FLOAT, 0.01, 0.005, 16 * 16).reshape(16, 16));
        SDVariable projected = graph.mmul("projected", input, weights);
        SDVariable activated = graph.nn.relu("activated", projected, 0);
        graph.reshape(OUTPUT, activated, 2, 32);
        graph.setOutputs(OUTPUT);
        return graph;
    }

    private static INDArray input() {
        return Nd4j.linspace(DataType.FLOAT, -0.5, 0.01, 4 * 16).reshape(4, 16);
    }
}
