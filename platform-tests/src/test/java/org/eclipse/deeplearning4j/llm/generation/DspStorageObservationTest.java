/* SPDX-License-Identifier: Apache-2.0 */
package org.eclipse.deeplearning4j.llm.generation;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class DspStorageObservationTest {
    @Test
    void reusedSlotObservationDoesNotAttributeFailureToOriginalProducer() {
        String diagnostic = GenerationPipeline.dspStorageObservation(149);
        assertTrue(diagnostic.contains("dspObservation=post-execution-storage"));
        assertTrue(diagnostic.contains("firstObservedNonFiniteStorageSlot=149"));
        assertTrue(diagnostic.contains("scanOrder=output-slot-index"));
        assertTrue(diagnostic.contains("causalOp=unknown"));
        assertTrue(diagnostic.contains("buffer coloring may reuse storage"));
        assertTrue(diagnostic.contains("not values at execution time"));
        assertFalse(diagnostic.contains("firstNonFiniteOp="));
    }

    @Test
    void noObservedSlotStillDoesNotClaimTemporalEvidence() {
        String diagnostic = GenerationPipeline.dspStorageObservation(-1);
        assertTrue(diagnostic.contains("firstObservedNonFiniteStorageSlot=-1"));
        assertTrue(diagnostic.contains("causalOp=unknown"));
        assertTrue(diagnostic.contains("dspObservation=post-execution-storage"));
    }
}
