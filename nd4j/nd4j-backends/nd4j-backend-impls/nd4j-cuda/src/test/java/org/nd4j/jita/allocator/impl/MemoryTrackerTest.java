package org.nd4j.jita.allocator.impl;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.junit.Test;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.*;

@Slf4j
public class MemoryTrackerTest {

    @Test
    public void testAllocatedDelta() {
        val precBefore = MemoryTracker.getInstance().getPreciseFreeMemory(0);
        val approxBefore = MemoryTracker.getInstance().getApproximateFreeMemory(0);
        val deltaBefore = precBefore - approxBefore;

        for (int i = 0; i < 100; i++) {
            val buffer = Nd4j.createBuffer(DataType.FLOAT, 100000, false);
        }

        val precAfter = MemoryTracker.getInstance().getPreciseFreeMemory(0);
        val approxAfter = MemoryTracker.getInstance().getApproximateFreeMemory(0);
        val deltaAfter =  precAfter - approxAfter;

        log.info("Initial delta: {}; Allocation delta: {}", deltaBefore, deltaAfter);
        log.info("BEFORE: Precise: {}; Approx: {};", precBefore, approxBefore);
        log.info("AFTER: Precise: {}; Approx: {};", precAfter, approxAfter);
        log.info("Precise allocated: {}", precBefore - precAfter);
        log.info("Approx allocated: {}", MemoryTracker.getInstance().getActiveMemory(0));
    }
}