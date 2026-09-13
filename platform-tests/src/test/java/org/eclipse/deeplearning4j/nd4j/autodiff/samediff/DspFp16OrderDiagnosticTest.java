package org.eclipse.deeplearning4j.nd4j.autodiff.samediff;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Opt-in, bounded order/data attribution for the intermittent FP16 replay NaN.
 * Invokes the existing test and its lifecycle verbatim; does not alter its assertions.
 * Run each predecessor setting in a separate JVM with identical seed ranges.
 */
@Slf4j
@EnabledIfSystemProperty(named = "nd4j.fp16.orderDiagnostic", matches = "true")
public class DspFp16OrderDiagnosticTest {
    @Test
    public void seededOriginalTestSequence() {
        boolean predecessor = Boolean.getBoolean("nd4j.fp16.predecessor");
        long firstSeed = Long.getLong("nd4j.fp16.firstSeed", 0L);
        int repetitions = Integer.getInteger("nd4j.fp16.repetitions", 16);
        if (repetitions < 1 || repetitions > 64) {
            throw new IllegalArgumentException("repetitions must be in [1,64]");
        }
        for (int repetition = 0; repetition < repetitions; repetition++) {
            long seed = firstSeed + repetition;
            if (predecessor) {
                DspMixedPrecisionReplayTest prior = new DspMixedPrecisionReplayTest();
                prior.setUp();
                try {
                    prior.testTritonDpaV2GqaPrefillProbabilitiesMatchNative();
                } finally {
                    prior.tearDown();
                }
            }
            Nd4j.getRandom().setSeed(seed);
            log.info("FP16_ORDER_START predecessor={} repetition={} seed={}",
                    predecessor, repetition, seed);
            DspMixedPrecisionReplayTest original = new DspMixedPrecisionReplayTest();
            original.setUp();
            try {
                original.testFp16WeightChainNoNaN();
                log.info("FP16_ORDER_PASS predecessor={} repetition={} seed={} steps=15",
                        predecessor, repetition, seed);
            } catch (AssertionError failure) {
                log.error("FP16_ORDER_FAIL predecessor={} repetition={} seed={}",
                        predecessor, repetition, seed, failure);
                throw failure;
            } finally {
                original.tearDown();
            }
        }
    }
}
