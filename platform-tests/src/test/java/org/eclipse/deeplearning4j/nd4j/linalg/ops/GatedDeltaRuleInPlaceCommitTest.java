package org.eclipse.deeplearning4j.nd4j.linalg.ops;

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.impl.transforms.custom.GatedDeltaRule;
import org.nd4j.linalg.api.ops.impl.transforms.custom.GatedDeltaRuleWithPrefix;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * The gated-delta-rule in-place commit flag (an INT32 scalar input read at run time).
 *
 * <p>Flag 0 must be exactly the flagless op: stateIn is only read and stateOut receives the
 * final state. Flag 1 must commit that same final state, bit for bit, into stateIn itself;
 * the activation (and prefix checkpoints) are unaffected. Covers the decode step (L=1), a
 * verification window, the chunked prefill (L=64 without actualLen) and BFLOAT16, whose
 * chunked path stores the state through a precision conversion.</p>
 */
class GatedDeltaRuleInPlaceCommitTest {
    private static final int H = 4;
    private static final int DK = 128;
    private static final int DV = 128;

    @ParameterizedTest(name = "prefix={0} L={1} actualLen={2} {3}")
    @CsvSource({
            "false,1,true,FLOAT", "false,5,true,FLOAT", "false,64,false,FLOAT", "false,64,false,BFLOAT16",
            "false,1,true,BFLOAT16", "true,1,true,FLOAT", "true,5,true,FLOAT", "true,5,true,BFLOAT16"})
    void commitFlagWritesTheFinalStateIntoStateIn(boolean prefix, int length, boolean withActualLen, DataType type) {
        Random rng = new Random(20260929L + length * 31L + (prefix ? 7 : 0) + type.ordinal());
        INDArray q = random(rng, type, 1f, 1, length, H, DK);
        INDArray k = random(rng, type, 0.2f, 1, length, H, DK);
        INDArray v = random(rng, type, 4f, 1, length, H, DV);
        INDArray beta = Nd4j.rand(DataType.FLOAT, 1, length, H).castTo(type);
        INDArray gate = random(rng, type, 0.1f, 1, length, H).subi(0.2);
        INDArray state = random(rng, type, 1f, 1, H, DK, DV);
        INDArray actualLen = withActualLen ? Nd4j.scalar(DataType.INT64, (long) length) : null;

        // Flagless reference.
        INDArray[] reference = Nd4j.exec(build(prefix, q, k, v, beta, gate, state.dup(), actualLen, null));

        // Flag 0: identical to the flagless op, stateIn untouched.
        INDArray stateFlag0 = state.dup();
        INDArray[] flag0 = Nd4j.exec(build(prefix, q, k, v, beta, gate, stateFlag0, actualLen,
                Nd4j.scalar(DataType.INT, 0)));
        assertBits(reference[0], flag0[0], "flag 0 activation");
        assertBits(reference[1], flag0[1], "flag 0 stateOut");
        if (prefix && length > 1) assertBits(writtenCheckpoints(reference[2], length), writtenCheckpoints(flag0[2], length),
                "flag 0 prefix checkpoints");
        assertBits(state, stateFlag0, "flag 0 must not write stateIn");

        // Flag 1: stateIn receives the final state; activation and checkpoints unchanged.
        INDArray stateFlag1 = state.dup();
        INDArray[] flag1 = Nd4j.exec(build(prefix, q, k, v, beta, gate, stateFlag1, actualLen,
                Nd4j.scalar(DataType.INT, 1)));
        assertBits(reference[0], flag1[0], "flag 1 activation");
        if (prefix && length > 1) assertBits(writtenCheckpoints(reference[2], length), writtenCheckpoints(flag1[2], length),
                "flag 1 prefix checkpoints");
        assertBits(reference[1], stateFlag1, "flag 1 committed state");
    }

    private static DynamicCustomOp build(boolean prefix, INDArray q, INDArray k, INDArray v, INDArray beta,
                                         INDArray gate, INDArray state, INDArray actualLen, INDArray flag) {
        if (prefix) {
            return flag == null
                    ? new GatedDeltaRuleWithPrefix(q, k, v, beta, gate, state, actualLen)
                    : new GatedDeltaRuleWithPrefix(q, k, v, beta, gate, state, actualLen, flag);
        }
        return flag == null
                ? new GatedDeltaRule(q, k, v, beta, gate, state, actualLen)
                : new GatedDeltaRule(q, k, v, beta, gate, state, actualLen, flag);
    }

    /**
     * Checkpoint slots t < L - 1: the op writes the state after every consumed row except the
     * last, whose state is stateOut (the final slot is defined-but-unselected storage).
     */
    private static INDArray writtenCheckpoints(INDArray prefixOut, int length) {
        return prefixOut.get(NDArrayIndex.interval(0, Math.max(0, length - 1)), NDArrayIndex.all(),
                NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all());
    }

    private static INDArray random(Random rng, DataType type, float scale, long... shape) {
        long n = 1;
        for (long d : shape) n *= d;
        float[] values = new float[(int) n];
        for (int i = 0; i < values.length; i++) values[i] = (rng.nextFloat() * 2f - 1f) * scale;
        return Nd4j.create(values, shape, 'c').castTo(type);
    }

    private static void assertBits(INDArray expected, INDArray actual, String message) {
        assertArrayEquals(expected.shape(), actual.shape(), message + " shape");
        assertEquals(expected.dataType(), actual.dataType(), message + " dtype");
        INDArray e = expected.castTo(DataType.FLOAT).dup('c');
        INDArray a = actual.castTo(DataType.FLOAT).dup('c');
        float[] ev = e.data().asFloat();
        float[] av = a.data().asFloat();
        for (int i = 0; i < ev.length; i++) {
            assertEquals(Float.floatToRawIntBits(ev[i]), Float.floatToRawIntBits(av[i]),
                    message + " at " + i + ": expected " + ev[i] + " got " + av[i]);
        }
    }
}
