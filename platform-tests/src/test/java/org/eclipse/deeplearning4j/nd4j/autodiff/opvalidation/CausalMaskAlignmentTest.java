package org.eclipse.deeplearning4j.nd4j.autodiff.opvalidation;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.validation.OpValidation;
import org.nd4j.autodiff.validation.TestCase;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;

/**
 * Causal attention with fewer queries than keys is aligned to the bottom-right in every
 * path: query i sits at key position i + (keys - queries) and sees keys j &lt;= that position.
 * The unmasked forward (FlashAttentionHelper), the masked forward (AttentionHelper) and the
 * backward must agree; the backward once used a top-left mask (a single query saw only key
 * 0), so its gradient belonged to a different function than the forward computed.
 */
class CausalMaskAlignmentTest {
    private DataType previousDefault;
    private DataType previousFloating;

    @BeforeEach
    void doubleDefaults() {
        // Gradient checks require DOUBLE defaults.
        previousDefault = Nd4j.dataType();
        previousFloating = Nd4j.defaultFloatingPointType();
        Nd4j.setDefaultDataTypes(DataType.DOUBLE, DataType.DOUBLE);
    }

    @AfterEach
    void restoreDefaults() {
        Nd4j.setDefaultDataTypes(previousDefault, previousFloating);
    }

    @ParameterizedTest(name = "queries={0} keys={1}")
    @CsvSource({"1,3", "2,5", "3,3", "4,2"})
    void forwardPathsMatchBottomRightReference(int queries, int keys) {
        Nd4j.getRandom().setSeed(20260929L + queries * 7L + keys);
        int batch = 2, dim = 4;
        INDArray q = Nd4j.rand(DataType.DOUBLE, batch, queries, dim);
        INDArray k = Nd4j.rand(DataType.DOUBLE, batch, keys, dim);
        INDArray v = Nd4j.rand(DataType.DOUBLE, batch, keys, dim);
        INDArray expected = reference(q, k, v);

        INDArray unmasked = attention(q, k, v, null);
        INDArray masked = attention(q, k, v, Nd4j.ones(DataType.DOUBLE, batch, keys));
        assertClose(expected, unmasked, "unmasked (flash) forward");
        assertClose(expected, masked, "masked (helper) forward");
    }

    @ParameterizedTest(name = "queries={0} keys={1}")
    @CsvSource({"1,3", "2,5", "3,3"})
    void gradientMatchesForward(int queries, int keys) {
        Nd4j.getRandom().setSeed(1234L + queries * 11L + keys);
        SameDiff sd = SameDiff.create();
        SDVariable q = sd.var("q", Nd4j.rand(DataType.DOUBLE, 3, queries, 4));
        SDVariable k = sd.var("k", Nd4j.rand(DataType.DOUBLE, 3, keys, 4));
        SDVariable v = sd.var("v", Nd4j.rand(DataType.DOUBLE, 3, keys, 4));
        SDVariable out = sd.nn.dotProductAttentionV2(q, v, k, null, null, 1.0, 0.0, true, true);
        out.norm1("loss").markAsLoss();
        assertNull(OpValidation.validate(new TestCase(sd).gradientCheck(true)));
        sd.close();
    }

    private static INDArray attention(INDArray q, INDArray k, INDArray v, INDArray keyMask) {
        SameDiff sd = SameDiff.create();
        // Graph-owned copies: closing the graph frees its constants.
        SDVariable sq = sd.constant("q", q.dup()), sk = sd.constant("k", k.dup()), sv = sd.constant("v", v.dup());
        SDVariable mask = keyMask == null ? null : sd.constant("mask", keyMask.dup());
        SDVariable out = sd.nn.dotProductAttentionV2(sq, sv, sk, null, mask, 1.0, 0.0, true, false);
        INDArray result = out.eval().dup();
        sd.close();
        return result;
    }

    /** Host DOUBLE reference: scale 1, bottom-right causal mask, softmax over visible keys. */
    private static INDArray reference(INDArray q, INDArray k, INDArray v) {
        long batch = q.size(0), queries = q.size(1), keys = k.size(1), dim = q.size(2);
        long offset = Math.max(0, keys - queries);
        INDArray out = Nd4j.zeros(DataType.DOUBLE, batch, queries, dim);
        for (long b = 0; b < batch; b++) {
            for (long i = 0; i < queries; i++) {
                long visible = Math.min(keys, i + offset + 1);
                double[] scores = new double[(int) visible];
                double max = Double.NEGATIVE_INFINITY;
                for (long j = 0; j < visible; j++) {
                    double s = 0;
                    for (long d = 0; d < dim; d++) s += q.getDouble(b, i, d) * k.getDouble(b, j, d);
                    scores[(int) j] = s;
                    max = Math.max(max, s);
                }
                double sum = 0;
                for (int j = 0; j < visible; j++) {
                    scores[j] = Math.exp(scores[j] - max);
                    sum += scores[j];
                }
                for (long d = 0; d < dim; d++) {
                    double acc = 0;
                    for (int j = 0; j < visible; j++) acc += scores[j] / sum * v.getDouble(b, j, d);
                    out.putScalar(new long[]{b, i, d}, acc);
                }
            }
        }
        return out;
    }

    private static void assertClose(INDArray expected, INDArray actual, String label) {
        assertEquals(expected.length(), actual.length(), label + " length");
        double[] e = expected.dup('c').data().asDouble();
        double[] a = actual.castTo(DataType.DOUBLE).dup('c').data().asDouble();
        for (int i = 0; i < e.length; i++) {
            assertEquals(e[i], a[i], 1e-9 * Math.max(1.0, Math.abs(e[i])), label + " element " + i);
        }
    }
}
