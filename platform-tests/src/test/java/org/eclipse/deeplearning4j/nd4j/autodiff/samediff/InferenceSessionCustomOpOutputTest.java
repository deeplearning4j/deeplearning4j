package org.eclipse.deeplearning4j.nd4j.autodiff.samediff;

import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Collections;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Regression test for InferenceSession's CustomOp output allocation branch
 * (getAndParameterizeOp / lines ~6000-6025).
 *
 * <p>Two bugs were fixed there:
 * <ol>
 *     <li>{@code isOutput} was computed but a hardcoded {@code false} was passed to
 *         {@code mmgr.allocateFromDescriptor(...)} instead, meaning custom-op outputs that
 *         are session-required results were allocated exactly like a throwaway intermediate.
 *         Every sibling allocation site in the same class passes the computed {@code isOutput}
 *         flag through.</li>
 *     <li>The output shape's embedded dtype was never corrected to the SDVariable's declared
 *         dtype when the two diverged (a documented workaround for
 *         https://github.com/eclipse/deeplearning4j/issues/6872, where many ops have multiple
 *         valid output dtypes and the native shape-calc cannot know which was requested) - the
 *         code read the shape's extras and wrote the exact same value back (a no-op), silently
 *         leaving the allocated output array's actual dtype out of sync with the declared
 *         SDVariable dtype.</li>
 * </ol>
 *
 * <p>This test exercises a {@code DynamicCustomOp} (matmul) that goes through that exact
 * branch, executes the graph repeatedly with different inputs (to force the memory manager to
 * allocate/recycle buffers across executions), and verifies that:
 * <ul>
 *     <li>a previously returned session-output array is not corrupted by a later execution
 *         (isOutput retention), and</li>
 *     <li>a custom op with an explicitly requested non-default output dtype (one_hot) actually
 *         produces an array of that dtype (dtype-extras fix).</li>
 * </ul>
 */
public class InferenceSessionCustomOpOutputTest {

    @Test
    public void testCustomOpOutputNotCorruptedByLaterExecution() {
        SameDiff sd = SameDiff.create();
        SDVariable a = sd.var("a", Nd4j.create(new float[]{1, 2, 3, 4}, new long[]{2, 2}));
        SDVariable b = sd.var("b", Nd4j.create(new float[]{1, 0, 0, 1}, new long[]{2, 2}));
        SDVariable out = sd.mmul("out", a, b);

        Map<String, INDArray> firstResult = sd.output(Collections.emptyMap(), "out");
        // Intentionally NOT duplicated: if the output array is allocated as a throwaway
        // (isOutput=false) instead of a retained session output, a later execution reusing
        // the memory manager's buffer pool could silently corrupt this reference.
        INDArray first = firstResult.get("out");
        float[] firstExpected = {1, 2, 3, 4};
        assertArrayEquals(firstExpected, first.dup().toFloatVector(), 1e-5f);

        // Force a second, differently-shaped/valued execution through the same session's
        // memory manager to try to trigger buffer reuse.
        a.setArray(Nd4j.create(new float[]{5, 6, 7, 8, 9, 10}, new long[]{3, 2}));
        b.setArray(Nd4j.create(new float[]{1, 0, 0, 1}, new long[]{2, 2}));
        Map<String, INDArray> secondResult = sd.output(Collections.emptyMap(), "out");
        INDArray second = secondResult.get("out");
        float[] secondExpected = {5, 6, 7, 8, 9, 10};
        assertArrayEquals(secondExpected, second.dup().toFloatVector(), 1e-5f);

        // The first execution's output must still hold its original values.
        assertArrayEquals(firstExpected, first.toFloatVector(), 1e-5f,
                "First execution's custom-op output was corrupted by a later session execution - " +
                        "indicates the output array was not retained as a session output (isOutput flag).");
    }

    @Test
    public void testCustomOpOutputDtypeMatchesRequestedDtype() {
        // one_hot's native shape-calc always yields FLOAT regardless of the requested output
        // dtype (see comment/issue 6872 in InferenceSession); the SDVariable's declared dtype
        // (as requested here, INT32) must be the dtype actually written into the allocated array.
        SameDiff sd = SameDiff.create();
        INDArray indicesArr = Nd4j.createFromArray(0, 2, 1);
        SDVariable indices = sd.constant("indices", indicesArr);
        SDVariable oneHot = sd.oneHot("oneHot", indices, 3, -1, 1.0, 0.0, DataType.INT32);

        Map<String, INDArray> result = sd.output(Collections.emptyMap(), "oneHot");
        INDArray out = result.get("oneHot");

        assertEquals(DataType.INT32, out.dataType(),
                "Custom op output array dtype must match the SDVariable's declared dtype");
        assertEquals(DataType.INT32, sd.getVariable("oneHot").dataType());
    }
}
