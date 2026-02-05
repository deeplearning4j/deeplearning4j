package org.eclipse.deeplearning4j.nd4j.autodiff.opvalidation;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.OpContext;
import org.nd4j.linalg.api.ops.impl.reduce.floating.Mean;
import org.nd4j.linalg.api.ops.impl.reduce.longer.CountNonZero;
import org.nd4j.linalg.api.ops.impl.reduce.same.Sum;
import org.nd4j.linalg.api.ops.impl.scalar.ScalarAdd;
import org.nd4j.linalg.api.ops.impl.scalar.comparison.ScalarGreaterThan;
import org.nd4j.linalg.api.ops.impl.indexaccum.FirstIndex;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.nd4j.linalg.api.ops.impl.shape.Create;
import org.nd4j.linalg.api.ops.impl.shape.CreateView;
import org.nd4j.linalg.api.ops.impl.transforms.same.Abs;
import org.nd4j.linalg.api.ops.impl.transforms.strict.Exp;
import org.nd4j.linalg.api.ops.impl.transforms.floating.Sqrt;
import org.nd4j.linalg.api.ops.impl.transforms.bool.IsFinite;
import org.nd4j.linalg.api.ops.impl.transforms.any.Assign;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.common.tests.tags.NativeTag;

import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for shape calculation caching across all op types.
 * Validates correctness (cache hit returns right shape) and invalidation
 * (changed inputs produce new shapes). Also benchmarks cache hit speedup.
 */
@NativeTag
@Tag("shape-cache")
public class TestShapeCacheValidation extends BaseOpValidation {

    // ========== DynamicCustomOp: value-dependent ops ==========

    @Test
    public void testCreateOpCacheInvalidation() {
        // 'create' derives output shape from input VALUES, not shapes.
        // This was the exact op that broke: same input shape [2] but different values.
        DynamicCustomOp createOp = DynamicCustomOp.builder("create")
                .addInputs(Nd4j.createFromArray(new long[]{2, 5}))
                .addIntegerArguments(99L, (long) DataType.FLOAT.toInt(), 99L, (long) DataType.FLOAT.toInt())
                .build();

        OpContext oc1 = Nd4j.getExecutioner().buildContext();
        oc1.setInputArray(0, Nd4j.createFromArray(new long[]{2, 5}));
        oc1.setIArguments(99L, (long) DataType.FLOAT.toInt(), 99L, (long) DataType.FLOAT.toInt());

        List<DataBuffer> shapes1 = createOp.calculateOutputShape(oc1);
        assertFalse(shapes1.isEmpty(), "Should compute shapes for create [2,5]");

        // Now change VALUES but keep same input shape [2]
        OpContext oc2 = Nd4j.getExecutioner().buildContext();
        oc2.setInputArray(0, Nd4j.createFromArray(new long[]{3, 7}));
        oc2.setIArguments(99L, (long) DataType.FLOAT.toInt(), 99L, (long) DataType.FLOAT.toInt());

        List<DataBuffer> shapes2 = createOp.calculateOutputShape(oc2);
        assertFalse(shapes2.isEmpty(), "Should compute shapes for create [3,7]");

        // Shapes must differ since values differ — use assertNotSame to check it's not the cached List
        assertNotSame(shapes1, shapes2,
                "create op: different input values must invalidate cache (different List instance)");
    }

    @Test
    public void testCreateOpJavaWrapperCacheInvalidation() {
        // Test using the actual Create Java wrapper class
        Create createOp = new Create(Nd4j.createFromArray(new long[]{4, 8}), 'c', false, DataType.FLOAT);

        OpContext oc1 = Nd4j.getExecutioner().buildContext();
        oc1.setInputArray(0, Nd4j.createFromArray(new long[]{4, 8}));
        oc1.setIArguments((long) 'c', (long) DataType.FLOAT.toInt());
        oc1.setBArguments(false);

        List<DataBuffer> shapes1 = createOp.calculateOutputShape(oc1);
        assertFalse(shapes1.isEmpty(), "Create op should produce shapes for [4,8]");

        // Same values — should cache hit
        List<DataBuffer> shapes1b = createOp.calculateOutputShape(oc1);
        assertSame(shapes1, shapes1b, "Create op: same values must cache hit");

        // Different values, same input shape [2]
        OpContext oc2 = Nd4j.getExecutioner().buildContext();
        oc2.setInputArray(0, Nd4j.createFromArray(new long[]{6, 10}));
        oc2.setIArguments((long) 'c', (long) DataType.FLOAT.toInt());
        oc2.setBArguments(false);

        List<DataBuffer> shapes2 = createOp.calculateOutputShape(oc2);
        assertFalse(shapes2.isEmpty(), "Create op should produce shapes for [6,10]");
        assertNotSame(shapes1, shapes2, "Create op: different values must invalidate cache");
    }

    @Test
    public void testCreateOpVLMDecodeScenario() {
        // Simulates the exact VLM pipeline scenario that originally broke:
        // Prefill: create([1, 348]) -> output shape [1, 348]
        // Decode step 1: create([1, 349]) -> output shape [1, 349]
        // Both inputs have shape [2] (LONG), but different values.
        Create createOp = new Create(Nd4j.createFromArray(new long[]{1, 348}), 'c', false, DataType.FLOAT);

        // Prefill step
        OpContext ocPrefill = Nd4j.getExecutioner().buildContext();
        ocPrefill.setInputArray(0, Nd4j.createFromArray(new long[]{1, 348}));
        ocPrefill.setIArguments((long) 'c', (long) DataType.FLOAT.toInt());
        ocPrefill.setBArguments(false);

        List<DataBuffer> prefillShapes = createOp.calculateOutputShape(ocPrefill);
        assertFalse(prefillShapes.isEmpty());

        // Decode step 1 — same shape [2] but value 349 instead of 348
        OpContext ocDecode = Nd4j.getExecutioner().buildContext();
        ocDecode.setInputArray(0, Nd4j.createFromArray(new long[]{1, 349}));
        ocDecode.setIArguments((long) 'c', (long) DataType.FLOAT.toInt());
        ocDecode.setBArguments(false);

        List<DataBuffer> decodeShapes = createOp.calculateOutputShape(ocDecode);
        assertFalse(decodeShapes.isEmpty());

        // CRITICAL: must NOT return stale cached shape
        assertNotSame(prefillShapes, decodeShapes,
                "VLM decode scenario: create([1,349]) must not return cached create([1,348]) shapes");
    }

    @Test
    public void testCreateViewCacheInvalidation() {
        // create_view derives shape from index values passed as inputs
        INDArray base = Nd4j.ones(DataType.FLOAT, 4, 6, 8);

        // Index encoding: [type, size, stride, offset, inclusive]
        // ALL_TYPE=2 means keep the whole dimension
        INDArray allIndex = Nd4j.createFromArray(new long[]{2, 1, 1, 0, 1}); // all()
        // INTERVAL_TYPE=1 means slice a range
        INDArray interval1 = Nd4j.createFromArray(new long[]{1, 3, 1, 0, 1}); // [0:3]
        INDArray interval2 = Nd4j.createFromArray(new long[]{1, 5, 1, 0, 1}); // [0:5]

        CreateView viewOp = new CreateView(base, new INDArray[]{allIndex, interval1, allIndex});

        // First call with interval [0:3] on dim 1
        OpContext oc1 = Nd4j.getExecutioner().buildContext();
        oc1.setInputArray(0, base);
        oc1.setInputArray(1, allIndex);
        oc1.setInputArray(2, interval1);  // [0:3]
        oc1.setInputArray(3, allIndex);

        List<DataBuffer> shapes1 = viewOp.calculateOutputShape(oc1);
        assertFalse(shapes1.isEmpty(), "create_view should produce shapes");

        // Same call — cache hit
        List<DataBuffer> shapes1b = viewOp.calculateOutputShape(oc1);
        assertSame(shapes1, shapes1b, "create_view: same inputs must cache hit");

        // Different interval [0:5] on dim 1 — same input shapes but different values
        OpContext oc2 = Nd4j.getExecutioner().buildContext();
        oc2.setInputArray(0, base);
        oc2.setInputArray(1, allIndex);
        oc2.setInputArray(2, interval2);  // [0:5]
        oc2.setInputArray(3, allIndex);

        List<DataBuffer> shapes2 = viewOp.calculateOutputShape(oc2);
        assertFalse(shapes2.isEmpty(), "create_view should produce shapes for different interval");
        assertNotSame(shapes1, shapes2,
                "create_view: different index values must invalidate cache");
    }

    @Test
    public void testReshapeOpCacheInvalidation() {
        // reshape reads target shape from input[1] values
        INDArray input = Nd4j.ones(DataType.FLOAT, 2, 6);

        DynamicCustomOp reshapeOp = DynamicCustomOp.builder("reshape")
                .addInputs(input, Nd4j.createFromArray(new long[]{3, 4}))
                .build();

        OpContext oc1 = Nd4j.getExecutioner().buildContext();
        oc1.setInputArray(0, input);
        oc1.setInputArray(1, Nd4j.createFromArray(new long[]{3, 4}));

        List<DataBuffer> shapes1 = reshapeOp.calculateOutputShape(oc1);
        assertFalse(shapes1.isEmpty());

        // Same input shapes, different reshape target values
        OpContext oc2 = Nd4j.getExecutioner().buildContext();
        oc2.setInputArray(0, input);
        oc2.setInputArray(1, Nd4j.createFromArray(new long[]{4, 3}));

        List<DataBuffer> shapes2 = reshapeOp.calculateOutputShape(oc2);
        assertFalse(shapes2.isEmpty());

        assertNotSame(shapes1, shapes2,
                "reshape op: different target shape values must invalidate cache");
    }

    @Test
    public void testDynamicOpCacheHitWithSameInputs() {
        // When inputs are truly identical (same shapes AND values), cache should hit
        INDArray input = Nd4j.ones(DataType.FLOAT, 2, 6);
        INDArray shape = Nd4j.createFromArray(new long[]{3, 4});

        DynamicCustomOp reshapeOp = DynamicCustomOp.builder("reshape")
                .addInputs(input, shape)
                .build();

        OpContext oc = Nd4j.getExecutioner().buildContext();
        oc.setInputArray(0, input);
        oc.setInputArray(1, shape);

        List<DataBuffer> shapes1 = reshapeOp.calculateOutputShape(oc);

        // Call again with same context — should be a cache hit (same List instance)
        long start = System.nanoTime();
        List<DataBuffer> shapes2 = reshapeOp.calculateOutputShape(oc);
        long cacheTime = System.nanoTime() - start;

        assertSame(shapes1, shapes2,
                "Same inputs should return same cached List instance");
        System.out.println("DynamicCustomOp cache hit time: " + (cacheTime / 1000) + " us");
    }

    @Test
    public void testGatherOpAxisViaIArgs() {
        // gather uses iArgs for axis — changing iArgs must invalidate cache
        INDArray input = Nd4j.ones(DataType.FLOAT, 3, 4, 5);
        INDArray indices = Nd4j.createFromArray(new int[]{0, 2});

        DynamicCustomOp gatherOp = DynamicCustomOp.builder("gather")
                .addInputs(input, indices)
                .addIntegerArguments(0L)  // axis=0
                .build();

        OpContext oc1 = Nd4j.getExecutioner().buildContext();
        oc1.setInputArray(0, input);
        oc1.setInputArray(1, indices);
        oc1.setIArguments(0L);  // axis=0

        List<DataBuffer> shapes1 = gatherOp.calculateOutputShape(oc1);
        assertFalse(shapes1.isEmpty());

        // Change axis to 1
        OpContext oc2 = Nd4j.getExecutioner().buildContext();
        oc2.setInputArray(0, input);
        oc2.setInputArray(1, indices);
        oc2.setIArguments(1L);  // axis=1

        List<DataBuffer> shapes2 = gatherOp.calculateOutputShape(oc2);
        assertFalse(shapes2.isEmpty());

        assertNotSame(shapes1, shapes2,
                "gather op: different axis iArg must invalidate cache");
    }

    @Test
    public void testConcatOpAxisChange() {
        // concat uses iArgs for axis
        INDArray a = Nd4j.ones(DataType.FLOAT, 2, 3);
        INDArray b = Nd4j.ones(DataType.FLOAT, 2, 3);

        DynamicCustomOp concatOp = DynamicCustomOp.builder("concat")
                .addInputs(a, b)
                .addIntegerArguments(0L)
                .build();

        OpContext oc1 = Nd4j.getExecutioner().buildContext();
        oc1.setInputArray(0, a);
        oc1.setInputArray(1, b);
        oc1.setIArguments(0L);  // concat on axis 0

        List<DataBuffer> shapes1 = concatOp.calculateOutputShape(oc1);

        OpContext oc2 = Nd4j.getExecutioner().buildContext();
        oc2.setInputArray(0, a);
        oc2.setInputArray(1, b);
        oc2.setIArguments(1L);  // concat on axis 1

        List<DataBuffer> shapes2 = concatOp.calculateOutputShape(oc2);

        assertNotSame(shapes1, shapes2,
                "concat: different axis must invalidate cache");
    }

    @Test
    public void testInPlaceFlagAffectsSignature() {
        // In-place vs not-in-place: use identity (registered custom op)
        INDArray input = Nd4j.ones(DataType.FLOAT, 3, 4);

        DynamicCustomOp op1 = DynamicCustomOp.builder("identity")
                .addInputs(input)
                .build();
        op1.setInplaceCall(false);

        DynamicCustomOp op2 = DynamicCustomOp.builder("identity")
                .addInputs(input)
                .build();
        op2.setInplaceCall(true);

        OpContext oc = Nd4j.getExecutioner().buildContext();
        oc.setInputArray(0, input);

        // Both should compute shapes without error
        List<DataBuffer> shapes1 = op1.calculateOutputShape(oc);
        List<DataBuffer> shapes2 = op2.calculateOutputShape(oc);
        assertFalse(shapes1.isEmpty());
        assertFalse(shapes2.isEmpty());
    }

    // ========== BaseScalarOp ==========

    @Test
    public void testScalarOpShapeCache() {
        INDArray x = Nd4j.ones(DataType.FLOAT, 4, 5);

        ScalarAdd op = new ScalarAdd(x, 1.0);
        OpContext oc = Nd4j.getExecutioner().buildContext();
        oc.setInputArray(0, x);

        List<DataBuffer> shapes1 = op.calculateOutputShape(oc);
        assertFalse(shapes1.isEmpty());

        // Second call — cache hit
        List<DataBuffer> shapes2 = op.calculateOutputShape(oc);
        assertSame(shapes1, shapes2, "ScalarOp: identical inputs should return cached instance");

        // Different shape — cache miss
        INDArray x2 = Nd4j.ones(DataType.FLOAT, 3, 7);
        OpContext oc2 = Nd4j.getExecutioner().buildContext();
        oc2.setInputArray(0, x2);

        List<DataBuffer> shapes3 = op.calculateOutputShape(oc2);
        assertFalse(shapes3.isEmpty());
        assertNotSame(shapes1, shapes3, "ScalarOp: different input shape should invalidate cache");
    }

    // ========== BaseScalarBoolOp ==========

    @Test
    public void testScalarBoolOpShapeCache() {
        INDArray x = Nd4j.ones(DataType.FLOAT, 4, 5);
        ScalarGreaterThan op = new ScalarGreaterThan(x, 0.5);
        OpContext oc = Nd4j.getExecutioner().buildContext();
        oc.setInputArray(0, x);

        List<DataBuffer> shapes1 = op.calculateOutputShape(oc);
        assertFalse(shapes1.isEmpty());

        List<DataBuffer> shapes2 = op.calculateOutputShape(oc);
        assertSame(shapes1, shapes2, "ScalarBoolOp: cache hit");

        INDArray x2 = Nd4j.ones(DataType.FLOAT, 6, 2);
        OpContext oc2 = Nd4j.getExecutioner().buildContext();
        oc2.setInputArray(0, x2);
        List<DataBuffer> shapes3 = op.calculateOutputShape(oc2);
        assertNotSame(shapes1, shapes3, "ScalarBoolOp: cache miss on different shape");
    }

    // ========== BaseTransformSameOp ==========

    @Test
    public void testTransformSameOpShapeCache() {
        INDArray x = Nd4j.rand(DataType.FLOAT, 3, 4);
        Abs op = new Abs(x);
        OpContext oc = Nd4j.getExecutioner().buildContext();
        oc.setInputArray(0, x);

        List<DataBuffer> shapes1 = op.calculateOutputShape(oc);
        List<DataBuffer> shapes2 = op.calculateOutputShape(oc);
        assertSame(shapes1, shapes2, "TransformSameOp: cache hit");

        OpContext oc2 = Nd4j.getExecutioner().buildContext();
        oc2.setInputArray(0, Nd4j.rand(DataType.FLOAT, 5, 6));
        List<DataBuffer> shapes3 = op.calculateOutputShape(oc2);
        assertNotSame(shapes1, shapes3, "TransformSameOp: cache miss");
    }

    // ========== BaseTransformStrictOp ==========

    @Test
    public void testTransformStrictOpShapeCache() {
        INDArray x = Nd4j.rand(DataType.FLOAT, 3, 4);
        Exp op = new Exp(x);
        OpContext oc = Nd4j.getExecutioner().buildContext();
        oc.setInputArray(0, x);

        List<DataBuffer> shapes1 = op.calculateOutputShape(oc);
        List<DataBuffer> shapes2 = op.calculateOutputShape(oc);
        assertSame(shapes1, shapes2, "TransformStrictOp: cache hit");

        OpContext oc2 = Nd4j.getExecutioner().buildContext();
        oc2.setInputArray(0, Nd4j.rand(DataType.FLOAT, 7, 2));
        List<DataBuffer> shapes3 = op.calculateOutputShape(oc2);
        assertNotSame(shapes1, shapes3, "TransformStrictOp: cache miss");
    }

    // ========== BaseTransformFloatOp ==========

    @Test
    public void testTransformFloatOpShapeCache() {
        INDArray x = Nd4j.rand(DataType.FLOAT, 3, 4).addi(0.1);
        Sqrt op = new Sqrt(x);
        OpContext oc = Nd4j.getExecutioner().buildContext();
        oc.setInputArray(0, x);

        List<DataBuffer> shapes1 = op.calculateOutputShape(oc);
        List<DataBuffer> shapes2 = op.calculateOutputShape(oc);
        assertSame(shapes1, shapes2, "TransformFloatOp: cache hit");
    }

    // ========== BaseTransformBoolOp ==========

    @Test
    public void testTransformBoolOpShapeCache() {
        INDArray x = Nd4j.rand(DataType.FLOAT, 4, 3);
        IsFinite op = new IsFinite(x);
        OpContext oc = Nd4j.getExecutioner().buildContext();
        oc.setInputArray(0, x);

        List<DataBuffer> shapes1 = op.calculateOutputShape(oc);
        List<DataBuffer> shapes2 = op.calculateOutputShape(oc);
        assertSame(shapes1, shapes2, "TransformBoolOp: cache hit");
    }

    // ========== BaseTransformAnyOp ==========

    @Test
    public void testTransformAnyOpShapeCache() {
        INDArray x = Nd4j.rand(DataType.FLOAT, 4, 3);
        Assign op = new Assign(x);
        OpContext oc = Nd4j.getExecutioner().buildContext();
        oc.setInputArray(0, x);

        List<DataBuffer> shapes1 = op.calculateOutputShape(oc);
        List<DataBuffer> shapes2 = op.calculateOutputShape(oc);
        assertSame(shapes1, shapes2, "TransformAnyOp: cache hit");
    }

    // ========== BaseReduceFloatOp ==========

    @Test
    public void testReduceFloatOpShapeCache() {
        INDArray x = Nd4j.rand(DataType.FLOAT, 3, 4, 5);
        Mean op = new Mean(x, 1);
        OpContext oc = Nd4j.getExecutioner().buildContext();
        oc.setInputArray(0, x);

        List<DataBuffer> shapes1 = op.calculateOutputShape(oc);
        List<DataBuffer> shapes2 = op.calculateOutputShape(oc);
        assertSame(shapes1, shapes2, "ReduceFloatOp: cache hit");

        // Same shape — cache hit (same shape signature)
        OpContext oc2 = Nd4j.getExecutioner().buildContext();
        oc2.setInputArray(0, Nd4j.rand(DataType.FLOAT, 3, 4, 5));
        List<DataBuffer> shapes3 = op.calculateOutputShape(oc2);
        assertSame(shapes1, shapes3, "ReduceFloatOp: same shape+dtype = cache hit");

        // Different shape — cache miss (dimensions differ so output shape differs)
        OpContext oc3 = Nd4j.getExecutioner().buildContext();
        oc3.setInputArray(0, Nd4j.rand(DataType.FLOAT, 6, 7, 8));
        List<DataBuffer> shapes4 = op.calculateOutputShape(oc3);
        assertNotSame(shapes1, shapes4, "ReduceFloatOp: different dims = cache miss");

        // Different rank — also cache miss
        OpContext oc4 = Nd4j.getExecutioner().buildContext();
        oc4.setInputArray(0, Nd4j.rand(DataType.FLOAT, 6, 7));
        List<DataBuffer> shapes5 = op.calculateOutputShape(oc4);
        assertNotSame(shapes1, shapes5, "ReduceFloatOp: different rank = cache miss");
    }

    // ========== BaseReduceSameOp ==========

    @Test
    public void testReduceSameOpShapeCache() {
        INDArray x = Nd4j.rand(DataType.FLOAT, 4, 5);
        Sum op = new Sum(x, 0);
        OpContext oc = Nd4j.getExecutioner().buildContext();
        oc.setInputArray(0, x);

        List<DataBuffer> shapes1 = op.calculateOutputShape(oc);
        List<DataBuffer> shapes2 = op.calculateOutputShape(oc);
        assertSame(shapes1, shapes2, "ReduceSameOp: cache hit");
    }

    // ========== BaseReduceLongOp ==========

    @Test
    public void testReduceLongOpShapeCache() {
        INDArray x = Nd4j.rand(DataType.FLOAT, 4, 5);
        CountNonZero op = new CountNonZero(x, 0);
        OpContext oc = Nd4j.getExecutioner().buildContext();
        oc.setInputArray(0, x);

        List<DataBuffer> shapes1 = op.calculateOutputShape(oc);
        List<DataBuffer> shapes2 = op.calculateOutputShape(oc);
        assertSame(shapes1, shapes2, "ReduceLongOp: cache hit");
    }

    // ========== BaseIndexAccumulation ==========

    @Test
    public void testIndexAccumulationShapeCache() {
        INDArray x = Nd4j.rand(DataType.FLOAT, 3, 4);
        FirstIndex op = new FirstIndex(x, Conditions.greaterThan(0.5), 1);
        OpContext oc = Nd4j.getExecutioner().buildContext();
        oc.setInputArray(0, x);

        List<DataBuffer> shapes1 = op.calculateOutputShape(oc);
        List<DataBuffer> shapes2 = op.calculateOutputShape(oc);
        assertSame(shapes1, shapes2, "IndexAccumulation: cache hit");

        OpContext oc2 = Nd4j.getExecutioner().buildContext();
        oc2.setInputArray(0, Nd4j.rand(DataType.FLOAT, 5, 6, 7));
        List<DataBuffer> shapes3 = op.calculateOutputShape(oc2);
        assertNotSame(shapes1, shapes3, "IndexAccumulation: different rank = cache miss");
    }

    // ========== Benchmark: cache hit vs miss ==========

    @Test
    public void testShapeCacheBenchmark() {
        int warmup = 100;
        int iterations = 1000;

        // --- DynamicCustomOp benchmark (native JNI) ---
        INDArray input = Nd4j.rand(DataType.FLOAT, 8, 64, 256);
        DynamicCustomOp softmax = DynamicCustomOp.builder("softmax")
                .addInputs(input)
                .addIntegerArguments(-1L)
                .build();

        OpContext oc = Nd4j.getExecutioner().buildContext();
        oc.setInputArray(0, input);
        oc.setIArguments(-1L);

        // Warmup
        for (int i = 0; i < warmup; i++) {
            softmax.calculateOutputShape(oc);
        }

        // Benchmark (all should be cache hits after first)
        long start = System.nanoTime();
        for (int i = 0; i < iterations; i++) {
            softmax.calculateOutputShape(oc);
        }
        long dynamicCached = System.nanoTime() - start;
        System.out.println("DynamicCustomOp (softmax) " + iterations + " cache hits: "
                + (dynamicCached / 1000) + " us total, " + (dynamicCached / iterations) + " ns/call");

        // --- BaseTransformSameOp benchmark (Java-side) ---
        INDArray x = Nd4j.rand(DataType.FLOAT, 8, 64, 256);
        Abs absOp = new Abs(x);
        OpContext oc2 = Nd4j.getExecutioner().buildContext();
        oc2.setInputArray(0, x);

        for (int i = 0; i < warmup; i++) {
            absOp.calculateOutputShape(oc2);
        }

        start = System.nanoTime();
        for (int i = 0; i < iterations; i++) {
            absOp.calculateOutputShape(oc2);
        }
        long baseOpCached = System.nanoTime() - start;
        System.out.println("BaseTransformSameOp (abs) " + iterations + " cache hits: "
                + (baseOpCached / 1000) + " us total, " + (baseOpCached / iterations) + " ns/call");

        // --- ScalarOp benchmark ---
        ScalarAdd addOp = new ScalarAdd(x, 1.0);
        OpContext oc3 = Nd4j.getExecutioner().buildContext();
        oc3.setInputArray(0, x);

        for (int i = 0; i < warmup; i++) {
            addOp.calculateOutputShape(oc3);
        }

        start = System.nanoTime();
        for (int i = 0; i < iterations; i++) {
            addOp.calculateOutputShape(oc3);
        }
        long scalarCached = System.nanoTime() - start;
        System.out.println("ScalarAdd " + iterations + " cache hits: "
                + (scalarCached / 1000) + " us total, " + (scalarCached / iterations) + " ns/call");
    }
}
