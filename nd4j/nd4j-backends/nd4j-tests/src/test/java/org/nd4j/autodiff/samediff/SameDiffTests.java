package org.nd4j.autodiff.samediff;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.blas.params.MMulTranspose;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.api.ops.impl.accum.Mean;
import org.nd4j.linalg.api.ops.impl.accum.distances.*;
import org.nd4j.linalg.api.ops.impl.controlflow.While;
import org.nd4j.linalg.api.ops.impl.layers.Linear;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.*;
import org.nd4j.linalg.api.ops.impl.scalar.ScalarDivision;
import org.nd4j.linalg.api.ops.impl.shape.OnesLike;
import org.nd4j.linalg.api.ops.impl.transforms.IsMax;
import org.nd4j.linalg.api.ops.impl.transforms.SoftMaxDerivative;
import org.nd4j.linalg.api.ops.impl.transforms.arithmetic.AddOp;
import org.nd4j.linalg.api.ops.impl.transforms.arithmetic.DivOp;
import org.nd4j.linalg.api.ops.impl.transforms.arithmetic.MulOp;
import org.nd4j.linalg.api.ops.impl.transforms.arithmetic.SubOp;
import org.nd4j.linalg.api.ops.impl.transforms.arithmetic.bp.MulBpOp;
import org.nd4j.linalg.api.ops.impl.transforms.arithmetic.bp.SubBpOp;
import org.nd4j.linalg.api.ops.impl.transforms.comparison.*;
import org.nd4j.linalg.api.ops.impl.transforms.gradient.GradientBackwardsMarker;
import org.nd4j.linalg.api.ops.random.impl.BernoulliDistribution;
import org.nd4j.linalg.checkutil.NDArrayCreationUtil;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.linalg.util.ArrayUtil;
import org.nd4j.nativeblas.NativeOpsHolder;
import org.nd4j.weightinit.impl.OneInitScheme;
import org.nd4j.weightinit.impl.UniformInitScheme;
import org.nd4j.weightinit.impl.ZeroInitScheme;

import java.lang.reflect.Field;
import java.util.*;

import static org.junit.Assert.*;
import static org.junit.Assume.assumeNotNull;
import static org.nd4j.linalg.indexing.NDArrayIndex.*;

/**
 * Created by agibsonccc on 4/11/17.
 */
@Slf4j
public class SameDiffTests {
    private DataBuffer.Type initialType;

    @Before
    public void before() throws Exception {
        Nd4j.create(1);
        initialType = Nd4j.dataType();

        Nd4j.setDataType(DataBuffer.Type.DOUBLE);
        Nd4j.getRandom().setSeed(123);
    }

    @After
    public void after() throws Exception {
        Nd4j.setDataType(initialType);
    }


    @After
    public void tearDown() throws Exception {
        NativeOpsHolder.getInstance().getDeviceNativeOps().enableDebugMode(false);
        NativeOpsHolder.getInstance().getDeviceNativeOps().enableVerboseMode(false);
    }

    public Map<String, INDArray> variablesForInput() {
        INDArray inputs = Nd4j.create(new double[][]{
                {0.52, 1.12, 0.77},
                {0.88, -1.08, 0.15},
                {0.52, 0.06, -1.30},
                {0.74, -2.49, 1.39}
        });

        INDArray labels = Nd4j.create(new double[]{1, 1, 0, 1}).reshape(4, 1);

        INDArray weights = Nd4j.zeros(3, 1);

        Map<String, INDArray> inputMap = new HashMap<>();
        inputMap.put("x", inputs);
        inputMap.put("w", weights);
        inputMap.put("y", labels);
        return inputMap;
    }


    @Test
    public void testScalarMul() {
        int d0 = 2;
        int d1 = 3;
        int d2 = 4;

        int n = d0 * d1 * d2;

        for (char inOrder : new char[]{'c', 'f'}) {
            String msg = "Order: " + inOrder + ", Nd4J order: " + Nd4j.order();

            SameDiff sd = SameDiff.create();

            INDArray inArr = Nd4j.linspace(1, n, n).reshape(inOrder, d0, d1, d2);
            INDArray inMul2Exp = inArr.mul(2);

            SDVariable in = sd.var("in", inArr);
            SDVariable inMul2 = in.mul(2.0);

            sd.exec();

            System.out.println("*** Expected ***");
            System.out.println(inMul2Exp.shapeInfoToString());
            System.out.println(Arrays.toString(inMul2Exp.data().asFloat()));

            System.out.println("*** Actual ***");
            System.out.println(inMul2.getArr().shapeInfoToString());
            System.out.println(Arrays.toString(inMul2.getArr().data().asFloat()));

            assertEquals(msg, inArr, in.getArr());
            assertEquals(msg, inMul2Exp, inMul2.getArr());
        }
    }


    @Test
    public void testAddArgsAndOutput() {
        SameDiff sameDiff = SameDiff.create();
        val varOne = sameDiff.var("one", Nd4j.ones(2));
    }


    @Test
    public void testReductionsBackwards() {

        for (int i = 0; i < 7; i++) {

            SameDiff sd = SameDiff.create();

            int nOut = 4;
            int minibatch = 3;
            SDVariable input = sd.var("in", new long[]{-1, nOut});
            SDVariable label = sd.var("label", new long[]{-1, nOut});

            SDVariable diff = input.sub(label);
            SDVariable sqDiff = diff.mul(diff);
            SDVariable msePerEx = sd.mean("msePerEx", sqDiff, 1);

            SDVariable loss;
            String name;
            switch (i) {
                case 0:
                    loss = sd.mean("loss", msePerEx, 0);
                    name = "mean";
                    break;
                case 1:
                    loss = sd.sum("loss", msePerEx, 0);
                    name = "sum";
                    break;
                case 2:
                    loss = sd.standardDeviation("loss", msePerEx, true, 0);
                    name = "stdev";
                    break;
                case 3:
                    loss = sd.min("loss", msePerEx, 0);
                    name = "min";
                    break;
                case 4:
                    loss = sd.max("loss", msePerEx, 0);
                    name = "max";
                    break;
                case 5:
                    loss = sd.variance("loss", msePerEx, true, 0);
                    name = "variance";
                    break;
                case 6:
                    loss = sd.prod("loss", msePerEx, 0);
                    name = "prod";
                    break;
                default:
                    throw new RuntimeException();
            }


            String msg = "test: " + i + " - " + name;
            log.info("*** Starting test: " + msg);

            INDArray inputArr = Nd4j.rand(minibatch, nOut);
            INDArray labelArr = Nd4j.rand(minibatch, nOut);

            sd.associateArrayWithVariable(inputArr, input);
            sd.associateArrayWithVariable(labelArr, label);

            INDArray result = sd.execAndEndResult();
            assertEquals(1, result.length());

            Pair<Map<SDVariable, DifferentialFunction>, List<DifferentialFunction>> p = sd.execBackwards();
        }
    }

    @Test
    public void testMseBackwards() {

        SameDiff sd = SameDiff.create();

        int nOut = 4;
        int minibatch = 3;
        SDVariable input = sd.var("in", new long[]{-1, nOut});
        SDVariable label = sd.var("label", new long[]{-1, nOut});

        SDVariable diff = input.sub(label);
        SDVariable sqDiff = diff.mul(diff);
        SDVariable msePerEx = sd.mean("msePerEx", sqDiff, 1);
        SDVariable avgMSE = sd.mean("loss", msePerEx, 0);

        INDArray inputArr = Nd4j.rand(minibatch, nOut);
        INDArray labelArr = Nd4j.rand(minibatch, nOut);

        sd.associateArrayWithVariable(inputArr, input);
        sd.associateArrayWithVariable(labelArr, label);

        INDArray result = sd.execAndEndResult();
        assertEquals(1, result.length());

        Pair<Map<SDVariable, DifferentialFunction>, List<DifferentialFunction>> p = sd.execBackwards();
    }

    @Test
    public void testEvalVariable() {
        SameDiff sameDiff = SameDiff.create();
        INDArray ones = Nd4j.ones(4);
        INDArray twos = ones.add(ones);
        SDVariable inputOne = sameDiff.var("inputone", ones);
        SDVariable inputResult = inputOne.add("extravarname", inputOne);
        assertEquals(twos, inputResult.eval());
    }


    @Test
    public void testSum() {
        SameDiff sameDiff = SameDiff.create();
        INDArray arr = Transforms.sigmoid(Nd4j.linspace(1, 4, 4));
        SDVariable x = sameDiff.var("x", arr);
        SDVariable result = sameDiff.sum(x, 1); //[1,4].sum(1) == [1,1]

        sameDiff.exec();

        INDArray exp = Nd4j.scalar(1+2+3+4);
        INDArray resultArr = result.getArr();
        assertEquals(exp, resultArr);
    }


    @Test
    public void testXwPlusB() {
        SameDiff sameDiff = SameDiff.create();
        INDArray input = Nd4j.create(new long[]{2, 2});
        INDArray weights = Nd4j.create(new long[]{2, 2});
        INDArray b = Nd4j.create(new long[]{1, 2});

        SDVariable sdInput = sameDiff.var("input", input);
        SDVariable sdWeights = sameDiff.var("weights", weights);
        SDVariable sdBias = sameDiff.var("bias", b);

        SDVariable res = sameDiff.xwPlusB(sdInput, sdWeights, sdBias);
        sameDiff.exec();

        INDArray out = res.getArr();

        assertArrayEquals(new long[]{2, 2}, res.getShape());

    }

    @Test
    public void testReluLayer() {
        SameDiff sameDiff = SameDiff.create();
        INDArray input = Nd4j.create(new long[]{2, 2});
        INDArray weights = Nd4j.create(new long[]{2, 2});
        INDArray b = Nd4j.create(new long[]{1, 2});

        SDVariable sdInput = sameDiff.var("input", input);
        SDVariable sdWeights = sameDiff.var("weights", weights);
        SDVariable sdBias = sameDiff.var("bias", b);

        SDVariable res = sameDiff.reluLayer(sdInput, sdWeights, sdBias);
        sameDiff.exec();

        INDArray out = res.getArr();

        assertArrayEquals(new long[]{2, 2}, res.getShape());

    }


    @Test
    public void testBiasAdd() {
        SameDiff sameDiff = SameDiff.create();
        INDArray ia = Nd4j.create(new long[]{2, 2});
        INDArray b = Nd4j.create(new long[]{1, 2});

        SDVariable input = sameDiff.var("input", ia);
        SDVariable bias = sameDiff.var("bias", b);

        SDVariable res = sameDiff.biasAdd(input, bias);
        assertArrayEquals(new long[]{2, 2}, res.getShape());

    }

    @Test
    public void testSoftmaxXentWithLogits() {

        SameDiff sameDiff = SameDiff.create();
        INDArray logits = Nd4j.create(new long[]{1, 1});
        INDArray weights = Nd4j.create(new long[]{1, 1});
        INDArray labels = Nd4j.create(new long[]{1, 1});

        SDVariable sdLogits = sameDiff.var("logits", logits);
        SDVariable sdWeights = sameDiff.var("weights", weights);
        SDVariable sdLabels = sameDiff.var("labels", labels);

        int mode = 0;
        double labelSmoothing = 0.0;

        SDVariable res = sameDiff.softmaxCrossEntropyWithLogits(sdLogits, sdWeights, sdLabels, mode, labelSmoothing);
        sameDiff.exec();

        INDArray resultArray = res.getArr();
        assertArrayEquals(new long[]{1, 1}, res.getShape());
    }

    @Test
    public void testWeightedXentWithLogits() {
        SameDiff sameDiff = SameDiff.create();
        INDArray targets = Nd4j.create(new long[]{1, 5});
        INDArray inputs = Nd4j.create(new long[]{1, 5});
        INDArray weights = Nd4j.create(new long[]{1, 5});

        SDVariable sdInputs = sameDiff.var("inputs", inputs);
        SDVariable sdWeights = sameDiff.var("weights", weights);
        SDVariable sdTargets = sameDiff.var("targets", targets);

        SDVariable res = sameDiff.weightedCrossEntropyWithLogits(sdTargets, sdInputs, sdWeights);
        sameDiff.exec();

        INDArray resultArray = res.getArr();
        assertArrayEquals(new long[]{1, 5}, res.getShape());
    }

    @Test
    public void testSigmoidXentWithLogits() {
        SameDiff sameDiff = SameDiff.create();
        INDArray logits = Nd4j.create(new long[]{1, 5});
        INDArray weights = Nd4j.create(new long[]{1, 5});
        INDArray labels = Nd4j.create(new long[]{1, 5});

        SDVariable sdLogits = sameDiff.var("logits", logits);
        SDVariable sdWeights = sameDiff.var("weights", weights);
        SDVariable sdLabels = sameDiff.var("labels", labels);

        int mode = 0;
        double labelSmoothing = 0.0;

        SDVariable res = sameDiff.sigmoidCrossEntropyWithLogits(sdLogits, sdWeights, sdLabels, mode, labelSmoothing);
        sameDiff.exec();

        INDArray resultArray = res.getArr();
        assertArrayEquals(new long[]{1, 5}, res.getShape());

    }

    @Test
    public void testDropout() {
        SameDiff sd = SameDiff.create();
        double p = 0.5;
        INDArray ia = Nd4j.create(new long[]{2, 2});

        SDVariable input = sd.var("input", ia);

        SDVariable res = sd.dropout(input, p);
        assertArrayEquals(new long[]{2, 2}, res.getShape());
    }

    @Test
    public void testMseForward() {

        SameDiff sd = SameDiff.create();

        int nOut = 4;
        int minibatch = 3;
        SDVariable input = sd.var("in", new long[]{-1, nOut});
        SDVariable label = sd.var("label", new long[]{-1, nOut});

        SDVariable diff = input.sub(label);
        SDVariable sqDiff = diff.mul(diff);
        SDVariable msePerEx = sd.mean("msePerEx", sqDiff, 1);
        SDVariable score = sd.mean("score", msePerEx);

        INDArray inputArr = Nd4j.rand(minibatch, nOut);
        INDArray labelArr = Nd4j.rand(minibatch, nOut);

        sd.associateArrayWithVariable(inputArr, input);
        sd.associateArrayWithVariable(labelArr, label);

        INDArray result = sd.execAndEndResult();
        assertNotNull(result);                          //*** Fails Here - Null output ***
        assertEquals(1, result.length());
    }


    @Test
    public void testReshape() {
        SameDiff sameDiff = SameDiff.create();
        INDArray arr = Transforms.sigmoid(Nd4j.linspace(1, 4, 4)).reshape(2, 2);
        SDVariable x = sameDiff.var("x", arr);
        SDVariable result1 = sameDiff.reshape(x, 2, 2);
        assertArrayEquals(new long[]{2, 2}, result1.eval().shape());
        INDArray arr_shape = Nd4j.create(new double[]{2, 2}, new int[]{2});
        SDVariable shape = sameDiff.var("shape", arr_shape);
        SDVariable result2 = sameDiff.reshape(x, shape);
        assertArrayEquals(new long[]{2, 2}, result2.eval().shape());

    }

    @Test
    public void testTranspose() {
        SameDiff sameDiff = SameDiff.create();
        INDArray arr = Transforms.sigmoid(Nd4j.linspace(1, 4, 4));
        SDVariable x = sameDiff.var("x", arr);
        SDVariable result = sameDiff.transpose(x);
        sameDiff.exec();
        assertArrayEquals(new long[]{4, 1}, result.getArr().shape());

    }

    @Test

    public void testShape() {
        SameDiff sameDiff = SameDiff.create();
        val shape = new long[]{2, 3};
        SDVariable x = sameDiff.var("x", shape);
        SDVariable result = sameDiff.shape(x);
        assertArrayEquals(result.eval().toLongVector(), shape);
}
    @Test
    public void testGather() {
        SameDiff sameDiff = SameDiff.create();
        INDArray arr = Nd4j.create(new float[]{1, 2, 3, 4}, new long[]{2, 2});
        SDVariable x = sameDiff.var("x", arr);
        SDVariable result = sameDiff.gather(x, new int[]{1, 0}, 1);
        INDArray expected = Nd4j.create(new float[]{2, 1, 4, 3}, new long[]{2, 2});
        assertEquals(expected, result.eval());

    }

    @Test
    public void testGatherNd() {
        SameDiff sameDiff = SameDiff.create();
        INDArray arr1 = Transforms.sigmoid(Nd4j.linspace(1, 24, 24)).reshape(2, 3, 4);
        INDArray arr2 = Nd4j.create(new float[]{1, 2, 3, 0, 1, 3, 1, 0, 2}, new long[]{3, 3});
        SDVariable x = sameDiff.var("x", arr1);
        SDVariable idxs = sameDiff.var("idxs", arr2);
        SDVariable result = sameDiff.gatherNd(x, idxs);
        // build expected output array
        INDArray expected  = Nd4j.zeros(3);
        for (int i=0; i<3; i++){
            INDArray idx = arr2.get(NDArrayIndex.point(i));
            expected.get(NDArrayIndex.point(i)).assign(
                    arr1.get(NDArrayIndex.point(idx.getInt(0)),
                            NDArrayIndex.point(idx.getInt(1)),
                            NDArrayIndex.point(idx.getInt(2))));
            }
        assertEquals(expected, result.eval());
        }

    @Test
    public void testStack() {
        SameDiff sameDiff = SameDiff.create();
        INDArray arr1 = Transforms.sigmoid(Nd4j.linspace(1, 6, 6)).reshape(3, 2);
        INDArray arr2 = Transforms.sigmoid(Nd4j.linspace(7, 12, 6)).reshape(3, 2);
        SDVariable x1 = sameDiff.var("x1", arr1);
        SDVariable x2 = sameDiff.var("x2", arr2);
        SDVariable result = sameDiff.stack(1, x1, x2);
        assertArrayEquals(new long[]{3, 2, 2}, result.eval().shape());
    }

    @Test
    public void testParallelStack() {
        SameDiff sameDiff = SameDiff.create();
        INDArray arr1 = Transforms.sigmoid(Nd4j.linspace(1, 6, 6)).reshape(3, 2);
        INDArray arr2 = Transforms.sigmoid(Nd4j.linspace(7, 12, 6)).reshape(3, 2);
        SDVariable x1 = sameDiff.var("x1", arr1);
        SDVariable x2 = sameDiff.var("x2", arr2);
        SDVariable result = sameDiff.parallel_stack(new SDVariable[]{x1, x2});
        assertArrayEquals(new long[]{2, 3, 2}, result.eval().shape());
        assertEquals(Nd4j.concat(0, arr1, arr2).reshape(2, 3, 2), result.eval());
    }

    @Test
    public void testUnStack() {
        Nd4j.getExecutioner().enableDebugMode(true);
        Nd4j.getExecutioner().enableVerboseMode(true);
        SameDiff sameDiff = SameDiff.create();
        INDArray arr1 = Nd4j.zeros(3, 2);
        INDArray arr2 = Nd4j.ones(3, 2);
        SDVariable x1 = sameDiff.var("x1", arr1);
        SDVariable x2 = sameDiff.var("x2", arr2);
        SDVariable stacked = sameDiff.stack(0, x1, x2);
        SDVariable[] result = sameDiff.unstack(stacked, 0, 2);
        assertEquals(arr1, result[0].eval());
        assertEquals(arr2, result[1].eval());
    }

    @Test
    public void testPermute() {
        SameDiff sameDiff = SameDiff.create();
        INDArray arr = Transforms.sigmoid(Nd4j.linspace(1, 6, 6).reshape(2, 3));
        SDVariable x = sameDiff.var("x", arr);
        SDVariable result = sameDiff.permute(x, 1, 0);
        assertArrayEquals(new long[]{3, 2}, result.getShape());

    }

    @Test
    public void testConcat() {
        SameDiff sameDiff = SameDiff.create();
        INDArray arr1 = Transforms.sigmoid(Nd4j.linspace(1, 4, 4));
        INDArray arr2 = Transforms.sigmoid(Nd4j.linspace(4, 8, 4));
        SDVariable x1 = sameDiff.var("x1", arr1);
        SDVariable x2 = sameDiff.var("x2", arr2);
        SDVariable result = sameDiff.concat(0, x1, x2);
        assertArrayEquals(new long[]{2, 4}, result.eval().shape());

    }

    @Test
    public void testTile() {
        SameDiff sameDiff = SameDiff.create();
        INDArray arr = Transforms.sigmoid(Nd4j.linspace(1, 4, 4));
        SDVariable x = sameDiff.var("x", arr);
        SDVariable result = sameDiff.tile(x, new int[]{2, 2});
        assertArrayEquals(new long[]{2, 8}, result.eval().shape());
        INDArray arr2 = Nd4j.concat(0, arr, arr);  // (1, 4), (1, 4) -> (2, 4)
        INDArray expected = Nd4j.concat(1, arr2, arr2);  // (2, 4), (2, 4) -> (2, 8)
        assertEquals(expected, result.eval());

    }

    @Test
    public void testDistance() {
        SameDiff sameDiff = SameDiff.create();
        INDArray arr = Transforms.sigmoid(Nd4j.linspace(1, 4, 4)).reshape(2, 2);
        SDVariable x = sameDiff.var("x", arr);
        SDVariable y = sameDiff.var("y", arr);
        SDVariable result = sameDiff.cosineSimilarity(x, y, 1);
        SDVariable addResult = result.add(result);
        SDVariable finalReshape = sameDiff.reshape(addResult, 1, 2);
        assertArrayEquals(new long[]{1, 2}, finalReshape.getShape());
    }

    @Test
    public void testReverseSequence() {
        SameDiff sameDiff = SameDiff.create();
        float[] input_data = new float[]{
                1, 2, 3,
                4, 5, 6,
                7, 8, 9,
                0, 0, 0,
                0, 0, 0,

                1, 2, 3,
                4, 5, 6,
                0, 0, 0,
                0, 0, 0,
                0, 0, 0
        };
        float[] expected_output = new float[]{
                7, 8, 9,
                4, 5, 6,
                1, 2, 3,
                0, 0, 0,
                0, 0, 0,

                4, 5, 6,
                1, 2, 3,
                0, 0, 0,
                0, 0, 0,
                0, 0, 0
        };
        INDArray arr1 = Nd4j.create(input_data, new long[]{2, 5, 3});
        INDArray arr2 = Nd4j.create(new float[]{3, 2}).reshape(2);
        SDVariable x = sameDiff.var("x", arr1);
        SDVariable seq_lengths = sameDiff.var("seq_lengths", arr2);
        SDVariable result = sameDiff.reverseSequence(x, seq_lengths, 1, 0);
        INDArray expected = Nd4j.create(expected_output, new long[]{2, 5, 3});
        assertArrayEquals(arr1.shape(), result.eval().shape());
        assertEquals(expected, result.eval());
    }

    @Test
    public void testSequenceMask() {
        SameDiff sameDiff = SameDiff.create();
        INDArray arr = Nd4j.create(new float[] {1, 3, 2}).reshape(3);
        SDVariable lengths = sameDiff.var("lengths", arr);

        // Test with static max len
        int maxlen = 5;
        INDArray expected = Nd4j.create(new float[] {1, 0, 0, 0, 0,
                                                     1, 1, 1, 0, 0,
                                                     1, 1, 0, 0, 0},
                                                   new long[]{3, 5});
        SDVariable result1 = sameDiff.sequenceMask(lengths, maxlen);
        assertArrayEquals(expected.shape(), result1.eval().shape());
        assertEquals(expected, result1.eval());

        // Test with dynamic maxlen
        lengths = sameDiff.var("lengths2", arr); // required because of an internal samediff bug
        SDVariable maxLen = sameDiff.var("maxLen", Nd4j.create(new float[]{5}).reshape(1));
        SDVariable result2 = sameDiff.sequenceMask(lengths, maxLen);
        assertArrayEquals(expected.shape(), result2.eval().shape());
        assertEquals(expected, result2.eval());
    }

    @Test
    public void testTensorGradMmul() {
        SameDiff sameDiff = SameDiff.create();
        INDArray arr = Transforms.sigmoid(Nd4j.linspace(1, 4, 4)).reshape(2, 2);
        SDVariable x = sameDiff.var("x", arr);
        SDVariable y = sameDiff.var("y", arr);
        SDVariable result = sameDiff.mmul(x, y);
        SDVariable otherResult = result.add(result);
        assertArrayEquals(new long[]{2, 2}, result.getShape());
    }


    @Test
    public void testEval() {
        SameDiff sameDiff = SameDiff.create();
        INDArray arr = Nd4j.linspace(1, 4, 4);
        SDVariable x = sameDiff.var("x", arr);
        SDVariable sigmoid = sameDiff.sigmoid(x);
        INDArray assertion = Transforms.sigmoid(arr);
        INDArray[] eval = sameDiff.eval(Collections.singletonMap("x", arr));
        assertEquals(assertion, eval[0]);

    }


    @Test
    public void testUpdateVariableName() throws Exception {
        INDArray inArr = Nd4j.create(1, 4);
        SameDiff sd = SameDiff.create();
        SDVariable in = sd.var("in", inArr);
        SDVariable s = sd.tanh("s", in);

        List<SDVariable> l = sd.variables();
        assertEquals(2, l.size());      //Fails here: returns 3 (inc "tanh" variable that should have been replaced)

        for (SDVariable sdv : l) {
            String n = sdv.getVarName();
            assertTrue(n.equals("in") || n.equals("s"));
        }

        Field f = SameDiff.class.getDeclaredField("incomingArgsReverse");
        f.setAccessible(true);
        Map<String, String[]> incomingArgsReverse = (Map<String, String[]>) f.get(sd);

        for (Map.Entry<String, String[]> e : incomingArgsReverse.entrySet()) {
            for (String str : e.getValue()) {
                assertTrue(str, str.equals("in") || str.equals("s"));
            }
        }

        f = SameDiff.class.getDeclaredField("outgoingArgsReverse");      //Also: typo in the SameDiff class field name
        f.setAccessible(true);
        Map<String, String[]> outgoingArgsReverse = (Map<String, String[]>) f.get(sd);
        for (Map.Entry<String, String[]> e : outgoingArgsReverse.entrySet()) {
            for (String str : e.getValue()) {
                assertTrue(str, str.equals("in") || str.equals("s"));  //Also fails here due to "tanh" variable
            }
        }
    }

    @Test
    public void testFunctionInputsAndArgs() {
        SameDiff sameDiff = SameDiff.create();
        SDVariable var = sameDiff.var("one", Nd4j.scalar(1.0));
        SDVariable variable2 = sameDiff.var("two", Nd4j.scalar(1.0));
        val sum = var.add(variable2);
        assertArrayEquals(new long[]{1, 1}, sum.getShape());


    }


    @Test
    public void testCrossSameDiffVariableInitWithAlloc() {
        SameDiff first = SameDiff.create();
        SameDiff second = SameDiff.create();


        SDVariable firstVar = first.var("one", new long[]{2, 2});
        SDVariable secondVar = second.var(firstVar);
        assertTrue(firstVar.getArr() == secondVar.getArr());
        assertEquals(firstVar.getVarName(), secondVar.getVarName());

    }


    @Test
    public void testCrossSameDiffVariableInitWithPlaceHolder() {
        SameDiff first = SameDiff.create();
        SameDiff second = SameDiff.create();


        SDVariable firstVar = first.var("one", new long[]{2, 2});
        SDVariable secondVar = second.var(firstVar);
        assumeNotNull(firstVar.getArr());

        assertTrue(firstVar.getArr() == secondVar.getArr());
        assertEquals(firstVar.getVarName(), secondVar.getVarName());

    }


    @Test
    public void testVariableArrayReference() {
        SameDiff sameDiff = SameDiff.create();
        SDVariable arr = sameDiff.var("one", new long[]{2, 2});
        assertArrayEquals(new long[]{2, 2}, arr.getShape());
        assumeNotNull(arr.getArr());
        assertArrayEquals(new long[]{2, 2}, arr.getArr().shape());

    }

    @Test
    public void testEvalAddSelf() {
        /**
         * Note this test fails yet due to needing
         * to validate simple cases like x * x
         * matching number of inputs.
         */
        SameDiff sameDiff = SameDiff.create();
        INDArray arr = Nd4j.linspace(1, 4, 4);
        SDVariable x = sameDiff.var("x", arr);
        SDVariable sigmoid = x.mul(x);
        INDArray assertion = arr.mul(arr);
        INDArray[] eval = sameDiff.eval(Collections.singletonMap("x", arr));
        assertEquals(assertion, eval[0]);

    }

    @Test
    public void testEvalAdd() {
        SameDiff sameDiff = SameDiff.create();
        INDArray arr = Nd4j.linspace(1, 4, 4);
        INDArray yArr = arr.dup();
        SDVariable x = sameDiff.var("x", arr);
        SDVariable y = sameDiff.var("y", yArr);

        SDVariable sigmoid = x.mul(y);
        INDArray assertion = arr.mul(arr);
        Map<String, INDArray> vars = new HashMap<>();
        vars.put("x", arr);
        vars.put("y", yArr);
        INDArray[] eval = sameDiff.eval(vars);
        assertEquals(assertion, eval[0]);

    }


    @Test
    public void testTensorGradTensorMmul() {
        SameDiff sameDiff = SameDiff.create();
        INDArray arr = Transforms.sigmoid(Nd4j.linspace(1, 8, 8)).reshape(2, 2, 2);
        SDVariable x = sameDiff.var("x", arr);
        SDVariable y = sameDiff.var("y", arr);
        SDVariable result = sameDiff.tensorMmul(x, y, new int[][]{{0}, {1}});
        assertArrayEquals(ArrayUtil.getTensorMmulShape(new long[]{2, 2, 2}, new long[]{2, 2, 2}, new int[][]{{0}, {1}}), result.getShape());
        assertEquals(32, sameDiff.numElements());
    }

    @Test
    public void testDup() {
        SameDiff sameDiff = SameDiff.create();
        INDArray arr = Transforms.sigmoid(Nd4j.linspace(1, 8, 8)).reshape(2, 2, 2);
        SDVariable x = sameDiff.var("x", arr);
        SDVariable y = sameDiff.var("y", arr);
        SameDiff tg2 = sameDiff.dup();
    }


    @Test
    public void testLogGrad() {
        SameDiff sameDiff = SameDiff.create();
        SDVariable input = sameDiff.var("x", Nd4j.linspace(1, 4, 4));
        SDVariable log = sameDiff.log(input);
        SDVariable sum = sameDiff.sum(log, Integer.MAX_VALUE);
        INDArray result = null;
        Pair<Map<SDVariable, DifferentialFunction>, List<DifferentialFunction>> execBackwards = sameDiff.execBackwards();
        System.out.println(execBackwards);
        //INDArray assertion = Nd4j.create(new double[]{1, 0.5, 0.33, 0.25});
        // assertTrue(assertion.equalsWithEps(result, 1e-2));
    }


    @Test
    public void testElementWiseDivAndRDiv() {
        SameDiff sameDiff = SameDiff.create();
        INDArray ones = Nd4j.ones(4);
        INDArray toDivBy = Nd4j.valueArrayOf(4, 0.25);
        Map<String, INDArray> xAndY = new HashMap<>();
        xAndY.put("x", ones);
        xAndY.put("y", toDivBy);
        sameDiff.defineFunction("div", new SameDiff.SameDiffFunctionDefinition() {
            @Override
            public SDVariable[] define(SameDiff sameDiff, Map<String, INDArray> inputs, SDVariable[] variableInputs) {
                SDVariable x = sameDiff.var("x", inputs.get("x"));
                SDVariable y = sameDiff.var("y", inputs.get("y"));
                return new SDVariable[]{x.div(y)};
            }
        }, xAndY);

        sameDiff.defineFunction("rdiv", new SameDiff.SameDiffFunctionDefinition() {
            @Override
            public SDVariable[] define(SameDiff sameDiff, Map<String, INDArray> inputs, SDVariable[] variableInputs) {
                SDVariable x = sameDiff.var("x", inputs.get("x"));
                SDVariable y = sameDiff.var("y", inputs.get("y"));
                return new SDVariable[]{x.rdiv(y)};
            }
        }, xAndY);


        INDArray assertionForDiv = Nd4j.valueArrayOf(4, 4.0);
        INDArray assertionForRDiv = Nd4j.valueArrayOf(4, 0.25);
        assertEquals(assertionForDiv, sameDiff.getFunction("div").execAndEndResult());
        assertEquals(assertionForRDiv, sameDiff.getFunction("rdiv").execAndEndResult());

    }


    @Test
    public void testNegativeGradient() {
        SameDiff sameDiff = SameDiff.create();
        INDArray ones = Nd4j.ones(4);
        Map<String, INDArray> xAndY = new HashMap<>();
        xAndY.put("x", ones);
        sameDiff.defineFunction("neg", new SameDiff.SameDiffFunctionDefinition() {
            @Override
            public SDVariable[] define(SameDiff sameDiff, Map<String, INDArray> inputs, SDVariable[] variableInputs) {
                SDVariable x = sameDiff.var("x", inputs.get("x"));
                return new SDVariable[]{sameDiff.neg(x)};
            }
        }, xAndY);

        INDArray assertionForDiv = Nd4j.valueArrayOf(4, -1);
        assertEquals(assertionForDiv, sameDiff.getFunction("neg").execAndEndResult());

    }


    @Test
    public void testSumOp() {
        SameDiff sameDiff = SameDiff.create();
        INDArray sumInput = Nd4j.linspace(1, 4, 4).reshape(2, 2);
        Map<String, INDArray> inputs = new HashMap<>();
        inputs.put("x", sumInput);
        sameDiff.defineFunction("sum", new SameDiff.SameDiffFunctionDefinition() {
            @Override
            public SDVariable[] define(SameDiff sameDiff, Map<String, INDArray> inputs, SDVariable[] variableInputs) {
                SDVariable input = sameDiff.var("x", inputs.get("x"));
                SDVariable sum = sameDiff.sum(input, 1);
                return new SDVariable[]{sum};
            }
        }, inputs);

        INDArray assertion = sumInput.sum(1);
        INDArray executions = sameDiff.execAndEndResult("sum");
        assertEquals(assertion, executions);
    }


    @Test
    public void testVariableReferenceNoFunction() {
        /**
         * Creating a variable should not create a differential function.
         */
        SameDiff sameDiff = SameDiff.create();
        SDVariable sdVariable = sameDiff.var("one", Nd4j.scalar(1.0));
        assumeNotNull(sameDiff.getVariable(sdVariable.getVarName()));
    }


    @Test
    public void testVariableWithFunction() {
        /**
         * A variable's function should be null
         * when just a variable but
         * have a function result
         * when the variable itself is the result of a function.
         *
         */
        SameDiff sameDiff = SameDiff.create();
        SDVariable sdVariable = sameDiff.var("one", Nd4j.scalar(1.0));
        SDVariable add = sdVariable.add(1.0);
        assertEquals(sameDiff.getVariable(add.getVarName()), add);
    }


    @Test
    public void testUpdateVariable() {
        SameDiff sameDiff = SameDiff.create();
        SDVariable one = sameDiff.one("one", new long[]{1, 1});
        sameDiff.updateVariableName(one.getVarName(), "one-diff");
        assertEquals(one.getArr(), sameDiff.getVariable("one-diff").getArr());
    }


    @Test
    public void testMulGradient() {
        INDArray arr1 = Nd4j.linspace(1, 4, 4).reshape(2, 2);
        INDArray arr2 = Nd4j.linspace(1, 4, 4).reshape(2, 2);

        INDArray gradAssertion = Nd4j.ones(arr1.shape());
        INDArray scalar = Nd4j.scalar(1.0);
        INDArray aGradAssertion = Nd4j.create(new double[][]{
                {1, 4},
                {9, 16}
        });

        INDArray cGradAssertion = Nd4j.create(new double[][]{
                {1, 2},
                {3, 4}
        });

        INDArray wGradAssertion = Nd4j.create(new double[][]{
                {2, 8},
                {18, 32}
        });

        INDArray dGradAssertion = Nd4j.ones(2, 2);

        SameDiff sameDiff = SameDiff.create();

        SDVariable sdVariable = sameDiff.var("a", arr1);
        SDVariable sdVariable1 = sameDiff.var("w", arr2);
        SDVariable varMulPre = sdVariable.mul("c", sdVariable1);
        SDVariable varMul = varMulPre.mul("d", sdVariable1);
        SDVariable sum = sameDiff.sum("ret", varMul, Integer.MAX_VALUE);

        Pair<Map<SDVariable, DifferentialFunction>, List<DifferentialFunction>> mapListPair = sameDiff.execBackwards();

        SDVariable finalResult = sameDiff.grad(sum.getVarName());

        SDVariable cGrad = sameDiff.grad(varMulPre.getVarName());

        SDVariable mulGradResult = sameDiff.grad(varMul.getVarName());
        SDVariable aGrad = sameDiff.grad(sdVariable.getVarName());
        SDVariable wGrad = sameDiff.grad(sdVariable1.getVarName());
        SDVariable dGrad = sameDiff.grad(varMul.getVarName());

        INDArray scalarGradTest = finalResult.getArr();
        assertEquals(scalar, scalarGradTest);


        INDArray gradTest = mulGradResult.getArr();
        assertEquals(gradAssertion, gradTest);

        INDArray aGradTest = aGrad.getArr();
        assertEquals(aGradAssertion, aGradTest);

        INDArray cGradTest = cGrad.getArr();
        assertEquals(cGradAssertion, cGradTest);

        INDArray wGradTest = wGrad.getArr();
        assertEquals(wGradAssertion, wGradTest);

        INDArray dGradTest = dGrad.getArr();
        assertEquals(dGradAssertion, dGradTest);


    }


    @Test(expected = ND4JIllegalStateException.class)
    public void testPlaceHolderWithFullShape() {
        val sd = SameDiff.create();
        val placeholder = sd.var("somevar", new long[]{2, 2});
        sd.addAsPlaceHolder(placeholder.getVarName());
        assertTrue(sd.isPlaceHolder(placeholder.getVarName()));
        sd.resolveVariablesWith(Collections.singletonMap(placeholder.getVarName(), Nd4j.linspace(1, 4, 4)));
    }


    @Test
    public void testLinearModule() {
        int nIn = 5;
        Linear linear = Linear.execBuilder()
                .nIn(nIn)
                .nOut(4)
                .weightInitScheme(new UniformInitScheme('f', nIn))
                .biasWeightInitScheme(new ZeroInitScheme('f'))
                .build();
        linear.exec(Nd4j.linspace(1, 20, 20).reshape(4, 5));
        assertEquals(1, linear.numOutputArguments());

    }


    @Test
    public void testLinearModule2() {
        Linear linear = Linear.execBuilder()
                .nIn(3)
                .nOut(2)
                .weightInitScheme(new OneInitScheme('f'))
                .biasWeightInitScheme(new ZeroInitScheme('f'))
                .build();
        linear.exec(Nd4j.linspace(1, 6, 6).reshape(2, 3));
        INDArray assertion = Nd4j.create(new double[][]{
                {6, 6},
                {15, 15}
        });
        assertEquals(assertion, linear.outputArguments()[0]);

    }


    @Test
    public void testInPlaceAdd() {
        SameDiff sameDiff = SameDiff.create();
        SDVariable toAdd = sameDiff.var("arr1", Nd4j.ones(2, 2));
        SDVariable add = sameDiff.var("arr2", Nd4j.valueArrayOf(2, 2, 2.0));
        SDVariable result = toAdd.addi(add);
        INDArray result2 = sameDiff.execAndEndResult();
        INDArray arr = result.getArr();
        INDArray assertion = Nd4j.ones(2, 2).addi(Nd4j.valueArrayOf(2, 2, 2.0));
        assertEquals(assertion, result2);
    }


    @Test
    public void testDefineFunctionArrayExistence() {
        SameDiff sameDiff = SameDiff.create();
        String testFunctionName = "testfunction";
        SDVariable[] inputVars = new SDVariable[]{
                sameDiff.var("one", new long[]{1, 1}),
                sameDiff.var("two", new long[]{1, 1}),

        };

        SameDiff functionDef = sameDiff.defineFunction(testFunctionName, new SameDiff.SameDiffFunctionDefinition() {
            @Override
            public SDVariable[] define(SameDiff sameDiff, Map<String, INDArray> inputs, SDVariable[] variableInputs) {
                return new SDVariable[]{variableInputs[0].add(variableInputs[1])};
            }
        }, inputVars);


        //1 input plus 2 outputs
        assertEquals(3, functionDef.variables().size());


    }


    @Test(timeout = 10000L)
    public void testWhileLoop() {
        SameDiff sameDiff = SameDiff.create();
        sameDiff.whileStatement(new SameDiff.DefaultSameDiffConditional(), new SameDiff.SameDiffFunctionDefinition() {
            @Override
            public SDVariable[] define(SameDiff sameDiff, Map<String, INDArray> inputs, SDVariable[] variableInputs) {
                SDVariable eqResult = sameDiff.neq(variableInputs[0], variableInputs[1]);
                return new SDVariable[]{eqResult};
            }
        }, new SameDiff.SameDiffFunctionDefinition() {
            @Override
            public SDVariable[] define(SameDiff sameDiff, Map<String, INDArray> inputs, SDVariable[] variableInputs) {
                SDVariable ret = variableInputs[1].addi(1.0);
                return new SDVariable[]{variableInputs[0], ret};
            }
        }, new SDVariable[]{
                sameDiff.one("one", new long[]{1, 1}),
                sameDiff.var("two", new long[]{1, 1}),

        });

        Pair<Map<SDVariable, DifferentialFunction>, List<DifferentialFunction>> exec = sameDiff.exec();
        assertFalse(exec.getRight().isEmpty());
        While function = (While) exec.getRight().get(exec.getRight().size() - 1);
        assumeNotNull(function.getOutputVars());
        assertEquals(1, function.getNumLooped());
        sameDiff.toString();
    }


    @Test
    public void testIfStatementTrueBodyBackwards() {
        SameDiff sameDiff = SameDiff.create();
        SameDiff.SameDiffFunctionDefinition conditionBody = new SameDiff.SameDiffFunctionDefinition() {
            @Override
            public SDVariable[] define(SameDiff sameDiff, Map<String, INDArray> inputs, SDVariable[] variableInputs) {
                SDVariable sum = sameDiff.sum(variableInputs[0], Integer.MAX_VALUE);
                SDVariable result = sameDiff.gt(sum, 1.0);
                return new SDVariable[]{result};
            }
        };


        SameDiff.SameDiffFunctionDefinition trueBody = new SameDiff.SameDiffFunctionDefinition() {
            @Override
            public SDVariable[] define(SameDiff sameDiff, Map<String, INDArray> inputs, SDVariable[] variableInputs) {
                SDVariable add = variableInputs[0].add(1.0);
                return new SDVariable[]{add};
            }
        };

        SameDiff.SameDiffFunctionDefinition falseBody = new SameDiff.SameDiffFunctionDefinition() {
            @Override
            public SDVariable[] define(SameDiff sameDiff, Map<String, INDArray> inputs, SDVariable[] variableInputs) {
                SDVariable sub = variableInputs[0].sub(1.0);
                return new SDVariable[]{sub};
            }
        };

        //true body trigger
        SDVariable[] firstInputs = new SDVariable[]{
                sameDiff.var("one", new long[]{1, 1})

        };


        sameDiff.ifStatement(new SameDiff.DefaultSameDiffConditional(), conditionBody, trueBody, falseBody, firstInputs);
        sameDiff.execBackwards();
        SameDiff grad = sameDiff.getFunction("grad");
       /* If ifBlock = (If) grad.getFunction(new long[]{1},new long[]{2});
        SameDiff assertComparision = SameDiff.create();
        SDVariable initialInput = assertComparision.zero("zero",new long[]{1,1});
        initialInput.addi(1.0);
        assumeNotNull(ifBlock.getTrueBodyExecuted());
        assertTrue(ifBlock.getTrueBodyExecuted());
        assertEquals(Nd4j.scalar(1.00),initialInput.getArr());
        assertEquals(Nd4j.scalar(1.0),ifBlock.getLoopBodyExecution().getVariableForVertexId(2).getArr());
*/
    }


    @Test(timeout = 10000L)
    public void testWhileBackwards() {
        SameDiff sameDiff = SameDiff.create();
        sameDiff.whileStatement(new SameDiff.DefaultSameDiffConditional(), new SameDiff.SameDiffFunctionDefinition() {
            @Override
            public SDVariable[] define(SameDiff sameDiff, Map<String, INDArray> inputs, SDVariable[] variableInputs) {
                SDVariable eqResult = sameDiff.neq(variableInputs[0], variableInputs[1]);
                return new SDVariable[]{eqResult};
            }
        }, new SameDiff.SameDiffFunctionDefinition() {
            @Override
            public SDVariable[] define(SameDiff sameDiff, Map<String, INDArray> inputs, SDVariable[] variableInputs) {
                SDVariable ret = variableInputs[1].addi(1.0);
                return new SDVariable[]{variableInputs[0], ret};
            }
        }, new SDVariable[]{
                sameDiff.one("one", new long[]{1, 1}),
                sameDiff.var("two", new long[]{1, 1}),

        });

        sameDiff.execBackwards();
        SameDiff exec = sameDiff.getFunction("grad");
        System.out.println(exec);
    }


    @Test
    public void testIfStatementTrueBody() {
        SameDiff sameDiff = SameDiff.create();

        SameDiff.SameDiffFunctionDefinition conditionBody = new SameDiff.SameDiffFunctionDefinition() {
            @Override
            public SDVariable[] define(SameDiff sameDiff, Map<String, INDArray> inputs, SDVariable[] variableInputs) {
                SDVariable sum = sameDiff.sum(variableInputs[0], Integer.MAX_VALUE);
                SDVariable result = sameDiff.gt(sum, 1.0);
                return new SDVariable[]{result};
            }
        };


        SameDiff.SameDiffFunctionDefinition trueBody = new SameDiff.SameDiffFunctionDefinition() {
            @Override
            public SDVariable[] define(SameDiff sameDiff, Map<String, INDArray> inputs, SDVariable[] variableInputs) {
                SDVariable add = variableInputs[0].add(1.0);
                return new SDVariable[]{add};
            }
        };

        SameDiff.SameDiffFunctionDefinition falseBody = new SameDiff.SameDiffFunctionDefinition() {
            @Override
            public SDVariable[] define(SameDiff sameDiff, Map<String, INDArray> inputs, SDVariable[] variableInputs) {
                SDVariable sub = variableInputs[0].sub(1.0);
                return new SDVariable[]{sub};
            }
        };

        //true body trigger
        SDVariable[] firstInputs = new SDVariable[]{
                sameDiff.var("one", new long[]{1, 1})

        };


        sameDiff.ifStatement(new SameDiff.DefaultSameDiffConditional(), conditionBody, trueBody, falseBody, firstInputs);
        Pair<Map<SDVariable, DifferentialFunction>, List<DifferentialFunction>> exec = sameDiff.exec();

    }


    @Test
    public void testIfStatementFalseBody() {
        SameDiff sameDiff = SameDiff.create();

        SameDiff.SameDiffFunctionDefinition conditionBody = new SameDiff.SameDiffFunctionDefinition() {
            @Override
            public SDVariable[] define(SameDiff sameDiff, Map<String, INDArray> inputs, SDVariable[] variableInputs) {
                SDVariable sum = sameDiff.sum(variableInputs[0], Integer.MAX_VALUE);
                SDVariable result = sameDiff.gt(sum, 1.0);
                return new SDVariable[]{result};
            }
        };


        SameDiff.SameDiffFunctionDefinition trueBody = new SameDiff.SameDiffFunctionDefinition() {
            @Override
            public SDVariable[] define(SameDiff sameDiff, Map<String, INDArray> inputs, SDVariable[] variableInputs) {
                SDVariable add = variableInputs[0].add(1.0);
                return new SDVariable[]{add};
            }
        };

        SameDiff.SameDiffFunctionDefinition falseBody = new SameDiff.SameDiffFunctionDefinition() {
            @Override
            public SDVariable[] define(SameDiff sameDiff, Map<String, INDArray> inputs, SDVariable[] variableInputs) {
                SDVariable sub = variableInputs[0].sub(1.0);
                return new SDVariable[]{sub};
            }
        };


        //false body trigger
        SDVariable[] secondInputs = new SDVariable[]{
                sameDiff.setupFunction(sameDiff.var("two", new long[]{1, 1}))

        };

        sameDiff.ifStatement(new SameDiff.DefaultSameDiffConditional(), conditionBody, trueBody, falseBody, secondInputs);

        Pair<Map<SDVariable, DifferentialFunction>, List<DifferentialFunction>> exec = sameDiff.exec();


    }


    @Test
    public void testAutoBroadcastAddMatrixector() {
        SameDiff sameDiff = SameDiff.create();
        INDArray arr = Nd4j.linspace(1, 4, 4).reshape(2, 2);
        INDArray row = Nd4j.ones(2);
        INDArray assertion = arr.add(1.0);
        SDVariable left = sameDiff.var("arr", arr);
        SDVariable right = sameDiff.var("row", row);
        SDVariable test = left.add(right);
        sameDiff.exec();
        assertEquals(assertion, test.getArr());
    }


    @Test
    public void testNegativeOneShape() {
        val sd = SameDiff.create();
        val var = sd.var("test", new long[]{-1, 3});
        assertNull(var.getShape());
        assertTrue(var.isPlaceHolder());
    }

    @Test
    public void testShapeResolutionMinus1() {
        int nIn = 3;
        int nOut = 4;

        int minibatch = 3;

        for (boolean useMinus1 : new boolean[]{false, true}) {
            log.info("Starting: {}", (useMinus1 ? "minibatch -1" : "minibatch 3"));

            long[] inShape;
            if (useMinus1) {
                inShape = new long[]{-1, nIn};
            } else {
                inShape = new long[]{minibatch, nIn};
            }
            val wShape = new long[]{nIn, nOut};
            val bShape = new long[]{1, nOut};

            SameDiff sd = SameDiff.create();
            SDVariable layerInput = sd.var("in", inShape);
            SDVariable weights = sd.var("W", wShape);
            SDVariable bias = sd.var("b", bShape);

            SDVariable mmul = sd.mmul("mmul", layerInput, weights);
            SDVariable z = mmul.add("z", bias);
            SDVariable out = sd.sigmoid("out", z);

            Map<String, INDArray> m = new HashMap<>();
            INDArray in = Nd4j.rand(new long[]{minibatch, nIn});
            INDArray w = Nd4j.rand(wShape);
            INDArray b = Nd4j.rand(bShape);

            sd.associateArrayWithVariable(in, sd.getVariable("in"));
            assertNotNull(sd.getArrForVarName("in"));
            sd.associateArrayWithVariable(w, sd.getVariable("W"));
            sd.associateArrayWithVariable(b, sd.getVariable("b"));

            INDArray outArr = sd.execAndEndResult();

            assertArrayEquals(new long[]{minibatch, nOut}, outArr.shape());
        }
    }

    @Test
    public void testLabelInputPlaceHolderSgd() {

        SameDiff sd = SameDiff.create();

        int nIn = 3;
        int nOut = 4;
        int minibatch = 3;
        SDVariable input = sd.var("in", new long[]{-1, nIn});
        SDVariable label = sd.var("label", new long[]{-1, nOut});
        assertTrue(input.isPlaceHolder());
        assertTrue(label.isPlaceHolder());
        SDVariable weights = sd.var("W", new long[]{nIn, nOut});
        SDVariable bias = sd.var("b", new long[]{1, nOut});


        SDVariable mmul = sd.mmul("mmul", input, weights);
        SDVariable z = mmul.add("z", bias);
        SDVariable out = sd.tanh(z);

        SDVariable diff = out.sub(label);
        SDVariable sqDiff = diff.mul(diff);
        SDVariable msePerEx = sd.mean("msePerEx", sqDiff, 1);
        SDVariable avgMSE = sd.mean("loss", msePerEx, 0);

        INDArray inputArr = Nd4j.rand(minibatch, nIn);
        INDArray labelArr = Nd4j.rand(minibatch, nOut);
        INDArray weightsArr = Nd4j.rand(nIn, nOut);
        INDArray biasArr = Nd4j.rand(1, nOut);

        sd.associateArrayWithVariable(inputArr, input);
        sd.associateArrayWithVariable(labelArr, label);
        sd.associateArrayWithVariable(weightsArr, weights);
        sd.associateArrayWithVariable(biasArr, bias);

        INDArray result = sd.execAndEndResult();
    }


    @Test
    public void testSequentialMeansPlaceholder() {
        for (int dim0 : new int[]{10, -1}) {
            String msg = "Dimension 0 = " + dim0;
            System.out.println(msg);
            SameDiff sd = SameDiff.create();
            SDVariable in = sd.var("in", new long[]{dim0, 9, 8});
            SDVariable mean1 = sd.mean(in, 2);                  //[10,9,8] -> [10,9]
            SDVariable mean2 = sd.mean(mean1, 1);               //[10,9] -> [10,1]

            INDArray inArr = Nd4j.create(10, 9, 8);
            sd.associateArrayWithVariable(inArr, in);

            INDArray out = sd.execAndEndResult();     //Exception here, dim0=-1 case only

            assertArrayEquals(msg, new long[]{10, 1}, out.shape());
        }
    }


    @Test
    public void testReductionShapes1() {

        SameDiff sd = SameDiff.create();
        SDVariable in = sd.var("in", new long[]{10, 9, 8});
        SDVariable mean1 = sd.mean(in, 2);      //[10,9] out
        SDVariable mean2 = sd.mean(mean1, 1);   //[10,1] out
        sd.execAndEndResult();  //***Exception***

        INDArray m1 = mean1.getArr();
        INDArray m2 = mean2.getArr();

        assertArrayEquals(new long[]{10, 9}, m1.shape());
        assertArrayEquals(new long[]{10, 1}, m2.shape());
    }


    @Test
    public void testReductionShapes2() {

        SameDiff sd2 = SameDiff.create();
        SDVariable in2 = sd2.var("in", new long[]{10, 9, 8});
        SDVariable meanA = sd2.mean(in2, 0);      //[9,8] out
        assertArrayEquals(new long[]{9, 8}, meanA.getShape());

        SDVariable meanB = sd2.mean(meanA, 0);   //[1,8] out
        assertArrayEquals(new long[]{1, 8}, meanB.getShape());

        assertArrayEquals(meanA.getShape(), meanA.getArr().shape());
        assertArrayEquals(meanB.getShape(), meanB.getArr().shape());

        sd2.exec(); //***Exception***

        INDArray mA = meanA.getArr();
        INDArray mB = meanB.getArr();

        assertArrayEquals(new long[]{9, 8}, mA.shape());
        assertArrayEquals(new long[]{1, 8}, mB.shape());
    }

    @Test
    public void testNames() {
        SameDiff sd = SameDiff.create();
        SDVariable in1 = sd.var("in", new long[]{3, 2});
        SDVariable in2 = sd.var("in", new long[]{3, 3});

        val m = in1.add(1.0);
        val f = m.add(2.0);
        val s = in2.add(5.0);


        val arr = sd.execAndEndResult();
        log.info("Result M: {}", m.getArr());
        log.info("Result F: {}", f.getArr());
        log.info("Result S: {}", s.getArr());
    }


    @Test
    public void testBroadcast() {
        SameDiff sd = SameDiff.create();
        SDVariable in = sd.var("in", Nd4j.rand(3, 4));
        SDVariable broadcast = sd.f().broadcast(in, 3, 4, 5);

        INDArray out = sd.execAndEndResult();
        assertArrayEquals(new long[]{3, 4, 5}, out.shape());

        for (int i = 0; i < 5; i++) {
            assertEquals(in.getArr(), out.get(all(), all(), point(i)));
        }
    }

    @Test
    public void testRunLogisticRegression() {
        Map<String, INDArray> vars = this.variablesForInput();
        SameDiff outside = SameDiff.create();
        outside.defineFunction("activate", new SameDiff.SameDiffFunctionDefinition() {
            @Override
            public SDVariable[] define(SameDiff sameDiff, Map<String, INDArray> inputs, SDVariable[] variableInputs) {
                sameDiff.enableDebugMode();
                SDVariable x = sameDiff.var("x", inputs.get("x"));
                SDVariable w = sameDiff.var("w", inputs.get("w"));
                SDVariable y = sameDiff.var("y", inputs.get("y"));
                SDVariable activation = sameDiff.sigmoid("activation", sameDiff.mmul("mmul", x, w));
                SDVariable oneMinusY = y.rsub("oneminusy", 1.0);
                SDVariable oneMinusPredictions = activation.rsub("oneminusactivations", 1.0);
                SDVariable outputTimesY = y.mul("output * y", activation);
                SDVariable yHat = oneMinusPredictions.mul("yhat", oneMinusY);
                SDVariable probs = outputTimesY.add("probs", yHat);
                SDVariable logProbs = sameDiff.log("logprob", probs);
                SDVariable ret = sameDiff.sum("totalsum", logProbs, Integer.MAX_VALUE);
                SDVariable ret2 = sameDiff.neg("negtotalsum", ret);
                return new SDVariable[]{ret2};
            }
        }, vars);

        SameDiff activation = outside.getFunction("activate");
        int epochsToRun = 5;
        double lr = 0.1;
     /*   for(int i = 0; i < epochsToRun; i++) {
            activation.execBackwards();
            INDArray wGrad = activation.grad("w").getArr().reshape(vars.get("w").shape());
            vars.get("w").subi(wGrad.mul(lr));
            System.out.println("Score: " + activation.getVariable("negtotalsum").getArr());
        }*/

    }


    @Test
    public void testSoftmaxRegression() {
        Map<String, INDArray> vars = this.variablesForInput();
        SameDiff outside = SameDiff.create();
        outside.defineFunction("activate", new SameDiff.SameDiffFunctionDefinition() {
            @Override
            public SDVariable[] define(SameDiff sameDiff, Map<String, INDArray> inputs, SDVariable[] variableInputs) {
                sameDiff.enableDebugMode();
                SDVariable x = sameDiff.var("x", inputs.get("x"));
                SDVariable w = sameDiff.var("w", inputs.get("w"));
                SDVariable y = sameDiff.var("y", inputs.get("y"));
                SDVariable activation = sameDiff.softmax("activation", sameDiff.mmul("mmul", x, w));
                SDVariable ret = sameDiff.sum("totalsum", activation, Integer.MAX_VALUE);
                SDVariable ret2 = sameDiff.neg("negtotalsum", ret);
                return new SDVariable[]{ret2};
            }
        }, vars);


        /**
         * Backwards should be:
         * neg score
         * sum sum of log
         * log (log probs)
         * add
         * mul
         * mul
         * rsub (predictions)
         * sigmoid
         * rsub
         * matrix multiply
         *
         */


        Pair<Map<SDVariable, DifferentialFunction>, List<DifferentialFunction>> opsBackward = outside.getFunction("activate").execBackwards();
        SameDiff gradSameDiff = outside.getFunction("activate").getFunction("grad");

        SDVariable gradWrtX = outside.getFunction("activate").grad("x");
        SDVariable gradWrtW = outside.getFunction("activate").grad("w");
        assertNotNull(gradWrtX);
        assertNotNull(gradWrtW);

        INDArray wGradAssertion = Nd4j.create(new double[]{0, 0, 0}).reshape(3, 1);
        assertEquals(wGradAssertion, outside.getFunction("activate").grad("w").getArr());
        //note here that the gradients here end up being some weird really low eps where it
        //isn't exactly zero
        //        assertEquals(inputAssertion,outside.getFunction("activate").grad("x").getArr());


        System.out.println(gradWrtX);
        System.out.println(gradWrtW);


    }


    @Test
    public void testTransposeWithVector() {
        val sd = SameDiff.create();
        val matrix = Nd4j.linspace(1, 12, 12).reshape(4, 3);
        val vector = Nd4j.linspace(1, 4, 4).reshape(4, 1);
        val input1 = sd.var("input", matrix);
        val input2 = sd.var("input2", vector);
        val output = sd.mmul("output", input1, input2, MMulTranspose.builder().transposeA(true).transposeB(false).build());
        assertArrayEquals(new long[]{3, 1}, output.getShape());
        val result = sd.exec();
    }


    @Test
    public void testLogisticRegression() {
        Map<String, INDArray> vars = this.variablesForInput();
        SameDiff outside = SameDiff.create();

        outside.defineFunction("activate", new SameDiff.SameDiffFunctionDefinition() {
            @Override
            public SDVariable[] define(SameDiff sameDiff, Map<String, INDArray> inputs, SDVariable[] variableInputs) {
                sameDiff.enableDebugMode();
                SDVariable x = sameDiff.var("x", inputs.get("x"));
                SDVariable w = sameDiff.var("w", inputs.get("w"));
                SDVariable y = sameDiff.var("y", inputs.get("y"));
                SDVariable activation = sameDiff.sigmoid("activation", sameDiff.mmul("mmul", x, w));
                SDVariable oneMinusY = y.rsub("oneminusy", 1.0);
                SDVariable oneMinusPredictions = activation.rsub("oneminusactivations", 1.0);
                SDVariable outputTimesY = y.mul("output * y", activation);
                SDVariable yHat = oneMinusPredictions.mul("yhat", oneMinusY);
                SDVariable probs = outputTimesY.add("probs", yHat);
                SDVariable logProbs = sameDiff.log("logprob", probs);
                SDVariable ret = sameDiff.sum("totalsum", logProbs, Integer.MAX_VALUE);
                SDVariable ret2 = sameDiff.neg("negtotalsum", ret);
                return new SDVariable[]{ret2};
            }
        }, vars);


        /**
         * Backwards should be:
         * neg score
         * sum sum of log
         * log (log probs)
         * add
         * mul
         * mul
         * rsub (predictions)
         * sigmoid
         * rsub
         * matrix multiply
         *
         */

        Nd4j.getExecutioner().enableDebugMode(true);
        Nd4j.getExecutioner().enableVerboseMode(true);


        Pair<Map<SDVariable, DifferentialFunction>, List<DifferentialFunction>> opsBackward = outside.getFunction("activate").execBackwards();
        SameDiff gradSameDiff = outside.getFunction("activate").getFunction("grad");

        SDVariable gradWrtX = outside.getFunction("activate").grad("x");
        SDVariable gradWrtW = outside.getFunction("activate").grad("w");
        assertNotNull(gradWrtX);
        assertNotNull(gradWrtW);

        INDArray wGradAssertion = Nd4j.create(new double[]{-0.81, 1.255, -1.80499983}).reshape(3, 1);
        INDArray inputAssertion = Nd4j.valueArrayOf(vars.get("x").shape(), 1e-1);
        INDArray yGradAssertion = Nd4j.zeros(vars.get("y").shape());
        INDArray mmulGrad = Nd4j.create(new double[]{-0.5, -0.5, 0.5, -0.5}).reshape(4, 1);
        INDArray predsGradAssertion = Nd4j.create(new double[]{-2, -2, 2, -2}).reshape(4, 1);
        INDArray oneMinusPredsGradAssertion = Nd4j.create(new double[]{0, 0, -2, 0}).reshape(4, 1);
        INDArray oneMinusLabelsAssertion = Nd4j.valueArrayOf(4, -1).reshape(4, 1);
        INDArray outputTimesYGradAssertion = Nd4j.valueArrayOf(4, -2).reshape(4, 1);
        INDArray yHatAssertion = outputTimesYGradAssertion.dup();
        INDArray labelProbsGradAssertion = yHatAssertion.dup();
        INDArray logProbsGradAssertion = Nd4j.valueArrayOf(4, -1).reshape(4, 1);

        assertEquals(logProbsGradAssertion, outside.getFunction("activate").grad("logprob").getArr());
        assertEquals(labelProbsGradAssertion, outside.getFunction("activate").grad("probs").getArr());
        assertEquals(yHatAssertion, outside.getFunction("activate").grad("yhat").getArr());
        assertEquals(outputTimesYGradAssertion, outside.getFunction("activate").grad("output * y").getArr());
        assertEquals(oneMinusLabelsAssertion, outside.getFunction("activate").grad("oneminusy").getArr());
        assertEquals(oneMinusPredsGradAssertion, outside.getFunction("activate").grad("oneminusactivations").getArr());
        assertEquals(predsGradAssertion, outside.getFunction("activate").grad("activation").getArr());
        assertEquals(mmulGrad, outside.getFunction("activate").grad("mmul").getArr());
        assertEquals(yGradAssertion, outside.getFunction("activate").grad("y").getArr());
        assertEquals(wGradAssertion, outside.getFunction("activate").grad("w").getArr());
        //note here that the gradients here end up being some weird really low eps where it
        //isn't exactly zero
        //        assertEquals(inputAssertion,outside.getFunction("activate").grad("x").getArr());


        System.out.println(gradWrtX);
        System.out.println(gradWrtW);


    }


    @Test
    public void testNestedExecution() {
        final SameDiff outer = SameDiff.create();
        Map<String, INDArray> input = new HashMap<>();
        input.put("x", Nd4j.ones(2));
        outer.defineFunction("firstadd", new SameDiff.SameDiffFunctionDefinition() {
            @Override
            public SDVariable[] define(SameDiff sameDiff, Map<String, INDArray> inputs, SDVariable[] variableInputs) {
                SDVariable input = sameDiff.var("x", inputs.get("x"));
                SDVariable ret = input.add(input);
                return new SDVariable[]{ret};
            }
        }, input);

        outer.defineFunction("secondadd", new SameDiff.SameDiffFunctionDefinition() {
            @Override
            public SDVariable[] define(SameDiff sameDiff, Map<String, INDArray> inputs, SDVariable[] variableInputs) {
                SDVariable result = outer.invokeFunctionOn("firstadd", sameDiff);
                return new SDVariable[]{result.add(1.0)};
            }
        });

        SameDiff secondAdd = outer.getFunction("secondadd");
        INDArray[] outputs = secondAdd.eval(input);
        INDArray outputsAssertion = Nd4j.valueArrayOf(2, 2.0);
        assertEquals(outputsAssertion, outputs[0]);
    }


    @Test
    public void testResultPropagation() {
        SameDiff sameDiff = SameDiff.create();
        INDArray inputs = Nd4j.create(new double[][]{
                {0.52, 1.12, 0.77},
                {0.88, -1.08, 0.15},
                {0.52, 0.06, -1.30},
                {0.74, -2.49, 1.39}
        });


        INDArray weights = Nd4j.randn(3, 1);

        SDVariable x = sameDiff.var("x", inputs);
        SDVariable w = sameDiff.var("w", weights);
        SDVariable preOutput = sameDiff.mmul(x, w);

        SDVariable outputs = sameDiff.sigmoid(preOutput);
        List<DifferentialFunction> ops = sameDiff.exec().getRight();
        DifferentialFunction firstOp = ops.get(0);
        val firstResult = sameDiff.getVariable(firstOp.outputVariables()[0].getVarName()).getArr();

    }

    @Test
    public void testSimpleDefineFunction() {
        SameDiff sameDiffOuter = SameDiff.create();
        Map<String, INDArray> inputs = variablesForInput();
        inputs.remove("y");
        String logisticForward = "logisticPredictions";
        sameDiffOuter.defineFunction(logisticForward, new SameDiff.SameDiffFunctionDefinition() {
            @Override
            public SDVariable[] define(SameDiff sameDiff, Map<String, INDArray> inputs, SDVariable[] variableInputs) {

                SDVariable input = sameDiff.var("x", inputs.get("x"));
                SDVariable w = sameDiff.var("w", inputs.get("w"));
                SDVariable preOutput = sameDiff.mmul(input, w);
                SDVariable sigmoid = sameDiff.sigmoid(preOutput);
                return new SDVariable[]{sigmoid};
            }

        }, inputs);

        assertEquals(1, sameDiffOuter.definedFunctionNames().size());

        //note here that we don't add the duplicate ops with define function anymore
    }


    @Test
    public void testSoftmax() {
        SameDiff sameDiff = SameDiff.create();
        INDArray sumInput = Nd4j.linspace(1, 4, 4).reshape(2, 2);
        Map<String, INDArray> inputs = new HashMap<>();
        inputs.put("x", sumInput);
        sameDiff.defineFunction("softmax", new SameDiff.SameDiffFunctionDefinition() {
            @Override
            public SDVariable[] define(SameDiff sameDiff, Map<String, INDArray> inputs, SDVariable[] variableInputs) {
                SDVariable input = sameDiff.var("x", inputs.get("x").dup());
                SDVariable softmax = sameDiff.softmax(input);
                //original shape ends up being 2,2
                return new SDVariable[]{softmax};
            }
        }, inputs);

        INDArray executions = sameDiff.execAndEndResult("softmax");
        INDArray assertions = Transforms.softmax(sumInput.dup());
        assertArrayEquals(sumInput.shape(), executions.shape());
        System.out.println(executions);
        assertEquals(assertions, executions);


        SoftMaxDerivative softMaxDerivative = new SoftMaxDerivative(sumInput);
        Nd4j.getExecutioner().exec(softMaxDerivative);
        System.out.println(softMaxDerivative.z());
    }


    @Test
    public void testSigmoidBackwards() {
        SameDiff sameDiff = SameDiff.create();
        INDArray sumInput = Nd4j.linspace(1, 4, 4).reshape(2, 2);
        Map<String, INDArray> inputs = new HashMap<>();
        inputs.put("x", sumInput);
        SDVariable input = sameDiff.var("x", inputs.get("x"));
        SDVariable sigmoid = sameDiff.sigmoid(input);
        SDVariable sum = sameDiff.sum(sigmoid, Integer.MAX_VALUE);
        List<DifferentialFunction> backwardsOps = sameDiff.execBackwards().getRight();
        Op finalOp = (Op) backwardsOps.get(backwardsOps.size() - 1);
        assertTrue(Nd4j.create(new double[][]{
                {0.1966, 0.1050},
                {0.0452, 0.0177}
        }).equalsWithEps(
                finalOp.z(), 1e-2));
        System.out.println(backwardsOps);
    }


    @Test
    public void testSumGradient() {
        SameDiff sameDiff = SameDiff.create();
        SDVariable twoByTwo = sameDiff.var("initial", Nd4j.linspace(1, 4, 4).reshape(2, 2));
        SDVariable sum = sameDiff.sum(twoByTwo, Integer.MAX_VALUE);
        Pair<Map<SDVariable, DifferentialFunction>, List<DifferentialFunction>> execBackwards = sameDiff.execBackwards();
        SameDiff grad = sameDiff.getFunction("grad");
        SDVariable gradArr = sameDiff.grad(twoByTwo.getVarName());
        assertEquals(Nd4j.ones(2, 2), gradArr.getArr());
    }


    @Test
    public void testMmulGradient() {
        SameDiff sameDiff = SameDiff.create();
        INDArray sumInput = Nd4j.linspace(1, 4, 4).reshape(2, 2);
        Map<String, INDArray> inputs = new HashMap<>();
        inputs.put("x", sumInput);
        inputs.put("y", sumInput.dup());

        sameDiff.defineFunction("mmulGradient", new SameDiff.SameDiffFunctionDefinition() {
            @Override
            public SDVariable[] define(SameDiff sameDiff, Map<String, INDArray> inputs, SDVariable[] variableInputs) {
                SDVariable input = sameDiff.var("x", inputs.get("x"));
                SDVariable input2 = sameDiff.var("y", inputs.get("y"));
                SDVariable exp = sameDiff.mmul(input, input2);
                SDVariable sum = sameDiff.sum(exp, Integer.MAX_VALUE);
                return new SDVariable[]{sum};
            }
        }, inputs);

        List<DifferentialFunction> ops = sameDiff.getFunction("mmulGradient").execBackwards().getRight();
        String print = sameDiff.asFlatPrint();


        assumeNotNull(sameDiff.getFunction("mmulGradient").getFunction("grad"));
        assumeNotNull(sameDiff.getFunction("mmulGradient").grad("x"));
        assumeNotNull(sameDiff.getFunction("mmulGradient").grad("y"));

        SDVariable gradWrtX = sameDiff.getFunction("mmulGradient").grad("x");
        SDVariable gradWrtY = sameDiff.getFunction("mmulGradient").grad("y");
        assumeNotNull(gradWrtX.getArr());
        assumeNotNull(gradWrtY.getArr());


        INDArray xGradAssertion = Nd4j.create(new double[][]{
                {3, 7},
                {3, 7}
        });

        INDArray yGradAssertion = Nd4j.create(new double[][]{
                {4, 4},
                {6, 6}
        });

        assertEquals(xGradAssertion, gradWrtX.getArr());
        assertEquals(yGradAssertion, gradWrtY.getArr());

    }

    @Test
    public void testExpGradient() {
        SameDiff sameDiff = SameDiff.create();
        INDArray sumInput = Nd4j.linspace(1, 4, 4).reshape(2, 2);
        Map<String, INDArray> inputs = new HashMap<>();
        inputs.put("x", sumInput);
        sameDiff.defineFunction("expGradient", new SameDiff.SameDiffFunctionDefinition() {
            @Override
            public SDVariable[] define(SameDiff sameDiff, Map<String, INDArray> inputs, SDVariable[] variableInputs) {
                SDVariable input = sameDiff.var("x", inputs.get("x"));
                SDVariable exp = sameDiff.exp(input);
                SDVariable sum = sameDiff.sum(exp, Integer.MAX_VALUE);
                return new SDVariable[]{sum};
            }
        }, inputs);


        List<DifferentialFunction> ops = sameDiff.getFunction("expGradient").execBackwards().getRight();

        INDArray executions = ops.get(ops.size() - 1).outputVariables()[0].getArr();
        INDArray assertion = Nd4j.create(new double[][]{
                {2.7183, 7.3891},
                {20.0855, 54.5981}
        });
        assertArrayEquals(sumInput.shape(), executions.shape());
        assertEquals(assertion, executions);
        System.out.println(executions);
        //assertEquals(Nd4j.ones(2,2),executions);
    }


/*    @Test
    public void testDepth() {
        SameDiff sameDiff = SameDiff.create();
        SDVariable x = sameDiff.one("one",new long[]{2,2});
        assertEquals(0,x.depth());
        SDVariable sigmoid = sameDiff.sigmoid("sigmoid",x);
        assertEquals(1,sigmoid.depth());
    }*/


    @Test
    public void testTanhGradient() {
        SameDiff sameDiff = SameDiff.create();
        INDArray sumInput = Nd4j.linspace(1, 4, 4).reshape(2, 2);
        Map<String, INDArray> inputs = new HashMap<>();
        inputs.put("x", sumInput);
        sameDiff.defineFunction("tanhGradient", new SameDiff.SameDiffFunctionDefinition() {
            @Override
            public SDVariable[] define(SameDiff sameDiff, Map<String, INDArray> inputs, SDVariable[] variableInputs) {
                SDVariable input = sameDiff.var("x", inputs.get("x"));
                SDVariable tanh = sameDiff.tanh(input);
                SDVariable sum = sameDiff.sum(tanh, Integer.MAX_VALUE);
                return new SDVariable[]{tanh};
            }
        }, inputs);

        INDArray executions = sameDiff.getFunction("tanhGradient").execBackwardAndEndResult();
        //[0.41997434161402614,0.07065082485316443,0.009866037165440211,0.0013409506830258655]
        INDArray assertion = Nd4j.create(new double[][]{
                {0.41997434161402614, 0.07065082485316443},
                {0.009866037165440211, 0.0013409506830258655}
        });

        assertTrue(assertion.equalsWithEps(
                executions, 1e-3));

        assertArrayEquals(sumInput.shape(), executions.shape());
        assertEquals(assertion, executions);
        System.out.println(executions);
        //assertEquals(Nd4j.ones(2,2),executions);
    }


    @Test
    public void testRsubScalar() {
        SameDiff sameDiff = SameDiff.create();
        Map<String, INDArray> params = new HashMap<>();
        INDArray var = Nd4j.valueArrayOf(4, 2);
        params.put("x", var);
        sameDiff.defineFunction("rsubop", new SameDiff.SameDiffFunctionDefinition() {
            @Override
            public SDVariable[] define(SameDiff sameDiff, Map<String, INDArray> inputs, SDVariable[] variableInputs) {
                SDVariable input = sameDiff.var("x", inputs.get("x"));
                SDVariable ret = input.rsub(1.0);
                return new SDVariable[]{ret};
            }
        }, params);

        SameDiff logisticGraph = sameDiff.getFunction("rsubop");
        INDArray[] outputs = logisticGraph.eval(params);
        assertEquals(Nd4j.ones(4).muli(-1), outputs[0]);
        System.out.println(Arrays.toString(outputs));


    }


    @Test
    public void testFunctionScalarResultPropagation() {
        SameDiff sameDiffOuter = SameDiff.create();
        Map<String, INDArray> inputs = variablesForInput();

        sameDiffOuter.defineFunction("logisticPredictions", new SameDiff.SameDiffFunctionDefinition() {
            @Override
            public SDVariable[] define(SameDiff sameDiff, Map<String, INDArray> inputs, SDVariable[] variableInputs) {
                SDVariable input = sameDiff.var("x", inputs.get("x"));
                SDVariable w = sameDiff.var("w", inputs.get("w"));
                SDVariable preOutput = sameDiff.mmul(input, w);
                SDVariable sigmoid = sameDiff.sigmoid(preOutput);
                return new SDVariable[]{sigmoid};
            }
        }, inputs);

        sameDiffOuter.defineFunction("oneminuspredictions", new SameDiff.SameDiffFunctionDefinition() {
            @Override
            public SDVariable[] define(SameDiff sameDiff, Map<String, INDArray> inputs, SDVariable[] variableInputs) {
                SDVariable y = sameDiff.var("y", inputs.get("y"));
                SDVariable oneMinusPredictions = y.rsub(1.0);
                return new SDVariable[]{oneMinusPredictions};
            }
        }, inputs);


        SameDiff logisticGraph = sameDiffOuter.getFunction("oneminuspredictions");
        INDArray[] outputs = logisticGraph.eval(inputs);
        INDArray assertion = Nd4j.create(new double[]{0, 0, 1, 0});
        assertEquals(assertion, outputs[outputs.length - 1]);

    }

    @Test
    public void testInplaceSubi() {
        SameDiff sameDiffOuter = SameDiff.create();
        Map<String, INDArray> params = new HashMap<>();
        params.put("x", Nd4j.ones(4));
        sameDiffOuter.defineFunction("inplacesubi", new SameDiff.SameDiffFunctionDefinition() {
            @Override
            public SDVariable[] define(SameDiff sameDiff, Map<String, INDArray> inputs, SDVariable[] variableInputs) {
                SDVariable inplace = sameDiff.var("x", inputs.get("x"));
                return new SDVariable[]{inplace.subi(1.0)};
            }
        }, params);

        sameDiffOuter.getFunction("inplacesubi").eval(params);
        assertEquals(Nd4j.zeros(4), params.get("x"));
    }


    @Test
    public void testMmul() {
        SameDiff sameDiffOuter = SameDiff.create();
        Map<String, INDArray> inputs = variablesForInput();
        SDVariable x = sameDiffOuter.var("x", inputs.get("x"));
        SDVariable w = sameDiffOuter.var("w", inputs.get("w"));
        SDVariable output = sameDiffOuter.mmul(x, w);
    }


    @Test
    public void testGraphBuilding() {
        final SameDiff sameDiffOuter = SameDiff.create();
        Map<String, INDArray> inputs = variablesForInput();

        sameDiffOuter.defineFunction("logisticPredictions", new SameDiff.SameDiffFunctionDefinition() {
            @Override
            public SDVariable[] define(SameDiff sameDiff, Map<String, INDArray> inputs, SDVariable[] variableInputs) {
                SDVariable input = sameDiff.var("x", inputs.get("x"));
                SDVariable w = sameDiff.var("w", inputs.get("w"));
                SDVariable y = sameDiff.var("y", inputs.get("y"));
                SDVariable preOutput = sameDiff.mmul(input, w);
                SDVariable sigmoid = sameDiff.sigmoid(preOutput);

                return new SDVariable[]{sigmoid};
            }
        }, inputs);

        sameDiffOuter.defineFunction("loss", new SameDiff.SameDiffFunctionDefinition() {
            @Override
            public SDVariable[] define(SameDiff sameDiff, Map<String, INDArray> inputs, SDVariable[] variableInputs) {
                SDVariable outputs = sameDiffOuter.invokeFunctionOn("logisticPredictions", sameDiff);
                SDVariable y = sameDiff.getVariable("y");
                SDVariable outputTimesY = outputs.mul(y);
                return new SDVariable[]{outputTimesY};

            }
        }, inputs);


        SameDiff logisticPrediction = sameDiffOuter.getFunction("logisticPredictions");
        List<String> logisticOpNameAssertions = Arrays.asList("mmul", "sigmoid");


    }


    @Test
    public void testScalarAdd() {
        SameDiff sameDiff = SameDiff.create();
        SDVariable twoByTwo = sameDiff.var("first", Nd4j.linspace(1, 4, 4).reshape('c', 2, 2));
        SDVariable add = twoByTwo.add(1.0);
        INDArray test = sameDiff.execAndEndResult();
        INDArray assertion = Nd4j.linspace(1, 4, 4).reshape('c', 2, 2).add(1.0);
        assertEquals(assertion, test);
    }


    @Test
    public void testSums() {
        SameDiff sameDiff = SameDiff.create();
        INDArray ones = Nd4j.ones(4);
        SDVariable sdVariable = sameDiff.var("ones", ones);
        SDVariable result = sdVariable.addi(1.0);
        SDVariable total = sameDiff.sum(result, Integer.MAX_VALUE);
        List<DifferentialFunction> ops = sameDiff.exec().getRight();
        INDArray output = null;
        for (int i = 0; i < 5; i++) {
            output = sameDiff.execAndEndResult(ops);
            System.out.println("Ones " + ones);
            System.out.println(output);
        }

        assertEquals(Nd4j.valueArrayOf(4, 7), ones);
        assertEquals(28, output.getDouble(0), 1e-1);
    }


    @Test
    public void testDenseLayerForwardPass() {
        Nd4j.getRandom().setSeed(12345);

        SameDiff sd = SameDiff.create();

        INDArray iInput = Nd4j.rand(3, 4);
        INDArray iWeights = Nd4j.rand(4, 5);
        INDArray iBias = Nd4j.rand(1, 5);

        SDVariable input = sd.var("input", iInput);
        SDVariable weights = sd.var("weights", iWeights);
        SDVariable bias = sd.var("bias", iBias);

        SDVariable mmul = sd.mmul("mmul", input, weights);
        SDVariable z = mmul.add("z", bias);
        SDVariable out = sd.sigmoid("out", z);

        INDArray expMmul = iInput.mmul(iWeights);
        INDArray expZ = expMmul.addRowVector(iBias);
        INDArray expOut = Transforms.sigmoid(expZ, true);

        sd.exec();

        assertEquals(expMmul, mmul.getArr());
        assertEquals(expZ, z.getArr());
        assertEquals(expOut, out.getArr());
    }

    @Test
    public void testActivationBackprop() {

        Activation[] afns = new Activation[]{
                Activation.TANH,
                Activation.SIGMOID,
                Activation.ELU,
                Activation.SOFTPLUS,
                Activation.SOFTSIGN,
                Activation.HARDTANH,
                Activation.CUBE,            //WRONG output - see issue https://github.com/deeplearning4j/nd4j/issues/2426
                Activation.RELU,            //JVM crash
                Activation.LEAKYRELU        //JVM crash
        };

        for (Activation a : afns) {

            SameDiff sd = SameDiff.create();
            INDArray inArr = Nd4j.linspace(-3, 3, 7);
            INDArray labelArr = Nd4j.linspace(-3, 3, 7).muli(0.5);
            SDVariable in = sd.var("in", inArr.dup());

//            System.out.println("inArr: " + inArr);

            INDArray outExp;
            SDVariable out;
            switch (a) {
                case ELU:
                    out = sd.elu("out", in);
                    outExp = Transforms.elu(inArr, true);
                    break;
                case HARDTANH:
                    out = sd.hardTanh("out", in);
                    outExp = Transforms.hardTanh(inArr, true);
                    break;
                case LEAKYRELU:
                    out = sd.leakyRelu("out", in, 0.01);
                    outExp = Transforms.leakyRelu(inArr, true);
                    break;
                case RELU:
                    out = sd.relu("out", in, 0.0);
                    outExp = Transforms.relu(inArr, true);
                    break;
                case SIGMOID:
                    out = sd.sigmoid("out", in);
                    outExp = Transforms.sigmoid(inArr, true);
                    break;
                case SOFTPLUS:
                    out = sd.softplus("out", in);
                    outExp = Transforms.softPlus(inArr, true);
                    break;
                case SOFTSIGN:
                    out = sd.softsign("out", in);
                    outExp = Transforms.softsign(inArr, true);
                    break;
                case TANH:
                    out = sd.tanh("out", in);
                    outExp = Transforms.tanh(inArr, true);
                    break;
                case CUBE:
                    out = sd.cube("out", in);
                    outExp = Transforms.pow(inArr, 3, true);
                    break;
                default:
                    throw new RuntimeException(a.toString());
            }

            //Sum squared error loss:
            SDVariable label = sd.var("label", labelArr.dup());
            SDVariable diff = label.sub("diff", out);
            SDVariable sqDiff = diff.mul("sqDiff", diff);
            SDVariable totSum = sd.sum("totSum", sqDiff, Integer.MAX_VALUE);    //Loss function...

            sd.exec();
            INDArray outAct = sd.getVariable("out").getArr();
            assertEquals(a.toString(), outExp, outAct);

            // L = sum_i (label - out)^2
            //dL/dOut = 2(out - label)
            INDArray dLdOutExp = outExp.sub(labelArr).mul(2);
            INDArray dLdInExp = a.getActivationFunction().backprop(inArr.dup(), dLdOutExp.dup()).getFirst();

            sd.execBackwards();
            SameDiff gradFn = sd.getFunction("grad");
            INDArray dLdOutAct = gradFn.getVariable("out-grad").getArr();
            INDArray dLdInAct = gradFn.getVariable("in-grad").getArr();

            assertEquals(a.toString(), dLdOutExp, dLdOutAct);
            assertEquals(a.toString(), dLdInExp, dLdInAct);
        }
    }

    @Test
    public void testMmulWithTranspose() {
        //Here: [x,3]^T * [x,4] = [3,4]

        for (int i : new int[]{2, 1}) {
            System.out.println("i = " + i);
            INDArray first = Nd4j.linspace(1, 3 * i, 3 * i).reshape('c', i, 3);      //To [1,3] or [2,3]
            INDArray second = Nd4j.linspace(4, 4 + 4 * i, 4 * i).reshape('c', i, 4);  //To [1,4] or [2,4]

            System.out.println("Shapes: " + Arrays.toString(first.shape()) + "\t" + Arrays.toString(second.shape()));

            SameDiff sd = SameDiff.create();
            SDVariable f = sd.var("in1", first);
            SDVariable s = sd.var("in2", second);

            MMulTranspose mt = MMulTranspose.builder()
                    .transposeA(true)
                    .transposeB(false)
                    .transposeResult(false)
                    .a(first)
                    .b(second)
                    .build();
            SDVariable mmul = sd.f().mmul(f, s, mt);
            sd.updateVariableNameAndReference(mmul, "mmul");

            INDArray out = sd.execAndEndResult();

            INDArray exp = first.transpose().mmul(second);
            assertEquals(exp, out);
            System.out.println("----- Finished: i = " + i + " ------");
        }
    }

    @Test
    public void testPlaceholderReduceSimple() {
        SameDiff sd = SameDiff.create();
        SDVariable v = sd.var("in", new long[]{-1, 10});
        SDVariable vSum = sd.sum(v, 1);                             //Exception here
    }


    @Test
    public void testSequentialMeans() {
        SameDiff sd = SameDiff.create();
        SDVariable in = sd.var("in", new long[]{10, 10, 10});
        SDVariable mean1 = sd.mean(in, 2);      //[10,10] out
        SDVariable mean2 = sd.mean(mean1, 1);   //[10,1] out - ***exception here***
    }

    @Test
    public void testConv3dBasic() {
        int nIn = 3;
        int nOut = 4;
        int kH = 2;
        int kW = 2;
        int kT = 2;

        int mb = 3;
        int imgH = 28;
        int imgW = 28;
        int imgT = 28;

        SameDiff sd = SameDiff.create();
        INDArray wArr = Nd4j.create(nOut, nIn, kT, kH, kW); //As per DL4J
        INDArray bArr = Nd4j.create(1, nOut);
        INDArray inArr = Nd4j.create(mb, nIn, imgT, imgH, imgW);

        SDVariable in = sd.var("in", inArr);
        SDVariable w = sd.var("W", wArr);
        SDVariable b = sd.var("b", bArr);

        //Order: https://github.com/deeplearning4j/libnd4j/blob/6c41ea5528bb1f454e92a9da971de87b93ff521f/include/ops/declarable/generic/convo/conv2d.cpp#L20-L22
        //in, w, b - bias is optional
        SDVariable[] vars = new SDVariable[]{in, w, b};

        Conv3DConfig conv3DConfig = Conv3DConfig.builder()
                .kH(kH).kW(kW).kT(kT)
                .dilationH(1).dilationW(1).dilationT(1)
                .isValidMode(false)
                .biasUsed(false)
                .build();

        SDVariable out = sd.conv3d(vars, conv3DConfig);
        out = sd.tanh("out", out);

        INDArray outArr = sd.execAndEndResult();
        //Expected output size: out = (in - k)/d + 1 = (28-2+0)/1+1 = 27
        val outShape = outArr.shape();
        assertArrayEquals(new long[]{mb, nOut, 27, 27, 27}, outShape);
    }

    @Test
    public void testBatchNormTest() {
        SameDiff sd = SameDiff.create();

        INDArray input = Nd4j.rand(1, 10);
        INDArray mean = Nd4j.rand(1, 10);
        INDArray var = Nd4j.rand(1, 10);
        INDArray gamma = Nd4j.rand(1, 10);
        INDArray beta = Nd4j.rand(1, 10);

        SDVariable sdInput = sd.var("input", input);
        SDVariable sdMean = sd.var("mean", mean);
        SDVariable sdVar = sd.var("var", var);
        SDVariable sdGamma = sd.var("gamma", gamma);
        SDVariable sdBeta = sd.var("beta", beta);

        SDVariable out = sd.batchNorm(sdInput, sdMean, sdVar, sdGamma, sdBeta,
                true, true, 0.0);
        out = sd.tanh("out", out);

        INDArray outArr = sd.execAndEndResult();
        assertArrayEquals(new long[]{1, 10}, outArr.shape());

    }

    @Test
    public void testLrn() {
        SameDiff sd = SameDiff.create();

        INDArray input = Nd4j.create(new float[]{4, 4, 4, 4}, new long[]{1, 4, 1, 1});

        SDVariable sdInput = sd.var("input", input);

        LocalResponseNormalizationConfig lrn = LocalResponseNormalizationConfig.builder()
                .alpha(1.0)
                .beta(.5)
                .bias(0.0)
                .depth(1).build();

        SDVariable out = sd.localResponseNormalization(sdInput, lrn);
        SDVariable sdOut = sd.tanh("out", out);

        sd.exec();

        for (int i = 0; i < 4; i++)
           assert out.getArr().get(all(), NDArrayIndex.point(i), all(), all()).getDouble(0) == 1;

    }

    @Test
    public void testMoments() {
        SameDiff sd = SameDiff.create();

        INDArray input = Nd4j.create(new float[]{1, 2, 3, 4}, new long[]{2, 2});

        SDVariable sdInput = sd.var("input", input);

        val axis = new int[]{0, 1};
        SDVariable[] moments = sd.moments(sdInput, axis);
        SDVariable mean = moments[0];
        SDVariable variance = moments[1];

        SDVariable sum = mean.add(variance);
        SDVariable out = sd.tanh("out", sum);

        INDArray outArr = sd.execAndEndResult();

        INDArray meanArray = mean.getArr();
        INDArray varArray = variance.getArr();

        assert meanArray.getDouble(0) == 2.5;
        assert varArray.getDouble(0) == 1.25;
    }

    @Test
    public void testNormalizeMoments() {
        SameDiff sd = SameDiff.create();

        INDArray counts = Nd4j.create(new float[]{2}, new long[]{1, 1});
        INDArray means = Nd4j.create(new float[]{2, 4}, new long[]{1, 2});
        INDArray vars = Nd4j.create(new float[]{6, 8}, new long[]{1, 2});


        SDVariable sdCounts = sd.var("counts", counts);
        SDVariable sdMeans = sd.var("means", means);
        SDVariable sdVars = sd.var("vars", vars);
        double shift = 0.0;

        SDVariable[] moments = sd.normalizeMoments(sdCounts, sdMeans, sdVars, shift);
        SDVariable normMean = moments[0];
        SDVariable normVariance = moments[1];

        SDVariable sum = normMean.add(normVariance);
        SDVariable out = sd.tanh("out", sum);

        INDArray outArr = sd.execAndEndResult();

        INDArray meanArray = normMean.getArr();
        INDArray varArray = normVariance.getArr();

        assert meanArray.getDouble(0, 0) == 1;
        assert meanArray.getDouble(0, 1) == 2;
        assert Arrays.equals(meanArray.shape(), varArray.shape());

    }


    @Test
    public void testDepthWiseConv2dBasic() {
        int nIn = 3;
        int depthWise = 4;
        int kH = 2;
        int kW = 2;

        int mb = 3;
        int imgH = 28;
        int imgW = 28;


        SameDiff sd = SameDiff.create();
        INDArray depthWeightArr = Nd4j.create(depthWise, nIn, kH, kW);

        INDArray bArr = Nd4j.create(1, depthWise * nIn);
        INDArray inArr = Nd4j.create(mb, nIn, imgH, imgW);

        SDVariable in = sd.var("in", inArr);
        SDVariable dW = sd.var("dW", depthWeightArr);
        SDVariable b = sd.var("b", bArr);

        SDVariable[] vars = new SDVariable[]{in, dW, b};

        Conv2DConfig c = Conv2DConfig.builder()
                .kh(kH).kw(kW)
                .ph(0).pw(0)
                .sy(1).sx(1)
                .dh(1).dw(1)
                .isSameMode(false)
                .build();

        SDVariable out = sd.sconv2d(vars, c);
        out = sd.tanh("out", out);

        INDArray outArr = sd.execAndEndResult();
        //Expected output size: out = (in - k + 2*p)/s + 1 = (28-2+0)/1+1 = 27
        val outShape = outArr.shape();
        assertArrayEquals(new long[]{mb, depthWise * nIn, 27, 27}, outShape);
    }

    @Test
    public void testSeparableConv2dBasic() {
        int nIn = 3;
        int nOut = 4;
        int kH = 2;
        int kW = 2;

        int mb = 3;
        int imgH = 28;
        int imgW = 28;

        int depthWise = 3;

        SameDiff sd = SameDiff.create();
        INDArray depthWeightArr = Nd4j.create(depthWise, nIn, kH, kW);
        INDArray pointWeightArr = Nd4j.create(nOut, depthWise, 1, 1);

        INDArray bArr = Nd4j.create(1, nOut);
        INDArray inArr = Nd4j.create(mb, nIn, imgH, imgW);

        SDVariable in = sd.var("in", inArr);
        SDVariable dW = sd.var("dW", depthWeightArr);
        SDVariable pW = sd.var("pW", pointWeightArr);
        SDVariable b = sd.var("b", bArr);

        SDVariable[] vars = new SDVariable[]{in, dW, pW, b};

        Conv2DConfig c = Conv2DConfig.builder()
                .kh(kH).kw(kW)
                .ph(0).pw(0)
                .sy(1).sx(1)
                .dh(1).dw(1)
                .isSameMode(false)
                .build();

        SDVariable out = sd.sconv2d(vars, c);
        out = sd.tanh("out", out);

        INDArray outArr = sd.execAndEndResult();
        //Expected output size: out = (in - k + 2*p)/s + 1 = (28-2+0)/1+1 = 27
        val outShape = outArr.shape();
        assertArrayEquals(new long[]{mb, nOut, 27, 27}, outShape);
    }

    @Test
    public void testDeconv2dBasic() {
        int nIn = 3;
        int nOut = 4;
        int kH = 2;
        int kW = 2;

        int mb = 3;
        int imgH = 28;
        int imgW = 28;

        SameDiff sd = SameDiff.create();
        INDArray wArr = Nd4j.create(nOut, nIn, kH, kW);
        INDArray bArr = Nd4j.create(1, nOut);
        INDArray inArr = Nd4j.create(mb, nIn, imgH, imgW);

        SDVariable in = sd.var("in", inArr);
        SDVariable w = sd.var("W", wArr);
        SDVariable b = sd.var("b", bArr);

        SDVariable[] vars = new SDVariable[]{in, w, b};

        DeConv2DConfig deconv = DeConv2DConfig.builder()
                .kX(kH).kY(kW)
                .pX(0).pY(0)
                .sX(1).sY(1)
                .dX(1).dY(1)
                .isSameMode(false)
                .build();

        SDVariable out = sd.deconv2d(vars, deconv);
        out = sd.tanh("out", out);

        INDArray outArr = sd.execAndEndResult();
        //Expected output size: out = (in + k + 2*p)/ s - 1 = (28 + 2+0)/1 - 1 = 29
        val outShape = outArr.shape();
        assertArrayEquals(new long[]{mb, nOut, 29, 29}, outShape);
    }


    @Test
    public void testConv2dBasic() {
        int nIn = 3;
        int nOut = 4;
        int kH = 2;
        int kW = 2;

        int mb = 3;
        int imgH = 28;
        int imgW = 28;

        SameDiff sd = SameDiff.create();
        INDArray wArr = Nd4j.create(nOut, nIn, kH, kW); //As per DL4J
        INDArray bArr = Nd4j.create(1, nOut);
        INDArray inArr = Nd4j.create(mb, nIn, imgH, imgW);

        SDVariable in = sd.var("in", inArr);
        SDVariable w = sd.var("W", wArr);
        SDVariable b = sd.var("b", bArr);

        //Order: https://github.com/deeplearning4j/libnd4j/blob/6c41ea5528bb1f454e92a9da971de87b93ff521f/include/ops/declarable/generic/convo/conv2d.cpp#L20-L22
        //in, w, b - bias is optional
        SDVariable[] vars = new SDVariable[]{in, w, b};

        Conv2DConfig c = Conv2DConfig.builder()
                .kh(kH).kw(kW)
                .ph(0).pw(0)
                .sy(1).sx(1)
                .dh(1).dw(1)
                .isSameMode(false)
                .build();

        SDVariable out = sd.conv2d(vars, c);
        out = sd.tanh("out", out);

        INDArray outArr = sd.execAndEndResult();
        //Expected output size: out = (in - k + 2*p)/s + 1 = (28-2+0)/1+1 = 27
        val outShape = outArr.shape();
        assertArrayEquals(new long[]{mb, nOut, 27, 27}, outShape);
        // sd.execBackwards(); // TODO: test failing here
    }

    @Test
    public void testMaxPooling2dBasic() {
        int nIn = 3;
        int kH = 2;
        int kW = 2;

        int mb = 3;
        int imgH = 28;
        int imgW = 28;

        SameDiff sd = SameDiff.create();
        INDArray inArr = Nd4j.create(mb, nIn, imgH, imgW);

        SDVariable in = sd.var("in", inArr);

        Pooling2DConfig pooling2DConfig = Pooling2DConfig.builder()
                .kh(kH).kw(kW)
                .ph(0).pw(0)
                .sy(1).sx(1)
                .dh(1).dw(1)
                .isSameMode(false)
                .build();

        SDVariable out = sd.maxPooling2d(in, pooling2DConfig);
        out = sd.tanh("out", out);

        INDArray outArr = sd.execAndEndResult();
        val outShape = outArr.shape();
        // oH = (iH - (kH + (kH-1)*(dH-1)) + 2*pH)/sH + 1;
        assertArrayEquals(new long[]{mb, nIn, 27, 27}, outShape);
    }

    @Test
    public void testAvgPooling2dBasic() {
        int nIn = 3;
        int kH = 2;
        int kW = 2;

        int mb = 3;
        int imgH = 28;
        int imgW = 28;

        SameDiff sd = SameDiff.create();
        INDArray inArr = Nd4j.create(mb, nIn, imgH, imgW);

        SDVariable in = sd.var("in", inArr);

        Pooling2DConfig pooling2DConfig = Pooling2DConfig.builder()
                .kh(kH).kw(kW)
                .ph(0).pw(0)
                .sy(1).sx(1)
                .dh(1).dw(1)
                .isSameMode(false)
                .build();

        SDVariable out = sd.avgPooling2d(in, pooling2DConfig);
        out = sd.tanh("out", out);

        INDArray outArr = sd.execAndEndResult();
        val outShape = outArr.shape();
        // oH = (iH - (kH + (kH-1)*(dH-1)) + 2*pH)/sH + 1;
        assertArrayEquals(new long[]{mb, nIn, 27, 27}, outShape);
    }

    @Test
    public void testAvgPooling3dBasic() {
        int nIn = 3;
        int kH = 2;
        int kW = 2;
        int kD = 2;

        int mb = 3;
        int imgH = 28;
        int imgW = 28;
        int imgD = 28;

        SameDiff sd = SameDiff.create();
        INDArray inArr = Nd4j.create(mb, nIn, imgD, imgH, imgW);

        SDVariable in = sd.var("in", inArr);


        SDVariable[] vars = new SDVariable[]{in};

        Pooling3DConfig pooling3DConfig = Pooling3DConfig.builder()
                .kH(kH).kW(kW).kT(kD)
                .pH(0).pH(0).pT(0)
                .sH(1).sW(1).sT(1)
                .dilationH(0).dilationW(0).dilationT(0)
                .ceilingMode(false)
                .build();

        SDVariable out = sd.avgPooling3d(vars, pooling3DConfig);
        out = sd.tanh("out", out);

        INDArray outArr = sd.execAndEndResult();
        val outShape = outArr.shape();
        // oH = (iH - (kH + (kH-1)*(dH-1)) + 2*pH)/sH + 1;
        assertArrayEquals(new long[]{mb, nIn, 27, 27, 27}, outShape);
    }

    @Test
    public void testMaxPooling3dBasic() {
        int nIn = 3;
        int kH = 2;
        int kW = 2;
        int kD = 2;

        int mb = 3;
        int imgH = 28;
        int imgW = 28;
        int imgD = 28;

        SameDiff sd = SameDiff.create();
        INDArray inArr = Nd4j.create(mb, nIn, imgD, imgH, imgW);

        SDVariable in = sd.var("in", inArr);


        SDVariable[] vars = new SDVariable[]{in};

        Pooling3DConfig pooling3DConfig = Pooling3DConfig.builder()
                .kH(kH).kW(kW).kT(kD)
                .pH(0).pH(0).pT(0)
                .sH(1).sW(1).sT(1)
                .dilationH(0).dilationW(0).dilationT(0)
                .ceilingMode(false)
                .build();

        SDVariable out = sd.maxPooling3d(vars, pooling3DConfig);
        out = sd.tanh("out", out);

        INDArray outArr = sd.execAndEndResult();
        val outShape = outArr.shape();
        // oH = (iH - (kH + (kH-1)*(dH-1)) + 2*pH)/sH + 1;
        assertArrayEquals(new long[]{mb, nIn, 27, 27, 27}, outShape);
    }

    @Test
    public void testConv1dBasic() {
        int nIn = 3;
        int nOut = 4;
        int k = 2;
        int mb = 3;
        int img = 28;

        SameDiff sd = SameDiff.create();
        INDArray wArr = Nd4j.create(nOut, nIn, k);
        INDArray inArr = Nd4j.create(mb, nIn, img);

        SDVariable in = sd.var("in", inArr);
        SDVariable w = sd.var("W", wArr);

        SDVariable[] vars = new SDVariable[]{in, w};

        Conv1DConfig conv1DConfig = Conv1DConfig.builder()
                .k(k).p(0).s(1)
                .isSameMode(false)
                .build();

        SDVariable out = sd.conv1d(vars, conv1DConfig);
        out = sd.tanh("out", out);

        INDArray outArr = sd.execAndEndResult();
        INDArray iOut = out.getArr();
        //Expected output size: out = (in - k + 2*p)/s + 1 = (28-2+0)/1+1 = 27
        val outShape = outArr.shape();
        assertArrayEquals(new long[]{mb, nOut, 27}, outShape);
    }

    @Test
    public void validateMeanDiff() {
        Nd4j.getRandom().setSeed(12345);

        INDArray arr = Nd4j.rand(3, 4);

        SameDiff sd = SameDiff.create();
        SDVariable v = sd.var("in", arr);
        SDVariable mean = sd.mean("mean", v);

        INDArray out = sd.execAndEndResult();
        assertEquals(out, arr.mean(Integer.MAX_VALUE));

        sd.execBackwards();
        INDArray dLdIn = sd.grad("in").getArr();

        //If L = mean(in)
        //then dL/dIn = 1/N

        assertEquals(Nd4j.valueArrayOf(arr.shape(), 1.0 / arr.length()), dLdIn);
    }

    @Test
    public void validateSumDiff() {
        Nd4j.getRandom().setSeed(12345);

        INDArray arr = Nd4j.rand(3, 4);

        SameDiff sd = SameDiff.create();
        SDVariable v = sd.var("in", arr);
        SDVariable mean = sd.sum("sum", v);

        INDArray out = sd.execAndEndResult();
        assertEquals(out, arr.sum(Integer.MAX_VALUE));

        sd.execBackwards();
        INDArray dLdIn = sd.grad("in").getArr();

        //If L = sum(in)
        //then dL/dIn = 1

        assertEquals(Nd4j.ones(arr.shape()), dLdIn);
    }

    @Test
    public void validateStdevDiff() {
        for (boolean biasCorrected : new boolean[]{true, false}) {
            Nd4j.getRandom().setSeed(12345);

            INDArray arr = Nd4j.rand(3, 4);

            SameDiff sd = SameDiff.create();
            SDVariable v = sd.var("in", arr);
            SDVariable stdev = sd.standardDeviation("stdev", v, biasCorrected);

            INDArray out = sd.execAndEndResult();
            assertEquals(out, arr.std(biasCorrected, Integer.MAX_VALUE));

            sd.execBackwards();
            INDArray dLdIn = sd.grad("in").getArr();

            //If L = stdev(in)
            //then dL/dIn = (in-mean) / (s*(N-1))
            // or /N for non-bias corrected

            double m = arr.meanNumber().doubleValue();
            double s = arr.stdNumber(biasCorrected).doubleValue();
            INDArray exp = arr.sub(m).div(s);
            exp.divi(biasCorrected ? arr.length() - 1 : arr.length());

            assertEquals(exp, dLdIn);
        }
    }

    @Test
    public void validateVarDiff() {
        for (boolean biasCorrected : new boolean[]{true, false}) {
            Nd4j.getRandom().setSeed(12345);

            INDArray arr = Nd4j.rand(3, 4);

            SameDiff sd = SameDiff.create();
            SDVariable v = sd.var("in", arr);
            SDVariable var = sd.variance("var", v, biasCorrected);

            INDArray out = sd.execAndEndResult();
            assertEquals(out, arr.var(biasCorrected, Integer.MAX_VALUE));

            sd.execBackwards();
            INDArray dLdIn = sd.grad("in").getArr();

            //If L = var(in)
            //then dL/dIn = 2/(N-1) * (in-mean)
            // or /N for non-bias corrected

            double m = arr.meanNumber().doubleValue();
            INDArray exp = arr.sub(m).mul(2);
            exp.divi(biasCorrected ? arr.length() - 1 : arr.length());

            assertEquals(exp, dLdIn);
        }
    }

    @Test
    public void validateMinDiff() {
        Nd4j.getRandom().setSeed(12345);

        INDArray arr = Nd4j.rand(3, 4);

        SameDiff sd = SameDiff.create();
        SDVariable v = sd.var("in", arr);
        SDVariable min = sd.min("min", v);

        INDArray out = sd.execAndEndResult();
        assertEquals(out, arr.min(Integer.MAX_VALUE));

        sd.execBackwards();
        INDArray dLdIn = sd.grad("in").getArr();

        //If L = min(in)
        //then dL/dIn = 1 if in_i == min(in) or 0 otherwise

        //Note that we don't have an "IsMin" op, so use IsMax(neg(in)) which is equivalent
        INDArray exp = Nd4j.getExecutioner().execAndReturn(new IsMax(arr.neg()));

        assertEquals(exp, dLdIn);
    }

    @Test
    public void validateMaxDiff() {
        Nd4j.getRandom().setSeed(12345);

        INDArray arr = Nd4j.rand(3, 4);

        SameDiff sd = SameDiff.create();
        SDVariable v = sd.var("in", arr);
        SDVariable min = sd.max("max", v);

        INDArray out = sd.execAndEndResult();
        assertEquals(out, arr.max(Integer.MAX_VALUE));

        sd.execBackwards();
        INDArray dLdIn = sd.grad("in").getArr();

        //If L = max(in)
        //then dL/dIn = 1 if in_i == max(in) or 0 otherwise

        INDArray exp = Nd4j.getExecutioner().execAndReturn(new IsMax(arr.dup()));

        assertEquals(exp, dLdIn);
    }

    @Test
    public void validateProdDiff() {
        Nd4j.getRandom().setSeed(12345);

        INDArray arr = Nd4j.rand(3, 4);

        SameDiff sd = SameDiff.create();
        SDVariable v = sd.var("in", arr);
        SDVariable prod = sd.prod("prod", v);

        double p = arr.prodNumber().doubleValue();
        INDArray out = sd.execAndEndResult();
        assertEquals(out, arr.prod(Integer.MAX_VALUE));

        sd.execBackwards();
        INDArray dLdIn = sd.grad("in").getArr();

        //If L = prod(in)
        //then dL/dIn = prod(in) / in       i.e., product of input *excluding* in_i as (d/dx(xyzabc) = yzabc

        INDArray exp = arr.rdiv(p);
        assertEquals(exp, dLdIn);
    }

    @Test
    public void testSquare() {
        Nd4j.getRandom().setSeed(12345);

        int mb = 5;
        int nOut = 4;

        SameDiff sd = SameDiff.create();
        SDVariable in = sd.var("in", Nd4j.rand(mb, nOut));
        SDVariable label = sd.var("label", Nd4j.rand(mb, nOut));
        SDVariable diff = in.sub(label);
        SDVariable sqDiff = sd.square(diff);

        INDArray expOut = in.getArr().sub(label.getArr());
        expOut.muli(expOut);

        System.out.println("About to exec");
        INDArray out = sd.execAndEndResult();   //JVM crash

        assertEquals(out, expOut);
    }


    @Test
    public void testExpandDims() {
        for (int i = 0; i <= 2; i++) {
            SameDiff sd = SameDiff.create();
            SDVariable in = sd.var("in", Nd4j.create(2, 3));
            SDVariable expanded = sd.f().expandDims(in, i);

            INDArray out = sd.execAndEndResult();
            switch (i) {
                case 0:
                    assertArrayEquals(new long[]{1, 2, 3}, out.shape());
                    break;
                case 1:
                    assertArrayEquals(new long[]{2, 1, 3}, out.shape());
                    break;
                case 2:
                    assertArrayEquals(new long[]{2, 3, 1}, out.shape());
                    break;
                default:
                    throw new RuntimeException();
            }
        }
    }

    @Test
    public void testZerosLike() {
        SameDiff sd = SameDiff.create();
        SDVariable var0 = sd.var("in", new long[]{3, 4});
        SDVariable out = sd.zerosLike("out", var0);

        INDArray out1 = sd.execAndEndResult();
        assertEquals(Nd4j.zeros(3, 4), out1);

        sd.associateArrayWithVariable(Nd4j.create(4, 5), var0);
        INDArray out2 = sd.execAndEndResult();
        assertEquals(Nd4j.zeros(4, 5), out2);
    }

    @Test
    public void testOnesLike() {
        SameDiff sd = SameDiff.create();
        SDVariable var0 = sd.var("in", new long[]{3, 4});
        SDVariable out = sd.onesLike("out", var0);

        INDArray out1 = sd.execAndEndResult();
        assertEquals(Nd4j.ones(3, 4), out1);

        sd.associateArrayWithVariable(Nd4j.create(4, 5), var0);
        INDArray out2 = sd.execAndEndResult();
        assertEquals(Nd4j.ones(4, 5), out2);
    }

    @Test
    public void testOneHot() {
        //indices = [[0, 2], [1, -1]]
        INDArray indicesArr = Nd4j.zeros(2, 2);
        indicesArr.put(0, 1, 2);
        indicesArr.put(1, 0, 1);
        indicesArr.put(1, 1, -1);
        INDArray expectedOut = Nd4j.zeros(new long[]{2, 2, 3});
        /*
        # output: [2 x 2 x 3]
        # [[[1.0, 0.0, 0.0],   # one_hot(0)
        #   [0.0, 0.0, 1.0]],  # one_hot(2)
        #  [[0.0, 1.0, 0.0],   # one_hot(1)
        #   [0.0, 0.0, 0.0]]]  # one_hot(-1)
        */
        expectedOut.putScalar(0, 0, 0, 1.0);
        expectedOut.putScalar(0, 1, 2, 1.0);
        expectedOut.putScalar(1, 0, 1, 1.0);

        SameDiff sd = SameDiff.create();
        SDVariable indices = sd.var("indices", new long[]{2, 2});
        sd.associateArrayWithVariable(indicesArr, indices);
        INDArray out1 = sd.execAndEndResult();
        log.info(out1.toString());
        assertEquals(expectedOut, out1);

    }

    @Test
    public void testOnesLikeBackprop() {
        SameDiff sd = SameDiff.create();
        SDVariable var0 = sd.var("in", new long[]{3, 4});
        SDVariable ones = sd.onesLike("ones", var0);
        SDVariable out = sd.sum("oun", ones);

        INDArray outArr = sd.execAndEndResult();
        assertEquals(Nd4j.valueArrayOf(1, 12.0), outArr);

        sd.execBackwards();

        assertEquals(Nd4j.create(3, 4), sd.grad("in").getArr());
    }


    @Test
    public void testReduce3() {

        Nd4j.getRandom().setSeed(12345);

        int d0 = 3;
        int d1 = 4;
        int d2 = 5;

        for (val reduceDims : new int[][]{{Integer.MAX_VALUE}, {0, 1, 2}, {0}, {1}, {2}, {0, 1}, {0, 2}, {1, 2}}) {
            for (int i = 0; i < 6; i++) {

                SameDiff sd = SameDiff.create();
                sd.setLogExecution(false);

                INDArray a = Nd4j.rand(new long[]{d0, d1, d2});
                INDArray b = Nd4j.rand(new long[]{d0, d1, d2});


                SDVariable in = sd.var("in", a);
                SDVariable in2 = sd.var("in2", b);

                INDArray expOut;
                SDVariable reduced;
                String name;
                switch (i) {
                    case 0:
                        reduced = sd.manhattanDistance(in, in2, reduceDims);
                        name = "manhattan";
                        expOut = Nd4j.getExecutioner().exec(new ManhattanDistance(a, b), reduceDims);
                        break;
                    case 1:
                        reduced = sd.euclideanDistance(in, in2, reduceDims);
                        name = "euclidean";
                        expOut = Nd4j.getExecutioner().exec(new EuclideanDistance(a, b), reduceDims);
                        break;
                    case 2:
                        reduced = sd.cosineSimilarity(in, in2, reduceDims);
                        name = "cosine";
                        expOut = Nd4j.getExecutioner().exec(new CosineSimilarity(a, b), reduceDims);
                        break;
                    case 3:
                        reduced = sd.jaccardDistance(in, in2, reduceDims);
                        name = "jaccard";
                        expOut = Nd4j.getExecutioner().exec(new JaccardDistance(a, b), reduceDims);
                        break;
                    case 4:
                        reduced = sd.hammingDistance(in, in2, reduceDims);
                        name = "hamming";
                        expOut = Nd4j.getExecutioner().exec(new HammingDistance(a, b), reduceDims);
                        break;
                    case 5:
                        reduced = sd.cosineDistance(in, in2, reduceDims);
                        name = "reduced";
                        expOut = Nd4j.getExecutioner().exec(new CosineDistance(a, b), reduceDims);
                        break;
                    default:
                        throw new RuntimeException();
                }

                long[] expShape;
                if (Arrays.equals(new int[]{0}, reduceDims)) {
                    expShape = new long[]{4, 5};
                } else if (Arrays.equals(new int[]{1}, reduceDims)) {
                    expShape = new long[]{3, 5};
                } else if (Arrays.equals(new int[]{2}, reduceDims)) {
                    expShape = new long[]{3, 4};
                } else if (Arrays.equals(new int[]{Integer.MAX_VALUE}, reduceDims)) {
                    expShape = new long[]{1, 1};
                } else if (Arrays.equals(new int[]{0, 1}, reduceDims)) {
                    expShape = new long[]{1, 5};
                } else if (Arrays.equals(new int[]{0, 2}, reduceDims)) {
                    expShape = new long[]{1, 4};
                } else if (Arrays.equals(new int[]{1, 2}, reduceDims)) {
                    expShape = new long[]{3, 1};
                } else if (Arrays.equals(new int[]{0, 1, 2}, reduceDims)) {
                    expShape = new long[]{1, 1};
                } else {
                    throw new RuntimeException();
                }

                String msg = name + " - dims=" + Arrays.toString(reduceDims);

                INDArray out = sd.execAndEndResult();

                log.info(msg + " - expected shape: " + Arrays.toString(expShape) + ", out=" + Arrays.toString(out.shape())
                        + ", outExp=" + Arrays.toString(expOut.shape()));

                assertArrayEquals(msg, expShape, out.shape());
                assertArrayEquals(msg, expShape, expOut.shape());

                assertEquals(msg, out, expOut);
            }
        }
    }


    @Test
    public void testManhattanAlongDim0() {
        Nd4j.getRandom().setSeed(12345);

        INDArray a = Nd4j.rand(new long[]{3, 4, 5});
        INDArray b = Nd4j.rand(new long[]{3, 4, 5});

        INDArray expOut = Nd4j.getExecutioner().exec(new ManhattanDistance(a, b), 0);

        val expShape = new long[]{4, 5};

        assertArrayEquals(expShape, expOut.shape());
    }


    @Test
    public void testJaccardDistance() {
        Nd4j.getRandom().setSeed(12345);

        INDArray a = Nd4j.rand(new long[]{3, 4}).addi(0.1);
        INDArray b = Nd4j.rand(new long[]{3, 4}).addi(0.1);

        SameDiff sd = SameDiff.create();
        SDVariable in1 = sd.var("in1", a);
        SDVariable in2 = sd.var("in2", b);

        SDVariable jaccard = sd.jaccardDistance("out", in1, in2);

        INDArray min = Transforms.min(a, b);
        INDArray max = Transforms.max(a, b);

        double minSum = min.sumNumber().doubleValue();
        double maxSum = max.sumNumber().doubleValue();
        double jd = 1.0 - minSum / maxSum;

        INDArray out = sd.execAndEndResult();
        assertEquals(1, out.length());

        assertEquals(jd, out.getDouble(0), 1e-6);
    }


    @Test
    public void testSlice2d() {
        INDArray inArr = Nd4j.linspace(1, 12, 12).reshape('c', 3, 4);

        SameDiff sd = SameDiff.create();
        SDVariable in = sd.var("in", inArr);
        SDVariable slice_full = sd.slice(in, new int[]{0, 0}, new int[]{3, 4});
        SDVariable subPart = sd.slice(in, new int[]{1, 2}, new int[]{2, 2});

        sd.execAndEndResult();

        assertEquals(inArr, slice_full.getArr());
        assertEquals(inArr.get(interval(1, 3), interval(2, 4)), subPart.getArr());
    }


    @Test
    public void testSlice3d() {
        INDArray inArr = Nd4j.linspace(1, 60, 60).reshape('c', 3, 4, 5);

        SameDiff sd = SameDiff.create();
        SDVariable in = sd.var("in", inArr);
        SDVariable slice_full = sd.slice(in, new int[]{0, 0, 0}, new int[]{3, 4, 5});
        SDVariable subPart = sd.slice(in, new int[]{1, 2, 3}, new int[]{2, 2, 1});

        sd.execAndEndResult();

        assertEquals(inArr, slice_full.getArr());
        assertEquals(inArr.get(interval(1, 3), interval(2, 4), interval(3, 4)), subPart.getArr());
    }

    @Test
    public void testStridedSlice2dBasic() {
        INDArray inArr = Nd4j.linspace(1, 12, 12).reshape('c', 3, 4);

        SameDiff sd = SameDiff.create();
        SDVariable in = sd.var("in", inArr);
        SDVariable slice_full = sd.stridedSlice(in, new int[]{0, 0}, new int[]{3, 4}, new int[]{1, 1});
        SDVariable subPart = sd.stridedSlice(in, new int[]{1, 2}, new int[]{3, 4}, new int[]{1, 1});
        SDVariable subPart2 = sd.stridedSlice(in, new int[]{0, 0}, new int[]{4, 5}, new int[]{2, 2});

        sd.execAndEndResult();

        assertEquals(inArr, slice_full.getArr());
        assertEquals(inArr.get(interval(1, 3), interval(2, 4)), subPart.getArr());
        assertEquals(inArr.get(interval(0, 2, 4), interval(0, 2, 5)), subPart2.getArr());
    }


    @Test
    public void testStridedSliceBeginEndMask() {
        INDArray inArr = Nd4j.linspace(1, 12, 12).reshape('c', 3, 4);

        SameDiff sd = SameDiff.create();
        SDVariable in = sd.var("in", inArr);
        SDVariable slice1 = sd.stridedSlice(in, new int[]{-999, 0}, new int[]{2, 4}, new int[]{1, 1}, 1 << 1, 0, 0, 0, 0);
        SDVariable slice2 = sd.stridedSlice(in, new int[]{1, 0}, new int[]{-999, 4}, new int[]{1, 1}, 0, 1, 0, 0, 0);

        sd.execAndEndResult();

        assertEquals(inArr.get(NDArrayIndex.interval(0, 2), NDArrayIndex.all()), slice1.getArr());
        assertEquals(inArr.get(NDArrayIndex.interval(1, 3), NDArrayIndex.all()), slice2.getArr());
    }

    @Test
    public void testStridedSliceEllipsisMask() {
        INDArray inArr = Nd4j.linspace(1, 60, 60).reshape('c', 3, 4, 5);
        SameDiff sd = SameDiff.create();
        SDVariable in = sd.var("in", inArr);

        //[1:3,...] -> [1:3,:,:]
        SDVariable slice = sd.stridedSlice(in, new int[]{1}, new int[]{3}, new int[]{1}, 0, 0, 1 << 1, 0, 0);
        //[1:3,...,1:4] -> [1:3,:,1:4]
        SDVariable slice2 = sd.stridedSlice(in, new int[]{1, 1}, new int[]{3, 4}, new int[]{1, 1}, 0, 0, 1 << 1, 0, 0);

        sd.execAndEndResult();

        assertEquals(inArr.get(interval(1, 3), all(), all()), slice.getArr());
        assertEquals(inArr.get(interval(1, 3), all(), all()), slice2.getArr());
    }

    @Test
    public void testStridedSliceNewAxisMask() {
        INDArray inArr = Nd4j.linspace(1, 60, 60).reshape('c', 3, 4, 5);
        SameDiff sd = SameDiff.create();
        SDVariable in = sd.var("in", inArr);
        SDVariable slice = sd.stridedSlice(in, new int[]{-999, 0, 0, 0}, new int[]{-999, 3, 4, 5}, new int[]{-999, 1, 1, 1}, 0, 0, 0, 1, 0);

        INDArray out = sd.execAndEndResult();

        assertArrayEquals(new long[]{1, 3, 4, 5}, inArr.shape());
        assertEquals(inArr, out.get(point(0), all(), all(), all()));
    }

    @Test
    public void testStridedSliceNewAxisMask2() {
        INDArray inArr = Nd4j.linspace(1, 60, 60).reshape('c', 3, 4, 5);
        SameDiff sd = SameDiff.create();
        SDVariable in = sd.var("in", inArr);
        SDVariable slice = sd.stridedSlice(in, new int[]{1, 1, -999, 1}, new int[]{3, 3, -999, 4}, new int[]{1, 1, -999, 1}, 0, 0, 0, 1 << 2, 0);
        INDArray out = sd.execAndEndResult();

        assertArrayEquals(new long[]{2, 2, 1, 3}, slice.getArr().shape());
    }

    @Test
    public void testStridedSliceShrinkAxisMask() {

        INDArray inArr = Nd4j.linspace(1, 60, 60).reshape('c', 3, 4, 5);
        SameDiff sd = SameDiff.create();
        SDVariable in = sd.var("in", inArr);
        SDVariable slice = sd.stridedSlice(in, new int[]{0, 0, 0}, new int[]{-999, 4, 5}, new int[]{1, 1, 1}, 0, 0, 0, 0, 1);
        SDVariable slice2 = sd.stridedSlice(in, new int[]{2, 0, 0}, new int[]{-999, 4, 5}, new int[]{1, 1, 1}, 0, 0, 0, 0, 1);
        SDVariable slice3 = sd.stridedSlice(in, new int[]{1, 2, 1}, new int[]{-999, -999, 5}, new int[]{1, 1, 1}, 0, 0, 0, 0, 1 | 1 << 1);

        sd.execAndEndResult();

        assertEquals(inArr.get(point(0), all(), all()), slice.getArr());
        assertEquals(inArr.get(point(2), all(), all()), slice2.getArr());
        assertEquals(inArr.get(point(1), point(2), interval(1, 5)), slice3.getArr());
    }

    @Test
    public void testPairwiseBooleanTransforms() {
        /*
        eq, neq, gt, lt, gte, lte, or, and, xor
         */
        //Test transforms (pairwise)
        Nd4j.getRandom().setSeed(12345);

        for (int i = 0; i < 11; i++) {
            SameDiff sd = SameDiff.create();

            int nOut = 4;
            int minibatch = 5;

            INDArray ia = Nd4j.randn(minibatch, nOut);
            INDArray ib = Nd4j.randn(minibatch, nOut);

            SDVariable in1 = sd.var("in1", ia);
            SDVariable in2 = sd.var("in2", ib);


            SDVariable t;
            INDArray expOut;
            switch (i) {
                case 0:
                    t = sd.eq(in1, in2);
                    expOut = ia.eq(ib);
                    break;
                case 1:
                    t = sd.neq(in1, in2);
                    expOut = ia.neq(ib);
                    break;
                case 2:
                    t = sd.gt(in1, in2);
                    expOut = ia.gt(ib);
                    break;
                case 3:
                    t = sd.lt(in1, in2);
                    expOut = ia.lt(ib);
                    break;
                case 4:
                    t = sd.gte(in1, in2);
                    expOut = ia.dup();
                    Nd4j.getExecutioner().exec(new GreaterThanOrEqual(new INDArray[]{ia, ib}, new INDArray[]{expOut}));
                    break;
                case 5:
                    t = sd.lte(in1, in2);
                    expOut = ia.dup();
                    Nd4j.getExecutioner().exec(new LessThanOrEqual(new INDArray[]{ia, ib}, new INDArray[]{expOut}));
                    break;
                case 6:
                    ia = Nd4j.getExecutioner().exec(new BernoulliDistribution(ia, 0.5));
                    ib = Nd4j.getExecutioner().exec(new BernoulliDistribution(ib, 0.5));
                    t = sd.or(in1, in2);
                    expOut = Transforms.or(ia, ib);
                    break;
                case 7:
                    t = sd.max(in1, in2);
                    expOut = Nd4j.getExecutioner().execAndReturn(new OldMax(ia, ib, ia.dup(), ia.length()));
                    break;
                case 8:
                    t = sd.min(in1, in2);
                    expOut = Nd4j.getExecutioner().execAndReturn(new OldMin(ia, ib, ia.dup(), ia.length()));
                    break;
                case 9:
                    ia = Nd4j.getExecutioner().exec(new BernoulliDistribution(ia, 0.5));
                    ib = Nd4j.getExecutioner().exec(new BernoulliDistribution(ib, 0.5));
                    t = sd.and(in1, in2);
                    expOut = Transforms.and(ia, ib);
                    break;
                case 10:
                    ia = Nd4j.getExecutioner().exec(new BernoulliDistribution(ia, 0.5));
                    ib = Nd4j.getExecutioner().exec(new BernoulliDistribution(ib, 0.5));
                    t = sd.xor(in1, in2);
                    expOut = Transforms.xor(ia, ib);
                    break;
                default:
                    throw new RuntimeException();
            }

            log.info("Executing: " + i);
            INDArray out = sd.execAndEndResult();

            assertEquals(expOut, out);
        }
    }

    @Test
    public void testBooleanChecks() {
        /*
        isNonDecreasing,
         */
        Nd4j.getRandom().setSeed(12345);

        for (int i = 0; i < 3; i++) {
            SameDiff sd = SameDiff.create();

            int nOut = 4;
            int minibatch = 5;

            INDArray ia = Nd4j.randn(minibatch, nOut);

            SDVariable in1 = sd.var("in1", ia);
            INDArray expOut = Nd4j.create(new float[]{1});
            SDVariable t;

            switch (i) {
                case 0:
                    t = sd.isNonDecreasing(in1);
                    Nd4j.getExecutioner().exec(new IsNonDecreasing(new INDArray[]{ia}, new INDArray[]{expOut}));
                    break;
                case 1:
                    t = sd.isStrictlyIncreasing(in1);
                    Nd4j.getExecutioner().exec(new IsStrictlyIncreasing(new INDArray[]{ia}, new INDArray[]{expOut}));
                    break;
                case 2:
                    t = sd.isNumericTensor(in1);
                    Nd4j.getExecutioner().exec(new IsNumericTensor(new INDArray[]{ia}, new INDArray[]{expOut}));
                    break;
                default:
                    throw new RuntimeException();
            }

            log.info("Executing: " + i);
            INDArray out = sd.execAndEndResult();

            assertEquals(expOut, out);
        }
    }

    @Test
    public void testExpandDims2d() {
        val origShape = new long[]{3, 4};

        for (int i = 0; i < 3; i++) {
            for (Pair<INDArray, String> p : NDArrayCreationUtil.getAllTestMatricesWithShape(origShape[0], origShape[1], 12345)) {
                INDArray inArr = p.getFirst().muli(100);

                SameDiff sd = SameDiff.create();
                SDVariable in = sd.var("in", inArr);
                SDVariable expand = sd.f().expandDims(in, i);

                INDArray out = sd.execAndEndResult();

                INDArray expOut;
                switch (i) {
                    case 0:
                        expOut = inArr.dup('c').reshape('c', 1, origShape[0], origShape[1]);
                        break;
                    case 1:
                        expOut = inArr.dup('c').reshape('c', origShape[0], 1, origShape[1]);
                        break;
                    case 2:
                        expOut = inArr.dup('c').reshape('c', origShape[0], origShape[1], 1);
                        break;
                    default:
                        throw new RuntimeException();
                }

                String msg = "expandDim=" + i + ", source=" + p.getSecond();

                assertEquals(msg, out, expOut);
            }
        }
    }

    @Test
    public void testSqueezeDims() {
        val origShape = new long[]{3, 4, 5};

        for (int i = 0; i < 3; i++) {

            val shape = origShape.clone();
            shape[i] = 1;

            for (Pair<INDArray, String> p : NDArrayCreationUtil.getAll3dTestArraysWithShape(12345, shape)) {
                INDArray inArr = p.getFirst().muli(100);

                SameDiff sd = SameDiff.create();
                SDVariable in = sd.var("in", inArr);
                SDVariable squeeze = sd.f().squeeze(in, i);

                INDArray out = sd.execAndEndResult();

                INDArray expOut;
                switch (i) {
                    case 0:
                        expOut = inArr.dup('c').reshape('c', origShape[1], origShape[2]);
                        break;
                    case 1:
                        expOut = inArr.dup('c').reshape('c', origShape[0], origShape[2]);
                        break;
                    case 2:
                        expOut = inArr.dup('c').reshape('c', origShape[0], origShape[1]);
                        break;
                    default:
                        throw new RuntimeException();
                }

                String msg = "squeezeDim=" + i + ", source=" + p.getSecond();

                assertEquals(msg, out, expOut);
            }
        }
    }

    @Test
    public void testExpandSqueezeChain() {

        val origShape = new long[]{3, 4};

        for (int i = 0; i < 3; i++) {
            for (Pair<INDArray, String> p : NDArrayCreationUtil.getAllTestMatricesWithShape(origShape[0], origShape[1], 12345)) {
                INDArray inArr = p.getFirst().muli(100);

                SameDiff sd = SameDiff.create();
                SDVariable in = sd.var("in", inArr);
                SDVariable expand = sd.expandDims(in, i);
                SDVariable squeeze = sd.squeeze(expand, i);

                INDArray out = sd.execAndEndResult();

                String msg = "expand/Squeeze=" + i + ", source=" + p.getSecond();

                assertEquals(msg, out, inArr);  //expand -> squeeze: should be opposite ops
            }
        }
    }

    @Test
    public void testSqueezeExpandChain() {

        val origShape = new long[]{3, 4, 5};

        for (int i = 0; i < 3; i++) {

            val shape = origShape.clone();
            shape[i] = 1;

            for (Pair<INDArray, String> p : NDArrayCreationUtil.getAll3dTestArraysWithShape(12345, shape)) {
                INDArray inArr = p.getFirst().muli(100);

                SameDiff sd = SameDiff.create();
                SDVariable in = sd.var("in", inArr);
                SDVariable squeeze = sd.squeeze(in, i);
                SDVariable expand = sd.expandDims(squeeze, i);

                INDArray out = sd.execAndEndResult();

                String msg = "expand/Squeeze=" + i + ", source=" + p.getSecond();

                assertEquals(msg, out, inArr);  //squeeze -> expand: should be opposite ops
            }
        }
    }

    @Test
    public void testConfusionMatrix() {
        INDArray labels = Nd4j.create(new float[]{1, 2, 4});
        INDArray pred = Nd4j.create(new float[]{2, 2, 4});
        INDArray weights = Nd4j.create(new float[]{10, 100, 1000});
        Integer numClasses = 5;
        SameDiff sd = SameDiff.create();
        SDVariable labelsVar = sd.var("labels", labels);
        SDVariable predictionsVar = sd.var("predictions", pred);
        SDVariable weightsVar = sd.var("weights", weights);
        sd.confusionMatrix(labelsVar, predictionsVar, numClasses, weightsVar);
        INDArray out = sd.execAndEndResult();

        INDArray exp = Nd4j.create(new float[][]{{0, 0, 0, 0, 0}, {0, 0, 10, 0, 0}, {0, 0, 100, 0, 0},
                {0, 0, 0, 0, 0}, {0, 0, 0, 0, 1000}});

        assertEquals(exp, out);
    }

    @Test
    public void testArgMax() {
        Nd4j.getRandom().setSeed(12345);

        for (val dim : new int[][]{{0}, {1}, {Integer.MAX_VALUE}, {0, 1}, {}}) {
            INDArray inArr = Nd4j.rand(3, 4);
            SameDiff sd = SameDiff.create();

            SDVariable in = sd.var("in", inArr);
            SDVariable argmax = sd.argmax("argmax", in, dim);

            INDArray out = sd.execAndEndResult();

            INDArray exp = Nd4j.argMax(inArr, dim);

            assertEquals(exp, out);
        }
    }

    @Test
    public void testArgMin() {

        Nd4j.getRandom().setSeed(12345);

        for (val dim : new int[][]{{0}, {1}, {Integer.MAX_VALUE}, {0, 1}, {}}) {
            INDArray inArr = Nd4j.rand(3, 4);
            SameDiff sd = SameDiff.create();

            SDVariable in = sd.var("in", inArr);
            SDVariable argmin = sd.argmin("argmin", in, dim);

            INDArray out = sd.execAndEndResult();

            INDArray exp = Nd4j.argMax(inArr.neg(), dim);   //argmin(x) == argmax(-x)

            assertEquals(exp, out);
        }
    }

    @Test
    public void testScatterAdd() {
        INDArray arr1 = Nd4j.zeros(3, 3);
        INDArray arr2 = Nd4j.create(new float[]{0,1}, new long[]{2});
        INDArray arr3 = Nd4j.ones(3, 3);
        INDArray expected = Nd4j.create(new float[]{1, 1, 1,
                                                    1, 1, 1,
                                                    0, 0, 0},
                                            new long[]{3, 3});

        SameDiff sd  = SameDiff.create();
        SDVariable refs = sd.var("refs", arr1);
        SDVariable idxs = sd.var("idxs", arr2);
        SDVariable upds = sd.var("upds", arr3);

        SDVariable result = sd.scatterAdd(refs, idxs, upds);
        assertArrayEquals(new long[]{3, 3}, result.eval().shape());
        assertEquals(expected, result.eval());

    }

    @Test
    public void testScatterMul() {
        INDArray arr1 = Nd4j.ones(3, 3);
        INDArray arr2 = Nd4j.create(new float[]{0,1}, new long[]{2});
        INDArray arr3 = Nd4j.zeros(3, 3);
        INDArray expected = Nd4j.create(new float[]{0, 0, 0,
                                                    0, 0, 0,
                                                    1, 1, 1},
                                            new long[]{3, 3});

        SameDiff sd  = SameDiff.create();
        SDVariable refs = sd.var("refs", arr1);
        SDVariable idxs = sd.var("idxs", arr2);
        SDVariable upds = sd.var("upds", arr3);

        SDVariable result = sd.scatterMul(refs, idxs, upds);
        assertArrayEquals(new long[]{3, 3}, result.eval().shape());
        assertEquals(expected, result.eval());

    }

    @Test
    public void testScatterSub() {
        INDArray arr1 = Nd4j.ones(3, 3);
        INDArray arr2 = Nd4j.create(new float[]{0,1}, new long[]{2});
        INDArray arr3 = Nd4j.ones(3, 3);
        INDArray expected = Nd4j.create(new float[]{0, 0, 0,
                                                    0, 0, 0,
                                                    1, 1, 1},
                                            new long[]{3, 3});

        SameDiff sd  = SameDiff.create();
        SDVariable refs = sd.var("refs", arr1);
        SDVariable idxs = sd.var("idxs", arr2);
        SDVariable upds = sd.var("upds", arr3);

        SDVariable result = sd.scatterSub(refs, idxs, upds);
        assertArrayEquals(new long[]{3, 3}, result.eval().shape());
        assertEquals(expected, result.eval());

    }

    @Test
    public void testScatterDiv() {
        INDArray arr1 = Nd4j.ones(3, 3);
        INDArray arr2 = Nd4j.create(new float[]{0,1}, new long[]{2});
        INDArray arr3 = Nd4j.ones(3, 3).assign(2);
        INDArray expected = Nd4j.create(new float[]{0.5f, 0.5f, 0.5f,
                                                    0.5f, 0.5f, 0.5f,
                                                    1.0f, 1.0f, 1.0f},
                                            new long[]{3, 3});

        SameDiff sd  = SameDiff.create();
        SDVariable refs = sd.var("refs", arr1);
        SDVariable idxs = sd.var("idxs", arr2);
        SDVariable upds = sd.var("upds", arr3);

        SDVariable result = sd.scatterDiv(refs, idxs, upds);
        assertArrayEquals(new long[]{3, 3}, result.eval().shape());
        assertEquals(expected, result.eval());

    }

    @Test
    public void testRollAxis() {
        INDArray inArr = Nd4j.create(new long[]{2, 3, 4});
        SameDiff sd = SameDiff.create();
        SDVariable in = sd.var("in", inArr);
        SDVariable rolled = sd.rollAxis(in, 2);
        assertArrayEquals(new long[]{4, 2, 3}, rolled.eval().shape());
    }

    @Test
    public void testReciprocal() {
        INDArray inArr = Nd4j.linspace(1,4,4).reshape(2,2);
        INDArray expected = Nd4j.onesLike(inArr).divi(inArr);
        SameDiff sd = SameDiff.create();
        SDVariable in = sd.var("in", inArr);
        SDVariable reciprocal = sd.reciprocal(in);
        INDArray res = reciprocal.eval();
        assertEquals(expected,res);
    }


    @Test
    public void validateInternalState(){
        SameDiff sd = SameDiff.create();
        sd.enableDebugMode();

        int nOut = 4;
        int minibatch = 10;
        SDVariable input = sd.var("in", new int[]{minibatch, nOut});
        SDVariable label = sd.var("label", new int[]{minibatch, nOut});

        SDVariable diff = input.sub("diff", label);
        SDVariable sqDiff = diff.mul("sqDiff", diff);
        SDVariable msePerEx = sd.mean("msePerEx", sqDiff, 1);

        SDVariable out = sd.mean("loss", msePerEx, 0);

        assertEquals("diff", diff.getVarName());
        assertEquals("sqDiff", sqDiff.getVarName());

//        System.out.println(sd.summary());

        //Validate internal state:

        DifferentialFunction[] dfs = sd.functions();
        assertEquals(4, dfs.length);    //sub, mul, mean, mean
        assertEquals(SubOp.class, dfs[0].getClass());
        assertEquals(MulOp.class, dfs[1].getClass());
        assertEquals(Mean.class, dfs[2].getClass());
        assertEquals(Mean.class, dfs[3].getClass());

        //incomingArgsReverse: maps from function own name to input args (input SDVariables)
        Map<String, String[]> incomingArgsReverse = getObject("incomingArgsReverse", sd, SameDiff.class);
        assertEquals(4, incomingArgsReverse.size());

        Map<String, String[]> incomingArgsReverseExp = new LinkedHashMap<>();
        incomingArgsReverseExp.put(dfs[0].getOwnName(), new String[]{"in", "label"});
        incomingArgsReverseExp.put(dfs[1].getOwnName(), new String[]{"diff", "diff"});
        incomingArgsReverseExp.put(dfs[2].getOwnName(), new String[]{"sqDiff"});
        incomingArgsReverseExp.put(dfs[3].getOwnName(), new String[]{"msePerEx"});
        for (Map.Entry<String, String[]> e : incomingArgsReverseExp.entrySet()) {
            assertArrayEquals(e.getValue(), incomingArgsReverse.get(e.getKey()));
        }

        //outgoingArgsReverse: maps from function own name to outputs (output SDVariables)
        Map<String,String[]> outgoingArgsReverse = getObject("outgoingArgsReverse", sd, SameDiff.class);
        Map<String, String[]> outgoingArgsReverseExp = new LinkedHashMap<>();
        outgoingArgsReverseExp.put(dfs[0].getOwnName(), new String[]{"diff"});      //Sub
        outgoingArgsReverseExp.put(dfs[1].getOwnName(), new String[]{"sqDiff"});    //Mul
        outgoingArgsReverseExp.put(dfs[2].getOwnName(), new String[]{"msePerEx"});  //Mean
        outgoingArgsReverseExp.put(dfs[3].getOwnName(), new String[]{"loss"});      //Mean
        for (Map.Entry<String, String[]> e : outgoingArgsReverseExp.entrySet()) {
            assertArrayEquals(e.getValue(), outgoingArgsReverse.get(e.getKey()));
        }

        //==============================================================================================================
        //Check gradient function

        sd.createGradFunction();
        SameDiff sdGrad = sd.getFunction("grad");

        DifferentialFunction[] dfsBackward = sdGrad.functions();
        assertEquals(14, dfsBackward.length);    //sub, mul, mean, mean, backward marker, meanbp=(onesLike,scalardiv,mul), meanbp=(onesLike,scalardiv,mul), mulbp, add (from diff.mul(diff)), subbp

        List<Class> classesExp = Arrays.asList(
                SubOp.class, MulOp.class, Mean.class, Mean.class, GradientBackwardsMarker.class, OnesLike.class,
                ScalarDivision.class, MulOp.class, OnesLike.class, ScalarDivision.class, MulOp.class, MulBpOp.class, AddOp.class, SubBpOp.class);

        assertEquals(14, dfsBackward.length);
        for(int i=0; i<14; i++ ){
            assertEquals(classesExp.get(i), dfsBackward[i].getClass());
        }

        List<SDVariable> variables = sdGrad.variables();    //in, label, sub, multiply

        Map<String,String[]> incomingArgsReverseBP = getObject("incomingArgsReverse", sdGrad, SameDiff.class);
        System.out.println(incomingArgsReverseBP.keySet());
        //Should have 1 entry for each DifferentialFunction...
        assertEquals(14, incomingArgsReverseBP.size());

        Map<String,String[]> outgoingArgsReverseBP = getObject("outgoingArgsReverse", sdGrad, SameDiff.class);
        System.out.println(outgoingArgsReverseBP.keySet());
        //Should have 1 entry for each DifferentialFunction...
        assertEquals(14, outgoingArgsReverseBP.size());

    }

    @Test
    public void testGather2(){

        INDArray in = Nd4j.rand(10,10);
        INDArray indices = Nd4j.create(new double[]{0,1,5});

        SameDiff sd = SameDiff.create();

        SDVariable var = sd.var("in", in);
        SDVariable varIndices = sd.var("indices", indices);
        SDVariable gather = sd.gather(var, varIndices, 0);

        System.out.println(in);

        INDArray exp = Nd4j.pullRows(in, 1, new int[]{0,1,5});  //Along dimension 1 -> equiv to "indexes for axis 0"
        INDArray act = sd.execAndEndResult();

        assertEquals(exp, act);
    }

    @Test
    public void testGatherOp(){

        INDArray in = Nd4j.rand(10,10);
        INDArray indices = Nd4j.create(new double[]{0,1,5});
        INDArray out = Nd4j.create(3, 10);

        DynamicCustomOp op = DynamicCustomOp.builder("gather")
                .addIntegerArguments(0) //Indexes are for dimension 0
                .addInputs(in, indices)
                .addOutputs(out)
                .build();

        Nd4j.getExecutioner().exec(op);

        System.out.println(out);

        INDArray exp = Nd4j.pullRows(in, 1, new int[]{0,1,5});  //Along dimension 1 == indexes for dimension 0

        assertEquals(exp, out);

        //Shape function:
        List<long[]> shapes = Nd4j.getExecutioner().calculateOutputShape(op);
        long[] expShape = new long[]{3,10};

        assertEquals(1, shapes.size());

        assertArrayEquals(expShape, shapes.get(0));
    }


    @Test
    public void testConditions() {

        SameDiff sd = SameDiff.create();

        INDArray ia = Nd4j.create(new float[]{4, 2});
        SDVariable in = sd.var("in", new int[]{1, 2});
        sd.associateArrayWithVariable(ia, in);


        INDArray expFinite = Nd4j.create(new float[]{1, 1});
        SDVariable finite = sd.isFinite(in);

        INDArray expInfinite = Nd4j.create(new float[]{0, 0});
        SDVariable infinite = sd.isInfinite(in);

        INDArray expNaN =  Nd4j.create(new float[]{0, 0});
        SDVariable isnan = sd.isNaN(in);

        sd.exec();
        assertEquals(expFinite, finite.getArr());
        assertEquals(expInfinite, infinite.getArr());
        assertEquals(expNaN, isnan.getArr());

    }

    @Test
    public void invertPermutation() {
        SameDiff sd = SameDiff.create();

        INDArray ia = Nd4j.create(new float[] {3, 4, 0, 2, 1});
        INDArray expOut = Nd4j.create(new float[] {2, 4, 3, 0, 1});

        SDVariable input = sd.var("in", new int[] {1, 5});
        sd.associateArrayWithVariable(ia, input);

        SDVariable out = sd.invertPermutation(input);

        sd.exec();

        assertEquals(expOut, out.getArr());
    }


    private static <T> T getObject(String fieldName, Object from, Class<?> fromClass){
        try {
            Field f = fromClass.getDeclaredField(fieldName);
            f.setAccessible(true);
            return (T)f.get(from);
        } catch (Exception e){
            throw new RuntimeException(e);
        }
    }

}
