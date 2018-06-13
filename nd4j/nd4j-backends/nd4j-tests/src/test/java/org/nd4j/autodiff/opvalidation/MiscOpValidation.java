package org.nd4j.autodiff.opvalidation;

import lombok.Builder;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.junit.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.validation.OpValidation;
import org.nd4j.autodiff.validation.TestCase;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.checkutil.NDArrayCreationUtil;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.linalg.primitives.Triple;
import org.nd4j.linalg.util.ArrayUtil;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;
import static org.nd4j.linalg.indexing.NDArrayIndex.interval;

@Slf4j
public class MiscOpValidation extends BaseOpValidation {

    public MiscOpValidation(Nd4jBackend backend) {
        super(backend);
    }



    @Test
    public void testGradientAutoBroadcast1() {

        Nd4j.getRandom().setSeed(12345);

        List<String> failed = new ArrayList<>();

        for (int dim_sz1 : new int[]{0, 1, 2}) {

            int[] in2Shape = {3, 4, 5};
            in2Shape[dim_sz1] = 1;

            for (int i = 0; i < 8; i++) {

                SameDiff sd = SameDiff.create();

                SDVariable in3 = sd.var("in3", Nd4j.rand(new int[]{3, 4, 5}));
                SDVariable in2 = sd.var("in2", in2Shape);

                SDVariable bcOp;
                String name;
                switch (i) {
                    case 0:
                        bcOp = in3.add(in2);
                        name = "add";
                        break;
                    case 1:
                        bcOp = in3.sub(in2);
                        name = "sub";
                        break;
                    case 2:
                        bcOp = in3.mul(in2);
                        name = "mul";
                        break;
                    case 3:
                        bcOp = in3.div(in2);
                        name = "div";
                        break;
                    case 4:
                        bcOp = in3.rsub(in2);
                        name = "rsub";
                        break;
                    case 5:
                        bcOp = in3.rdiv(in2);
                        name = "rdiv";
                        break;
                    case 6:
                        bcOp = sd.f().floorDiv(in3, in2);
                        name = "floordiv";
                        break;
                    case 7:
                        bcOp = sd.f().floorMod(in3, in2);
                        name = "floormod";
                        break;
                    default:
                        throw new RuntimeException();
                }

                SDVariable outVar = sd.sum(bcOp);

                String msg = "(test " + i + ": " + name + ", dimension=" + dim_sz1 + ")";
                log.info("*** Starting test: " + msg);

                INDArray in3Arr = Nd4j.randn(new int[]{3, 4, 5}).muli(100);
                INDArray in2Arr = Nd4j.randn(in2Shape).muli(100);

                sd.associateArrayWithVariable(in3Arr, in3);
                sd.associateArrayWithVariable(in2Arr, in2);

                TestCase tc = new TestCase(sd);

                String error = OpValidation.validate(tc);
                if(error != null){
                    failed.add(name);
                }
            }
        }

        assertEquals("Failed: " + failed, 0, failed.size());
    }

    @Test
    public void testGradientAutoBroadcast2() {

        Nd4j.getRandom().setSeed(12345);

        List<String> failed = new ArrayList<>();

        for (int[] dim_sz1s : new int[][]{{0, 1}, {0, 2}, {1, 2}, {0, 1, 2}}) {

            int[] otherShape = {3, 4, 5};
            otherShape[dim_sz1s[0]] = 1;
            otherShape[dim_sz1s[1]] = 1;
            if (dim_sz1s.length == 3) {
                otherShape[dim_sz1s[2]] = 1;
            }

            for (int i = 0; i < 8; i++) {

                SameDiff sd = SameDiff.create();

                SDVariable in3 = sd.var("in3", new int[]{3, 4, 5});
                SDVariable in2 = sd.var("inToBc", otherShape);

                String name;
                SDVariable bcOp;
                switch (i) {
                    case 0:
                        bcOp = in3.add(in2);
                        name = "add";
                        break;
                    case 1:
                        bcOp = in3.sub(in2);
                        name = "sub";
                        break;
                    case 2:
                        bcOp = in3.mul(in2);
                        name = "mul";
                        break;
                    case 3:
                        bcOp = in3.div(in2);
                        name = "div";
                        break;
                    case 4:
                        bcOp = in3.rsub(in2);
                        name = "rsub";
                        break;
                    case 5:
                        bcOp = in3.rdiv(in2);
                        name = "rdiv";
                        break;
                    case 6:
                        bcOp = sd.f().floorDiv(in3, in2);
                        name = "floordiv";
                        break;
                    case 7:
                        bcOp = sd.f().floorMod(in3, in2);
                        name = "floormod";
                        break;
                    default:
                        throw new RuntimeException();
                }

                SDVariable outVar = sd.sum(bcOp);

                String msg = "(test " + i + ": " + name + ", dimensions=" + Arrays.toString(dim_sz1s) + ")";
                log.info("*** Starting test: " + msg);

                INDArray in3Arr = Nd4j.randn(new int[]{3, 4, 5}).muli(100);
                INDArray in2Arr = Nd4j.randn(otherShape).muli(100);

                sd.associateArrayWithVariable(in3Arr, in3);
                sd.associateArrayWithVariable(in2Arr, in2);

                TestCase tc = new TestCase(sd);
                String error = OpValidation.validate(tc);
                if(error != null){
                    failed.add(name);
                }
            }
        }

        assertEquals("Failed: " + failed, 0, failed.size());
    }

    @Test
    public void testGradientAutoBroadcast3() {
        //These tests: output size > input sizes

        fail("TEST CRASHES JVM");

        Nd4j.getRandom().setSeed(12345);

        List<String> failed = new ArrayList<>();

        //Test cases: in1Shape, in2Shape, shapeOf(op(in1,in2))
        List<Triple<long[], long[], long[]>> testCases = new ArrayList<>();
        testCases.add(new Triple<>(new long[]{3, 1}, new long[]{1, 4}, new long[]{3, 4}));
        testCases.add(new Triple<>(new long[]{3, 1}, new long[]{3, 4}, new long[]{3, 4}));
        testCases.add(new Triple<>(new long[]{3, 4}, new long[]{1, 4}, new long[]{3, 4}));
        testCases.add(new Triple<>(new long[]{3, 4, 1}, new long[]{1, 1, 5}, new long[]{3, 4, 5}));
        testCases.add(new Triple<>(new long[]{3, 4, 1}, new long[]{3, 1, 5}, new long[]{3, 4, 5}));
        testCases.add(new Triple<>(new long[]{3, 1, 5}, new long[]{1, 4, 1}, new long[]{3, 4, 5}));
        testCases.add(new Triple<>(new long[]{3, 1, 5}, new long[]{1, 4, 5}, new long[]{3, 4, 5}));
        testCases.add(new Triple<>(new long[]{3, 1, 5}, new long[]{3, 4, 5}, new long[]{3, 4, 5}));
        testCases.add(new Triple<>(new long[]{3, 1, 1, 1}, new long[]{1, 4, 5, 6}, new long[]{3, 4, 5, 6}));
        testCases.add(new Triple<>(new long[]{1, 1, 1, 6}, new long[]{3, 4, 5, 6}, new long[]{3, 4, 5, 6}));
        testCases.add(new Triple<>(new long[]{1, 4, 5, 1}, new long[]{3, 1, 1, 6}, new long[]{3, 4, 5, 6}));
        testCases.add(new Triple<>(new long[]{1, 6}, new long[]{3, 4, 5, 1}, new long[]{3, 4, 5, 6}));

        for (val p : testCases) {

            for (int i = 0; i < 8; i++) {

                SameDiff sd = SameDiff.create();

                SDVariable in3 = sd.var("in1", p.getFirst());
                SDVariable in2 = sd.var("in2", p.getSecond());

                String name;
                SDVariable bcOp;
                switch (i) {
                    case 0:
                        bcOp = in3.add(in2);
                        name = "add";
                        break;
                    case 1:
                        bcOp = in3.sub(in2);
                        name = "sub";
                        break;
                    case 2:
                        bcOp = in3.mul(in2);
                        name = "mul";
                        break;
                    case 3:
                        bcOp = in3.div(in2);
                        name = "div";
                        break;
                    case 4:
                        bcOp = in3.rsub(in2);
                        name = "rsub";
                        break;
                    case 5:
                        bcOp = in3.rdiv(in2);
                        name = "rdiv";
                        break;
                    case 6:
                        bcOp = sd.f().floorDiv(in3, in2);
                        name = "floordiv";
                        break;
                    case 7:
                        bcOp = sd.f().floorMod(in3, in2);
                        name = "floormod";
                        break;
                    default:
                        throw new RuntimeException();
                }

                SDVariable outVar = sd.sum(bcOp);

                String msg = "(test " + i + ": " + name + ", array 1 size =" + Arrays.toString(p.getFirst())
                        + ", array 2 size = " + Arrays.toString(p.getSecond()) + ")";
                log.info("*** Starting test: " + msg);

                INDArray in3Arr = Nd4j.randn(p.getFirst()).muli(100);
                INDArray in2Arr = Nd4j.randn(p.getSecond()).muli(100);

                sd.associateArrayWithVariable(in3Arr, in3);
                sd.associateArrayWithVariable(in2Arr, in2);

                TestCase tc = new TestCase(sd);
                String error = OpValidation.validate(tc);
                if(error != null){
                    failed.add(name);
                }
            }
        }

        assertEquals("Failed: " + failed, 0, failed.size());
    }



    @Test
    public void testScatterOpGradients() {


        List<String> failed = new ArrayList<>();

        for (int i = 0; i < 5; i++) {
            Nd4j.getRandom().setSeed(12345);

            SameDiff sd = SameDiff.create();

            SDVariable in = sd.var("in", new int[]{20, 10});
            SDVariable indices = sd.var("indices", new long[]{5});
            SDVariable updates = sd.var("updates", new int[]{5, 10});


            in.setArray(Nd4j.rand(20, 10));
            indices.setArray(Nd4j.create(new double[]{3, 4, 5, 10, 18}));
            updates.setArray(Nd4j.rand(5, 10).muli(2).subi(1));

            SDVariable scatter;
            String name;
            switch (i) {
                case 0:
                    scatter = sd.scatterAdd("s", in, indices, updates);
                    name = "scatterAdd";
                    break;
                case 1:
                    scatter = sd.scatterSub("s", in, indices, updates);
                    name = "scatterSub";
                    break;
                case 2:
                    scatter = sd.scatterMul("s", in, indices, updates);
                    name = "scatterMul";
                    break;
                case 3:
                    scatter = sd.scatterDiv("s", in, indices, updates);
                    name = "scatterDiv";
                    break;
                case 4:
                    scatter = sd.scatterUpdate("s", in, indices, updates);
                    name = "scatterUpdate";
                    break;
                default:
                    throw new RuntimeException();
            }

            SDVariable loss = sd.sum(scatter);  //.standardDeviation(scatter, true);  //.sum(scatter);  //TODO stdev might be better here as gradients are non-symmetrical...
            sd.execAndEndResult();

            TestCase tc = new TestCase(sd);
            String error = OpValidation.validate(tc);
            if(error != null){
                failed.add(name);
            }
        }

        assertEquals(failed.toString(), 0, failed.size());
    }

    @Test
    public void testGatherGradient() {
        Nd4j.getRandom().setSeed(12345);

        List<String> failed = new ArrayList<>();

        for (int rank = 2; rank <= 3; rank++) {
            for (int dim = 0; dim < rank; dim++) {
                SameDiff sd = SameDiff.create();

                int[] inShape;
                if (rank == 2) {
                    inShape = new int[]{10, 10};
                } else {
                    inShape = new int[]{10, 10, 10};
                }

                SDVariable in = sd.var("in", Nd4j.rand(inShape));
                SDVariable indices = sd.var("indices", Nd4j.create(new double[]{0, 3, 7}));

                SDVariable gather = sd.gather(in, indices, dim);
                sd.execAndEndResult();  //TODO REMOVE THIS

                SDVariable loss = sd.standardDeviation("loss", gather, true, Integer.MAX_VALUE);

                String msg = "rank=" + rank + ", dim=" + dim;

                TestCase tc = new TestCase(sd);
                String error = OpValidation.validate(tc);
                if(error != null){
                    failed.add(name);
                }
            }
        }

        assertEquals(failed.toString(), 0, failed.size());
    }
}
