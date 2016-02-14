package org.nd4j.linalg.cpu.javacpp;

import org.apache.commons.math3.util.Pair;
import org.junit.Test;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.executioner.OpExecutioner;
import org.nd4j.linalg.api.ops.executioner.OpExecutionerUtil;
import org.nd4j.linalg.api.ops.impl.accum.Bias;
import org.nd4j.linalg.api.ops.impl.accum.Mean;
import org.nd4j.linalg.api.ops.impl.accum.Variance;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.checkutil.NDArrayCreationUtil;
import org.nd4j.linalg.convolution.Convolution;
import org.nd4j.linalg.convolution.OldConvolution;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.Arrays;
import java.util.List;
import java.util.concurrent.TimeUnit;
import static org.junit.Assert.*;

/**
 * @author Adam Gibson
 */
public class LoopTest {
    @Test
    public void testLoop() {
        Loop loop = new Loop();
        float[] add1 = new float[1000000];
        float[] add2 = new float[1];
        long start = System.nanoTime();
        loop.execFloatTransform(add1,3,0,0,1,1,"exp",add2,add1);
        long end = System.nanoTime();
        System.out.println((TimeUnit.MILLISECONDS.convert(Math.abs(end - start),TimeUnit.NANOSECONDS)));
        loop.execScalarFloat(add1,add1,add1.length,0,0,1,1,"div_scalar",new float[]{1});

    }

    @Test
    public void testCreate() {
        INDArray arr = Nd4j.create(new double[10]);
    }

    @Test
    public void testShape() {
        INDArray arr = Nd4j.create(new int[]{1,10});
        System.out.println(Arrays.toString(arr.shape()));
    }



    @Test
    public void testSumWithRow2() {
        //All sums in this method execute without exceptions.
        INDArray array3d = Nd4j.ones(2,10,10);
        array3d.sum(0);
        array3d.sum(1);
        array3d.sum(2);

        INDArray array4d = Nd4j.ones(2, 10, 10, 10);
        array4d.sum(0);
        array4d.sum(1);
        array4d.sum(2);
        array4d.sum(3);

        INDArray array5d = Nd4j.ones(2, 10, 10, 10, 10);
        array5d.sum(0);
        array5d.sum(1);
        array5d.sum(2);
        array5d.sum(3);
        array5d.sum(4);
    }

    @Test
    public void testTensorStats() {
        List<Pair<INDArray,String>> testInputs = NDArrayCreationUtil.getAllTestMatricesWithShape(9, 13, 123);

        for(int j = 0; j < testInputs.size(); j++) {
            Pair<INDArray,String> pair = testInputs.get(j);
            INDArray arr = pair.getFirst();
            String msg = pair.getSecond();

            int nTAD0 = arr.tensorssAlongDimension(0);
            int nTAD1 = arr.tensorssAlongDimension(1);

            OpExecutionerUtil.Tensor1DStats t0 = OpExecutionerUtil.get1DTensorStats(arr, 0);
            OpExecutionerUtil.Tensor1DStats t1 = OpExecutionerUtil.get1DTensorStats(arr, 1);

            assertEquals(nTAD0,t0.getNumTensors());
            assertEquals(nTAD1, t1.getNumTensors());

            INDArray tFirst0 = arr.tensorAlongDimension(0,0);
            INDArray tSecond0 = arr.tensorAlongDimension(1,0);

            INDArray tFirst1 = arr.tensorAlongDimension(0,1);
            INDArray tSecond1 = arr.tensorAlongDimension(1,1);

            assertEquals(tFirst0.offset(),t0.getFirstTensorOffset());
            assertEquals(tFirst1.offset(),t1.getFirstTensorOffset());
            int separation0 = tSecond0.offset() - tFirst0.offset();
            int separation1 = tSecond1.offset() - tFirst1.offset();
            assertEquals(separation0,t0.getTensorStartSeparation());
            assertEquals(separation1,t1.getTensorStartSeparation());

            for(int i = 0; i < nTAD0; i++) {
                INDArray tad0 = arr.tensorAlongDimension(i,0);
                assertEquals(tad0.length(), t0.getTensorLength());
                assertEquals(tad0.elementWiseStride(),t0.getElementWiseStride());

                int offset = tad0.offset();
                int calcOffset = t0.getFirstTensorOffset() + i * t0.getTensorStartSeparation();
                assertEquals(offset,calcOffset);
            }

            for(int i = 0; i<nTAD1; i++) {
                INDArray tad1 = arr.tensorAlongDimension(i,1);
                assertEquals(tad1.length(), t1.getTensorLength());
                assertEquals(tad1.elementWiseStride(),t1.getElementWiseStride());

                int offset = tad1.offset();
                int calcOffset = t1.getFirstTensorOffset() + i*t1.getTensorStartSeparation();
                assertEquals(offset,calcOffset);
            }
        }
    }



    @Test
    public void testGetColumnFortran() {
        Nd4j.factory().setOrder('f');
        INDArray n = Nd4j.create(Nd4j.linspace(1, 4, 4).data(), new int[]{2, 2});
        INDArray column = Nd4j.create(new float[]{1, 2});
        INDArray column2 = Nd4j.create(new float[]{3, 4});
        INDArray testColumn = n.getColumn(0);
        INDArray testColumn1 = n.getColumn(1);
        assertEquals(column, testColumn);
        assertEquals(column2, testColumn1);

    }

    @Test
    public void testDescriptiveStats() {
        OpExecutioner opExecutioner = Nd4j.getExecutioner();
        INDArray x = Nd4j.linspace(1, 5, 5);

        Mean mean = new Mean(x);
        opExecutioner.exec(mean);
        assertEquals(3.0, mean.currentResult().doubleValue(), 1e-1);

        Variance variance = new Variance(x.dup(), true);
        opExecutioner.exec(variance);
        assertEquals(2.5, variance.currentResult().doubleValue(), 1e-1);
    }


    @Test
    public void testBias() {
        INDArray bias = Nd4j.linspace(1, 4, 4);
        Bias biaOp = new Bias(bias);
        Nd4j.getExecutioner().exec(biaOp);
        assertEquals(0.0,biaOp.currentResult().doubleValue(),1e-1);
    }

    @Test
    public void testDup() {

        for(int x = 0; x < 100; x++) {
            INDArray orig = Nd4j.linspace(1, 4, 4);
            INDArray dup = orig.dup();
            assertEquals(orig, dup);

            INDArray matrix = Nd4j.create(new float[]{1, 2, 3, 4}, new int[]{2, 2});
            INDArray dup2 = matrix.dup();
            assertEquals(matrix, dup2);

            INDArray row1 = matrix.getRow(1);
            INDArray dupRow = row1.dup();
            assertEquals(row1, dupRow);


            INDArray columnSorted = Nd4j.create(new float[]{2, 1, 4, 3}, new int[]{2, 2});
            INDArray dup3 = columnSorted.dup();
            assertEquals(columnSorted, dup3);
        }
    }

    @Test
    public void testPutRow() {
        INDArray d = Nd4j.linspace(1, 4, 4).reshape(2, 2);
        INDArray slice1 = d.slice(1);
        INDArray n = d.dup();

        //works fine according to matlab, let's go with it..
        //reproduce with:  A = newShapeNoCopy(linspace(1,4,4),[2 2 ]);
        //A(1,2) % 1 index based
        float nFirst = 2;
        float dFirst = d.getFloat(0, 1);
        assertEquals(nFirst, dFirst, 1e-1);
        assertEquals(d.data(), n.data());
        assertEquals(true, Arrays.equals(new int[]{2, 2}, n.shape()));

        INDArray newRow = Nd4j.linspace(5, 6, 2);
        n.putRow(0, newRow);
        d.putRow(0, newRow);


        INDArray testRow = n.getRow(0);
        assertEquals(newRow.length(), testRow.length());
        assertEquals(true, Shape.shapeEquals(new int[]{1, 2}, testRow.shape()));


        INDArray nLast = Nd4j.create(Nd4j.linspace(1, 4, 4).data(), new int[]{2, 2});
        INDArray row = nLast.getRow(1);
        INDArray row1 = Nd4j.create(new double[]{3, 4}, new int[]{1,2});
        assertEquals(row, row1);


        INDArray arr = Nd4j.create(new int[]{3, 2});
        INDArray evenRow = Nd4j.create(new double[]{1, 2}, new int[]{1,2});
        arr.putRow(0, evenRow);
        INDArray firstRow = arr.getRow(0);
        assertEquals(true, Shape.shapeEquals(new int[]{1,2}, firstRow.shape()));
        INDArray testRowEven = arr.getRow(0);
        assertEquals(evenRow, testRowEven);


        INDArray row12 = Nd4j.create(new double[]{5, 6}, new int[]{1,2});
        arr.putRow(1, row12);
        assertEquals(true, Shape.shapeEquals(new int[]{1,2}, arr.getRow(0).shape()));
        INDArray testRow1 = arr.getRow(1);
        assertEquals(row12, testRow1);


        INDArray multiSliceTest = Nd4j.create(Nd4j.linspace(1, 16, 16).data(), new int[]{4, 2, 2});
        INDArray test = Nd4j.create(new double[]{5,6}, new int[]{1,2});
        INDArray test2 = Nd4j.create(new double[]{7,8}, new int[]{1,2});

        INDArray multiSliceRow1 = multiSliceTest.slice(1).getRow(0);
        INDArray multiSliceRow2 = multiSliceTest.slice(1).getRow(1);

        assertEquals(test, multiSliceRow1);
        assertEquals(test2,multiSliceRow2);



        INDArray threeByThree = Nd4j.create(3,3);
        INDArray threeByThreeRow1AndTwo = threeByThree.get(NDArrayIndex.interval(1, 3),NDArrayIndex.all());
        threeByThreeRow1AndTwo.putRow(1,Nd4j.ones(3));
        assertEquals(Nd4j.ones(3),threeByThreeRow1AndTwo.getRow(1));

    }

    @Test
    public void testCompareIm2Col() throws Exception {

        int[] miniBatches = {1, 3, 5};
        int[] depths = {1, 3, 5};
        int[] inHeights = {5,21};
        int[] inWidths = {5,21};
        int[] strideH = {1,2};
        int[] strideW = {1,2};
        int[] sizeW = {1,2,3};
        int[] sizeH = {1,2,3};
        int[] padH = {0,1,2};
        int[] padW = {0,1,2};

        DataBuffer.Type[] types = new DataBuffer.Type[]{DataBuffer.Type.FLOAT, DataBuffer.Type.DOUBLE, DataBuffer.Type.FLOAT, DataBuffer.Type.DOUBLE};
        DataBuffer.AllocationMode[] modes = new DataBuffer.AllocationMode[]{DataBuffer.AllocationMode.HEAP, DataBuffer.AllocationMode.HEAP,
                DataBuffer.AllocationMode.DIRECT, DataBuffer.AllocationMode.DIRECT};

        String factoryClassName = Nd4j.factory().getClass().toString().toLowerCase();
        if( factoryClassName.contains("jcublas") || factoryClassName.contains("cuda") ){
            //Only test direct for CUDA; test all for CPU
            types = new DataBuffer.Type[]{DataBuffer.Type.FLOAT, DataBuffer.Type.DOUBLE};
            modes = new DataBuffer.AllocationMode[]{DataBuffer.AllocationMode.DIRECT, DataBuffer.AllocationMode.DIRECT};
        }

        for( int i = 0; i < types.length; i++) {
            DataBuffer.Type type = types[i];
            DataBuffer.AllocationMode mode = modes[i];

            Nd4j.factory().setDType(type);
            Nd4j.dtype = type;
            Nd4j.alloc = mode;
            System.out.println("Testing allocation mode " + mode + " and data type " + type);

            for (int m : miniBatches) {
                for (int d : depths) {
                    for (int h : inHeights) {
                        for (int w : inWidths) {
                            for (int sh : strideH) {
                                for (int sw : strideW) {
                                    for (int kh : sizeH) {
                                        for (int kw : sizeW) {
                                            for (int ph : padH) {
                                                for (int pw : padW) {
                                                    if ((w - kw + 2 * pw) % sw != 0 || (h - kh + 2 * ph) % sh != 0)
                                                        continue;   //(w-kp+2*pw)/sw + 1 is not an integer, i.e., number of outputs doesn't fit

                                                    INDArray in = Nd4j.rand(new int[]{m, d, h, w});
                                                    assertEquals(in.data().allocationMode(), mode);
                                                    assertEquals(in.data().dataType(), type);
                                                    INDArray im2col = Convolution.im2col(in, kh, kw, sh, sw, ph, pw, false);    //Cheating, to get correct shape for input

                                                    INDArray imgOutOld = OldConvolution.col2im(im2col, sh, sw, ph, pw, h, w);
                                                    INDArray imgOutNew = Convolution.col2im(im2col, sh, sw, ph, pw, h, w);
                                                    assertEquals(imgOutOld, imgOutNew);
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }


}
