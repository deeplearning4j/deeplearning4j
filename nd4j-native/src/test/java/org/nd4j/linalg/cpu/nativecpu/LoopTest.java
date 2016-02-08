package org.nd4j.linalg.cpu.nativecpu;

import org.apache.commons.math3.util.FastMath;
import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.Accumulation;
import org.nd4j.linalg.api.ops.impl.scalar.ScalarAdd;
import org.nd4j.linalg.api.ops.impl.transforms.*;
import org.nd4j.linalg.api.ops.impl.transforms.comparison.Eps;
import org.nd4j.linalg.api.rng.distribution.Distribution;
import org.nd4j.linalg.convolution.Convolution;
import org.nd4j.linalg.cpu.nativecpu.CBLAS;
import org.nd4j.linalg.cpu.nativecpu.util.LibUtils;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.learning.*;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.util.ArrayUtil;

import java.util.*;

import static org.junit.Assert.*;
/**
 * @author Adam Gibson
 */
public class LoopTest {
    static {
        LibUtils.loadLibrary("libnd4j");
    }
    @Test
    public void testLoop() {
        INDArray linspace = Nd4j.linspace(1,4,4);
        double sum = Nd4j.getBlasWrapper().asum(linspace);
        assertEquals(10,sum,1e-1);

    }

    @Test
    public void testPutSlice() {
        INDArray n = Nd4j.linspace(1,27,27).reshape(3, 3, 3);
        INDArray newSlice = Nd4j.zeros(3, 3);
        n.putSlice(0, newSlice);
        assertEquals(newSlice, n.slice(0));

        INDArray firstDimensionAs1 = newSlice.reshape(1, 3, 3);
        n.putSlice(0, firstDimensionAs1);


    }

    @Test
    public void testStdev() {
        INDArray arr = Nd4j.create(new float[]{0.9296161f, 0.31637555f, 0.1839188f}, new int[]{1, 3}, 'c');
        double stdev = arr.stdNumber().doubleValue();
        double stdev2 = arr.std(1).getDouble(0);
        assertEquals(stdev,stdev2,0.0);

        double exp = 0.397842772f;
        assertEquals(exp,stdev,1e-7f);
    }

    @Test
    public void testDup() {
        Nd4j.getRandom().setSeed(12345L);
        INDArray twoByEight = Nd4j.linspace(1,16,16).reshape(2,8);

        INDArray view = twoByEight.get(NDArrayIndex.all(), NDArrayIndex.interval(0, 2));
        int eleStride = view.elementWiseStride();
        INDArray viewCopy = view.dup();
        assertEquals(view,viewCopy);

    }

    @Test
    public void testMultiDimSum() {
        double[] data = new double[]{22.,  26.,  30};
        INDArray assertion = Nd4j.create(data);
        for(int i = 0; i < data.length; i++) {
            assertEquals(data[i],assertion.getDouble(i),1e-1);
        }

        INDArray twoTwoByThree = Nd4j.linspace(1,12,12).reshape(2, 2, 3);
        INDArray multiSum = twoTwoByThree.sum(0, 1);
        assertEquals(assertion,multiSum);

    }

    @Test
    public void testArrCreationShape() {
        INDArray arr = Nd4j.linspace(1,4,4).reshape(2,2);
        for(int i = 0; i < 2; i++)
            assertEquals(2,arr.size(i));
        int[] stride = ArrayUtil.calcStrides(new int[]{2, 2});
        for(int i = 0; i < stride.length; i++) {
            assertEquals(stride[i],arr.stride(i));
        }

        assertArrayEquals(new int[]{2,2},arr.shape());
        assertArrayEquals(new int[]{2,1},arr.stride());
    }

    @Test
    public void testColumnSumDouble() {
        INDArray twoByThree = Nd4j.linspace(1, 600, 600).reshape(150, 4);
        INDArray columnVar = twoByThree.sum(0);
        INDArray assertion = Nd4j.create(new float[]{44850.0f, 45000.0f, 45150.0f, 45300.0f});
        assertEquals(assertion, columnVar);

    }

    @Test
    public void testEight() {
        INDArray baseArr = Nd4j.linspace(1,8,8).reshape(2, 2, 2);
        assertEquals(2,baseArr.tensorssAlongDimension(0,1));
        INDArray columnVectorFirst = Nd4j.create(new double[]{1,5,3,7}, new int[]{2,2});
        INDArray columnVectorSecond = Nd4j.create(new double[]{2, 6,4, 8}, new int[]{2, 2});
        INDArray test1 = baseArr.tensorAlongDimension(0, 0, 1);
        test1.toString();
        assertEquals(columnVectorFirst, test1);
        INDArray test2 = baseArr.tensorAlongDimension(1, 0, 1);
        assertEquals(columnVectorSecond,test2);

    }




    @Test
    public void testAdaDeltaCombining() {
        int n = 7;
        Nd4j.getRandom().setSeed(12345);
        Random r = new Random(12345);
        double[] rhos = new double[n];
        AdaDelta[] adaDeltas = new AdaDelta[n];
        INDArray[] msgs = new INDArray[n];
        INDArray[] msdxs = new INDArray[n];

        double avgRho = 0.0;
        INDArray avgStateMsg = Nd4j.zeros(1,10);
        INDArray avgStateMsdxs = Nd4j.zeros(1,10);
        for( int i = 0; i < msgs.length; i++ ){
            rhos[i] = r.nextDouble();
            avgRho += rhos[i];
            adaDeltas[i] = new AdaDelta(rhos[i]);
            msgs[i] = Nd4j.rand(1, 10);
            msdxs[i] = Nd4j.rand(1, 10);
            avgStateMsg.addi(msgs[i]);
            avgStateMsdxs.addi(msdxs[i]);
            adaDeltas[i].setMsg(msgs[i].dup());
            adaDeltas[i].setMsdx(msdxs[i].dup());
        }
        avgRho /= n;
        avgStateMsg.divi(n);
        avgStateMsdxs.divi(n);

        GradientUpdaterAggregator ag = adaDeltas[0].getAggregator(true);
        for( int i=1; i<n; i++ ) ag.aggregate(adaDeltas[i]);

        AdaDelta combined = (AdaDelta)ag.getUpdater();

        assertEquals(avgRho,combined.getRho(),1e-10);
        assertEquals(avgStateMsg,combined.getMsg());
        assertEquals(avgStateMsdxs,combined.getMsdx());

        //Check merging of AdaDelta:
        GradientUpdaterAggregator first = adaDeltas[0].getAggregator(false);
        GradientUpdaterAggregator second = adaDeltas[2].getAggregator(false);
        for(int i = 0; i < n; i++ ){
            if(i < 2){
                first.aggregate(adaDeltas[i]);
            } else {
                second.aggregate(adaDeltas[i]);
            }
        }

        GradientUpdaterAggregator agMerged = first.combine(second);
        AdaDelta combined2 = (AdaDelta) agMerged.getUpdater();
        assertEquals(avgRho,combined2.getRho(),1e-10);
        assertEquals(avgStateMsg,combined2.getMsg());
        assertEquals(avgStateMsdxs,combined2.getMsdx());
    }

    @Test
    public void testAdamCombining() {
        int n = 7;
        Nd4j.getRandom().setSeed(12345);
        Random r = new Random(12345);
        double[] lrs = new double[n];
        double[] beta1s = new double[n];
        double[] beta2s = new double[n];
        double[] eps = new double[n];
        Adam[] adams = new Adam[n];
        INDArray[] ms = new INDArray[n];
        INDArray[] vs = new INDArray[n];

        double avgLr = 0.0;
        double avgBeta1 = 0.0;
        double avgBeta2 = 0.0;
        double avgEps = 0.0;
        INDArray avgStateM = Nd4j.zeros(1,10);
        INDArray avgStateV = Nd4j.zeros(1,10);
        for( int i=0; i<n; i++ ){
            lrs[i] = r.nextDouble();
            beta1s[i] = r.nextDouble();
            beta2s[i] = r.nextDouble();
            eps[i] = r.nextDouble();
            avgLr += lrs[i];
            avgBeta1 += beta1s[i];
            avgBeta2 += beta2s[i];
            avgEps += eps[i];
            adams[i] = new Adam(lrs[i]);
            adams[i].setBeta1(beta1s[i]);
            adams[i].setBeta2(beta2s[i]);
            adams[i].setEpsilon(eps[i]);
            ms[i] = Nd4j.rand(1, 10);
            vs[i] = Nd4j.rand(1, 10);
            avgStateM.addi(ms[i]);
            avgStateV.addi(vs[i]);
            adams[i].setM(ms[i].dup());
            adams[i].setV(vs[i].dup());
        }
        avgLr /= n;
        avgBeta1 /= n;
        avgBeta2 /= n;
        avgEps /= n;
        avgStateM.divi(n);
        avgStateV.divi(n);

        GradientUpdaterAggregator ag = adams[0].getAggregator(true);
        for( int i=1; i<n; i++) ag.aggregate(adams[i]);

        Adam combined = (Adam)ag.getUpdater();

        assertEquals(avgLr,combined.getLearningRate(),1e-10);
        assertEquals(avgBeta1,combined.getBeta1(),1e-10);
        assertEquals(avgBeta2,combined.getBeta2(),1e-10);
        assertEquals(avgEps,combined.getEpsilon(),1e-10);
        assertEquals(avgStateM,combined.getM());
        assertEquals(avgStateV,combined.getV());

        //Check merging of AdamAggregators:
        GradientUpdaterAggregator first = adams[0].getAggregator(false);
        GradientUpdaterAggregator second = adams[2].getAggregator(false);
        for(int i=0; i < n; i++ ){
            if(i < 2){
                first.aggregate(adams[i]);
            } else {
                second.aggregate(adams[i]);
            }
        }
        GradientUpdaterAggregator agMerged = first.combine(second);
        Adam combined2 = (Adam) agMerged.getUpdater();
        assertEquals(avgLr,combined2.getLearningRate(),1e-10);
        assertEquals(avgBeta1,combined2.getBeta1(),1e-10);
        assertEquals(avgBeta2,combined2.getBeta2(),1e-10);
        assertEquals(avgEps,combined2.getEpsilon(),1e-10);
        assertEquals(avgStateM,combined2.getM());
        assertEquals(avgStateV,combined2.getV());
    }


    @Test
    public void testSgdCombining() {
        int n = 7;
        Nd4j.getRandom().setSeed(12345);
        Random r = new Random(12345);
        double[] lrs = new double[n];
        Sgd[] sgds = new Sgd[n];

        double avgLr = 0.0;
        for( int i=0; i<n; i++ ){
            lrs[i] = r.nextDouble();
            avgLr += lrs[i];
            sgds[i] = new Sgd(lrs[i]);
        }
        avgLr /= n;

        GradientUpdaterAggregator ag = sgds[0].getAggregator(true);
        for( int i=1; i<n; i++) ag.aggregate(sgds[i]);

        Sgd combined = (Sgd)ag.getUpdater();

        assertEquals(avgLr,combined.getLearningRate(),1e-10);

        //Check merging of SgdAggregators:
        GradientUpdaterAggregator first = sgds[0].getAggregator(false);
        GradientUpdaterAggregator second = sgds[2].getAggregator(false);
        for(int i=0; i<n; i++ ){
            if(i<2){
                first.aggregate(sgds[i]);
            } else {
                second.aggregate(sgds[i]);
            }
        }
        GradientUpdaterAggregator agMerged = first.combine(second);
        Sgd combined2 = (Sgd) agMerged.getUpdater();
        assertEquals(avgLr,combined2.getLearningRate(),1e-10);
    }


    @Test
    public void testRmsPropCombining() {
        int n = 7;
        Nd4j.getRandom().setSeed(12345);
        Random r = new Random(12345);
        double[] lrs = new double[n];
        double[] rmsDecays = new double[n];
        RmsProp[] rmsProps = new RmsProp[n];
        INDArray[] lastGradients = new INDArray[n];

        double avgLr = 0.0;
        double avgRmsDecay = 0.0;
        INDArray avgLastGradient = Nd4j.zeros(1,10);
        for( int i=0; i<lastGradients.length; i++ ){
            lrs[i] = r.nextDouble();
            rmsDecays[i] = r.nextDouble();
            avgLr += lrs[i];
            avgRmsDecay += rmsDecays[i];
            rmsProps[i] = new RmsProp(lrs[i],rmsDecays[i]);
            lastGradients[i] = Nd4j.rand(1, 10);
            avgLastGradient.addi(lastGradients[i]);
            rmsProps[i].setLastGradient(lastGradients[i].dup());
        }
        avgLr /= n;
        avgRmsDecay /= n;
        avgLastGradient.divi(n);

        GradientUpdaterAggregator ag = rmsProps[0].getAggregator(true);
        for( int i=1; i<n; i++) ag.aggregate(rmsProps[i]);

        RmsProp combined = (RmsProp)ag.getUpdater();

        assertEquals(avgLr,combined.getLearningRate(),1e-10);
        assertEquals(avgRmsDecay,combined.getRmsDecay(),1e-10);
        assertEquals(avgLastGradient,combined.getLastGradient());

        //Check merging of RmsPropAggregators:
        GradientUpdaterAggregator first = rmsProps[0].getAggregator(false);
        GradientUpdaterAggregator second = rmsProps[2].getAggregator(false);
        for(int i=0; i<n; i++ ){
            if(i<2){
                first.aggregate(rmsProps[i]);
            } else {
                second.aggregate(rmsProps[i]);
            }
        }
        GradientUpdaterAggregator agMerged = first.combine(second);
        RmsProp combined2 = (RmsProp) agMerged.getUpdater();
        assertEquals(avgLr,combined2.getLearningRate(),1e-10);
        assertEquals(avgRmsDecay,combined2.getRmsDecay(),1e-10);
        assertEquals(avgLastGradient,combined2.getLastGradient());
    }



    @Test
    public void testNesterovCombining() {
        int n = 7;
        Nd4j.getRandom().setSeed(12345);
        Random r = new Random(12345);
        double[] lrs = new double[n];
        double[] momentums = new double[n];
        Nesterovs[] nesterovs = new Nesterovs[n];
        INDArray[] vs = new INDArray[n];

        double avgLr = 0.0;
        double avgMomentums = 0.0;
        INDArray avgState = Nd4j.zeros(1,10);
        for( int i = 0; i < vs.length; i++){
            lrs[i] = r.nextDouble();
            momentums[i] = r.nextDouble();
            avgLr += lrs[i];
            avgMomentums += momentums[i];
            nesterovs[i] = new Nesterovs(momentums[i],lrs[i]);
            vs[i] = Nd4j.rand(1, 10);
            avgState.addi(vs[i]);
            nesterovs[i].setV(vs[i].dup());
        }
        avgLr /= n;
        avgMomentums /= n;
        avgState.divi(n);

        GradientUpdaterAggregator ag = nesterovs[0].getAggregator(true);
        for( int i=1; i<n; i++) ag.aggregate(nesterovs[i]);

        Nesterovs combined = (Nesterovs)ag.getUpdater();

        assertEquals(avgLr,combined.getLearningRate(),1e-10);
        assertEquals(avgMomentums,combined.getMomentum(),1e-10);
        assertEquals(avgState,combined.getV());

        //Check merging of NesterovsAggregators:
        GradientUpdaterAggregator first = nesterovs[0].getAggregator(false);
        GradientUpdaterAggregator second = nesterovs[2].getAggregator(false);
        for(int i=0; i<n; i++ ){
            if(i<2){
                first.aggregate(nesterovs[i]);
            } else {
                second.aggregate(nesterovs[i]);
            }
        }

        GradientUpdaterAggregator agMerged = first.combine(second);
        Nesterovs combined2 = (Nesterovs) agMerged.getUpdater();
        assertEquals(avgLr,combined2.getLearningRate(),1e-10);
        assertEquals(avgMomentums,combined2.getMomentum(),1e-10);
        assertEquals(avgState,combined2.getV());
    }

    @Test
    @Ignore
    public void testCompareIm2ColImpl() {

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
        boolean[] coverall = {false,true};

        DataBuffer.Type[] types = new DataBuffer.Type[]{DataBuffer.Type.FLOAT, DataBuffer.Type.DOUBLE, DataBuffer.Type.FLOAT, DataBuffer.Type.DOUBLE};
        DataBuffer.AllocationMode[] modes = new DataBuffer.AllocationMode[]{DataBuffer.AllocationMode.HEAP, DataBuffer.AllocationMode.HEAP,
                DataBuffer.AllocationMode.DIRECT, DataBuffer.AllocationMode.DIRECT};

        String factoryClassName = Nd4j.factory().getClass().toString().toLowerCase();
        if( factoryClassName.contains("jcublas") || factoryClassName.contains("cuda") || factoryClassName.contains("native") ){
            //Only test direct for CUDA; test all for CPU
            types = new DataBuffer.Type[]{DataBuffer.Type.FLOAT, DataBuffer.Type.DOUBLE};
            modes = new DataBuffer.AllocationMode[]{DataBuffer.AllocationMode.DIRECT, DataBuffer.AllocationMode.DIRECT};
        }

        for( int i = 0; i<types.length; i++ ) {
            DataBuffer.Type type = types[i];
            DataBuffer.AllocationMode mode = modes[i];

            Nd4j.factory().setDType(type);
            Nd4j.dtype = type;
            Nd4j.alloc = mode;


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
                                                        continue;   //(w-kp+2*pw)/sw + 1 is not an integer,  i.e., number of outputs doesn't fit

                                                    for(boolean cAll : coverall) {

                                                        INDArray in = Nd4j.rand(new int[]{m, d, h, w});
                                                        assertEquals(in.data().allocationMode(), mode);
                                                        assertEquals(in.data().dataType(), type);

                                                        INDArray outOrig = OldConvolution.im2col(in, kh, kw, sh, sw, ph, pw, -1, cAll); //Old implementation
                                                        INDArray outNew = Convolution.im2col(in, kh, kw, sh, sw, ph, pw, cAll);         //Current implementation

                                                        assertEquals(outOrig,outNew);
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



    @Test
    public void testExecSubArray() {
        INDArray nd = Nd4j.create(new double[]{1, 2, 3, 4, 5, 6}, new int[]{2, 3});

        INDArray sub = nd.get(NDArrayIndex.all(), NDArrayIndex.interval(0, 2));
        Nd4j.getExecutioner().exec(new ScalarAdd(sub, 2));
        assertEquals(Nd4j.create(new double[][]{
                {3, 4}, {6, 7}
        }), sub);

    }

    @Test
    public void testLength() {
        INDArray values = Nd4j.create(2, 2);
        INDArray values2 = Nd4j.create(2, 2);

        values.put(0, 0, 0);
        values2.put(0, 0, 2);
        values.put(1, 0, 0);
        values2.put(1, 0, 2);
        values.put(0, 1, 0);
        values2.put(0, 1, 0);
        values.put(1, 1, 2);
        values2.put(1, 1, 2);


        INDArray expected = Nd4j.repeat(Nd4j.scalar(2), 2).reshape(2,1);

        Accumulation accum = Nd4j.getOpFactory().createAccum("euclidean", values, values2);
        INDArray results = Nd4j.getExecutioner().exec(accum, 1);
        assertEquals(expected, results);

    }


    @Test
    public void testBroadCasting() {
        INDArray first = Nd4j.arange(0, 3).reshape(3, 1);
        INDArray ret = first.broadcast(3, 4);
        INDArray testRet = Nd4j.create(new double[][]{
                {0, 0, 0, 0},
                {1, 1, 1, 1},
                {2, 2, 2, 2}
        });
        assertEquals(testRet, ret);
        INDArray r = Nd4j.arange(0, 4).reshape(1, 4);
        INDArray r2 = r.broadcast(4, 4);
        INDArray testR2 = Nd4j.create(new double[][]{
                {0, 1, 2, 3},
                {0, 1, 2, 3},
                {0, 1, 2, 3},
                {0, 1, 2, 3}
        });
        assertEquals(testR2, r2);

    }


    @Test
    public void testSortRows() {
        int nRows = 10;
        int nCols = 5;
        java.util.Random r = new java.util.Random(12345);

        for( int i=0; i < nCols; i++) {
            INDArray in = Nd4j.rand(new int[]{nRows,nCols});

            List<Integer> order = new ArrayList<>(nRows);
            //in.row(order(i)) should end up as out.row(i) - ascending
            //in.row(order(i)) should end up as out.row(nRows-j-1) - descending
            for( int j=0; j<nRows; j++ ) order.add(j);
            Collections.shuffle(order, r);
            for( int j = 0; j<nRows; j++ )
                in.putScalar(new int[]{j,i},order.get(j));

            INDArray outAsc = Nd4j.sortRows(in, i, true);
            INDArray outDesc = Nd4j.sortRows(in, i, false);

            for( int j = 0; j<nRows; j++ ){
                assertTrue(outAsc.getDouble(j,i)==j);
                int origRowIdxAsc = order.indexOf(j);
                assertTrue(outAsc.getRow(j).equals(in.getRow(origRowIdxAsc)));

                assertTrue(outDesc.getDouble(j,i)==(nRows-j-1));
                int origRowIdxDesc = order.indexOf(nRows-j-1);
                assertTrue(outDesc.getRow(j).equals(in.getRow(origRowIdxDesc)));
            }
        }
    }

    @Test
    public void testSortColumns() {
        int nRows = 5;
        int nCols = 10;
        java.util.Random r = new java.util.Random(12345);

        for( int i=0; i<nRows; i++ ){
            INDArray in = Nd4j.rand(new int[]{nRows,nCols});

            List<Integer> order = new ArrayList<>(nRows);
            for( int j=0; j<nCols; j++ ) order.add(j);
            Collections.shuffle(order, r);
            for( int j=0; j<nCols; j++ ) in.putScalar(new int[]{i,j},order.get(j));

            INDArray outAsc = Nd4j.sortColumns(in, i, true);
            INDArray outDesc = Nd4j.sortColumns(in, i, false);

            for( int j = 0; j < nCols; j++ ){
                assertTrue(outAsc.getDouble(i,j)==j);
                int origColIdxAsc = order.indexOf(j);
                assertTrue(outAsc.getColumn(j).equals(in.getColumn(origColIdxAsc)));

                assertTrue(outDesc.getDouble(i,j)==(nCols-j-1));
                int origColIdxDesc = order.indexOf(nCols-j-1);
                assertTrue(outDesc.getColumn(j).equals(in.getColumn(origColIdxDesc)));
            }
        }
    }



    @Test
    public void testEps() {
        INDArray ones = Nd4j.ones(5);
        double sum = Nd4j.getExecutioner().exec(new Eps(ones, ones, ones, ones.length())).z().sumNumber().doubleValue();
        assertEquals(5, sum, 1e-1);
    }

    @Test
    public void testLogDouble() {
        INDArray linspace = Nd4j.linspace(1, 6, 6);
        INDArray log = Transforms.log(linspace);
        INDArray assertion = Nd4j.create(new double[]{0, 0.6931471805599453, 1.0986122886681098, 1.3862943611198906, 1.6094379124341005, 1.791759469228055});
        assertEquals(assertion, log);
    }


    @Test
    public void testTile() {
        INDArray x = Nd4j.linspace(1,4,4).reshape(2, 2);
        INDArray repeated = x.repeat(new int[]{2});
        assertEquals(8,repeated.length());
        INDArray repeatAlongDimension = x.repeat(1,new int[]{2});
        INDArray assertionRepeat = Nd4j.create(new double[][]{
                {1, 1, 2, 2},
                {3, 3, 4, 4}
        });
        assertArrayEquals(new int[]{2,4},assertionRepeat.shape());
        assertEquals(assertionRepeat,repeatAlongDimension);
        System.out.println(repeatAlongDimension);
        INDArray ret = Nd4j.create(new double[]{0, 1, 2});
        INDArray tile = Nd4j.tile(ret, 2, 2);
        INDArray assertion = Nd4j.create(new double[][]{
                {0, 1, 2, 0, 1, 2}
                , {0, 1, 2, 0, 1, 2}
        });
        assertEquals(assertion,tile);
    }

    @Test
    public void testTensorDot() {
        INDArray oneThroughSixty = Nd4j.arange(60).reshape(3, 4, 5);
        INDArray oneThroughTwentyFour = Nd4j.arange(24).reshape(4, 3, 2);
        INDArray result = Nd4j.tensorMmul(oneThroughSixty, oneThroughTwentyFour, new int[][]{{1, 0}, {0, 1}});
        assertArrayEquals(new int[]{5, 2}, result.shape());
        INDArray assertion = Nd4j.create(new double[][]{
                {   4400 ,  4730},
                {  4532 ,  4874},
                {  4664  , 5018},
                {  4796 ,  5162},
                {  4928 , 5306}
        });
        assertEquals(assertion, result);

        INDArray w = Nd4j.valueArrayOf(new int[]{2, 1, 2, 2}, 0.5);
        INDArray col = Nd4j.create(new double[]{
                1, 1, 1, 1, 3, 3, 3, 3, 1, 1, 1, 1, 3, 3, 3, 3, 1, 1, 1, 1, 3, 3, 3,
                3, 1, 1, 1, 1, 3, 3, 3, 3, 2, 2, 2, 2, 4, 4, 4, 4, 2, 2, 2, 2, 4, 4,
                4, 4, 2, 2, 2, 2, 4, 4, 4, 4, 2, 2, 2, 2, 4, 4, 4, 4
        }, new int[]{1, 1, 2, 2, 4, 4});

        INDArray test = Nd4j.tensorMmul(col, w, new int[][]{{1, 2, 3}, {1, 2, 3}});
        INDArray assertion2 = Nd4j.create(new double[]{3., 3., 3., 3., 3., 3., 3., 3., 7., 7., 7., 7., 7., 7., 7., 7., 3., 3.
                , 3., 3., 3., 3., 3., 3., 7., 7., 7., 7., 7., 7., 7., 7.}, new int[]{1, 4, 4, 2}, new int[]{16, 8, 2, 1}, 0, 'f');
        assertion2.setOrder('f');
        assertEquals(assertion2,test);
    }

    @Test
    public void testHardTanhDerivative(){
        assertTrue( Nd4j.getOpFactory().createTransform("hardtanh", Nd4j.ones(1)).derivative() instanceof HardTanhDerivative);

        //HardTanh:
        //f(x) = 1 if x > 1
        //f(x) = -1 if x < -1
        //f(x) = x otherwise
        //This is piecewise differentiable.
        //f'(x) = 0 if |x|>1
        //f'(x) = 1 otherwise
        //Note for x= +/- 1, HardTanh is not differentiable. Choose f'(+/- 1) = 1

        INDArray z = Nd4j.zeros(100);
        double[] expOut = new double[100];
        for( int i=0; i < 100; i++) {
            double x = 0.1 * (i - 50);
            z.putScalar(i, x);
            expOut[i] = (Math.abs(x) <= 1.0 ? 1 : 0);
        }

        INDArray zPrime = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform("hardtanh", z).derivative());

        for( int i = 0; i < 100; i++ ){
            assertEquals(expOut[i],zPrime.getDouble(i),1e-1);
        }
    }


    @Test
    public void testRectifiedLinearDerivative() {
        assertTrue( Nd4j.getOpFactory().createTransform("relu", Nd4j.ones(1)).derivative() instanceof Step);

        //ReLU:
        //f(x) = max(0,x)
        //Piecewise differentiable; choose f'(0) = 0
        //f'(x) = 1 if x > 0
        //f'(x) = 0 if x <= 0

        INDArray z = Nd4j.zeros(100);
        double[] expOut = new double[100];
        for( int i = 0; i < 100; i++) {
            double x = 0.1 * (i - 50);
            z.putScalar(i, x);
            expOut[i] = (x  > 0 ? 1 : 0);
        }

        INDArray zPrime = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform("relu", z).derivative());

        for( int i = 0; i < 100; i++ ){
            assertTrue(expOut[i] == zPrime.getDouble(i));
        }
    }


    @Test
    public void testSigmoidDerivative() {
        assertTrue( Nd4j.getOpFactory().createTransform("sigmoid", Nd4j.ones(1)).derivative() instanceof SigmoidDerivative);

        //Derivative of sigmoid: ds(x)/dx = s(x)*(1-s(x))
        //s(x) = 1 / (exp(-x) + 1)
        INDArray z = Nd4j.zeros(100);
        double[] expOut = new double[100];
        for( int i = 0; i < 100; i++ ){
            double x = 0.1 * (i - 50);
            z.putScalar(i, x);
            double sigmoid = 1.0 / (FastMath.exp(-x)+1);
            expOut[i] = sigmoid * (1 - sigmoid);
        }

        INDArray zPrime = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform("sigmoid", z).derivative());

        for( int i = 0; i < 100; i++) {
            double relError = Math.abs(expOut[i] - zPrime.getDouble(i)) / (Math.abs(expOut[i]) + Math.abs(zPrime.getDouble(i)));
            assertTrue(relError < 1e-3);
        }
    }


    @Test
    public void testMmulGet(){
        Nd4j.dtype = DataBuffer.Type.DOUBLE;
        Nd4j.getRandom().setSeed(12345L);
        INDArray elevenByTwo = Nd4j.linspace(1,22,22).reshape(11,2);
        INDArray twoByEight = Nd4j.linspace(1,16,16).reshape(2,8);

        INDArray view = twoByEight.get(NDArrayIndex.all(), NDArrayIndex.interval(0, 2));
        INDArray assertion = Nd4j.create(new double[]{
                19.0,22.0,39.0,46.0,59.0,70.0,79.0,94.0,99.0,118.0,119.0,142.0,139.0,166.0,159.0,190.0,179.0,214.0,199.0,238.0,219.0,262.0,
        },new int[]{11,2});

        INDArray viewCopy = view.dup();
        assertEquals(view,viewCopy);

        INDArray mmul1 = elevenByTwo.mmul(view);
        INDArray mmul2 = elevenByTwo.mmul(viewCopy);
        assertEquals(assertion,mmul1);
        assertEquals(assertion,mmul2);
        assertTrue(mmul1.equals(mmul2));
    }

    @Test
    public void testELUDerivative(){
        assertTrue( Nd4j.getOpFactory().createTransform("elu",Nd4j.ones(1)).derivative() instanceof ELUDerivative);

        //f(x) = x if x>=0
        //f(x) = 1.0*(exp(x)-1)
        INDArray z = Nd4j.zeros(100);
        double[] out = new double[100];
        double[] outDeriv = new double[100];
        for( int i = 0; i < 100; i++) {
            double x = 0.1 * (i - 50);
            z.putScalar(i, x);
            if(x>=0){
                out[i] = x;
                outDeriv[i] = 1.0;
            } else {
                out[i] = FastMath.exp(x)-1.0;
                outDeriv[i] = FastMath.exp(x);
            }
        }

        INDArray act = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform("elu", z.dup()));
        INDArray actDeriv = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform("elu", z.dup()).derivative());

        System.out.println(act);

        for( int i = 0; i < 100; i++ ){
            double relError1 = Math.abs(out[i] - act.getDouble(i)) / (Math.abs(out[i]) + Math.abs(act.getDouble(i)));
            if(out[i] == 0.0 && act.getDouble(i) == 0.0) relError1 = 0.0;
            double relError2 = Math.abs(outDeriv[i] - actDeriv.getDouble(i)) / (Math.abs(outDeriv[i]) + Math.abs(actDeriv.getDouble(i)));
            if(outDeriv[i] == 0.0 && actDeriv.getDouble(i) == 0.0) relError2 = 0.0;
            assertTrue(relError1 < 1e-3);
            assertTrue(relError2 < 1e-3);
        }
    }


    @Test
    public void testSoftSignDerivative(){
        assertTrue( Nd4j.getOpFactory().createTransform("softsign", Nd4j.ones(1)).derivative() instanceof SoftSignDerivative );

        //Derivative: 1 / (1+abs(x))^2
        INDArray z = Nd4j.zeros(100);
        double[] expOut = new double[100];
        for( int i = 0; i < 100; i++) {
            double x = 0.1 * (i - 50);
            z.putScalar(i, x);
            double temp = 1 + Math.abs(x);
            expOut[i] = 1.0 / (temp*temp);
        }

        INDArray zPrime = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform("softsign", z).derivative());

        for( int i = 0; i < 100; i++ ){
            double relError = Math.abs(expOut[i] - zPrime.getDouble(i)) / (Math.abs(expOut[i]) + Math.abs(zPrime.getDouble(i)));
            assertTrue(relError < 1e-3);
        }
    }


    @Test
    public void testLeakyReLUDerivative(){
        assertTrue(Nd4j.getOpFactory().createTransform("leakyrelu", Nd4j.ones(1)).derivative() instanceof LeakyReLUDerivative);

        //Derivative: 0.01 if x<0, 1 otherwise
        INDArray z = Nd4j.zeros(100);
        double[] expOut = new double[100];
        for( int i = 0; i < 100; i++) {
            double x = 0.1 * (i - 50);
            z.putScalar(i, x);
            expOut[i] = (x >= 0 ? 1 : 0.01);
        }

        INDArray zPrime = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform("leakyrelu", z).derivative());

        for( int i = 0; i < 100; i++ ){
            double relError = Math.abs(expOut[i] - zPrime.getDouble(i)) / (Math.abs(expOut[i]) + Math.abs(zPrime.getDouble(i)));
           if(relError >= 1e-3) {
               System.out.println("uhh");
           }
            assertTrue(relError < 1e-3);
        }
    }



    @Test
    public void testTanhDerivative(){
        assertTrue(Nd4j.getOpFactory().createTransform("tanh", Nd4j.ones(1)).derivative() instanceof TanhDerivative);

        //Derivative of sigmoid: ds(x)/dx = s(x)*(1-s(x))
        //s(x) = 1 / (exp(-x) + 1)
        INDArray z = Nd4j.zeros(100);
        double[] expOut = new double[100];
        for( int i = 0; i < 100; i++) {
            double x = 0.1 * (i - 50);
            z.putScalar(i, x);
            double tanh = FastMath.tanh(x);
            expOut[i] = 1.0 - tanh * tanh;
        }

        INDArray zPrime = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform("tanh", z).derivative());

        for( int i = 0; i < 100; i++ ){
            double relError = Math.abs(expOut[i]-zPrime.getDouble(i)) / (Math.abs(expOut[i]) + Math.abs(zPrime.getDouble(i)));
            assertTrue(relError < 1e-3);
        }
    }



    @Test
    public void testSoftPlusDerivative(){
        //Derivative of softplus in sigmoid
        assertTrue( Nd4j.getOpFactory().createTransform("softplus", Nd4j.ones(1)).derivative() instanceof Sigmoid);

        //s(x) = 1 / (exp(-x) + 1)
        INDArray z = Nd4j.zeros(100);
        double[] expOut = new double[100];
        for( int i=0; i < 100; i++ ){
            double x = 0.1 * (i - 50);
            z.putScalar(i, x);
            expOut[i] = 1.0 / (1.0 + FastMath.exp(-x));
        }

        INDArray zPrime = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform("softplus", z).derivative());

        for( int i=0; i<100; i++ ){
            double relError = Math.abs(expOut[i]-zPrime.getDouble(i)) / (Math.abs(expOut[i]) + Math.abs(zPrime.getDouble(i)));
            assertTrue(relError < 1e-3);
        }
    }


    @Test
    public void testSoftMaxDerivative() {
        assertTrue( Nd4j.getOpFactory().createTransform("softmax", Nd4j.ones(1)).derivative() instanceof SoftMaxDerivative);

        Random r = new Random(12345L);

        INDArray z = Nd4j.zeros(20,5);
        double[][] in = new double[20][5];
        double[][] softmax = new double[20][5];
        double[][] expOut = new double[20][5];
        for( int i = 0; i < 20; i++) {
            double rowSumExp = 0.0;
            for( int j = 0; j < 5; j++) {
                in[i][j] = 10*r.nextDouble();
                z.putScalar(new int[]{i,j}, in[i][j]);
                rowSumExp += FastMath.exp(in[i][j]);
            }
            for(int j = 0; j < 5; j++){
                softmax[i][j] = FastMath.exp(in[i][j]) / rowSumExp;
                expOut[i][j] = softmax[i][j] * (1.0 - softmax[i][j]);
            }
        }

        INDArray sm = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform("softmax", z.dup()), 1);
        INDArray zPrime = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform("softmax", z).derivative());
        System.out.println(Arrays.toString(sm.data().asDouble()));
        System.out.println(Arrays.toString(zPrime.data().asDouble()));
        assertNotEquals(sm,zPrime);

        for( int i = 0; i < 20; i++) {
            for( int j = 0; j < 5; j++) {
                double relError = Math.abs(expOut[i][j]-zPrime.getDouble(i,j)) / (Math.abs(expOut[i][j]) + Math.abs(zPrime.getDouble(i,j)));
                assertTrue(relError < 1e-3);
            }
        }
    }



    @Test
    public void testAdaGrad1() {
        int rows = 1;
        int cols = 1;


        AdaGrad grad = new AdaGrad(rows, cols, 1e-3);
        INDArray W = Nd4j.ones(rows, cols);
        assertEquals(1e-1, grad.getGradient(W, 0).getDouble(0), 1e-1);
    }

    @Test
    public void testAdaGradCombining() {
        int n = 7;
        Nd4j.getRandom().setSeed(12345);
        Random r = new Random(12345);
        double[] lrs = new double[n];
        AdaGrad[] adaGrads = new AdaGrad[n];
        INDArray[] arr = new INDArray[n];

        double avgLr = 0.0;
        INDArray avgState = Nd4j.zeros(1,10);
        for( int i=0; i<arr.length; i++ ){
            lrs[i] = r.nextDouble();
            avgLr += lrs[i];
            adaGrads[i] = new AdaGrad(lrs[i]);
            arr[i] = Nd4j.rand(1, 10);
            avgState.addi(arr[i]);
            adaGrads[i].setHistoricalGradient(arr[i].dup());
        }
        avgLr /= n;
        avgState.divi(n);

        GradientUpdaterAggregator ag = adaGrads[0].getAggregator(true);
        for( int i=1; i<n; i++ ){
            ag.aggregate(adaGrads[i]);
        }

        AdaGrad combined = (AdaGrad)ag.getUpdater();

        double lrCombined = combined.getLearningRate();
        INDArray histCombined = combined.getHistoricalGradient();

        assertEquals(avgLr,lrCombined,1e-10);
        assertEquals(avgState,histCombined);

        //Check merging of AdaGradAggregators:
        GradientUpdaterAggregator first = adaGrads[0].getAggregator(false);
        GradientUpdaterAggregator second = adaGrads[2].getAggregator(false);
        for(int i=0; i<n; i++ ){
            if(i<2){
                first.aggregate(adaGrads[i]);
            } else {
                second.aggregate(adaGrads[i]);
            }
        }

        GradientUpdaterAggregator agMerged = first.combine(second);
        AdaGrad combined2 = (AdaGrad) agMerged.getUpdater();
        assertEquals(avgLr,combined2.getLearningRate(),1e-10);
        assertEquals(avgState,combined2.getHistoricalGradient());
    }

    @Test
    public void testNesterovs() {
        int rows = 10;
        int cols = 2;


        Nesterovs grad = new Nesterovs(0.5);
        INDArray W = Nd4j.zeros(rows, cols);
        Distribution dist = Nd4j.getDistributions().createNormal(1, 1);
        for (int i = 0; i < W.rows(); i++)
            W.putRow(i, Nd4j.create(dist.sample(W.columns())));

        for (int i = 0; i < 5; i++) {
            String learningRates = String.valueOf("\nAdagrad\n " + grad.getGradient(W, i)).replaceAll(";", "\n");
            System.out.println(learningRates);
            W.addi(Nd4j.randn(rows, cols));
        }
    }


    @Test
    public void testAdaGrad() {
        int rows = 10;
        int cols = 2;


        AdaGrad grad = new AdaGrad(rows, cols, 0.1);
        INDArray W = Nd4j.zeros(rows, cols);
        Distribution dist = Nd4j.getDistributions().createNormal(1, 1);
        for (int i = 0; i < W.rows(); i++)
            W.putRow(i, Nd4j.create(dist.sample(W.columns())));

        for (int i = 0; i < 5; i++) {
            String learningRates = String.valueOf("\nAdagrad\n " + grad.getGradient(W, i)).replaceAll(";", "\n");
            System.out.println(learningRates);
            W.addi(Nd4j.randn(rows, cols));
        }

    }

    @Test
    public void testAdaDelta() {
        int rows = 10;
        int cols = 2;


        AdaDelta grad = new AdaDelta();
        INDArray W = Nd4j.zeros(rows, cols);
        Distribution dist = Nd4j.getDistributions().createNormal(1e-3, 1e-3);
        for (int i = 0; i < W.rows(); i++)
            W.putRow(i, Nd4j.create(dist.sample(W.columns())));

        for (int i = 0; i < 5; i++) {
            String learningRates = String.valueOf("\nAdaelta\n " + grad.getGradient(W, i)).replaceAll(";", "\n");
            System.out.println(learningRates);
            W.addi(Nd4j.randn(rows, cols));
        }
    }


    @Test
    public void testAdam() {
        int rows = 10;
        int cols = 2;


        Adam grad = new Adam();
        INDArray W = Nd4j.zeros(rows, cols);
        Distribution dist = Nd4j.getDistributions().createNormal(1e-3, 1e-3);
        for (int i = 0; i < W.rows(); i++)
            W.putRow(i, Nd4j.create(dist.sample(W.columns())));

        for (int i = 0; i < 5; i++) {
            String learningRates = String.valueOf("\nAdam\n " + grad.getGradient(W, i)).replaceAll(";", "\n");
            System.out.println(learningRates);
            W.addi(Nd4j.randn(rows, cols));
        }
    }


}
