package org.nd4j.linalg.dataset;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

public class MultiDataSetTest {

    @Test
    public void testMerging2d() {
        //Simple test: single input/output arrays; 5 MultiDataSets to merge
        int nCols = 3;
        int nRows = 5;
        INDArray expIn = Nd4j.linspace(0, nCols * nRows - 1, nCols * nRows).reshape(nRows, nCols);
        INDArray expOut = Nd4j.linspace(100, 100 + nCols * nRows - 1, nCols * nRows).reshape(nRows, nCols);

        INDArray[] in = new INDArray[nRows];
        INDArray[] out = new INDArray[nRows];
        for (int i = 0; i < nRows; i++) in[i] = expIn.getRow(i).dup();
        for (int i = 0; i < nRows; i++) out[i] = expOut.getRow(i).dup();

        List<MultiDataSet> list = new ArrayList<>(nRows);
        for (int i = 0; i < nRows; i++) {
            list.add(new MultiDataSet(in[i], out[i]));
        }

        MultiDataSet merged = MultiDataSet.merge(list);
        assertEquals(1, merged.getFeatures().length);
        assertEquals(1, merged.getLabels().length);

        assertEquals(expIn, merged.getFeatures(0));
        assertEquals(expOut, merged.getLabels(0));
    }

    @Test
    public void testMerging2dMultipleInOut() {
        //Test merging: Multiple input/output arrays; 5 MultiDataSets to merge

        int nRows = 5;
        int nColsIn0 = 3;
        int nColsIn1 = 4;
        int nColsOut0 = 5;
        int nColsOut1 = 6;

        INDArray expIn0 = Nd4j.linspace(0, nRows * nColsIn0 - 1, nRows * nColsIn0).reshape(nRows, nColsIn0);
        INDArray expIn1 = Nd4j.linspace(0, nRows * nColsIn1 - 1, nRows * nColsIn1).reshape(nRows, nColsIn1);
        INDArray expOut0 = Nd4j.linspace(0, nRows * nColsOut0 - 1, nRows * nColsOut0).reshape(nRows, nColsOut0);
        INDArray expOut1 = Nd4j.linspace(0, nRows * nColsOut1 - 1, nRows * nColsOut1).reshape(nRows, nColsOut1);

        List<MultiDataSet> list = new ArrayList<>(nRows);
        for (int i = 0; i < nRows; i++) {
            if (i == 0) {
                //For first MultiDataSet: have 2 rows, not just 1
                INDArray in0 = expIn0.get(NDArrayIndex.interval(0, 1, true), NDArrayIndex.all()).dup();
                INDArray in1 = expIn1.get(NDArrayIndex.interval(0, 1, true), NDArrayIndex.all()).dup();
                INDArray out0 = expOut0.get(NDArrayIndex.interval(0, 1, true), NDArrayIndex.all()).dup();
                INDArray out1 = expOut1.get(NDArrayIndex.interval(0, 1, true), NDArrayIndex.all()).dup();
                list.add(new MultiDataSet(new INDArray[]{in0, in1}, new INDArray[]{out0, out1}));
                i++;
            } else {
                INDArray in0 = expIn0.getRow(i).dup();
                INDArray in1 = expIn1.getRow(i).dup();
                INDArray out0 = expOut0.getRow(i).dup();
                INDArray out1 = expOut1.getRow(i).dup();
                list.add(new MultiDataSet(new INDArray[]{in0, in1}, new INDArray[]{out0, out1}));
            }
        }

        MultiDataSet merged = MultiDataSet.merge(list);
        assertEquals(2, merged.getFeatures().length);
        assertEquals(2, merged.getLabels().length);

        assertEquals(expIn0, merged.getFeatures(0));
        assertEquals(expIn1, merged.getFeatures(1));
        assertEquals(expOut0, merged.getLabels(0));
        assertEquals(expOut1, merged.getLabels(1));
    }

    @Test
    public void testMerging4dMultipleInOut() {
        int nRows = 5;
        int depthIn0 = 3;
        int widthIn0 = 4;
        int heightIn0 = 5;

        int depthIn1 = 4;
        int widthIn1 = 5;
        int heightIn1 = 6;

        int nColsOut0 = 5;
        int nColsOut1 = 6;

        int lengthIn0 = nRows * depthIn0 * widthIn0 * heightIn0;
        int lengthIn1 = nRows * depthIn1 * widthIn1 * heightIn1;
        INDArray expIn0 = Nd4j.linspace(0, lengthIn0 - 1, lengthIn0).reshape(nRows, depthIn0, widthIn0, heightIn0);
        INDArray expIn1 = Nd4j.linspace(0, lengthIn1 - 1, lengthIn1).reshape(nRows, depthIn1, widthIn1, heightIn1);
        INDArray expOut0 = Nd4j.linspace(0, nRows * nColsOut0 - 1, nRows * nColsOut0).reshape(nRows, nColsOut0);
        INDArray expOut1 = Nd4j.linspace(0, nRows * nColsOut1 - 1, nRows * nColsOut1).reshape(nRows, nColsOut1);

        List<MultiDataSet> list = new ArrayList<>(nRows);
        for (int i = 0; i < nRows; i++) {
            if (i == 0) {
                //For first MultiDataSet: have 2 rows, not just 1
                INDArray in0 = expIn0.get(NDArrayIndex.interval(0, 1, true), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all()).dup();
                INDArray in1 = expIn1.get(NDArrayIndex.interval(0, 1, true), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all()).dup();
                INDArray out0 = expOut0.get(NDArrayIndex.interval(0, 1, true), NDArrayIndex.all()).dup();
                INDArray out1 = expOut1.get(NDArrayIndex.interval(0, 1, true), NDArrayIndex.all()).dup();
                list.add(new MultiDataSet(new INDArray[]{in0, in1}, new INDArray[]{out0, out1}));
                i++;
            } else {
                INDArray in0 = expIn0.get(NDArrayIndex.interval(i, i, true), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all()).dup();
                INDArray in1 = expIn1.get(NDArrayIndex.interval(i, i, true), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all()).dup();
                INDArray out0 = expOut0.getRow(i).dup();
                INDArray out1 = expOut1.getRow(i).dup();
                list.add(new MultiDataSet(new INDArray[]{in0, in1}, new INDArray[]{out0, out1}));
            }
        }

        MultiDataSet merged = MultiDataSet.merge(list);
        assertEquals(2, merged.getFeatures().length);
        assertEquals(2, merged.getLabels().length);

        assertEquals(expIn0, merged.getFeatures(0));
        assertEquals(expIn1, merged.getFeatures(1));
        assertEquals(expOut0, merged.getLabels(0));
        assertEquals(expOut1, merged.getLabels(1));
    }

    @Test
    public void testMergingTimeSeriesEqualLength() {
        int tsLength = 8;
        int nRows = 5;
        int nColsIn0 = 3;
        int nColsIn1 = 4;
        int nColsOut0 = 5;
        int nColsOut1 = 6;

        int n0 = nRows * nColsIn0 * tsLength;
        int n1 = nRows * nColsIn1 * tsLength;
        int nOut0 = nRows * nColsOut0 * tsLength;
        int nOut1 = nRows * nColsOut1 * tsLength;
        INDArray expIn0 = Nd4j.linspace(0, n0 - 1, n0).reshape(nRows, nColsIn0, tsLength);
        INDArray expIn1 = Nd4j.linspace(0, n1 - 1, n1).reshape(nRows, nColsIn1, tsLength);
        INDArray expOut0 = Nd4j.linspace(0, nOut0 - 1, nOut0).reshape(nRows, nColsOut0, tsLength);
        INDArray expOut1 = Nd4j.linspace(0, nOut1 - 1, nOut1).reshape(nRows, nColsOut1, tsLength);

        List<MultiDataSet> list = new ArrayList<>(nRows);
        for (int i = 0; i < nRows; i++) {
            if (i == 0) {
                //For first MultiDataSet: have 2 rows, not just 1
                INDArray in0 = expIn0.get(NDArrayIndex.interval(0, 1, true), NDArrayIndex.all(), NDArrayIndex.all()).dup();
                INDArray in1 = expIn1.get(NDArrayIndex.interval(0, 1, true), NDArrayIndex.all(), NDArrayIndex.all()).dup();
                INDArray out0 = expOut0.get(NDArrayIndex.interval(0, 1, true), NDArrayIndex.all(), NDArrayIndex.all()).dup();
                INDArray out1 = expOut1.get(NDArrayIndex.interval(0, 1, true), NDArrayIndex.all(), NDArrayIndex.all()).dup();
                list.add(new MultiDataSet(new INDArray[]{in0, in1}, new INDArray[]{out0, out1}));
                i++;
            } else {
                INDArray in0 = expIn0.get(NDArrayIndex.interval(i, i, true), NDArrayIndex.all(), NDArrayIndex.all()).dup();
                INDArray in1 = expIn1.get(NDArrayIndex.interval(i, i, true), NDArrayIndex.all(), NDArrayIndex.all()).dup();
                INDArray out0 = expOut0.get(NDArrayIndex.interval(i, i, true), NDArrayIndex.all(), NDArrayIndex.all()).dup();
                INDArray out1 = expOut1.get(NDArrayIndex.interval(i, i, true), NDArrayIndex.all(), NDArrayIndex.all()).dup();
                list.add(new MultiDataSet(new INDArray[]{in0, in1}, new INDArray[]{out0, out1}));
            }
        }

        MultiDataSet merged = MultiDataSet.merge(list);
        assertEquals(2, merged.getFeatures().length);
        assertEquals(2, merged.getLabels().length);

        assertEquals(expIn0, merged.getFeatures(0));
        assertEquals(expIn1, merged.getFeatures(1));
        assertEquals(expOut0, merged.getLabels(0));
        assertEquals(expOut1, merged.getLabels(1));
    }

    @Test
    public void testMergingTimeSeriesWithMasking() {
        //Mask arrays, and different lengths

        int tsLengthIn0 = 8;
        int tsLengthIn1 = 9;
        int tsLengthOut0 = 10;
        int tsLengthOut1 = 11;

        int nRows = 5;
        int nColsIn0 = 3;
        int nColsIn1 = 4;
        int nColsOut0 = 5;
        int nColsOut1 = 6;

        INDArray expectedIn0 = Nd4j.zeros(nRows, nColsIn0, tsLengthIn0);
        INDArray expectedIn1 = Nd4j.zeros(nRows, nColsIn1, tsLengthIn1);
        INDArray expectedOut0 = Nd4j.zeros(nRows, nColsOut0, tsLengthOut0);
        INDArray expectedOut1 = Nd4j.zeros(nRows, nColsOut1, tsLengthOut1);

        INDArray expectedMaskIn0 = Nd4j.zeros(nRows, tsLengthIn0);
        INDArray expectedMaskIn1 = Nd4j.zeros(nRows, tsLengthIn1);
        INDArray expectedMaskOut0 = Nd4j.zeros(nRows, tsLengthOut0);
        INDArray expectedMaskOut1 = Nd4j.zeros(nRows, tsLengthOut1);


        Random r = new Random(12345);
        List<MultiDataSet> list = new ArrayList<>(nRows);
        for (int i = 0; i < nRows; i++) {
            int thisRowIn0Length = tsLengthIn0 - i;
            int thisRowIn1Length = tsLengthIn1 - i;
            int thisRowOut0Length = tsLengthOut0 - i;
            int thisRowOut1Length = tsLengthOut1 - i;

            int in0NumElem = thisRowIn0Length * nColsIn0;
            INDArray in0 = Nd4j.linspace(0, in0NumElem - 1, in0NumElem).reshape(1, nColsIn0, thisRowIn0Length);

            int in1NumElem = thisRowIn1Length * nColsIn1;
            INDArray in1 = Nd4j.linspace(0, in1NumElem - 1, in1NumElem).reshape(1, nColsIn1, thisRowIn1Length);

            int out0NumElem = thisRowOut0Length * nColsOut0;
            INDArray out0 = Nd4j.linspace(0, out0NumElem - 1, out0NumElem).reshape(1, nColsOut0, thisRowOut0Length);

            int out1NumElem = thisRowOut1Length * nColsOut1;
            INDArray out1 = Nd4j.linspace(0, out1NumElem - 1, out1NumElem).reshape(1, nColsOut1, thisRowOut1Length);

            INDArray maskIn0 = null;
            INDArray maskIn1 = Nd4j.zeros(1, thisRowIn1Length);
            for (int j = 0; j < thisRowIn1Length; j++) {
                if (r.nextBoolean()) maskIn1.putScalar(j, 1.0);
            }
            INDArray maskOut0 = null;
            INDArray maskOut1 = Nd4j.zeros(1, thisRowOut1Length);
            for (int j = 0; j < thisRowOut1Length; j++) {
                if (r.nextBoolean()) maskOut1.putScalar(j, 1.0);
            }

            expectedIn0.put(new INDArrayIndex[]{NDArrayIndex.point(i), NDArrayIndex.all(), NDArrayIndex.interval(0, thisRowIn0Length)}, in0);
            expectedIn1.put(new INDArrayIndex[]{NDArrayIndex.point(i), NDArrayIndex.all(), NDArrayIndex.interval(0, thisRowIn1Length)}, in1);
            expectedOut0.put(new INDArrayIndex[]{NDArrayIndex.point(i), NDArrayIndex.all(), NDArrayIndex.interval(0, thisRowOut0Length)}, out0);
            expectedOut1.put(new INDArrayIndex[]{NDArrayIndex.point(i), NDArrayIndex.all(), NDArrayIndex.interval(0, thisRowOut1Length)}, out1);

            expectedMaskIn0.put(new INDArrayIndex[]{NDArrayIndex.point(i), NDArrayIndex.interval(0, thisRowIn0Length)},
                    Nd4j.ones(1, thisRowIn0Length));
            expectedMaskIn1.put(new INDArrayIndex[]{NDArrayIndex.point(i), NDArrayIndex.interval(0, thisRowIn1Length)},
                    maskIn1);
            expectedMaskOut0.put(new INDArrayIndex[]{NDArrayIndex.point(i), NDArrayIndex.interval(0, thisRowOut0Length)},
                    Nd4j.ones(1, thisRowOut0Length));
            expectedMaskOut1.put(new INDArrayIndex[]{NDArrayIndex.point(i), NDArrayIndex.interval(0, thisRowOut1Length)},
                    maskOut1);

            list.add(new MultiDataSet(new INDArray[]{in0, in1}, new INDArray[]{out0, out1}, new INDArray[]{maskIn0, maskIn1},
                    new INDArray[]{maskOut0, maskOut1}));
        }

        MultiDataSet merged = MultiDataSet.merge(list);

        assertEquals(2, merged.getFeatures().length);
        assertEquals(2, merged.getLabels().length);
        assertEquals(2, merged.getFeaturesMaskArrays().length);
        assertEquals(2, merged.getLabelsMaskArrays().length);

        assertEquals(expectedIn0, merged.getFeatures(0));
        assertEquals(expectedIn1, merged.getFeatures(1));
        assertEquals(expectedOut0, merged.getLabels(0));
        assertEquals(expectedOut1, merged.getLabels(1));

        assertEquals(expectedMaskIn0, merged.getFeaturesMaskArray(0));
        assertEquals(expectedMaskIn1, merged.getFeaturesMaskArray(1));
        assertEquals(expectedMaskOut0, merged.getLabelsMaskArray(0));
        assertEquals(expectedMaskOut1, merged.getLabelsMaskArray(1));
    }
}
