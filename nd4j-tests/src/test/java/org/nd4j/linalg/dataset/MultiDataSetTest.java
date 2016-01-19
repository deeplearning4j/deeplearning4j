package org.nd4j.linalg.dataset;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.ArrayList;
import java.util.List;

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

        fail();
    }

    @Test
    public void testMergingTimeSeriesWithMasking() {

        fail();
    }

    @Test
    public void testMixedDataTypes() {
        //Test merging with FF, CNN and RNN data

        fail();
    }


}
