package org.datavec.common;

import com.google.common.collect.Lists;
import org.datavec.api.writable.IntWritable;
import org.datavec.api.writable.Writable;
import org.datavec.common.data.NDArrayWritable;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

import java.util.List;

import static org.junit.Assert.assertEquals;

public class RecordConverterTest {
    @Test
    public void toRecord_PassInClassificationDataSet_ExpectNDArrayAndIntWritables() {
        INDArray feature1 = Nd4j.create(new double[] { 4, -5.7, 10, -0.1 });
        INDArray feature2 = Nd4j.create(new double[] { 11, .7, -1.3, 4});
        INDArray label1 = Nd4j.create(new double[] { 0, 0, 1, 0 });
        INDArray label2 = Nd4j.create(new double[] { 0, 1, 0, 0 });
        DataSet dataSet = new DataSet(Nd4j.vstack(Lists.newArrayList(feature1, feature2)), Nd4j.vstack(Lists.newArrayList(label1, label2)));

        List<List<Writable>> writableList = RecordConverter.toRecords(dataSet);

        assertEquals(2, writableList.size());
        testClassificationWritables(feature1, 2, writableList.get(0));
        testClassificationWritables(feature2, 1, writableList.get(1));
    }

    private void testClassificationWritables(INDArray expectedFeatureVector, int expectLabelIndex, List<Writable> writables) {
        NDArrayWritable ndArrayWritable = (NDArrayWritable) writables.get(0);
        IntWritable intWritable = (IntWritable) writables.get(1);

        assertEquals(2, writables.size());
        assertEquals(expectedFeatureVector, ndArrayWritable.get());
        assertEquals(expectLabelIndex, intWritable.get());
    }
}
