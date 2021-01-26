/*
 *  ******************************************************************************
 *  * Copyright (c) 2021 Deeplearning4j Contributors
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.datavec.spark.transform;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.util.ndarray.RecordConverter;
import org.datavec.api.writable.DoubleWritable;
import org.datavec.api.writable.Writable;
import org.datavec.spark.BaseSparkTest;
import org.junit.Test;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.List;

import static junit.framework.TestCase.assertTrue;
import static org.junit.Assert.assertEquals;

/**
 * Created by agibsonccc on 10/22/16.
 */
public class NormalizationTests extends BaseSparkTest {


    @Test
    public void testMeanStdZeros() {
        List<List<Writable>> data = new ArrayList<>();
        Schema.Builder builder = new Schema.Builder();
        int numColumns = 6;
        for (int i = 0; i < numColumns; i++)
            builder.addColumnDouble(String.valueOf(i));

        Nd4j.getRandom().setSeed(12345);

        INDArray arr = Nd4j.rand(DataType.FLOAT, 5, numColumns);
        for (int i = 0; i < 5; i++) {
            List<Writable> record = new ArrayList<>(numColumns);
            data.add(record);
            for (int j = 0; j < numColumns; j++) {
                record.add(new DoubleWritable(arr.getDouble(i, j)));
            }
        }


        Schema schema = builder.build();
        JavaRDD<List<Writable>> rdd = sc.parallelize(data);
        Dataset<Row> dataFrame = DataFrames.toDataFrame(schema, rdd);

        //assert equivalent to the ndarray pre processing
        DataNormalization zeroToOne = new NormalizerMinMaxScaler();
        zeroToOne.fit(new DataSet(arr.dup(), arr.dup()));
        INDArray zeroToOnes = arr.dup();
        zeroToOne.transform(new DataSet(zeroToOnes, zeroToOnes));
        List<Row> rows = Normalization.stdDevMeanColumns(dataFrame, dataFrame.columns());
        INDArray assertion = DataFrames.toMatrix(rows);
        INDArray expStd = arr.std(true, true, 0);
        INDArray std = assertion.getRow(0, true);
        assertTrue(expStd.equalsWithEps(std, 1e-3));
        //compare mean
        INDArray expMean = arr.mean(true, 0);
        assertTrue(expMean.equalsWithEps(assertion.getRow(1, true), 1e-3));

    }



    @Test
    public void normalizationTests() {
        List<List<Writable>> data = new ArrayList<>();
        Schema.Builder builder = new Schema.Builder();
        int numColumns = 6;
        for (int i = 0; i < numColumns; i++)
            builder.addColumnDouble(String.valueOf(i));

        for (int i = 0; i < 5; i++) {
            List<Writable> record = new ArrayList<>(numColumns);
            data.add(record);
            for (int j = 0; j < numColumns; j++) {
                record.add(new DoubleWritable(1.0));
            }

        }

        INDArray arr = RecordConverter.toMatrix(DataType.DOUBLE, data);

        Schema schema = builder.build();
        JavaRDD<List<Writable>> rdd = sc.parallelize(data);
        assertEquals(schema, DataFrames.fromStructType(DataFrames.fromSchema(schema)));
        assertEquals(rdd.collect(), DataFrames.toRecords(DataFrames.toDataFrame(schema, rdd)).getSecond().collect());

        Dataset<Row> dataFrame = DataFrames.toDataFrame(schema, rdd);
        dataFrame.show();
        Normalization.zeromeanUnitVariance(dataFrame).show();
        Normalization.normalize(dataFrame).show();

        //assert equivalent to the ndarray pre processing
        NormalizerStandardize standardScaler = new NormalizerStandardize();
        standardScaler.fit(new DataSet(arr.dup(), arr.dup()));
        INDArray standardScalered = arr.dup();
        standardScaler.transform(new DataSet(standardScalered, standardScalered));
        DataNormalization zeroToOne = new NormalizerMinMaxScaler();
        zeroToOne.fit(new DataSet(arr.dup(), arr.dup()));
        INDArray zeroToOnes = arr.dup();
        zeroToOne.transform(new DataSet(zeroToOnes, zeroToOnes));

        INDArray zeroMeanUnitVarianceDataFrame =
                        RecordConverter.toMatrix(DataType.DOUBLE, Normalization.zeromeanUnitVariance(schema, rdd).collect());
        INDArray zeroMeanUnitVarianceDataFrameZeroToOne =
                        RecordConverter.toMatrix(DataType.DOUBLE, Normalization.normalize(schema, rdd).collect());
        assertEquals(standardScalered, zeroMeanUnitVarianceDataFrame);
        assertTrue(zeroToOnes.equalsWithEps(zeroMeanUnitVarianceDataFrameZeroToOne, 1e-1));

    }

}
