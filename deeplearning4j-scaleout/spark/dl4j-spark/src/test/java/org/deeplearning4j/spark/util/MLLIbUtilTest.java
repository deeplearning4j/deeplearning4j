/*
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */

package org.deeplearning4j.spark.util;

import static org.junit.Assert.*;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.VoidFunction;
import org.apache.spark.input.PortableDataStream;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.canova.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.spark.BaseSparkTest;
import org.junit.Test;
import org.nd4j.linalg.dataset.DataSet;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.ClassPathResource;

import java.util.List;

/**
 * Created by agibsonccc on 1/23/15.
 */
public class MLLIbUtilTest extends BaseSparkTest {
    private static final Logger log = LoggerFactory.getLogger(MLLIbUtilTest.class);

    @Test
    public void testMlLibTest() {
        DataSet dataSet = new IrisDataSetIterator(150,150).next();
        List<DataSet> list = dataSet.asList();
        JavaRDD<DataSet> data = sc.parallelize(list);
        JavaRDD<LabeledPoint> mllLibData = MLLibUtil.fromDataSet(sc,data);
    }

    @Test
    public void testMlLibBinaryFiles() throws Exception {
        JavaPairRDD<String, PortableDataStream> pairRdd = sc.binaryFiles(new ClassPathResource("img.jpg").getFile().toString());
        JavaRDD<LabeledPoint> rdd = MLLibUtil.fromBinary(pairRdd, new ImageRecordReader(28, 28));
        rdd.foreach(new VoidFunction<LabeledPoint>() {
            @Override
            public void call(LabeledPoint labeledPoint) throws Exception {

            }
        });

    }

}
