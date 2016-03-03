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

package org.deeplearning4j.spark.impl.layer;


import org.apache.spark.api.java.JavaRDD;
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.layers.OutputLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.spark.BaseSparkTest;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.List;
import java.util.UUID;

import static org.junit.Assert.assertEquals;


/**
 * Created by agibsonccc on 1/18/15.
 */
public class TestSparkLayer extends BaseSparkTest {

    private static final Logger log = LoggerFactory.getLogger(TestSparkLayer.class);



    @Test
    public void testIris2() throws Exception {


        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .iterations(10)
                .learningRate(1e-1)
                .layer(new org.deeplearning4j.nn.conf.layers.OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .nIn(4).nOut(3)
                        .weightInit(WeightInit.XAVIER)
                        .activation("softmax").build())
                .build();


        System.out.println("Initializing network");
        SparkDl4jLayer master = new SparkDl4jLayer(sc,conf);
        DataSet d = new IrisDataSetIterator(150,150).next();
        d.normalizeZeroMeanZeroUnitVariance();
        d.shuffle();
        List<DataSet> next = d.asList();


        JavaRDD<DataSet> data = sc.parallelize(next);



        OutputLayer network2 =(OutputLayer) master.fitDataSet(data);

        INDArray params = network2.params();
        File writeTo = new File(UUID.randomUUID().toString());
        Nd4j.writeTxt(params, writeTo.getAbsolutePath(), ",");
        INDArray load = Nd4j.readTxt(writeTo.getAbsolutePath());
        assertEquals(params, load);
        writeTo.delete();
        Evaluation evaluation = new Evaluation();
        evaluation.eval(d.getLabels(), network2.output(d.getFeatureMatrix()));
        System.out.println(evaluation.stats());
    }



}
