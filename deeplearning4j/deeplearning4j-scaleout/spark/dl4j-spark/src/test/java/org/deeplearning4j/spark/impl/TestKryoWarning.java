/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.spark.impl;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.spark.api.TrainingMaster;
import org.deeplearning4j.spark.impl.graph.SparkComputationGraph;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster;
import org.junit.Ignore;
import org.junit.Test;

/**
 * Created by Alex on 20/07/2016.
 */
public class TestKryoWarning {

    private static void doTestMLN(SparkConf sparkConf) {
        JavaSparkContext sc = new JavaSparkContext(sparkConf);

        try {

            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().list()
                            .layer(0, new OutputLayer.Builder().nIn(10).nOut(10).build()).pretrain(false).backprop(true)
                            .build();

            TrainingMaster tm = new ParameterAveragingTrainingMaster.Builder(1).build();

            SparkDl4jMultiLayer sml = new SparkDl4jMultiLayer(sc, conf, tm);
        } finally {
            sc.stop();
        }
    }

    private static void doTestCG(SparkConf sparkConf) {
        JavaSparkContext sc = new JavaSparkContext(sparkConf);

        try {

            ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder().graphBuilder().addInputs("in")
                            .addLayer("0", new OutputLayer.Builder().nIn(10).nOut(10).build(), "in").setOutputs("0")
                            .pretrain(false).backprop(true).build();

            TrainingMaster tm = new ParameterAveragingTrainingMaster.Builder(1).build();

            SparkListenable scg = new SparkComputationGraph(sc, conf, tm);
        } finally {
            sc.stop();
        }
    }

    @Test
    @Ignore
    public void testKryoMessageMLNIncorrectConfig() {
        //Should print warning message
        SparkConf sparkConf = new SparkConf().setMaster("local[*]").setAppName("sparktest").set("spark.serializer",
                        "org.apache.spark.serializer.KryoSerializer");

        doTestMLN(sparkConf);
    }

    @Test
    @Ignore
    public void testKryoMessageMLNCorrectConfigKryo() {
        //Should NOT print warning message
        SparkConf sparkConf = new SparkConf().setMaster("local[*]").setAppName("sparktest")
                        .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
                        .set("spark.kryo.registrator", "org.nd4j.Nd4jRegistrator");

        doTestMLN(sparkConf);
    }

    @Test
    @Ignore
    public void testKryoMessageMLNCorrectConfigNoKryo() {
        //Should NOT print warning message
        SparkConf sparkConf = new SparkConf().setMaster("local[*]").setAppName("sparktest");

        doTestMLN(sparkConf);
    }



    @Test
    @Ignore
    public void testKryoMessageCGIncorrectConfig() {
        //Should print warning message
        SparkConf sparkConf = new SparkConf().setMaster("local[*]").setAppName("sparktest").set("spark.serializer",
                        "org.apache.spark.serializer.KryoSerializer");

        doTestCG(sparkConf);
    }

    @Test
    @Ignore
    public void testKryoMessageCGCorrectConfigKryo() {
        //Should NOT print warning message
        SparkConf sparkConf = new SparkConf().setMaster("local[*]").setAppName("sparktest")
                        .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
                        .set("spark.kryo.registrator", "org.nd4j.Nd4jRegistrator");

        doTestCG(sparkConf);
    }

    @Test
    @Ignore
    public void testKryoMessageCGCorrectConfigNoKryo() {
        //Should NOT print warning message
        SparkConf sparkConf = new SparkConf().setMaster("local[*]").setAppName("sparktest");

        doTestCG(sparkConf);
    }

}
