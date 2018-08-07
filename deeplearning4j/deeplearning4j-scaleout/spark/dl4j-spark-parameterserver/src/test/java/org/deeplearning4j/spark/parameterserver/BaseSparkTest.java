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

package org.deeplearning4j.spark.parameterserver;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster;
import org.junit.After;
import org.junit.Before;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;


/**
 * Created by agibsonccc on 1/23/15.
 */
public abstract class BaseSparkTest implements Serializable {
    protected transient JavaSparkContext sc;
    protected transient INDArray labels;
    protected transient INDArray input;
    protected transient INDArray rowSums;
    protected transient int nRows = 200;
    protected transient int nIn = 4;
    protected transient int nOut = 3;
    protected transient DataSet data;
    protected transient JavaRDD<DataSet> sparkData;

    @Before
    public void before() {

        sc = getContext();
        Random r = new Random(12345);
        labels = Nd4j.create(nRows, nOut);
        input = Nd4j.rand(nRows, nIn);
        rowSums = input.sum(1);
        input.diviColumnVector(rowSums);

        for (int i = 0; i < nRows; i++) {
            int x1 = r.nextInt(nOut);
            labels.putScalar(new int[] {i, x1}, 1.0);
        }

        sparkData = getBasicSparkDataSet(nRows, input, labels);
    }

    @After
    public void after() {
        sc.close();
        sc = null;
    }

    /**
     *
     * @return
     */
    public JavaSparkContext getContext() {
        if (sc != null)
            return sc;
        // set to test mode
        SparkConf sparkConf = new SparkConf().setMaster("local[" + numExecutors() + "]").set("spark.driver.host", "localhost").setAppName("sparktest");


        sc = new JavaSparkContext(sparkConf);

        return sc;
    }

    protected JavaRDD<DataSet> getBasicSparkDataSet(int nRows, INDArray input, INDArray labels) {
        List<DataSet> list = new ArrayList<>();
        for (int i = 0; i < nRows; i++) {
            INDArray inRow = input.getRow(i).dup();
            INDArray outRow = labels.getRow(i).dup();

            DataSet ds = new DataSet(inRow, outRow);
            list.add(ds);
        }
        list.iterator();

        data = DataSet.merge(list);
        return sc.parallelize(list);
    }


    protected SparkDl4jMultiLayer getBasicNetwork() {
        return new SparkDl4jMultiLayer(sc, getBasicConf(),
                        new ParameterAveragingTrainingMaster(true, numExecutors(), 1, 10, 1, 0));
    }

    protected int numExecutors() {
        int numProc = Runtime.getRuntime().availableProcessors();
        return Math.min(4, numProc);
    }

    protected MultiLayerConfiguration getBasicConf() {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(123)
                .updater(new Nesterovs(0.1, 0.9)).list()
                        .layer(0, new org.deeplearning4j.nn.conf.layers.DenseLayer.Builder().nIn(nIn).nOut(3)
                                        .activation(Activation.TANH).build())
                        .layer(1, new org.deeplearning4j.nn.conf.layers.OutputLayer.Builder(
                                        LossFunctions.LossFunction.MCXENT).nIn(3).nOut(nOut)
                                                        .activation(Activation.SOFTMAX).build())
                        .backprop(true).pretrain(false).build();

        return conf;
    }


}
