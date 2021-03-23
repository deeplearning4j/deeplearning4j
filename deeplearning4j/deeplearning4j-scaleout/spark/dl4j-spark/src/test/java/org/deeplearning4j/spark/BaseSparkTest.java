/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.deeplearning4j.spark;

import com.sun.jna.Platform;
import lombok.SneakyThrows;
import lombok.extern.slf4j.Slf4j;
import org.apache.hadoop.conf.Configuration;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.datavec.spark.util.SerializableHadoopConfig;
import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.BeforeEach;
import org.nd4j.common.resources.Downloader;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.Serializable;
import java.net.URI;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

@Slf4j
public abstract class BaseSparkTest extends BaseDL4JTest implements Serializable {
    protected transient JavaSparkContext sc;
    protected transient INDArray labels;
    protected transient INDArray input;
    protected transient INDArray rowSums;
    protected transient int nRows = 200;
    protected transient int nIn = 4;
    protected transient int nOut = 3;
    protected transient DataSet data;
    protected transient JavaRDD<DataSet> sparkData;

    @Override
    public long getTimeoutMilliseconds() {
        return 120000L;
    }
    @BeforeAll
    @SneakyThrows
    public static void beforeAll() {
        if(Platform.isWindows()) {
            File hadoopHome = new File(System.getProperty("java.io.tmpdir"),"hadoop-tmp");
            File binDir = new File(hadoopHome,"bin");
            if(!binDir.exists())
                binDir.mkdirs();
            File outputFile = new File(binDir,"winutils.exe");
            if(!outputFile.exists()) {
                log.info("Fixing spark for windows");
                Downloader.download("winutils.exe",
                        URI.create("https://github.com/cdarlint/winutils/blob/master/hadoop-2.6.5/bin/winutils.exe?raw=true").toURL(),
                        outputFile,"db24b404d2331a1bec7443336a5171f1",3);
            }

            System.setProperty("hadoop.home.dir", hadoopHome.getAbsolutePath());
        }
    }

    @BeforeEach
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

    @AfterEach
    public void after() {
        if(sc != null) {
            sc.close();
        }
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
        SparkConf sparkConf = new SparkConf().setMaster("local[" + numExecutors() + "]")
                .set("spark.driver.host", "localhost").setAppName("sparktest");


        sc = new JavaSparkContext(sparkConf);

        return sc;
    }

    protected JavaRDD<DataSet> getBasicSparkDataSet(int nRows, INDArray input, INDArray labels) {
        List<DataSet> list = new ArrayList<>();
        for (int i = 0; i < nRows; i++) {
            INDArray inRow = input.getRow(i, true).dup();
            INDArray outRow = labels.getRow(i, true).dup();

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
        return 4;
    }

    protected MultiLayerConfiguration getBasicConf() {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(123)
                .updater(new Nesterovs(0.1, 0.9)).list()
                .layer(0, new org.deeplearning4j.nn.conf.layers.DenseLayer.Builder().nIn(nIn).nOut(3)
                        .activation(Activation.TANH).build())
                .layer(1, new org.deeplearning4j.nn.conf.layers.OutputLayer.Builder(
                        LossFunctions.LossFunction.MCXENT).nIn(3).nOut(nOut)
                        .activation(Activation.SOFTMAX).build())
                .build();

        return conf;
    }


}
