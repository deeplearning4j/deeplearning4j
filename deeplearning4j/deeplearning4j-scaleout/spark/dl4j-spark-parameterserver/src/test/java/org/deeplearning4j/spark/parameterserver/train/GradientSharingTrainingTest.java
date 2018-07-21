/*
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
 */

package org.deeplearning4j.spark.parameterserver.train;

import lombok.extern.slf4j.Slf4j;
import org.apache.spark.api.java.JavaRDD;
import org.deeplearning4j.datasets.iterator.EarlyTerminationDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.spark.api.RDDTrainingApproach;
import org.deeplearning4j.spark.api.TrainingMaster;
import org.deeplearning4j.spark.impl.graph.SparkComputationGraph;
import org.deeplearning4j.spark.parameterserver.BaseSparkTest;
import org.deeplearning4j.spark.parameterserver.training.SharedTrainingMaster;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.learning.config.AMSGrad;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.parameterserver.distributed.conf.VoidConfiguration;

import java.io.File;
import java.net.Inet4Address;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertTrue;

@Slf4j
public class GradientSharingTrainingTest extends BaseSparkTest {

    @Rule
    public TemporaryFolder testDir = new TemporaryFolder();

    @Test
    public void trainSanityCheck() throws Exception {

        for(String s : new String[]{"direct", "export", "paths"}) {
            log.info("Starting: {}", s);
            boolean isPaths = "paths".equals(s);

            RDDTrainingApproach rddTrainingApproach;
            switch (s){
                case "direct":
                    rddTrainingApproach = RDDTrainingApproach.Direct;
                    break;
                case "export":
                    rddTrainingApproach = RDDTrainingApproach.Export;
                    break;
                case "paths":
                    rddTrainingApproach = RDDTrainingApproach.Direct;   //Actualy not used for fitPaths
                    break;
                default:
                    throw new RuntimeException();
            }

            File temp = testDir.newFolder();


            //TODO this probably won't work everywhere...
            String controller = Inet4Address.getLocalHost().getHostAddress();
            String networkMask = controller.substring(0, controller.lastIndexOf('.')) + ".0" + "/16";

            VoidConfiguration voidConfiguration = VoidConfiguration.builder()
                    .unicastPort(40123) // Should be open for IN/OUT communications on all Spark nodes
                    .networkMask(networkMask) // Local network mask
                    .controllerAddress(controller)
                    .build();
            TrainingMaster tm = new SharedTrainingMaster.Builder(voidConfiguration, 2, 1e-4, 16)
                    .rngSeed(12345)
                    .collectTrainingStats(false)
                    .batchSizePerWorker(16) // Minibatch size for each worker
                    .updatesThreshold(1e-4) // Encoding threshold (see docs for details)
                    .workersPerNode(2) // Workers per node
                    .rddTrainingApproach(rddTrainingApproach)
                    .exportDirectory("file:///" + temp.getAbsolutePath().replaceAll("\\\\", "/"))
                    .build();


            ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
                    .seed(12345)
                    .updater(new AMSGrad(0.01))
                    .graphBuilder()
                    .addInputs("in")
                    .layer("out", new OutputLayer.Builder().nIn(784).nOut(10).activation(Activation.SOFTMAX)
                            .lossFunction(LossFunctions.LossFunction.MCXENT).build(), "in")
                    .setOutputs("out")
                    .build();


            SparkComputationGraph sparkNet = new SparkComputationGraph(sc, conf, tm);
            sparkNet.setCollectTrainingStats(tm.getIsCollectTrainingStats());

            System.out.println(Arrays.toString(sparkNet.getNetwork().params().get(NDArrayIndex.point(0), NDArrayIndex.interval(0, 256)).dup().data().asFloat()));
            File f = testDir.newFolder();
            DataSetIterator iter = new MnistDataSetIterator(16, true, 12345);
            int count = 0;
            List<String> paths = new ArrayList<>();
            List<DataSet> ds = new ArrayList<>();
            while (iter.hasNext() && count++ < 128) {
                DataSet d = iter.next();
                if(isPaths) {
                    File out = new File(f, count + ".bin");
                    d.save(out);
                    String path = "file:///" + out.getAbsolutePath().replaceAll("\\\\", "/");
                    paths.add(path);
                }
                ds.add(d);
            }

            int numIter = 1;
            double[] acc = new double[numIter+1];
            for( int i=0; i<numIter; i++ ) {
                //Check accuracy before:
                DataSetIterator testIter = new EarlyTerminationDataSetIterator(new MnistDataSetIterator(32, false, 12345), 10);
                Evaluation eBefore = sparkNet.getNetwork().evaluate(testIter);

                INDArray paramsBefore = sparkNet.getNetwork().params().dup();
                ComputationGraph after;
                switch (s) {
                    case "direct":
                    case "export":
                        JavaRDD<DataSet> dsRDD = sc.parallelize(ds);
                        after = sparkNet.fit(dsRDD);
                        break;
                    case "paths":
                        JavaRDD<String> pathRdd = sc.parallelize(paths);
                        after = sparkNet.fitPaths(pathRdd);
                        break;
                    default:
                        throw new RuntimeException();
                }

                INDArray paramsAfter = after.params();
//            System.out.println(Arrays.toString(paramsBefore.get(NDArrayIndex.point(0), NDArrayIndex.interval(0, 256)).dup().data().asFloat()));
//            System.out.println(Arrays.toString(paramsAfter.get(NDArrayIndex.point(0), NDArrayIndex.interval(0, 256)).dup().data().asFloat()));
//            System.out.println(Arrays.toString(
//                    Transforms.abs(paramsAfter.sub(paramsBefore)).get(NDArrayIndex.point(0), NDArrayIndex.interval(0, 256)).dup().data().asFloat()));
                assertNotEquals(paramsBefore, paramsAfter);


                testIter = new EarlyTerminationDataSetIterator(new MnistDataSetIterator(32, false, 12345), 10);
                Evaluation eAfter = after.evaluate(testIter);

                assertTrue(eAfter.accuracy() >= eBefore.accuracy() + 0.01);

                if(i == 0){
                    acc[0] = eBefore.accuracy();
                }
                acc[i+1] = eAfter.accuracy();
            }
            log.info("Accuracies: {}", Arrays.toString(acc));
        }
    }
}
