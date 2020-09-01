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

package org.deeplearning4j.samediff;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.junit.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.TrainingConfig;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.evaluation.regression.RegressionEvaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.*;
import org.nd4j.linalg.learning.regularization.L1Regularization;
import org.nd4j.linalg.learning.regularization.L2Regularization;
import org.nd4j.linalg.learning.regularization.Regularization;
import org.nd4j.linalg.learning.regularization.WeightDecay;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.weightinit.impl.XavierInitScheme;

import java.util.*;

import static org.junit.Assert.assertEquals;

@Slf4j
public class CompareTrainingImplementations extends BaseDL4JTest {

    @Test
    public void testCompareMlpTrainingIris(){
        DataSetIterator iter = new IrisDataSetIterator(150, 150);
        NormalizerStandardize std = new NormalizerStandardize();
        std.fit(iter);
        iter.setPreProcessor(std);

        DataSet ds = iter.next();
        INDArray f = ds.getFeatures();
        INDArray l = ds.getLabels();

        double[] l1 = new double[]{0.0, 0.0, 0.01, 0.01, 0.0};
        double[] l2 = new double[]{0.0, 0.02, 0.00, 0.02, 0.0};
        double[] wd = new double[]{0.0, 0.0, 0.0, 0.0, 0.03};

        for (String u : new String[]{"sgd", "adam", "nesterov", "adamax", "amsgrad"}) {
            for(int i=0; i<l1.length; i++ ) {
                Nd4j.getRandom().setSeed(12345);
                double l1Val = l1[i];
                double l2Val = l2[i];
                double wdVal = wd[i];

                String testName = u + ", l1=" + l1Val + ", l2=" + l2Val + ", wd=" + wdVal;

                log.info("Starting: {}", testName);
                SameDiff sd = SameDiff.create();

                SDVariable in = sd.placeHolder("input", DataType.DOUBLE, -1, 4);
                SDVariable label = sd.placeHolder("label", DataType.DOUBLE, -1, 3);

                SDVariable w0 = sd.var("w0", new XavierInitScheme('c', 4, 10), DataType.DOUBLE, 4, 10);
                SDVariable b0 = sd.var("b0", Nd4j.create(DataType.DOUBLE, 1, 10));

                SDVariable w1 = sd.var("w1", new XavierInitScheme('c', 10, 3), DataType.DOUBLE, 10, 3);
                SDVariable b1 = sd.var("b1", Nd4j.create(DataType.DOUBLE,  1, 3));

                SDVariable z0 = in.mmul(w0).add(b0);
                SDVariable a0 = sd.nn().tanh(z0);
                SDVariable z1 = a0.mmul(w1).add("prediction", b1);
                SDVariable a1 = sd.nn().softmax("softmax", z1);

                SDVariable diff = sd.math().squaredDifference(a1, label);
                SDVariable lossMse = diff.mean();
                lossMse.markAsLoss();

                IUpdater updater;
                double lr;
                switch (u) {
                    case "sgd":
                        lr = 3e-1;
                        updater = new Sgd(lr);
                        break;
                    case "adam":
                        lr = 1e-2;
                        updater = new Adam(lr);
                        break;
                    case "nesterov":
                        lr = 1e-1;
                        updater = new Nesterovs(lr);
                        break;
                    case "adamax":
                        lr = 1e-2;
                        updater = new AdaMax(lr);
                        break;
                    case "amsgrad":
                        lr = 1e-2;
                        updater = new AMSGrad(lr);
                        break;
                    default:
                        throw new RuntimeException();
                }

                List<Regularization> r = new ArrayList<>();
                if(l2Val > 0){
                    r.add(new L2Regularization(l2Val));
                }
                if(l1Val > 0){
                    r.add(new L1Regularization(l1Val));
                }
                if(wdVal > 0){
                    r.add(new WeightDecay(wdVal, true));
                }
                TrainingConfig conf = new TrainingConfig.Builder()
                        .updater(updater)
                        .regularization(r)
                        .dataSetFeatureMapping("input")
                        .dataSetLabelMapping("label")
                        .build();
                sd.setTrainingConfig(conf);


                //Create equivalent DL4J net
                MultiLayerConfiguration mlc = new NeuralNetConfiguration.Builder()
                        .dataType(DataType.DOUBLE)
                        .weightInit(WeightInit.XAVIER).seed(12345)
                        .l1(l1Val).l2(l2Val)
                        .l1Bias(l1Val).l2Bias(l2Val)
                        .weightDecay(wdVal, true).weightDecayBias(wdVal, true)
                        .updater(new Sgd(1.0))      //Exclicitly use SGD(1.0) for comparing PRE-UPDATE GRADIENTS (but with l1/l2/wd component added)
                        .list()
                        .layer(new DenseLayer.Builder().nIn(4).nOut(10).activation(Activation.TANH).build())
                        .layer(new OutputLayer.Builder().nIn(10).nOut(3).activation(Activation.SOFTMAX).lossFunction(LossFunctions.LossFunction.MSE).build())
                        .build();

                MultiLayerNetwork net = new MultiLayerNetwork(mlc);
                net.init();
                Map<String,INDArray> oldParams = net.paramTable();

                //Assign parameters so we have identical models at the start:
                w0.getArr().assign(net.getParam("0_W"));
                b0.getArr().assign(net.getParam("0_b"));
                w1.getArr().assign(net.getParam("1_W"));
                b1.getArr().assign(net.getParam("1_b"));

                //Check output (forward pass)
                Map<String,INDArray> placeholders = new HashMap<>();
                placeholders.put("input", f);
                placeholders.put("label", l);
                Map<String,INDArray> map = sd.output(placeholders, lossMse.name(), a1.name());
                INDArray outSd = map.get(a1.name());
                INDArray outDl4j = net.output(f);

                assertEquals(testName, outDl4j, outSd);

                net.setInput(f);
                net.setLabels(l);
                net.computeGradientAndScore();
                net.getUpdater().update(net, net.gradient(), 0, 0, 150, LayerWorkspaceMgr.noWorkspacesImmutable()); //Division by minibatch, apply L1/L2



                //Check score
                double scoreDl4j = net.score();
                double scoreSd = map.get(lossMse.name()).getDouble(0) + sd.calcRegularizationScore();
                assertEquals(testName, scoreDl4j, scoreSd, 1e-6);

                double lossRegScoreSD = sd.calcRegularizationScore();
                double lossRegScoreDL4J = net.calcRegularizationScore(true);

                assertEquals(lossRegScoreDL4J, lossRegScoreSD, 1e-6);

                //Check gradients (before updater applied)
                Map<String,INDArray> grads = net.gradient().gradientForVariable();
                Map<String,INDArray> gm = sd.calculateGradients(placeholders, b1.name(), w1.name(), b0.name(), w0.name());

                //Note that the SameDiff gradients don't include the L1/L2 terms at present just from execBackwards()... these are added in fitting only
                //We can check correctness though with training param checks later
                if(l1Val == 0 && l2Val == 0 && wdVal == 0) {
                    assertEquals(testName, grads.get("1_b"), gm.get(b1.name()));
                    assertEquals(testName, grads.get("1_W"), gm.get(w1.name()));
                    assertEquals(testName, grads.get("0_b"), gm.get(b0.name()));
                    assertEquals(testName, grads.get("0_W"), gm.get(w0.name()));
                }


                //Check training with updater
                mlc = new NeuralNetConfiguration.Builder()
                        .dataType(DataType.DOUBLE)
                        .weightInit(WeightInit.XAVIER).seed(12345)
                        .l1(l1Val).l2(l2Val)
                        .l1Bias(l1Val).l2Bias(l2Val)
                        .weightDecay(wdVal, true).weightDecayBias(wdVal, true)
                        .updater(updater.clone())
                        .list()
                        .layer(new DenseLayer.Builder().nIn(4).nOut(10).activation(Activation.TANH).build())
                        .layer(new OutputLayer.Builder().nIn(10).nOut(3).activation(Activation.SOFTMAX).lossFunction(LossFunctions.LossFunction.MSE).build())
                        .build();
                net = new MultiLayerNetwork(mlc);
                net.init();
                net.setParamTable(oldParams);

                for( int j=0; j<3; j++ ) {
                    net.fit(ds);
                    sd.fit(ds);

                    String s = testName + " - " + j;
                    INDArray dl4j_0W = net.getParam("0_W");
                    INDArray sd_0W = w0.getArr();
                    assertEquals(s, dl4j_0W, sd_0W);
                    assertEquals(s, net.getParam("0_b"), b0.getArr());
                    assertEquals(s, net.getParam("1_W"), w1.getArr());
                    assertEquals(s, net.getParam("1_b"), b1.getArr());
                }

                //Compare evaluations
                Evaluation evalDl4j = net.doEvaluation(iter, new Evaluation())[0];
                Evaluation evalSd = new Evaluation();
                sd.evaluate(iter, "softmax", evalSd);
                assertEquals(evalDl4j, evalSd);

                RegressionEvaluation rEvalDl4j = net.doEvaluation(iter, new RegressionEvaluation())[0];
                RegressionEvaluation rEvalSd = new RegressionEvaluation();
                sd.evaluate(iter, "softmax", rEvalSd);
                assertEquals(rEvalDl4j, rEvalSd);

//                System.out.println("---------------------------------");
            }
        }
    }
}
