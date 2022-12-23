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

package org.eclipse.deeplearning4j.nd4j.autodiff.samediff;

import static org.deeplearning4j.datasets.iterator.RandomDataSetIterator.Values.INTEGER_0_10;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.nd4j.linalg.api.buffer.DataType.FLOAT;

import java.io.File;
import java.nio.file.Path;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.deeplearning4j.datasets.iterator.RandomDataSetIterator;
import org.deeplearning4j.datasets.iterator.utilty.ListDataSetIterator;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.autodiff.listeners.impl.ScoreListener;
import org.nd4j.autodiff.listeners.records.History;
import org.nd4j.autodiff.loss.LossReduce;
import org.nd4j.autodiff.samediff.SDIndex;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.TrainingConfig;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.evaluation.IEvaluation;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.eclipse.deeplearning4j.nd4j.linalg.dataset.IrisDataSetIterator;
import org.nd4j.linalg.dataset.MultiDataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.adapter.SingletonDataSetIterator;
import org.nd4j.linalg.dataset.adapter.SingletonMultiDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.learning.config.AMSGrad;
import org.nd4j.linalg.learning.config.AdaMax;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.shade.guava.collect.Lists;
import org.nd4j.weightinit.impl.OneInitScheme;
import org.nd4j.weightinit.impl.XavierInitScheme;
import org.nd4j.weightinit.impl.ZeroInitScheme;

@Slf4j
@NativeTag
@Tag(TagNames.TRAINING)
@Tag(TagNames.SAMEDIFF)
public class SameDiffTrainingTest extends BaseNd4jTestWithBackends {
    @TempDir
    Path testDir;


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @Tag(TagNames.LONG_TEST)
    @Tag(TagNames.LARGE_RESOURCES)
    public void testTraining(Nd4jBackend backend) {
        int nIn = 4;
        int nOut = 1;
        int NUM_SAMPLES = 300;
        int epoches = 2;
        int minibatch = 3;

        SameDiff sd = SameDiff.create();

        //First: Let's create our placeholders. Shape: [minibatch, in/out]
        SDVariable input = sd.placeHolder("input", FLOAT, -1, nIn);
        SDVariable labels = sd.placeHolder("labels", FLOAT, -1, nOut);

        //Second: let's create our variables
        SDVariable weights = sd.var("weights", new XavierInitScheme('c', nIn, nOut), FLOAT, nIn, nOut);
        SDVariable bias = sd.var("bias");

        //And define our forward pass:
        SDVariable out = input.mmul(weights).add(bias);     //Note: it's broadcast add here

        //And our loss function
        SDVariable mse = sd.loss.meanSquaredError("mse", labels, out, null);
        mse.markAsLoss();
        //Let's create some mock data for this example:
        Nd4j.getRandom().setSeed(12345);
        INDArray inputArr = Nd4j.rand(minibatch, nIn);
        INDArray labelArr = Nd4j.rand(minibatch, nOut);

        Map<String,INDArray> placeholderData = new HashMap<>();
        placeholderData.put("input", inputArr);
        placeholderData.put("labels", labelArr);

        //Execute forward pass:
        INDArray loss = sd.output(placeholderData, "mse").get("mse");
        System.out.println("MSE: " + loss);

        //Calculate gradients:
        Map<String,INDArray> gradMap = sd.calculateGradients(placeholderData, "weights");
        System.out.println("Weights gradient:");
        System.out.println(gradMap.get("weights"));

        //Mock random dataset for training
        INDArray indFeature = Nd4j.rand(new long[] {NUM_SAMPLES, nIn});
        INDArray indLabel = Nd4j.rand(new long[] {NUM_SAMPLES, nOut});
        DataSet ds = new DataSet(indFeature, indLabel);
        SplitTestAndTrain train_test = ds.splitTestAndTrain(0.7);
        DataSet dsTrain = train_test.getTrain();
        DataSet dsTest = train_test.getTest();
        DataSetIterator trainIter = new ListDataSetIterator<>(Lists.newArrayList(dsTrain), minibatch);
        DataSetIterator testIter = new ListDataSetIterator<>(Lists.newArrayList(dsTest), minibatch);
        //Train model
        double learningRate = 1e-3;
        TrainingConfig config = new TrainingConfig.Builder()
                .updater(new Sgd(learningRate))
                .dataSetFeatureMapping("input")
                .dataSetLabelMapping("labels")
                .build();
        sd.setTrainingConfig(config);
        sd.setListeners(new ScoreListener(1));
        History hist = sd.fit(trainIter, epoches);
        INDArray lossValues = hist.getLossCurve().getLossValues();
        assertTrue(lossValues.sumNumber().doubleValue() > 0.0);
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @Tag(TagNames.LONG_TEST)
    @Tag(TagNames.LARGE_RESOURCES)
    public void testTrainSmall() {

        int batchSize = 4;
        int modelDim = 8;

        SameDiff sd = SameDiff.create();

        SDVariable features = sd.placeHolder("features", FLOAT, batchSize, modelDim);
        SDVariable labels = sd.placeHolder("labels", FLOAT, batchSize, modelDim);
        SDVariable weights = sd.var("weights", new XavierInitScheme('c', modelDim, modelDim), FLOAT, modelDim, modelDim);
        SDVariable bias = sd.var("bias", new ZeroInitScheme('c'), FLOAT, modelDim);
        SDVariable predictions = sd.nn.linear("predictions", features, weights, bias);
        SDVariable loss = sd.loss.meanSquaredError("loss", labels, predictions, null);
        loss.markAsLoss();
        TrainingConfig config = new TrainingConfig.Builder()
                .updater(new Adam(0.1))
                .dataSetFeatureMapping("features")
                .dataSetLabelMapping("labels")
                .build();
        sd.setTrainingConfig(config);

        DataSetIterator iterator = new RandomDataSetIterator(1, new long[]{batchSize, modelDim}, new long[]{batchSize, modelDim}, INTEGER_0_10, INTEGER_0_10);

        sd.fit(iterator, 10);

    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @Tag(TagNames.LONG_TEST)
    @Tag(TagNames.LARGE_RESOURCES)
    public void irisTrainingSanityCheck(Nd4jBackend backend) {

        DataSetIterator iter = new IrisDataSetIterator(150, 150);
        DataSet d = iter.next();
        NormalizerStandardize std = new NormalizerStandardize();
        std.fit(d);
        iter.setPreProcessor(std);
        std.preProcess(d);
        DataSetIterator singleton = new SingletonDataSetIterator(d);
        for (String u : new String[]{"adam"}) {
            Nd4j.getRandom().setSeed(12345);
            log.info("Starting: " + u);
            SameDiff sd = SameDiff.create();

            SDVariable in = sd.placeHolder("input", FLOAT,  -1, 4);
            SDVariable label = sd.placeHolder("label", FLOAT, -1, 3);

            SDVariable w0 = sd.var("w0", new XavierInitScheme('c', 4, 10), FLOAT, 4, 10);
            SDVariable b0 = sd.zero("b0", FLOAT, 1, 10);

            SDVariable w1 = sd.var("w1", new XavierInitScheme('c', 10, 3), FLOAT, 10, 3);
            SDVariable b1 = sd.zero("b1", FLOAT, 1, 3);

            SDVariable z0 = in.mmul(w0).add(b0);
            SDVariable a0 = sd.math().tanh(z0);
            SDVariable z1 = a0.mmul(w1).add("prediction", b1);
            SDVariable a1 = sd.nn().softmax(z1,-1);

            SDVariable diff = sd.math().squaredDifference(a1, label);
            SDVariable lossMse = diff.mul(diff).mean();

            IUpdater updater;
            switch (u) {
                case "sgd":
                    updater = new Sgd(3e-1);
                    break;
                case "adam":
                    updater = new Adam(1e-1);
                    break;
                case "nesterov":
                    updater = new Nesterovs(1e-1);
                    break;
                default:
                    throw new RuntimeException();
            }

            TrainingConfig conf = new TrainingConfig.Builder()
                    .updater(updater)
                    .dataSetFeatureMapping("input")
                    .dataSetLabelMapping("label")
                    .build();

            sd.setTrainingConfig(conf);

            sd.setListeners(new ScoreListener(1));

            sd.fit(singleton, 50);

            Evaluation e = new Evaluation();
            Map<String, List<IEvaluation>> evalMap = new HashMap<>();
            evalMap.put("prediction", Collections.singletonList(e));

            sd.evaluateMultiple(iter, evalMap);

            System.out.println(e.stats());

            double acc = e.accuracy();
            assertTrue( acc >= 0.75,u + " - " + acc);
        }
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testGradients() {
        SameDiff sd = SameDiff.create();

        SDVariable i0 = sd.placeHolder("i0", FLOAT, 2,5);
        SDVariable w0 = sd.var("w0", new OneInitScheme('c'), FLOAT, 5, 3);
        SDVariable b0 = sd.var("b0", new OneInitScheme('c'), FLOAT,3);

        SDVariable w1 = sd.var("w1", new OneInitScheme('c'), FLOAT, 3,3);
        SDVariable b1 = sd.var("b1", new OneInitScheme('c'), FLOAT,3);

        SDVariable i1 = i0.mmul(w0).add(b0);
        SDVariable i2 = i1.mmul(w1).add(b1).add(i1);
        SDVariable l = i2.sum();

        sd.setLossVariables(l);
        INDArray gd = sd.calculateGradients(Collections.singletonMap("i0",Nd4j.rand(2,5)),"w0").get("w0");
        assertTrue(gd.sumNumber().doubleValue() > 0.0);
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testLossReducePersist(Nd4jBackend backend) {
        SameDiff sd = SameDiff.create();
        SDVariable in = sd.placeHolder("in", FLOAT, -1, 784);
        SDVariable labels = sd.placeHolder("labels", FLOAT,-1,10);
        SDVariable w1 = sd.var("w1", Nd4j.rand(FLOAT, 784, 100));
        SDVariable b1 = sd.var("b1", Nd4j.rand(FLOAT, 100));
        SDVariable a1 = sd.nn.tanh(in.mmul(w1).add(b1));
        SDVariable w2 = sd.var("w2", Nd4j.rand(FLOAT, 100, 10));
        SDVariable b2 = sd.var("b2", Nd4j.rand(FLOAT, 10));
        SDVariable out = sd.nn.softmax("out", a1.mmul(w2).add(b2));
        sd.loss().logLoss("loss",labels,out,null, LossReduce.SUM,1e-3);
        File tmpDir = testDir.resolve("path.fb").toFile();
        sd.save(tmpDir,true);

        SameDiff load = SameDiff.load(tmpDir,true);
        assertEquals(sd,load);
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void irisTrainingEvalTest(Nd4jBackend backend) {

        DataSetIterator iter = new IrisDataSetIterator(150, 150);
        NormalizerStandardize std = new NormalizerStandardize();
        std.fit(iter);
        iter.setPreProcessor(std);

        Nd4j.getRandom().setSeed(12345);
        SameDiff sd = SameDiff.create();

        SDVariable in = sd.placeHolder("input", FLOAT, -1, 4);
        SDVariable label = sd.placeHolder("label", FLOAT, -1, 3);

        SDVariable w0 = sd.var("w0", new XavierInitScheme('c', 4, 10), FLOAT, 4, 10);
        SDVariable b0 = sd.zero("b0", FLOAT, 1, 10);

        SDVariable w1 = sd.var("w1", new XavierInitScheme('c', 10, 3), FLOAT, 10, 3);
        SDVariable b1 = sd.zero("b1", FLOAT, 1, 3);

        SDVariable z0 = in.mmul(w0).add(b0);
        SDVariable a0 = sd.math().tanh(z0);
        SDVariable z1 = a0.mmul(w1).add("prediction", b1);
        SDVariable a1 = sd.nn().softmax(z1);

        SDVariable diff = sd.math().squaredDifference(a1, label);
        SDVariable lossMse = diff.mul(diff).mean();

        TrainingConfig conf = new TrainingConfig.Builder()
                .l2(1e-4)
                .updater(new Adam(1e-2))
                .dataSetFeatureMapping("input")
                .dataSetLabelMapping("label")
                .trainEvaluation("prediction", 0, new Evaluation())
                .build();

        sd.setTrainingConfig(conf);

        History hist = sd.fit().train(iter, 50).exec();

        Evaluation e = hist.finalTrainingEvaluations().evaluation("prediction");

        System.out.println(e.stats());

        double acc = e.accuracy();

        assertTrue(acc >= 0.75,"Accuracy bad: " + acc);
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void irisTrainingValidationTest(Nd4jBackend backend) {

        DataSetIterator iter = new IrisDataSetIterator(150, 150);
        NormalizerStandardize std = new NormalizerStandardize();
        std.fit(iter);
        iter.setPreProcessor(std);

        DataSetIterator valIter = new IrisDataSetIterator(30, 60);
        NormalizerStandardize valStd = new NormalizerStandardize();
        valStd.fit(valIter);
        valIter.setPreProcessor(std);

        Nd4j.getRandom().setSeed(12345);
        SameDiff sd = SameDiff.create();

        SDVariable in = sd.placeHolder("input", FLOAT, -1, 4);
        SDVariable label = sd.placeHolder("label", FLOAT, -1, 3);

        SDVariable w0 = sd.var("w0", new XavierInitScheme('c', 4, 10), FLOAT, 4, 10);
        SDVariable b0 = sd.zero("b0", FLOAT, 1, 10);

        SDVariable w1 = sd.var("w1", new XavierInitScheme('c', 10, 3), FLOAT, 10, 3);
        SDVariable b1 = sd.zero("b1", FLOAT, 1, 3);

        SDVariable z0 = in.mmul(w0).add(b0);
        SDVariable a0 = sd.math().tanh(z0);
        SDVariable z1 = a0.mmul(w1).add("prediction", b1);
        SDVariable a1 = sd.nn().softmax(z1);

        SDVariable diff = sd.math().squaredDifference(a1, label);
        SDVariable lossMse = diff.mul(diff).mean();

        TrainingConfig conf = new TrainingConfig.Builder()
                .l2(1e-4)
                .updater(new Adam(1e-2))
                .dataSetFeatureMapping("input")
                .dataSetLabelMapping("label")
                .validationEvaluation("prediction", 0, new Evaluation())
                .build();

        sd.setTrainingConfig(conf);

        History hist = sd.fit().train(iter, 50).validate(valIter, 5).exec();

        Evaluation e = hist.finalValidationEvaluations().evaluation("prediction");

        System.out.println(e.stats());

        double acc = e.accuracy();

        assertTrue(acc >= 0.75,"Accuracy bad: " + acc);
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testTrainingMixedDtypes(){

        for (String u : new String[]{"adam", "nesterov", "adamax", "amsgrad"}) {

            SameDiff sd = SameDiff.create();
            SDVariable in = sd.placeHolder("in", FLOAT, -1, 4);

            SDVariable inHalf = in.castTo(DataType.HALF);
            SDVariable inDouble = in.castTo(DataType.DOUBLE);

            SDVariable wFloat = sd.var("wFloat", Nd4j.rand(FLOAT, 4, 3));
            SDVariable wDouble = sd.var("wDouble", Nd4j.rand(DataType.DOUBLE, 4, 3));
            SDVariable wHalf = sd.var("wHalf", Nd4j.rand(DataType.HALF, 4, 3));

            SDVariable outFloat = in.mmul(wFloat);
            SDVariable outDouble = inDouble.mmul(wDouble);
            SDVariable outHalf = inHalf.mmul(wHalf);

            SDVariable sum = outFloat.add(outDouble.castTo(FLOAT)).add(outHalf.castTo(FLOAT));

            SDVariable loss = sum.std(true);

            IUpdater updater;
            switch (u) {
                case "sgd":
                    updater = new Sgd(1e-2);
                    break;
                case "adam":
                    updater = new Adam(1e-2);
                    break;
                case "nesterov":
                    updater = new Nesterovs(1e-2);
                    break;
                case "adamax":
                    updater = new AdaMax(1e-2);
                    break;
                case "amsgrad":
                    updater = new AMSGrad(1e-2);
                    break;
                default:
                    throw new RuntimeException();
            }

            TrainingConfig conf = new TrainingConfig.Builder()
                    .l2(1e-4)
                    .updater(updater)
                    .dataSetFeatureMapping("in")
                    .markLabelsUnused()
                    .build();

            sd.setTrainingConfig(conf);

            DataSet ds = new DataSet(Nd4j.rand(FLOAT, 3, 4), null);

            for( int i=0; i<10; i++ ){
                sd.fit(ds);
            }
        }

    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void simpleClassification(Nd4jBackend backend) {
        double learning_rate = 0.001;
        int seed = 7;
        org.nd4j.linalg.api.rng.Random rng = Nd4j.getRandom();
        rng.setSeed(seed);
        INDArray x1_label1 = Nd4j.randn(3.0, 1.0, new long[]{1000}, rng);
        INDArray x2_label1 = Nd4j.randn(2.0, 1.0, new long[]{1000}, rng);
        INDArray x1_label2 = Nd4j.randn(7.0, 1.0, new long[]{1000}, rng);
        INDArray x2_label2 = Nd4j.randn(6.0, 1.0, new long[]{1000}, rng);

        INDArray x1s = Nd4j.concat(0, x1_label1, x1_label2);
        INDArray x2s = Nd4j.concat(0, x2_label1, x2_label2);

        SameDiff sd = SameDiff.create();
        INDArray ys = Nd4j.scalar(0.0).mul(x1_label1.length()).add(Nd4j.scalar(1.0).mul(x1_label2.length()));

        SDVariable X1 = sd.placeHolder("x1", DataType.DOUBLE, 2000);
        SDVariable X2 = sd.placeHolder("x2", DataType.DOUBLE, 2000);
        SDVariable y = sd.placeHolder("y", DataType.DOUBLE);
        SDVariable w = sd.var("w", DataType.DOUBLE, 3);

        // TF code:
        //cost = tf.reduce_mean(-tf.log(y_model * Y + (1 — y_model) * (1 — Y)))
        SDVariable y_model =
                sd.nn.sigmoid(w.get(SDIndex.point(2)).mul(X2).add(w.get(SDIndex.point(1)).mul(X1)).add(w.get(SDIndex.point(0))));
        SDVariable cost_fun =
                (sd.math.neg(sd.math.log(y_model.mul(y).add((sd.math.log(sd.constant(1.0).minus(y_model)).mul(sd.constant(1.0).minus(y)))))));
        SDVariable loss = sd.mean("loss", cost_fun);

        val updater = new Sgd(learning_rate);

        sd.setLossVariables("loss");
        sd.createGradFunction();
        val conf = new TrainingConfig.Builder()
                .updater(updater)
                .dataSetFeatureMapping("x1", "x2", "y")
                .markLabelsUnused()
                .build();

        MultiDataSet mds = new MultiDataSet(new INDArray[]{x1s, x2s, ys},null);

        sd.setTrainingConfig(conf);
        History history = sd.fit(new SingletonMultiDataSetIterator(mds), 1);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testTrainingEvalVarNotReqForLoss() {
        //If a variable is not required for the loss - normally it won't be calculated
        //But we want to make sure it IS calculated here - so we can perform evaluation on it

        SameDiff sd = SameDiff.create();
        SDVariable in = sd.placeHolder("in", FLOAT, -1, 4);
        SDVariable label = sd.placeHolder("label", FLOAT, -1, 3);
        SDVariable w = sd.var("w", Nd4j.rand(FLOAT, 4, 3));
        SDVariable z = in.mmul(w);
        SDVariable out = sd.nn.softmax("softmax", z);
        SDVariable loss = sd.loss.logLoss("loss", label, out);
        SDVariable notRequiredForLoss = sd.nn.softmax("notRequiredForLoss", z);

        sd.setTrainingConfig(TrainingConfig.builder()
                .updater(new Adam(0.001))
                .dataSetFeatureMapping("in")
                .dataSetLabelMapping("label")
                .trainEvaluation("notRequiredForLoss", 0, new Evaluation())
                .build());

//        sd.setListeners(new ScoreListener(1));

        DataSet ds = new DataSet(Nd4j.rand(FLOAT, 3, 4), Nd4j.createFromArray(new float[][]{{1,0,0}, {0,1,0}, {0,0,1}}));

        History h = sd.fit()
                .train(new SingletonDataSetIterator(ds), 4)
                .exec();

        List<Double> l = h.trainingEval(Evaluation.Metric.ACCURACY);
        assertEquals(4, l.size());
    }


    @Override
    public char ordering() {
        return 'c';
    }
}
