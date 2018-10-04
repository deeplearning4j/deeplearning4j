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

package org.nd4j.autodiff.opvalidation;

import lombok.extern.slf4j.Slf4j;
import org.junit.Test;
import org.nd4j.OpValidationSuite;
import org.nd4j.autodiff.loss.LossFunctions;
import org.nd4j.autodiff.loss.LossInfo;
import org.nd4j.autodiff.loss.LossReduce;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.validation.OpValidation;
import org.nd4j.autodiff.validation.TestCase;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.accum.distances.CosineDistance;
import org.nd4j.linalg.api.ops.random.impl.BernoulliDistribution;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertEquals;

@Slf4j
public class LossOpValidation extends BaseOpValidation {
    public LossOpValidation(Nd4jBackend backend) {
        super(backend);
    }

    @Test
    public void testLossSimple2d() {

        Nd4j.getRandom().setSeed(12345);

        List<String> failed = new ArrayList<>();

        for (String fn : new String[]{"absdiff", "cosine", "hinge", "huber", "log", "mse",
                "sigmoidxent", "sigmoidxent_smooth", "softmaxxent", "softmaxxent_smooth" /* "mpwse" */}) {

            for (LossReduce reduction : LossReduce.values()) {

                SameDiff sd = SameDiff.create();

                int nOut = 4;
                int minibatch = 10;
                SDVariable predictions = sd.var("in", new int[]{-1, nOut});
                SDVariable labels = sd.var("labels", new int[]{-1, nOut});

                INDArray predictionsArr = Nd4j.randn(minibatch, nOut);
                INDArray labelsArr = Nd4j.randn(minibatch, nOut);

                INDArray expOut = null;
                SDVariable loss = null;
                switch (fn) {
                    case "absdiff":
                        expOut = Transforms.abs(predictionsArr.sub(labelsArr));
                        loss = sd.lossAbsoluteDifference("loss", labels, predictions, reduction);
                        break;
                    case "cosine":
                        //Cosine _similarity_: dot(a,b)/(l2Norm(a) * l2Norm(b))
                        //Cosine distance = acos(cosine similarity) / pi
                        expOut = Nd4j.getExecutioner().exec(new CosineDistance(predictionsArr, labelsArr), 1);
                        loss = sd.lossCosineDistance("loss", labels, predictions, reduction, 1);
                        break;
                    case "hinge":
                        //0 or 1 labels, but -1 or 1 when calculating loss
                        //L = max(0, 1 - prediction * label)
                        Nd4j.getExecutioner().exec(new BernoulliDistribution(labelsArr, 0.5));
                        INDArray labelMinusOneToOne = labelsArr.mul(2).subi(1);
                        expOut = Transforms.max(predictionsArr.mul(labelMinusOneToOne).rsubi(1), 0);
                        loss = sd.lossHinge("loss", labels, predictions, reduction);
                        break;
                    case "huber":
                        //https://en.wikipedia.org/wiki/Huber_loss
                        double delta = 1.0;
                        INDArray absDiff = Transforms.abs(labelsArr.sub(predictionsArr));
                        INDArray diff = labelsArr.sub(predictionsArr);
                        INDArray lte = absDiff.lte(delta);
                        INDArray gt = absDiff.gt(delta);
                        expOut = diff.mul(diff).mul(0.5).muli(lte);
                        expOut.addi(absDiff.mul(delta).subi(-0.5 * delta * delta).mul(gt));
                        loss = sd.lossHuber("loss", labels, predictions, reduction, delta);
                        break;
                    case "log":
                        double eps = 1e-7;
                        //Loss loss aka negative log likelihood: For singe example, -sum_outputs label_i * log(p_i)
                        //Labels are random one-hot
                        labelsArr.assign(0);
                        for( int i=0; i<labelsArr.size(0); i++ ){
                            labelsArr.putScalar(i, i%labelsArr.size(1), 1.0);
                        }
                        predictionsArr = Nd4j.rand(predictionsArr.shape());
                        INDArray logP = Transforms.log(predictionsArr.add(eps), true);
                        expOut = labelsArr.mul(logP);
                        loss = sd.lossLog("loss", labels, predictions, null, reduction, eps);
                        break;
                    case "mse":
                        //To match TF, this is actually sum of squares - 1/numExamples (prediction-label)^2
                        INDArray sqDiff = labelsArr.sub(predictionsArr);
                        sqDiff.muli(sqDiff);
                        expOut = sqDiff;
                        loss = sd.lossMeanSquaredError("loss", labels, predictions, null, reduction);
                        break;
                    case "sigmoidxent_smooth":  //Sigmoid xent with label smoothing
                    case "sigmoidxent":
                        //-1/numExamples * (label * log(p) + (1-label) * log(1-p))
                        INDArray sigmoidPredictions = Transforms.sigmoid(predictionsArr, true);
                        Nd4j.getExecutioner().exec(new BernoulliDistribution(labelsArr, 0.5));
                        double lblSmoothing = fn.equals("sigmoidxent_smooth") ? 0.1 : 0.0;
                        if(fn.equals("sigmoidxent_smooth")){
                            labelsArr.muli(1.0 - lblSmoothing).addi(0.5 * lblSmoothing);
                        }
                        INDArray oneSubLabel = labelsArr.rsub(1.0);
                        INDArray logPr = Transforms.log(sigmoidPredictions);
                        INDArray log1SubPr = Transforms.log(sigmoidPredictions.rsub(1.0));
                        expOut = labelsArr.mul(logPr).addi(oneSubLabel.mul(log1SubPr)).negi();

                        loss = sd.lossSigmoidCrossEntropy("loss", labels, predictions, null, reduction, lblSmoothing);
                        break;
                    case "softmaxxent":
                    case "softmaxxent_smooth":
                        //Same as negative log likelihood, but apply softmax on predictions first: For singe example, -sum_outputs label_i * log(p_i)
                        //Labels are random one-hot
                        INDArray softmaxPredictions = Transforms.softmax(predictionsArr, true);
                        labelsArr.assign(0);
                        for( int i=0; i<labelsArr.size(0); i++ ){
                            labelsArr.putScalar(i, i%labelsArr.size(1), 1.0);
                        }
                        double lblSmooth2 = fn.equals("softmaxxent_smooth") ? 0.1 : 0.0;
                        if(fn.equals("softmaxxent_smooth")){
                            labelsArr.muli(1.0 - lblSmooth2).addi(0.5 * lblSmooth2);
                        }
                        double eps2 = 1e-5;
                        INDArray logP2 = Transforms.log(softmaxPredictions.add(eps2), true);
                        expOut = labelsArr.mul(logP2).negi();
                        loss = sd.lossSoftmaxCrossEntropy("loss", labels, predictions, null, reduction, lblSmooth2);
                        break;
                    case "mpwse":
                        throw new UnsupportedOperationException("Not implemented");
                    default:
                        throw new RuntimeException();
                }

                switch (reduction){
                    case SUM:
                        expOut = expOut.sum().reshape();
                        break;
                    case MEAN_BY_WEIGHT:
                    case MEAN_BY_NONZERO_WEIGHT_COUNT:
                        //No weights in this test
                        expOut = expOut.mean().reshape();
                        break;
                }


                String msg = "test: " + fn + ", reduction=" + reduction;
                log.info("*** Starting test: " + msg);


                sd.associateArrayWithVariable(predictionsArr, predictions);
                sd.associateArrayWithVariable(labelsArr, labels);

                TestCase tc = new TestCase(sd)
                        .expectedOutput("loss", expOut)
                        .gradientCheck(false)                       //TODO  https://github.com/deeplearning4j/deeplearning4j/issues/6517
                        .testFlatBufferSerialization(TestCase.TestSerialization.NONE)   //TODO Re-enable later
                        ;

                String error = OpValidation.validate(tc);
                if(error != null){
                    failed.add(msg + " - " + error);
                }
            }
        }

        assertEquals(failed.toString(), 0, failed.size());
    }

    @Test
    public void testLossWeights2d() {
        OpValidationSuite.ignoreFailing();

        String[] weightTypes = new String[]{"none", "per-example", "per-output", "per-example-output"};

        Nd4j.getRandom().setSeed(12345);

        int nOut = 4;
        int minibatch = 10;
        List<String> failed = new ArrayList<>();

        for (String weightType : weightTypes) {

            for (boolean binary : new boolean[]{true, false}) {  //Binary mask (like DL4J) or arbitrary weights?

                int[] weightShape;
                switch (weightType) {
                    case "none":
                        weightShape = null;
                        break;
                    case "per-example":
                        weightShape = new int[]{minibatch, 1};
                        break;
                    case "per-output":
                        weightShape = new int[]{1, nOut};
                        break;
                    case "per-example-output":
                        weightShape = new int[]{minibatch, nOut};
                        break;
                    default:
                        throw new RuntimeException("Unknown type: " + weightType);
                }

                INDArray weightArr = null;
                if (!"none".equals(weightType)) {
                    if (binary) {
                        weightArr = Nd4j.getExecutioner().exec(new BernoulliDistribution(Nd4j.createUninitialized(weightShape), 0.5));
                    } else {
                        weightArr = Nd4j.rand(weightShape).muli(2.0);
                    }
                }

                for (LossFunctions.Reduction reduction : new LossFunctions.Reduction[]{
                        LossFunctions.Reduction.MEAN_BY_COUNT, LossFunctions.Reduction.MEAN_BY_WEIGHT, LossFunctions.Reduction.SUM}) {

                    for (String fn : new String[]{"mse", "l1", "l2", "mcxent"}) {

                        SameDiff sd = SameDiff.create();


                        SDVariable input = sd.var("in", new int[]{-1, nOut});
                        SDVariable labels = sd.var("labels", new int[]{-1, nOut});
                        SDVariable weight = null;
                        if (!"none".equals(weightType)) {
                            weight = sd.var("weights", weightArr);
                        }

                        INDArray inputArr = Nd4j.randn(minibatch, nOut).muli(100);
                        INDArray labelsArr = Nd4j.randn(minibatch, nOut).muli(100);

                        LossInfo lossInfo;
                        switch (fn) {
                            case "mse":
                                lossInfo = LossFunctions.mse("out", input, labels, weight, reduction, 1);
                                break;
                            case "l1":
                                lossInfo = LossFunctions.l1("out", input, labels, weight, reduction, 1);
                                //L1 = sum abs error
                                break;
                            case "l2":
                                lossInfo = LossFunctions.l2("out", input, labels, weight, reduction, 1);
                                //L2 = sum squared error
                                break;
                            case "mcxent":
                                lossInfo = LossFunctions.mcxent("out", input, labels, weight, reduction, 1);
                                //mcxent = sum label * log(prob)
                                break;
                            default:
                                throw new RuntimeException();
                        }


                        String msg = "lossFn=" + fn + ", reduction=" + reduction + ", weightType=" + weightType + ", binaryWeight=" + binary;
                        log.info("*** Starting test: " + msg);

                        sd.associateArrayWithVariable(inputArr, input);
                        sd.associateArrayWithVariable(labelsArr, labels);
                        if (weight != null) {
                            sd.associateArrayWithVariable(weightArr, weight);
                        }

                        TestCase tc = new TestCase(sd);
                        String error = OpValidation.validate(tc);
                        if(error != null){
                            failed.add(name);
                        }
                    }
                }
            }
        }

        assertEquals(failed.toString(), 0, failed.size());
    }


    @Test
    public void testLossWeights3d() {
        OpValidationSuite.ignoreFailing();

        String[] weightTypes = new String[]{"none", "per-example", "per-output", "per-timestep",
                "per-example-output", "per-example-timestep", "per-output-timestep", "per-all"};

        Nd4j.getRandom().setSeed(12345);

        int nOut = 4;
        int minibatch = 10;
        int tsLength = 5;

        List<String> failed = new ArrayList<>();

        for (String weightType : weightTypes) {

            for (boolean binary : new boolean[]{true, false}) {  //Binary mask (like DL4J) or arbitrary weights?

                int[] weightShape;
                switch (weightType) {
                    case "none":
                        weightShape = null;
                        break;
                    case "per-example":
                        weightShape = new int[]{minibatch, 1, 1};
                        break;
                    case "per-output":
                        weightShape = new int[]{1, nOut, 1};
                        break;
                    case "per-timestep":
                        weightShape = new int[]{1,1, tsLength};
                        break;
                    case "per-example-output":
                        weightShape = new int[]{minibatch, nOut, 1};
                        break;
                    case "per-example-timestep":
                        weightShape = new int[]{minibatch, 1, nOut};
                        break;
                    case "per-output-timestep":
                        weightShape = new int[]{1, nOut, tsLength};
                        break;
                    case "per-all":
                        weightShape = new int[]{minibatch, nOut, tsLength};
                        break;
                    default:
                        throw new RuntimeException("Unknown type: " + weightType);
                }

                INDArray weightArr = null;
                if (!"none".equals(weightType)) {
                    if (binary) {
                        weightArr = Nd4j.getExecutioner().exec(new BernoulliDistribution(Nd4j.createUninitialized(weightShape), 0.5));
                    } else {
                        weightArr = Nd4j.rand(weightShape).muli(2.0);
                    }
                }

                for (LossFunctions.Reduction reduction : new LossFunctions.Reduction[]{
                        LossFunctions.Reduction.MEAN_BY_COUNT, LossFunctions.Reduction.MEAN_BY_WEIGHT, LossFunctions.Reduction.SUM}) {

                    for (String fn : new String[]{"mse", "l1", "l2", "mcxent"}) {

                        SameDiff sd = SameDiff.create();


                        SDVariable input = sd.var("in", new int[]{-1, nOut, -1});
                        SDVariable labels = sd.var("labels", new int[]{-1, nOut, -1});
                        SDVariable weight = null;
                        if (!"none".equals(weightType)) {
                            weight = sd.var("weights", weightArr);
                        }

                        INDArray inputArr = Nd4j.randn(new int[]{minibatch, nOut, tsLength}).muli(10);
                        INDArray labelsArr = Nd4j.randn(new int[]{minibatch, nOut, tsLength}).muli(10);

                        LossInfo lossInfo;
                        switch (fn) {
                            case "mse":
                                lossInfo = LossFunctions.mse("out", input, labels, weight, reduction, 1, 2);
                                break;
                            case "l1":
                                lossInfo = LossFunctions.l1("out", input, labels, weight, reduction, 1, 2);
                                //L1 = sum abs error
                                break;
                            case "l2":
                                lossInfo = LossFunctions.l2("out", input, labels, weight, reduction, 1, 2);
                                //L2 = sum squared error
                                break;
                            case "mcxent":
                                lossInfo = LossFunctions.mcxent("out", input, labels, weight, reduction, 1, 2);
                                //mcxent = sum label * log(prob)
                                break;
                            default:
                                throw new RuntimeException();
                        }


                        String msg = "lossFn=" + fn + ", reduction=" + reduction + ", weightType=" + weightType + ", binaryWeight=" + binary;
                        log.info("*** Starting test: " + msg);

                        sd.associateArrayWithVariable(inputArr, input);
                        sd.associateArrayWithVariable(labelsArr, labels);
                        if (weight != null) {
                            sd.associateArrayWithVariable(weightArr, weight);
                        }

                        INDArray out = sd.execAndEndResult();
                        assertEquals(1, out.length());

                        TestCase tc = new TestCase(sd);
                        String error = OpValidation.validate(tc);
                        if(error != null){
                            failed.add(name);
                        }
                    }
                }
            }
        }

        assertEquals(failed.toString(), 0, failed.size());
    }

    @Test
    public void testLossWeights4d() {
        OpValidationSuite.ignoreFailing();

        String[] weightTypes = new String[]{"none", "per-example", "per-depth", "per-height", "per-width",
                "per-height-width", "per-depth-height", "per-depth-width", "per-example-depth", "per-example-height",
                "per-example-height-width", "per-all"};

        Nd4j.getRandom().setSeed(12345);

        //Assume NCHW format here
        int minibatch = 10;
        int depth = 4;
        int h = 5;
        int w = 6;
        List<String> failed = new ArrayList<>();

        for (String weightType : weightTypes) {

            for (boolean binary : new boolean[]{true, false}) {  //Binary mask (like DL4J) or arbitrary weights?

                int[] weightShape;
                switch (weightType) {
                    case "none":
                        weightShape = null;
                        break;
                    case "per-example":
                        weightShape = new int[]{minibatch, 1, 1, 1};
                        break;
                    case "per-depth":
                        weightShape = new int[]{1, depth, 1, 1};
                        break;
                    case "per-height":
                        weightShape = new int[]{1,1, h, 1};
                        break;
                    case "per-width":
                        weightShape = new int[]{1, 1, 1, w};
                        break;
                    case "per-height-width":
                        weightShape = new int[]{1, 1, h, w};
                        break;
                    case "per-depth-height":
                        weightShape = new int[]{1,depth, h, 1};
                        break;
                    case "per-depth-width":
                        weightShape = new int[]{1,depth, 1, w};
                        break;
                    case "per-example-depth":
                        weightShape = new int[]{minibatch, depth, 1, 1};
                        break;
                    case "per-example-height":
                        weightShape = new int[]{minibatch, 1, h, 1};
                        break;
                    case "per-example-height-width":
                        weightShape = new int[]{minibatch, 1, h, w};
                        break;
                    case "per-all":
                        weightShape = new int[]{minibatch, depth, h, w};
                        break;
                    default:
                        throw new RuntimeException("Unknown type: " + weightType);
                }

                INDArray weightArr = null;
                if (!"none".equals(weightType)) {
                    if (binary) {
                        weightArr = Nd4j.getExecutioner().exec(new BernoulliDistribution(Nd4j.createUninitialized(weightShape), 0.5));
                    } else {
                        weightArr = Nd4j.rand(weightShape).muli(2.0);
                    }
                }

                for (LossFunctions.Reduction reduction : new LossFunctions.Reduction[]{
                        LossFunctions.Reduction.MEAN_BY_COUNT, LossFunctions.Reduction.MEAN_BY_WEIGHT, LossFunctions.Reduction.SUM}) {

                    for (String fn : new String[]{"mse", "l1", "l2", "mcxent"}) {

                        SameDiff sd = SameDiff.create();


                        SDVariable input = sd.var("in", new int[]{-1, depth, -1, -1});
                        SDVariable labels = sd.var("labels", new int[]{-1, depth, -1, -1});
                        SDVariable weight = null;
                        if (!"none".equals(weightType)) {
                            weight = sd.var("weights", weightArr);
                        }

                        INDArray inputArr = Nd4j.randn(new int[]{minibatch, depth, h, w}).muli(10);
                        INDArray labelsArr = Nd4j.randn(new int[]{minibatch, depth, h, w}).muli(10);

                        LossInfo lossInfo;
                        switch (fn) {
                            case "mse":
                                lossInfo = LossFunctions.mse("out", input, labels, weight, reduction, 1, 2, 3);
                                break;
                            case "l1":
                                lossInfo = LossFunctions.l1("out", input, labels, weight, reduction, 1, 2, 3);
                                //L1 = sum abs error
                                break;
                            case "l2":
                                lossInfo = LossFunctions.l2("out", input, labels, weight, reduction, 1, 2, 3);
                                //L2 = sum squared error
                                break;
                            case "mcxent":
                                lossInfo = LossFunctions.mcxent("out", input, labels, weight, reduction, 1, 2, 3);
                                //mcxent = sum label * log(prob)
                                break;
                            default:
                                throw new RuntimeException();
                        }


                        String msg = "lossFn=" + fn + ", reduction=" + reduction + ", weightType=" + weightType + ", binaryWeight=" + binary;
                        log.info("*** Starting test: " + msg);

                        sd.associateArrayWithVariable(inputArr, input);
                        sd.associateArrayWithVariable(labelsArr, labels);
                        if (weight != null) {
                            sd.associateArrayWithVariable(weightArr, weight);
                        }

                        INDArray out = sd.execAndEndResult();
                        assertEquals(1, out.length());

                        TestCase tc = new TestCase(sd);

                        String error = OpValidation.validate(tc);
                        if(error != null){
                            failed.add(name);
                        }
                    }
                }
            }
        }
        assertEquals(failed.toString(), 0, failed.size());
    }
}
