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

package org.nd4j.autodiff.opvalidation;

import lombok.extern.slf4j.Slf4j;
import org.junit.Test;
import org.nd4j.OpValidationSuite;
import org.nd4j.autodiff.loss.LossReduce;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.validation.OpValidation;
import org.nd4j.autodiff.validation.TestCase;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.CustomOp;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.random.impl.BernoulliDistribution;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.*;

import static org.junit.Assert.*;

@Slf4j
public class LossOpValidation extends BaseOpValidation {
    public LossOpValidation(Nd4jBackend backend) {
        super(backend);
    }

    @Override
    public long getTimeoutMilliseconds() {
        return Long.MAX_VALUE;
    }

    // All tested Loss Ops have backprop at the moment 2019/01/30
    public static final Set<String> NO_BP_YET = new HashSet<>();

    @Test
    public void testLoss2d() {
        final List<String> oneDimensionalOutputFns = Arrays.asList("cosine", "mpwse", "softmaxxent", "softmaxxent_smooth", "mpwse", "sparsesoftmax");

        Nd4j.getRandom().setSeed(12345);

        List<String> failed = new ArrayList<>();

        int totalRun = 0;
        for (String fn : new String[]{
                "log_poisson", "log_poisson_full",
                "absdiff", "cosine", "hinge", "huber", "log", "mse",
                "sigmoidxent", "sigmoidxent_smooth", "softmaxxent", "softmaxxent_smooth", "mpwse",
                "sparsesoftmax"
                }) {


            for(String weights : new String[]{"none", "scalar", "perExample", "perOutput"}) {
                if(weights.equals("perOutput") && oneDimensionalOutputFns.contains(fn))
                    continue;   //Skip this combination (not possible)

                for (LossReduce reduction : LossReduce.values()) {
                    if((fn.equals("softmaxxent") || fn.equals("softmaxxent_smooth")) && reduction == LossReduce.NONE)
                        continue;       //Combination not supported (doesn't make sense)

                    if(fn.equals("sparsesoftmax") && (!weights.equals("none") || reduction != LossReduce.SUM) )
                        continue;   //sparse softmax doesn't support weights or reduction confic

                    SameDiff sd = SameDiff.create();

                    int nOut = 4;
                    int minibatch = 10;
                    SDVariable predictions = sd.var("in", DataType.DOUBLE, minibatch, nOut);
                    SDVariable labels;
                    if("sparsesoftmax".equalsIgnoreCase(fn)){
                        labels = sd.var("labels", DataType.INT, -1);
                    } else {
                        //ALl other loss functions
                        labels = sd.var("labels", DataType.DOUBLE, -1, nOut);
                    }

                    SDVariable w;
                    INDArray wArrBroadcast;
                    switch (weights){
                        case "none":
                            w = null;
                            wArrBroadcast = Nd4j.ones(DataType.DOUBLE, minibatch, nOut);
                            break;
                        case "scalar":
                            w = sd.var("weights", Nd4j.scalar(DataType.DOUBLE, 1.0));
                            wArrBroadcast = Nd4j.valueArrayOf(minibatch, nOut, 1.0).castTo(DataType.DOUBLE);
                            break;
                        case "perExample":
                            INDArray wpe = Nd4j.create(new double[]{0,0,1,1,2,2,3,3,4,4});
                            if(!fn.equals("softmaxxent") && !fn.equals("softmaxxent_smooth")){
                                //Softmaxxent only supports rank 1 not rank 2??
                                wpe = wpe.reshape(minibatch, 1);
                            }
                            w = sd.var("weights", wpe);
                            wArrBroadcast = Nd4j.create(DataType.DOUBLE, minibatch, nOut).addiColumnVector(w.getArr());
                            break;
                        case "perOutput":
                            w = sd.var("weights", Nd4j.create(new double[][]{
                                    {0,0,0,0}, {0,0,1,1}, {1,1,0,0}, {1,1,1,1}, {1,1,1,1},
                                    {2,2,2,2}, {2,2,2,2}, {2,2,2,2}, {2,2,2,2}, {2,2,2,2}}));
                            wArrBroadcast = w.getArr();
                            break;
                        default:
                            throw new RuntimeException();
                    }
                    INDArray wArr = w == null ? Nd4j.scalar(DataType.DOUBLE, 1.0) : w.getArr();


                    INDArray predictionsArr = Nd4j.randn(DataType.DOUBLE, minibatch, nOut);
                    INDArray labelsArr = Nd4j.randn(DataType.DOUBLE, minibatch, nOut);

                    INDArray expOut = null;
                    SDVariable loss = null;
                    switch (fn) {
                        case "absdiff":
                            expOut = Transforms.abs(predictionsArr.sub(labelsArr));
                            loss = sd.loss().absoluteDifference("loss", labels, predictions, w, reduction);
                            break;
                        case "cosine":
                            //Cosine _similarity_: dot(a,b)/(l2Norm(a) * l2Norm(b))
                            //Cosine distance = 1 - cosineSimilarity
                            //NOTE: both we and TF assume the inputs are normalized
                            predictionsArr.diviColumnVector(predictionsArr.norm2(1));
                            labelsArr.diviColumnVector(labelsArr.norm2(1));
                            expOut = predictionsArr.mul(labelsArr).sum(1).rsub(1.0).reshape(10,1);
                            loss = sd.loss().cosineDistance("loss", labels, predictions, w, reduction, 1);
                            break;
                        case "hinge":
                            //0 or 1 labels, but -1 or 1 when calculating loss
                            //L = max(0, 1 - prediction * label)
                            Nd4j.getExecutioner().exec(new BernoulliDistribution(labelsArr, 0.5));
                            INDArray labelMinusOneToOne = labelsArr.mul(2).subi(1);
                            expOut = Transforms.max(predictionsArr.mul(labelMinusOneToOne).rsubi(1), 0);
                            loss = sd.loss().hingeLoss("loss", labels, predictions, w, reduction);
                            break;
                        case "huber":
                            //https://en.wikipedia.org/wiki/Huber_loss
                            double delta = 1.0;
                            INDArray diff = labelsArr.sub(predictionsArr);
                            INDArray absDiff = Transforms.abs(diff);
                            INDArray lte = absDiff.lte(delta).castTo(DataType.DOUBLE);
                            INDArray gt = absDiff.gt(delta).castTo(DataType.DOUBLE);
                            expOut = diff.mul(diff).mul(0.5).muli(lte);
                            expOut.addi(absDiff.mul(delta).subi(0.5 * delta * delta).mul(gt));
                            loss = sd.loss().huberLoss("loss", labels, predictions, w, reduction, delta);
                            break;
                        case "log":
                            double eps = 1e-7;
                            //Loss loss aka binary cross entropy loss
                            //Labels are random bernoulli
                            Nd4j.getExecutioner().exec(new BernoulliDistribution(labelsArr, 0.5));
                            predictionsArr = Nd4j.rand(predictionsArr.shape()).muli(0.8).addi(0.1);
                            INDArray logP = Transforms.log(predictionsArr.add(eps), true);
                            INDArray log1p = Transforms.log(predictionsArr.rsub(1.0).add(eps), true);
                            expOut = labelsArr.mul(logP).addi(labelsArr.rsub(1).mul(log1p)).negi();
                            loss = sd.loss().logLoss("loss", labels, predictions, w, reduction, eps);
                            break;
                        case "log_poisson":
                            predictionsArr = Transforms.log(Transforms.abs(predictionsArr));
                            labelsArr = Transforms.abs(labelsArr);
                            expOut = Transforms.exp(predictionsArr).sub(labelsArr.mul(predictionsArr));
                            loss = sd.loss().logPoisson("loss", labels, predictions, w, reduction,false);
                            break;
                        case "log_poisson_full":
                            predictionsArr = Transforms.log(Transforms.abs(predictionsArr));
                            labelsArr = Transforms.abs(labelsArr);
                            expOut = Transforms.exp(predictionsArr)
                                    .sub(labelsArr.mul(predictionsArr))
                                    .add(labelsArr.mul(Transforms.log(labelsArr)))
                                    .sub(labelsArr)
                                    .add(Transforms.log(labelsArr.mul(Math.PI * 2)).mul(0.5));
                            loss = sd.loss().logPoisson("loss", labels, predictions, w, reduction,true);
                            break;
                        case "mse":
                            //To match TF, this is actually sum of squares - 1/numExamples (prediction-label)^2
                            INDArray sqDiff = labelsArr.sub(predictionsArr);
                            sqDiff.muli(sqDiff);
                            expOut = sqDiff;
                            loss = sd.loss().meanSquaredError("loss", labels, predictions, w, reduction);
                            break;
                        case "sigmoidxent_smooth":  //Sigmoid xent with label smoothing
                        case "sigmoidxent":
                            //-1/numExamples * (label * log(p) + (1-label) * log(1-p))
                            Nd4j.getExecutioner().exec(new BernoulliDistribution(labelsArr, 0.5));
                            double lblSmoothing = fn.equals("sigmoidxent_smooth") ? 0.3 : 0.0;
                            INDArray labelArrCopy = labelsArr.dup();
                            if (fn.equals("sigmoidxent_smooth")) {
                                labelArrCopy.muli(1.0 - lblSmoothing).addi(0.5 * lblSmoothing);
                            }

                            INDArray onePlusExpNegX = Transforms.log(Transforms.exp(predictionsArr.neg()).add(1.0));
                            expOut = predictionsArr.mul(labelArrCopy.rsub(1.0)).add(onePlusExpNegX);

                            loss = sd.loss().sigmoidCrossEntropy("loss", labels, predictions, w, reduction, lblSmoothing);
                            break;
                        case "softmaxxent":
                        case "softmaxxent_smooth":
                            //Same as negative log likelihood, but apply softmax on predictions first: For singe example, -sum_outputs label_i * log(p_i)
                            //Labels are random one-hot
                            //Note that output is shape [minibatch] for NONE reduction, or scalar otherwise
                            INDArray softmaxPredictions = Transforms.softmax(predictionsArr, true);
                            labelsArr.assign(0);
                            for (int i = 0; i < labelsArr.size(0); i++) {
                                labelsArr.putScalar(i, i % labelsArr.size(1), 1.0);
                            }
                            double lblSmooth2 = fn.equals("softmaxxent_smooth") ? 0.1 : 0.0;
                            INDArray labelsArrCopy = labelsArr.dup();
                            if (fn.equals("softmaxxent_smooth")) {
                                labelsArrCopy.muli(1.0 - lblSmooth2).addi(lblSmooth2 / labelsArrCopy.size(1));
                            }
                            INDArray logP2 = Transforms.log(softmaxPredictions, true);
                            expOut = labelsArrCopy.mul(logP2).negi().sum(1);
                            loss = sd.loss().softmaxCrossEntropy("loss", labels, predictions, w, reduction, lblSmooth2);
                            break;
                        case "mpwse":
                            expOut = Nd4j.create(labelsArr.size(0), 1);
                            double n = (double) labelsArr.size(1);
                            for(int example = 0; example < labelsArr.size(0); example++){
                                for(int i = 0; i < labelsArr.size(1); i++){
                                    for(int k = 0; k < labelsArr.size(1); k++){
                                        if(i != k){
                                            double y_i = predictionsArr.getDouble(example, i);
                                            double y_k = predictionsArr.getDouble(example, k);
                                            double q_i = labelsArr.getDouble(example, i);
                                            double q_k = labelsArr.getDouble(example, k);
                                            double add = Math.pow(((y_i-y_k)-(q_i-q_k)), 2);
                                            expOut.putScalar(example, expOut.getDouble(example) + add);
                                        }
                                    }
                                }
                            }

                            expOut.muli(1/((n*(n-1)) / 2));

                            loss = sd.loss().meanPairwiseSquaredError("loss", labels, predictions,w, reduction);
                            break;
                        case "sparsesoftmax":
                            labelsArr = Nd4j.create(DataType.DOUBLE, minibatch);
                            INDArray oneHot = Nd4j.create(DataType.DOUBLE, minibatch, nOut);
                            for( int i=0; i<minibatch; i++ ){
                                labelsArr.putScalar(i, i%nOut);
                                oneHot.putScalar(i, i%nOut, 1.0);
                            }

                            INDArray softmaxPredictions2 = Transforms.softmax(predictionsArr, true);
                            INDArray logP2_2 = Transforms.log(softmaxPredictions2, true);
                            expOut = oneHot.mul(logP2_2).negi().sum(1);

                            loss = sd.loss().sparseSoftmaxCrossEntropy(predictions, labels).sum("loss");
                            break;

                        default:
                            throw new RuntimeException();
                    }

                    switch (weights){
                        case "none":    //No changes
                            break;
                        case "scalar":
                            expOut.muli(wArr.getDouble(0));
                            break;
                        case "perExample":
                            expOut.muliColumnVector(wArr);
                            break;
                        case "perOutput":
                            expOut.muli(wArr);
                            break;
                        default:
                            throw new RuntimeException();
                    }

                    INDArray expOutBefore = expOut;
                    switch (reduction) {
                        case SUM:
                            expOut = expOut.sum().reshape();
                            break;
                        case MEAN_BY_WEIGHT:
                            if(oneDimensionalOutputFns.contains(fn)){
                                //1d output, not 2d
                                expOut = expOut.sum().divi(wArrBroadcast.getColumn(0).sumNumber().doubleValue());
                            } else {
                                expOut = expOut.sum().divi(wArrBroadcast.sumNumber().doubleValue());
                            }
                            break;
                        case MEAN_BY_NONZERO_WEIGHT_COUNT:
                            if(oneDimensionalOutputFns.contains(fn)) {
                                //1d output, not 2d
                                int countNonZero = wArrBroadcast.getColumn(0).neq(0.0).castTo(DataType.DOUBLE).sumNumber().intValue();
                                expOut = expOut.sum().divi(countNonZero);
                            } else {
                                int countNonZero = wArrBroadcast.neq(0.0).castTo(DataType.DOUBLE).sumNumber().intValue();
                                expOut = expOut.sum().divi(countNonZero);
                            }
                            break;
                    }


                    String msg = "test: " + fn + ", reduction=" + reduction + ", weights=" + weights;
                    log.info("*** Starting test: " + msg);


                    sd.associateArrayWithVariable(predictionsArr, predictions);
                    sd.associateArrayWithVariable(labelsArr, labels);

                    if(reduction == LossReduce.NONE){
                        //Sum to make scalar output for gradient check...
                        loss = loss.sum();
                    }

                    boolean doGradCheck = true;
                    if (OpValidationSuite.IGNORE_FAILING && NO_BP_YET.contains(fn)) {
                        log.warn("--- Skipping gradient check for: {} ---", fn);
                        doGradCheck = false;
                    }

                    TestCase tc = new TestCase(sd)
                            .expectedOutput("loss", expOut)
                            .gradientCheck(doGradCheck)
                            .testFlatBufferSerialization(TestCase.TestSerialization.BOTH);

                    if(reduction == LossReduce.MEAN_BY_NONZERO_WEIGHT_COUNT && !weights.equals("none")){
                        tc = tc.gradCheckMask(Collections.singletonMap("weights", w.getArr().neq(0)));
                    }

                    if(fn.equals("sparsesoftmax")){
                        tc.gradCheckSkipVariables("labels");
                    }

                    String error;
                    try {
                        error = OpValidation.validate(tc);
                    } catch (Throwable t){
                        log.error("Failed: {}", msg, t);
                        error = msg + ": " + t.getMessage();
                    }
                    if (error != null) {
                        failed.add(msg + ": " + error);
                    }
                    totalRun++;
                }
            }
        }

        assertEquals(failed.size() + " of " + totalRun + " failed: " + failed.toString(), 0, failed.size());
    }


    @Test
    public void testCosineDistance(){
        INDArray arr = Nd4j.create(new double[][]{{-0.3, -0.2, -0.1}, {0, 0.1, 0.2}});
        INDArray label = Nd4j.create(new double[][]{{1.0, 2.0, 3.0}, {-1.0, 2.0, 1.0}});
        INDArray w = Nd4j.create(new double[][]{{0},{1}});
        INDArray out = Nd4j.scalar(0.0);

        CustomOp op = DynamicCustomOp.builder("cosine_distance_loss")
                .addInputs(arr, w, label)
                .addOutputs(out)
                .addIntegerArguments(2, 1) //weighted mean, dimension 1
                .build();
        Nd4j.getExecutioner().exec(op);

        INDArray exp = Nd4j.scalar(0.6);    //https://github.com/deeplearning4j/deeplearning4j/issues/6532
        assertEquals(exp, out);
    }

    @Test
    public void testL2Loss(){

        for( int rank=0; rank<=3; rank++ ){
            long[] shape;
            switch (rank){
                case 0:
                    shape = new long[0];
                    break;
                case 1:
                    shape = new long[]{5};
                    break;
                case 2:
                    shape = new long[]{3,4};
                    break;
                case 3:
                    shape = new long[]{2,3,4};
                    break;
                case 4:
                    shape = new long[]{2,3,2,3};
                    break;
                default:
                    throw new RuntimeException();
            }
            INDArray arr = Nd4j.rand(DataType.DOUBLE, shape);

            SameDiff sd = SameDiff.create();
            SDVariable in = sd.var("v", arr);
            SDVariable loss = sd.loss().l2Loss("loss", in);

            INDArray exp = arr.mul(arr).sum().muli(0.5);

            TestCase tc = new TestCase(sd)
                    .expectedOutput("loss", exp)
                    .gradientCheck(true)
                    .testFlatBufferSerialization(TestCase.TestSerialization.BOTH);

            String err = OpValidation.validate(tc);
            assertNull(err);
        }
    }

    @Test
    public void testNonZeroResult() {
        INDArray predictions = Nd4j.rand(DataType.DOUBLE, 10, 5);
        INDArray w = Nd4j.scalar(1.0);
        INDArray label = Nd4j.rand(DataType.DOUBLE, 10, 5);
        final INDArray zero = Nd4j.scalar(0.);
        final INDArray zeroBp = Nd4j.zerosLike(predictions);

        final String[] lossOps = {
                "absolute_difference_loss",
                "cosine_distance_loss",
                "mean_pairwssqerr_loss",
                "mean_sqerr_loss",
                "sigm_cross_entropy_loss",
                "hinge_loss",
                "huber_loss",
                "log_loss",
                "softmax_cross_entropy_loss"
        };

        for (String lossOp : lossOps) {
            for (int reductionMode : new int[]{1, 2, 3}) {
                INDArray out = Nd4j.scalar(0.0);
                CustomOp op = DynamicCustomOp.builder(lossOp)
                        .addInputs(predictions, w, label)
                        .addOutputs(out)
                        .addIntegerArguments(
                                reductionMode,
                                0 // for cosine_distance_loss
                        )
                        .addFloatingPointArguments(1.0) // for sigm_cross_entropy_loss
                        .build();
                Nd4j.getExecutioner().exec(op);

                assertNotEquals(lossOp + " returns zero result. Reduction Mode " + reductionMode, out, zero);
            }
        }

        final String[] lossBPOps = {"absolute_difference_loss", "cosine_distance_loss", "sigm_cross_entropy_loss", "log_loss", "mean_sqerr_loss", "sigm_cross_entropy_loss", "softmax_cross_entropy_loss"};
        for (String lossOp : lossBPOps) {
            for (int reductionMode : new int[]{1, 2, 3}) {
                INDArray outBP = Nd4j.zerosLike(predictions);
                CustomOp op = DynamicCustomOp.builder(lossOp + "_grad")
                        .addInputs(predictions, w, label)
                        .addOutputs(outBP, Nd4j.zerosLike(w), Nd4j.zerosLike(label))
                        .addIntegerArguments(
                                reductionMode,
                                0 // for cosine_distance_loss
                        )
                        .addFloatingPointArguments(1.0) // for sigm_cross_entropy_loss
                        .build();
                Nd4j.getExecutioner().exec(op);

                assertNotEquals(lossOp + "_grad returns zero result. Reduction Mode " + reductionMode, outBP, zeroBp);
            }
        }
    }

    @Test
    public void TestStdLossMixedDataType(){
        // Default Data Type in this test suite is Double.
        // This test used to throw an Exception that we have mixed data types.

        SameDiff sd = SameDiff.create();
        SDVariable v = sd.placeHolder("x", DataType.FLOAT, 3,4);
        SDVariable loss = v.std(true);
    }
}
