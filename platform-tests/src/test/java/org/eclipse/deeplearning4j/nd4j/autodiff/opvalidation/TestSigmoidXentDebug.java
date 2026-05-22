package org.eclipse.deeplearning4j.nd4j.autodiff.opvalidation;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.loss.LossReduce;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.internal.SameDiffOp;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.impl.loss.AbsoluteDifferenceLoss;
import org.nd4j.linalg.api.ops.random.impl.BernoulliDistribution;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Isolation test for sigmoid cross entropy forward pass accumulation bug.
 * Calls sd.output() multiple times with the same inputs and checks that
 * the output values remain constant (not accumulating).
 */
@Slf4j
@NativeTag
public class TestSigmoidXentDebug {

    @Test
    public void testSigmoidXentForwardStability() {
        // Enable debug/verbose to trace every op
        Nd4j.getEnvironment().setDebug(true);
        Nd4j.getEnvironment().setVerbose(true);

        try {
            int minibatch = 10;
            int nOut = 4;

            SameDiff sd = SameDiff.create();
            // Disable DSP to match gradient check conditions
            sd.setDspAutoCompileEnabled(false);
            sd.setDspNativeAutoCompileEnabled(false);

            SDVariable predictions = sd.var("in", DataType.DOUBLE, minibatch, nOut);
            SDVariable labels = sd.placeHolder("labels", DataType.DOUBLE, minibatch, nOut);
            SDVariable w = sd.var("weights", Nd4j.ones(DataType.DOUBLE));

            // Create loss with SUM reduction
            SDVariable loss = sd.loss().sigmoidCrossEntropy("loss", labels, predictions, w, LossReduce.SUM, 0.0);

            // Initialize arrays
            INDArray predictionsArr = Nd4j.randn(DataType.DOUBLE, minibatch, nOut).muli(0.5);
            INDArray labelsArr = Nd4j.zeros(DataType.DOUBLE, minibatch, nOut);
            Nd4j.getExecutioner().exec(new BernoulliDistribution(labelsArr, 0.5));

            sd.associateArrayWithVariable(predictionsArr, predictions);

            Map<String, INDArray> placeholders = new HashMap<>();
            placeholders.put("labels", labelsArr);

            log.info("=== Predictions: {}", predictionsArr);
            log.info("=== Labels: {}", labelsArr);

            // Call sd.output() 5 times — values should be identical
            double[] scores = new double[5];
            for (int i = 0; i < 5; i++) {
                log.info("=== sd.output() call #{} ===", i);
                Map<String, INDArray> m = sd.output(placeholders, "loss");
                INDArray lossArr = m.get("loss");
                scores[i] = lossArr.sumNumber().doubleValue();
                log.info("=== Call #{}: loss = {}, lossArr = {}", i, scores[i], lossArr);
            }

            // All scores should be the same
            for (int i = 1; i < 5; i++) {
                double relDiff = Math.abs(scores[i] - scores[0]) / (Math.abs(scores[0]) + 1e-10);
                log.info("Score[0]={}, Score[{}]={}, relDiff={}", scores[0], i, scores[i], relDiff);
                assertEquals(scores[0], scores[i], 1e-6,
                    "Forward pass output changed between call 0 and call " + i +
                    ": " + scores[0] + " vs " + scores[i]);
            }
        } finally {
            Nd4j.getEnvironment().setDebug(false);
            Nd4j.getEnvironment().setVerbose(false);
        }
    }

    @Test
    public void testSigmoidXentForwardStabilityNoneReduction() {
        // Test with NONE reduction + sum (matching gradient check path)
        Nd4j.getEnvironment().setDebug(true);
        Nd4j.getEnvironment().setVerbose(true);

        try {
            int minibatch = 10;
            int nOut = 4;

            SameDiff sd = SameDiff.create();
            sd.setDspAutoCompileEnabled(false);
            sd.setDspNativeAutoCompileEnabled(false);

            SDVariable predictions = sd.var("in", DataType.DOUBLE, minibatch, nOut);
            SDVariable labels = sd.placeHolder("labels", DataType.DOUBLE, minibatch, nOut);
            SDVariable w = sd.var("weights", Nd4j.ones(DataType.DOUBLE));

            // NONE reduction returns [10,4], then sum to scalar
            SDVariable lossNone = sd.loss().sigmoidCrossEntropy("loss", labels, predictions, w, LossReduce.NONE, 0.0);
            SDVariable lossSum = lossNone.sum("loss_sum");

            INDArray predictionsArr = Nd4j.randn(DataType.DOUBLE, minibatch, nOut).muli(0.5);
            INDArray labelsArr = Nd4j.zeros(DataType.DOUBLE, minibatch, nOut);
            Nd4j.getExecutioner().exec(new BernoulliDistribution(labelsArr, 0.5));

            sd.associateArrayWithVariable(predictionsArr, predictions);

            Map<String, INDArray> placeholders = new HashMap<>();
            placeholders.put("labels", labelsArr);

            // Call sd.output() 5 times
            double[] scores = new double[5];
            for (int i = 0; i < 5; i++) {
                log.info("=== sd.output() call #{} ===", i);
                Map<String, INDArray> m = sd.output(placeholders, "loss_sum");
                INDArray lossArr = m.get("loss_sum");
                scores[i] = lossArr.sumNumber().doubleValue();
                log.info("=== Call #{}: loss_sum = {}", i, scores[i]);
            }

            for (int i = 1; i < 5; i++) {
                log.info("Score[0]={}, Score[{}]={}", scores[0], i, scores[i]);
                assertEquals(scores[0], scores[i], 1e-6,
                    "Forward pass output changed between call 0 and call " + i);
            }
        } finally {
            Nd4j.getEnvironment().setDebug(false);
            Nd4j.getEnvironment().setVerbose(false);
        }
    }

    @Test
    public void testSigmoidXentAfterCalculateGradients() {
        // This reproduces the gradient check pattern:
        // 1. calculateGradients() (adds backward ops)
        // 2. Repeated sd.output() for numerical gradient

        try {
            int minibatch = 10;
            int nOut = 4;

            SameDiff sd = SameDiff.create();
            sd.setDspAutoCompileEnabled(false);
            sd.setDspNativeAutoCompileEnabled(false);

            SDVariable predictions = sd.var("in", DataType.DOUBLE, minibatch, nOut);
            SDVariable labels = sd.placeHolder("labels", DataType.DOUBLE, minibatch, nOut);
            SDVariable w = sd.var("weights", Nd4j.ones(DataType.DOUBLE));

            SDVariable loss = sd.loss().sigmoidCrossEntropy("loss", labels, predictions, w, LossReduce.SUM, 0.0);

            INDArray predictionsArr = Nd4j.randn(DataType.DOUBLE, minibatch, nOut).muli(0.5);
            INDArray labelsArr = Nd4j.zeros(DataType.DOUBLE, minibatch, nOut);
            Nd4j.getExecutioner().exec(new BernoulliDistribution(labelsArr, 0.5));

            sd.associateArrayWithVariable(predictionsArr, predictions);

            Map<String, INDArray> placeholders = new HashMap<>();
            placeholders.put("labels", labelsArr);

            // Step 1: Calculate gradients (this adds backward ops to the graph)
            Set<String> varsNeedingGrads = new HashSet<>(Arrays.asList("in", "weights", "labels"));
            log.info("=== Calling calculateGradients ===");
            Map<String, INDArray> grads = sd.calculateGradients(placeholders, varsNeedingGrads);
            log.info("=== calculateGradients done, grad keys: {}", grads.keySet());

            // Step 2: Call sd.output() repeatedly — should be stable
            double eps = 1e-5;

            // IMPORTANT: Use the SameDiff copy (not the external reference).
            // ThreadSafeArrayHolder.setArray() detaches (copies) arrays, so
            // predictionsArr and sd.getArrForVarName("in") are different objects.
            INDArray sdArr = sd.getArrForVarName("in");
            double orig = sdArr.getDouble(0);
            log.info("=== predictionsArr identity: {}, sdArr identity: {}, same object: {}",
                    System.identityHashCode(predictionsArr),
                    System.identityHashCode(sdArr),
                    predictionsArr == sdArr);
            log.info("=== predictionsArr[0] = {}, sdArr[0] = {}", predictionsArr.getDouble(0), orig);

            // Score at original value
            Map<String, INDArray> m0 = sd.output(placeholders, "loss");
            double score0 = m0.get("loss").sumNumber().doubleValue();
            log.info("=== Original score: {}", score0);

            // Perturb the SameDiff copy directly (like GradCheckUtil does)
            sdArr.putScalar(0, orig + eps);
            double afterPut = sdArr.getDouble(0);
            log.info("=== After putScalar(0, {}): sdArr.getDouble(0) = {} (expected {})",
                    orig + eps, afterPut, orig + eps);

            Map<String, INDArray> mPlus = sd.output(placeholders, "loss");
            double scorePlus = mPlus.get("loss").sumNumber().doubleValue();
            log.info("=== scorePlus: {} (should differ from score0={})", scorePlus, score0);

            // Perturb element 0 by -eps (use sdArr, not predictionsArr)
            sdArr.putScalar(0, orig - eps);
            Map<String, INDArray> mMinus = sd.output(placeholders, "loss");
            double scoreMinus = mMinus.get("loss").sumNumber().doubleValue();
            log.info("=== scoreMinus: {}", scoreMinus);

            // Restore
            sdArr.putScalar(0, orig);

            double numericalGrad = (scorePlus - scoreMinus) / (2 * eps);
            double analyticGrad = grads.get("in").getDouble(0);
            log.info("=== numericalGrad: {}, analyticGrad: {}", numericalGrad, analyticGrad);
            log.info("=== scorePlus-scoreMinus: {}", scorePlus - scoreMinus);

            // The numerical grad should be close to analytic
            // If the forward pass is accumulating, scorePlus will be >> score0
            double relError = Math.abs(numericalGrad - analyticGrad) / (Math.abs(numericalGrad) + Math.abs(analyticGrad) + 1e-10);
            log.info("=== relError: {}", relError);

            // Scores should be close to score0 (only ±eps change to one element)
            assertTrue(Math.abs(scorePlus - score0) < 1.0,
                "scorePlus should be close to score0: " + scorePlus + " vs " + score0);
            assertTrue(Math.abs(scoreMinus - score0) < 1.0,
                "scoreMinus should be close to score0: " + scoreMinus + " vs " + score0);
            assertTrue(relError < 0.01,
                "Gradient check should pass: numerical=" + numericalGrad + " analytic=" + analyticGrad + " relError=" + relError);
        } finally {
            Nd4j.getEnvironment().setDebug(false);
            Nd4j.getEnvironment().setVerbose(false);
        }
    }

    @Test
    public void testPerturbationWithoutGradients() {
        // Test perturbation WITHOUT calculateGradients to isolate
        int minibatch = 10;
        int nOut = 4;

        SameDiff sd = SameDiff.create();
        sd.setDspAutoCompileEnabled(false);
        sd.setDspNativeAutoCompileEnabled(false);

        SDVariable predictions = sd.var("in", DataType.DOUBLE, minibatch, nOut);
        SDVariable labels = sd.placeHolder("labels", DataType.DOUBLE, minibatch, nOut);
        SDVariable w = sd.var("weights", Nd4j.ones(DataType.DOUBLE));

        SDVariable loss = sd.loss().sigmoidCrossEntropy("loss", labels, predictions, w, LossReduce.SUM, 0.0);

        INDArray predictionsArr = Nd4j.randn(DataType.DOUBLE, minibatch, nOut).muli(0.5);
        INDArray labelsArr = Nd4j.zeros(DataType.DOUBLE, minibatch, nOut);
        Nd4j.getExecutioner().exec(new BernoulliDistribution(labelsArr, 0.5));

        sd.associateArrayWithVariable(predictionsArr, predictions);

        Map<String, INDArray> placeholders = new HashMap<>();
        placeholders.put("labels", labelsArr);

        // Use the SameDiff copy (not the external reference) for perturbation
        INDArray sdArr = sd.getArrForVarName("in");
        log.info("=== [NoGrad] predictionsArr == sdArr: {}", predictionsArr == sdArr);

        double eps = 1e-5;
        double orig = sdArr.getDouble(0);

        // NO calculateGradients — just output
        Map<String, INDArray> m0 = sd.output(placeholders, "loss");
        double score0 = m0.get("loss").sumNumber().doubleValue();
        log.info("=== [NoGrad] Original score: {}", score0);

        // Perturb the SameDiff copy and output
        sdArr.putScalar(0, orig + eps);
        log.info("=== [NoGrad] After putScalar: sdArr.getDouble(0) = {}", sdArr.getDouble(0));
        Map<String, INDArray> mPlus = sd.output(placeholders, "loss");
        double scorePlus = mPlus.get("loss").sumNumber().doubleValue();
        log.info("=== [NoGrad] scorePlus: {} (should differ from {})", scorePlus, score0);

        sdArr.putScalar(0, orig - eps);
        Map<String, INDArray> mMinus = sd.output(placeholders, "loss");
        double scoreMinus = mMinus.get("loss").sumNumber().doubleValue();
        log.info("=== [NoGrad] scoreMinus: {}", scoreMinus);

        sdArr.putScalar(0, orig);

        double numericalGrad = (scorePlus - scoreMinus) / (2 * eps);
        log.info("=== [NoGrad] numericalGrad: {} (should be non-zero)", numericalGrad);

        assertNotEquals(score0, scorePlus, 1e-10,
            "[NoGrad] scorePlus should differ from score0 after perturbation");
        assertNotEquals(0.0, numericalGrad, 1e-10,
            "[NoGrad] numerical gradient should be non-zero");
    }

    @Test
    public void testMseForwardStability() {
        // Control test: MSE should NOT have this problem
        int minibatch = 10;
        int nOut = 4;

        SameDiff sd = SameDiff.create();
        sd.setDspAutoCompileEnabled(false);
        sd.setDspNativeAutoCompileEnabled(false);

        SDVariable predictions = sd.var("in", DataType.DOUBLE, minibatch, nOut);
        SDVariable labels = sd.placeHolder("labels", DataType.DOUBLE, minibatch, nOut);
        SDVariable w = sd.var("weights", Nd4j.ones(DataType.DOUBLE));

        SDVariable loss = sd.loss().meanSquaredError("loss", labels, predictions, w, LossReduce.SUM);

        INDArray predictionsArr = Nd4j.randn(DataType.DOUBLE, minibatch, nOut).muli(0.5);
        INDArray labelsArr = Nd4j.randn(DataType.DOUBLE, minibatch, nOut);

        sd.associateArrayWithVariable(predictionsArr, predictions);

        Map<String, INDArray> placeholders = new HashMap<>();
        placeholders.put("labels", labelsArr);

        double[] scores = new double[5];
        for (int i = 0; i < 5; i++) {
            Map<String, INDArray> m = sd.output(placeholders, "loss");
            scores[i] = m.get("loss").sumNumber().doubleValue();
            log.info("MSE Call #{}: loss = {}", i, scores[i]);
        }

        for (int i = 1; i < 5; i++) {
            assertEquals(scores[0], scores[i], 1e-6,
                "MSE forward pass should be stable");
        }
    }

    @Test
    public void testMseNonzeroWeightCountCrash() {
        // Test MEAN_BY_NONZERO_WEIGHT_COUNT with perExample weights
        // This crashes in TestLossOpValidation with "NDArray::reduceNumber LongOps: target array should be scalar"
        int minibatch = 10;
        int nOut = 4;

        SameDiff sd = SameDiff.create();
        sd.setDspAutoCompileEnabled(false);
        sd.setDspNativeAutoCompileEnabled(false);

        SDVariable predictions = sd.var("in", DataType.DOUBLE, minibatch, nOut);
        SDVariable labels = sd.var("labels", DataType.DOUBLE, minibatch, nOut);
        // perExample weights: [10] reshaped to [10,1]
        INDArray wpe = Nd4j.create(new double[]{0,0,1,1,2,2,3,3,4,4}).reshape(minibatch, 1);
        SDVariable w = sd.var("weights", wpe);

        SDVariable loss = sd.loss().meanSquaredError("loss", labels, predictions, w, LossReduce.MEAN_BY_NONZERO_WEIGHT_COUNT);

        INDArray predictionsArr = Nd4j.randn(DataType.DOUBLE, minibatch, nOut).muli(0.5);
        INDArray labelsArr = Nd4j.randn(DataType.DOUBLE, minibatch, nOut);

        sd.associateArrayWithVariable(predictionsArr, predictions);
        sd.associateArrayWithVariable(labelsArr, labels);

        // This should NOT crash
        Map<String, INDArray> out = sd.output(Collections.emptyMap(), "loss");
        double score = out.get("loss").getDouble(0);
        log.info("=== MSE MEAN_BY_NONZERO_WEIGHT_COUNT perExample score: {}", score);
        assertTrue(Double.isFinite(score), "Score should be finite: " + score);
    }

    @Test
    public void testMseReductionModes() {
        // Diagnostic: compare SUM vs MEAN_BY_WEIGHT to see if C++ ops
        // actually receive the correct reduction mode.
        // With unit weights, MEAN_BY_WEIGHT should be SUM / numElements.
        Nd4j.getRandom().setSeed(12345);
        int minibatch = 10;
        int nOut = 4;

        INDArray predictionsArr = Nd4j.randn(DataType.DOUBLE, minibatch, nOut).muli(0.5);
        INDArray labelsArr = Nd4j.randn(DataType.DOUBLE, minibatch, nOut);

        // Test with SUM reduction
        SameDiff sdSum = SameDiff.create();
        sdSum.setDspAutoCompileEnabled(false);
        sdSum.setDspNativeAutoCompileEnabled(false);
        SDVariable predSum = sdSum.var("in", predictionsArr.dup());
        SDVariable labSum = sdSum.var("labels", labelsArr.dup());
        SDVariable wSum = sdSum.var("weights", Nd4j.ones(DataType.DOUBLE));
        SDVariable lossSum = sdSum.loss().meanSquaredError("loss", labSum, predSum, wSum, LossReduce.SUM);

        // Check iArguments on the op - find the loss op by iterating
        for (Map.Entry<String, org.nd4j.autodiff.samediff.internal.SameDiffOp> e : sdSum.getOps().entrySet()) {
            org.nd4j.linalg.api.ops.DynamicCustomOp dco = (org.nd4j.linalg.api.ops.DynamicCustomOp) e.getValue().getOp();
            log.info("=== SUM graph op '{}' opName='{}' iArgs={}", e.getKey(), dco.opName(), java.util.Arrays.toString(dco.iArgs()));
        }

        Map<String, INDArray> sumOut = sdSum.output(Collections.emptyMap(), "loss");
        double sumScore = sumOut.get("loss").getDouble(0);
        log.info("=== MSE SUM score: {}", sumScore);

        // Test with MEAN_BY_WEIGHT reduction
        SameDiff sdMean = SameDiff.create();
        sdMean.setDspAutoCompileEnabled(false);
        sdMean.setDspNativeAutoCompileEnabled(false);
        SDVariable predMean = sdMean.var("in", predictionsArr.dup());
        SDVariable labMean = sdMean.var("labels", labelsArr.dup());
        SDVariable wMean = sdMean.var("weights", Nd4j.ones(DataType.DOUBLE));
        SDVariable lossMean = sdMean.loss().meanSquaredError("loss", labMean, predMean, wMean, LossReduce.MEAN_BY_WEIGHT);

        // Check iArguments on the op
        for (Map.Entry<String, org.nd4j.autodiff.samediff.internal.SameDiffOp> e : sdMean.getOps().entrySet()) {
            org.nd4j.linalg.api.ops.DynamicCustomOp dco = (org.nd4j.linalg.api.ops.DynamicCustomOp) e.getValue().getOp();
            log.info("=== MEAN graph op '{}' opName='{}' iArgs={}", e.getKey(), dco.opName(), java.util.Arrays.toString(dco.iArgs()));
        }

        Map<String, INDArray> meanOut = sdMean.output(Collections.emptyMap(), "loss");
        double meanScore = meanOut.get("loss").getDouble(0);
        log.info("=== MSE MEAN_BY_WEIGHT score: {}", meanScore);

        // With unit weights and [10,4] arrays, MEAN should be SUM/40
        double expectedMean = sumScore / (minibatch * nOut);
        log.info("=== Expected MEAN (SUM/{}): {}", minibatch * nOut, expectedMean);
        log.info("=== Ratio actual/expected: {}", meanScore / expectedMean);

        // Also test NONE reduction to get per-element values
        SameDiff sdNone = SameDiff.create();
        sdNone.setDspAutoCompileEnabled(false);
        sdNone.setDspNativeAutoCompileEnabled(false);
        SDVariable predNone = sdNone.var("in", predictionsArr.dup());
        SDVariable labNone = sdNone.var("labels", labelsArr.dup());
        SDVariable wNone = sdNone.var("weights", Nd4j.ones(DataType.DOUBLE));
        SDVariable lossNone = sdNone.loss().meanSquaredError("loss", labNone, predNone, wNone, LossReduce.NONE);

        for (Map.Entry<String, org.nd4j.autodiff.samediff.internal.SameDiffOp> e : sdNone.getOps().entrySet()) {
            org.nd4j.linalg.api.ops.DynamicCustomOp dco = (org.nd4j.linalg.api.ops.DynamicCustomOp) e.getValue().getOp();
            log.info("=== NONE graph op '{}' opName='{}' iArgs={}", e.getKey(), dco.opName(), java.util.Arrays.toString(dco.iArgs()));
        }

        Map<String, INDArray> noneOut = sdNone.output(Collections.emptyMap(), "loss");
        INDArray noneArr = noneOut.get("loss");
        log.info("=== MSE NONE shape: {}, sum: {}", java.util.Arrays.toString(noneArr.shape()), noneArr.sumNumber());

        // Verify consistency: NONE.sum() should equal SUM
        double noneSum = noneArr.sumNumber().doubleValue();
        assertEquals(noneSum, sumScore, 1e-6,
            "NONE.sum() should equal SUM: " + noneSum + " vs " + sumScore);

        // Verify MEAN_BY_WEIGHT is correct
        assertEquals(expectedMean, meanScore, 1e-6,
            "MEAN_BY_WEIGHT should equal SUM/" + (minibatch * nOut) +
            ": expected=" + expectedMean + " actual=" + meanScore +
            " (ratio=" + (meanScore / expectedMean) + ")");
    }

    @Test
    public void testSigmoidXentDirect() {
        Nd4j.getRandom().setSeed(12345);
        int minibatch = 10;
        int nOut = 4;

        INDArray logits = Nd4j.randn(DataType.DOUBLE, minibatch, nOut);
        INDArray labelsArr = Nd4j.randn(DataType.DOUBLE, minibatch, nOut);
        Nd4j.getExecutioner().exec(new org.nd4j.linalg.api.ops.random.impl.BernoulliDistribution(labelsArr, 0.5));

        log.info("=== logits first row: {}", logits.getRow(0));
        log.info("=== labels first row: {}", labelsArr.getRow(0));

        // Expected: max(x,0) - x*y + log(1+exp(-|x|))
        INDArray maxX0 = Transforms.max(logits, 0.0);
        INDArray xTimesY = logits.mul(labelsArr);
        INDArray logTerm = Transforms.log(Transforms.exp(Transforms.abs(logits).neg()).add(1.0));
        INDArray expected = maxX0.sub(xTimesY).add(logTerm);
        log.info("=== maxX0 first row: {}", maxX0.getRow(0));
        log.info("=== xTimesY first row: {}", xTimesY.getRow(0));
        log.info("=== logTerm first row: {}", logTerm.getRow(0));
        log.info("=== Expected first row: {}", expected.getRow(0));

        // Direct C++ op
        INDArray weightsArr = Nd4j.scalar(DataType.DOUBLE, 1.0);
        INDArray outputArr = Nd4j.create(DataType.DOUBLE, minibatch, nOut);
        org.nd4j.linalg.api.ops.DynamicCustomOp op = org.nd4j.linalg.api.ops.DynamicCustomOp.builder("sigm_cross_entropy_loss")
                .addInputs(logits, weightsArr, labelsArr)
                .addOutputs(outputArr)
                .addIntegerArguments(0)  // NONE reduction
                .addFloatingPointArguments(0.0)  // no label smoothing
                .build();
        Nd4j.getExecutioner().exec(op);
        log.info("=== Actual first row: {}", outputArr.getRow(0));
        log.info("=== Match? {}", expected.equalsWithEps(outputArr, 1e-6));
    }

    @Test
    public void testLogLossExpectedComputation() {
        // Test the log loss expected value computation from testLoss2d
        Nd4j.getRandom().setSeed(12345);
        int minibatch = 10;
        int nOut = 4;

        INDArray predictionsArr = Nd4j.randn(DataType.DOUBLE, minibatch, nOut);
        INDArray labelsArr = Nd4j.randn(DataType.DOUBLE, minibatch, nOut);

        // Replicate testLoss2d's log case
        double eps = 1e-7;
        Nd4j.getExecutioner().exec(new org.nd4j.linalg.api.ops.random.impl.BernoulliDistribution(labelsArr, 0.5));
        predictionsArr = Nd4j.rand(DataType.DOUBLE, predictionsArr.shape()).muli(0.8).addi(0.1);

        // Method 1: Original test code (chained in-place ops)
        INDArray logP = Transforms.log(predictionsArr.add(eps), true);
        INDArray log1p = Transforms.log(predictionsArr.rsub(1.0).add(eps), true);
        INDArray expOutOriginal = labelsArr.mul(logP).addi(labelsArr.rsub(1).mul(log1p)).negi();
        log.info("=== Method 1 (chained in-place): first row = {}", expOutOriginal.getRow(0));

        // Method 2: No in-place ops
        INDArray term1 = labelsArr.mul(logP);
        INDArray term2 = labelsArr.rsub(1.0).mul(log1p);
        INDArray expOutSafe = term1.add(term2).neg();
        log.info("=== Method 2 (no in-place): first row = {}", expOutSafe.getRow(0));

        // Method 3: Direct DynamicCustomOp
        INDArray weightsArr = Nd4j.scalar(DataType.DOUBLE, 1.0);
        INDArray outputArr = Nd4j.create(DataType.DOUBLE, minibatch, nOut);
        org.nd4j.linalg.api.ops.DynamicCustomOp op = org.nd4j.linalg.api.ops.DynamicCustomOp.builder("log_loss")
                .addInputs(predictionsArr, weightsArr, labelsArr)
                .addOutputs(outputArr)
                .addIntegerArguments(0)  // NONE reduction
                .addFloatingPointArguments(eps)
                .build();
        Nd4j.getExecutioner().exec(op);
        log.info("=== Method 3 (C++ op NONE): first row = {}", outputArr.getRow(0));

        // Compare
        log.info("=== Methods match? m1==m2: {}, m2==m3: {}, m1==m3: {}",
                expOutOriginal.equalsWithEps(expOutSafe, 1e-6),
                expOutSafe.equalsWithEps(outputArr, 1e-6),
                expOutOriginal.equalsWithEps(outputArr, 1e-6));
    }

    @Test
    public void testAbsDiffViaOpValidation() {
        // Exact replica of testLoss2d for absdiff + MEAN_BY_WEIGHT + weights=none
        int minibatch = 10;
        int nOut = 4;

        SameDiff sd = SameDiff.create();
        sd.setDspAutoCompileEnabled(false);
        sd.setDspNativeAutoCompileEnabled(false);
        SDVariable predictions = sd.var("in", DataType.DOUBLE, minibatch, nOut);
        SDVariable labels = sd.var("labels", DataType.DOUBLE, -1, nOut);

        // weights="none" → w=null → BaseLoss creates constant scalar 1.0
        INDArray wArrBroadcast = Nd4j.ones(DataType.DOUBLE, minibatch, nOut);

        INDArray predictionsArr = Nd4j.randn(DataType.DOUBLE, minibatch, nOut);
        INDArray labelsArr = Nd4j.randn(DataType.DOUBLE, minibatch, nOut);

        // expOut computation exactly like testLoss2d
        INDArray expOut = Transforms.abs(predictionsArr.sub(labelsArr));
        log.info("=== expOut shape: {}, sum: {}", Arrays.toString(expOut.shape()), expOut.sumNumber());

        // weights="none" → no multiplication

        // MEAN_BY_WEIGHT reduction
        double wSum = wArrBroadcast.sumNumber().doubleValue();
        log.info("=== wArrBroadcast sum: {}", wSum);
        INDArray sumResult = expOut.sum();
        log.info("=== sum() result: getDouble(0)={}, toString={}, shape={}, length={}", sumResult.getDouble(0), sumResult, Arrays.toString(sumResult.shape()), sumResult.length());
        log.info("=== sumResult data buffer length={}", sumResult.data().length());
        log.info("=== expOut data buffer length={}", expOut.data().length());
        log.info("=== same data buffer? {}", sumResult.data() == expOut.data());
        log.info("=== sumResult isView={}", sumResult.isView());
        double manualExpected = sumResult.getDouble(0) / wSum;
        log.info("=== manual division: {} / {} = {}", sumResult.getDouble(0), wSum, manualExpected);

        // Test: use div (not in-place) instead
        INDArray divResult = sumResult.div(wSum);
        log.info("=== div result: getDouble(0)={}", divResult.getDouble(0));

        // Test: use sumNumber().doubleValue() / wSum as a scalar
        double sumVal = expOut.sumNumber().doubleValue();
        log.info("=== sumNumber()={}", sumVal);
        INDArray manualScalar = Nd4j.scalar(DataType.DOUBLE, sumVal / wSum);
        log.info("=== manual scalar: {}", manualScalar.getDouble(0));

        sumResult.divi(wSum);
        log.info("=== sumResult after divi: getDouble(0)={}, toString={}", sumResult.getDouble(0), sumResult);

        // Use the manual scalar as expected
        expOut = manualScalar;
        log.info("=== expOut (using manual scalar): {}", expOut);

        SDVariable loss = sd.loss().absoluteDifference("loss", labels, predictions, null, LossReduce.MEAN_BY_WEIGHT);

        sd.associateArrayWithVariable(predictionsArr, predictions);
        sd.associateArrayWithVariable(labelsArr, labels);

        // Now run via OpValidation (same as testLoss2d)
        org.nd4j.autodiff.validation.TestCase tc = new org.nd4j.autodiff.validation.TestCase(sd)
                .expectedOutput("loss", expOut)
                .gradientCheck(false)
                .testFlatBufferSerialization(org.nd4j.autodiff.validation.TestCase.TestSerialization.BOTH);

        String error = org.nd4j.autodiff.validation.OpValidation.validate(tc);
        log.info("=== OpValidation result: {}", error);

        // Also run manually to compare
        Map<String, INDArray> placeholders = new HashMap<>();
        placeholders.put("in", predictionsArr);
        placeholders.put("labels", labelsArr);
        Map<String, INDArray> out = sd.output(placeholders, "loss");
        log.info("=== Manual sd.output result: {}", out.get("loss"));
        log.info("=== Expected: {}", expOut);

        assertNull(error, "OpValidation failed: " + error);
    }

    @Test
    public void testAbsDiffAllReductions() {
        Nd4j.getRandom().setSeed(12345);
        int minibatch = 10;
        int nOut = 4;

        INDArray predictionsArr = Nd4j.randn(DataType.DOUBLE, minibatch, nOut);
        INDArray labelsArr = Nd4j.randn(DataType.DOUBLE, minibatch, nOut);
        INDArray weightsArr = Nd4j.scalar(DataType.DOUBLE, 1.0);

        // Test ALL reduction modes via direct DynamicCustomOp
        INDArray absDiff = Transforms.abs(predictionsArr.sub(labelsArr));
        double rawSum = absDiff.sumNumber().doubleValue();
        double numElems = minibatch * nOut; // 40

        for (int reductionMode = 1; reductionMode <= 3; reductionMode++) {
            INDArray directOut = Nd4j.scalar(DataType.DOUBLE, 0.0);
            DynamicCustomOp directOp = DynamicCustomOp.builder("absolute_difference_loss")
                    .addInputs(predictionsArr, weightsArr, labelsArr)
                    .addOutputs(directOut)
                    .addIntegerArguments(reductionMode)
                    .build();
            Nd4j.exec(directOp);
            double actual = directOut.getDouble(0);

            double expected;
            String name;
            switch (reductionMode) {
                case 1: expected = rawSum; name = "SUM"; break;
                case 2: expected = rawSum / numElems; name = "MEAN_BY_WEIGHT"; break;
                case 3: expected = rawSum / numElems; name = "MEAN_BY_NONZERO_WEIGHT_COUNT"; break;
                default: throw new RuntimeException();
            }

            double ratio = actual / expected;
            log.info("=== mode={} ({}): expected={}, actual={}, ratio={}", reductionMode, name, expected, actual, ratio);
            assertEquals(expected, actual, 1e-5, name + " mismatch");
        }

        // Now test via SameDiff for MEAN_BY_WEIGHT
        SameDiff sd = SameDiff.create();
        sd.setDspAutoCompileEnabled(false);
        sd.setDspNativeAutoCompileEnabled(false);
        SDVariable predictions = sd.var("in", DataType.DOUBLE, minibatch, nOut);
        SDVariable labels = sd.var("labels", DataType.DOUBLE, -1, nOut);
        SDVariable loss = sd.loss().absoluteDifference("loss", labels, predictions, null, LossReduce.MEAN_BY_WEIGHT);

        sd.associateArrayWithVariable(predictionsArr, predictions);
        sd.associateArrayWithVariable(labelsArr, labels);
        Map<String, INDArray> placeholders = new HashMap<>();
        placeholders.put("in", predictionsArr);
        placeholders.put("labels", labelsArr);
        Map<String, INDArray> out = sd.output(placeholders, "loss");
        double sdResult = out.get("loss").getDouble(0);
        double expectedMean = rawSum / numElems;
        log.info("=== SameDiff MEAN_BY_WEIGHT: expected={}, actual={}, ratio={}", expectedMean, sdResult, sdResult / expectedMean);
        assertEquals(expectedMean, sdResult, 1e-5, "SameDiff MEAN_BY_WEIGHT wrong");
    }
}
