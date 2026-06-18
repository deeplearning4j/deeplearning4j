package org.eclipse.deeplearning4j.nd4j.autodiff.opvalidation;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.HashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Minimal direct-op test for sparse_softmax_cross_entropy_loss_with_logits.
 */
@Tag(TagNames.FULL_CI)
public class TestSparseSoftmaxDirect {

    @Test
    public void testDirectOpExecution() {
        Nd4j.getRandom().setSeed(12345);

        int batch = 4;
        int classes = 5;

        INDArray labels = Nd4j.createFromArray(0, 1, 2, 3);  // INT32
        INDArray logits = Nd4j.randn(DataType.DOUBLE, batch, classes);

        INDArray softmax = Transforms.softmax(logits.dup(), false);
        INDArray logSoftmax = Transforms.log(softmax, false);
        INDArray negLogSoftmax = logSoftmax.neg();

        double[] expected = new double[batch];
        for (int i = 0; i < batch; i++) {
            expected[i] = negLogSoftmax.getDouble(i, labels.getInt(i));
        }

        INDArray output = Nd4j.create(DataType.DOUBLE, batch);
        DynamicCustomOp op = DynamicCustomOp.builder("sparse_softmax_cross_entropy_loss_with_logits")
                .addInputs(labels, logits)
                .addOutputs(output)
                .build();
        Nd4j.getExecutioner().exec(op);

        for (int i = 0; i < batch; i++) {
            assertEquals(expected[i], output.getDouble(i), 1e-4, "Mismatch at example " + i);
        }
    }

    /**
     * Reproduces the exact TestLossOpValidation#testLoss2d sparsesoftmax scenario.
     * batch=10, nOut=4, seed=12345, labels as DOUBLE array with INT SameDiff var.
     */
    @Test
    public void testLoss2dScenario() {
        Nd4j.getRandom().setSeed(12345);
        int minibatch = 10;
        int nOut = 4;

        INDArray predictionsArr = Nd4j.randn(DataType.DOUBLE, minibatch, nOut);
        System.out.println("predictionsArr shape: " + java.util.Arrays.toString(predictionsArr.shape()));

        // Build expected value: -sum_c(oneHot[i,c] * log(softmax(predictions)[i,c]))
        INDArray labelsArr = Nd4j.create(DataType.DOUBLE, minibatch);
        INDArray oneHot = Nd4j.create(DataType.DOUBLE, minibatch, nOut);
        for (int i = 0; i < minibatch; i++) {
            labelsArr.putScalar(i, i % nOut);
            oneHot.putScalar(new int[]{i, i % nOut}, 1.0);
        }
        System.out.println("labelsArr: " + labelsArr);
        System.out.println("labelsArr dtype: " + labelsArr.dataType());

        INDArray softmaxPred = Transforms.softmax(predictionsArr.dup(), true);
        INDArray logP = Transforms.log(softmaxPred, true);
        INDArray perExampleLoss = oneHot.mul(logP).neg().sum(1);
        double expectedSum = perExampleLoss.sumNumber().doubleValue();
        System.out.println("Expected perExampleLoss: " + perExampleLoss);
        System.out.println("Expected sum: " + expectedSum);

        // Direct op with INT labels (sparse_softmax op requires integer label type)
        INDArray labelsIntDirect = labelsArr.castTo(DataType.INT);
        INDArray directOutput = Nd4j.create(DataType.DOUBLE, minibatch);
        DynamicCustomOp directOp = DynamicCustomOp.builder("sparse_softmax_cross_entropy_loss_with_logits")
                .addInputs(labelsIntDirect, predictionsArr)
                .addOutputs(directOutput)
                .build();
        Nd4j.getExecutioner().exec(directOp);
        System.out.println("Direct op output (INT labels): " + directOutput);
        System.out.println("Direct sum (INT labels): " + directOutput.sumNumber().doubleValue());
        assertEquals(expectedSum, directOutput.sumNumber().doubleValue(), 1e-3,
                "Direct op with INT labels mismatch");

        // Direct op with INT labels
        INDArray labelsInt = labelsArr.castTo(DataType.INT);
        INDArray directOutput2 = Nd4j.create(DataType.DOUBLE, minibatch);
        DynamicCustomOp directOp2 = DynamicCustomOp.builder("sparse_softmax_cross_entropy_loss_with_logits")
                .addInputs(labelsInt, predictionsArr)
                .addOutputs(directOutput2)
                .build();
        Nd4j.getExecutioner().exec(directOp2);
        System.out.println("Direct op output (INT labels): " + directOutput2);
        System.out.println("Direct sum (INT labels): " + directOutput2.sumNumber().doubleValue());
        assertEquals(expectedSum, directOutput2.sumNumber().doubleValue(), 1e-3,
                "Direct op with INT labels mismatch");

        // SameDiff forward test (exactly as testLoss2d)
        Nd4j.getRandom().setSeed(12345);
        predictionsArr = Nd4j.randn(DataType.DOUBLE, minibatch, nOut);
        labelsArr = Nd4j.create(DataType.DOUBLE, minibatch);
        oneHot = Nd4j.create(DataType.DOUBLE, minibatch, nOut);
        for (int i = 0; i < minibatch; i++) {
            labelsArr.putScalar(i, i % nOut);
            oneHot.putScalar(new int[]{i, i % nOut}, 1.0);
        }
        softmaxPred = Transforms.softmax(predictionsArr.dup(), true);
        logP = Transforms.log(softmaxPred, true);
        perExampleLoss = oneHot.mul(logP).neg().sum(1);
        INDArray expOut = Nd4j.scalar(DataType.DOUBLE, perExampleLoss.sumNumber().doubleValue());
        System.out.println("SameDiff expected output: " + expOut);

        SameDiff sd = SameDiff.create();
        SDVariable predVar = sd.var("in", DataType.DOUBLE, minibatch, nOut);
        SDVariable labVar = sd.var("labels", DataType.INT, -1);
        INDArray labelsInt2 = labelsArr.castTo(DataType.INT);
        sd.associateArrayWithVariable(predictionsArr, predVar);
        sd.associateArrayWithVariable(labelsInt2, labVar);
        sd.loss().sparseSoftmaxCrossEntropy(predVar, labVar).sum("loss");

        Map<String, INDArray> placeholders = new HashMap<>();
        placeholders.put("in", predictionsArr);
        placeholders.put("labels", labelsInt2);
        Map<String, INDArray> result = sd.output(placeholders, "loss");
        INDArray lossVal = result.get("loss");
        System.out.println("SameDiff loss: " + lossVal);
        System.out.println("SameDiff loss value: " + (lossVal != null ? lossVal.getDouble(0) : "null"));
        assertNotNull(lossVal, "SameDiff loss should not be null");
        assertEquals(expOut.getDouble(0), lossVal.getDouble(0), 1e-3, "SameDiff forward mismatch");
    }
}
