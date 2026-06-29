/*
 * TEMPORARY DIAGNOSTIC — to be deleted after root-cause identified.
 * Finds exact epoch where gradients first go NaN/Inf on CUDA.
 * Key rule: do NOT call loss.eval() or variable.eval() inside the normal loop
 * (that masks the bug); only call eval when a bad gradient is first detected.
 */
package org.eclipse.deeplearning4j.nd4j.linalg.sparse;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.graph.GraphPooling;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Map;

@Tag("sparse")
@Tag("gnn")
@Tag("diagnostic")
public class GnnGraphClassificationDiagTest {

    private static final int TOTAL_NODES = 18;
    private static final int NUM_GRAPHS  = 6;
    private static final int F  = 2;
    private static final int H  = 4;
    private static final int C  = 2;

    private static String summarize(String label, INDArray arr) {
        if (arr == null) return label + "=null";
        double min = arr.minNumber().doubleValue();
        double max = arr.maxNumber().doubleValue();
        boolean anyNan = arr.isNaN().any();
        boolean anyInf = arr.isInfinite().any();
        return String.format("%s: min=%.6f max=%.6f nan=%b inf=%b",
                label, min, max, anyNan, anyInf);
    }

    private static boolean hasNanOrInf(INDArray arr) {
        if (arr == null) return false;
        return arr.isNaN().any() || arr.isInfinite().any();
    }

    /** Returns the absolute max of an INDArray (0 if null). */
    private static double absMax(INDArray arr) {
        if (arr == null) return 0.0;
        double mn = arr.minNumber().doubleValue();
        double mx = arr.maxNumber().doubleValue();
        return Math.max(Math.abs(mn), Math.abs(mx));
    }

    @Test
    public void diagNanOnset() {
        System.out.println("=== GNN NaN Diagnostic v3 ===");
        System.out.println("Backend: " + Nd4j.getBackend().getClass().getSimpleName());
        Nd4j.getRandom().setSeed(123L);
        Nd4j.getConstantHandler().purgeConstants();

        int[] colIdxData = {
                1, 0, 2, 1, 4, 5, 3, 3,
                7, 6, 8, 7, 10, 11, 9, 9,
                13, 12, 14, 13, 16, 17, 15, 15
        };
        int[] rowPtrData = {0, 1, 3, 4, 6, 7, 8, 9, 11, 12, 14, 15, 16, 17, 19, 20, 22, 23, 24};

        INDArray colIdxArr = Nd4j.createFromArray(colIdxData);
        INDArray rowPtrArr = Nd4j.createFromArray(rowPtrData);

        double[][] xData = new double[TOTAL_NODES][F];
        for (int g = 0; g < NUM_GRAPHS; g += 2)
            for (int k = 0; k < 3; k++) { xData[g*3+k][0] = 1.0; xData[g*3+k][1] = 0.0; }
        for (int g = 1; g < NUM_GRAPHS; g += 2)
            for (int k = 0; k < 3; k++) { xData[g*3+k][0] = 0.0; xData[g*3+k][1] = 1.0; }
        INDArray xArr = Nd4j.createFromArray(xData);

        int[] graphIdsData = new int[TOTAL_NODES];
        for (int g = 0; g < NUM_GRAPHS; g++)
            for (int k = 0; k < 3; k++) graphIdsData[g*3+k] = g;
        INDArray graphIdsArr = Nd4j.createFromArray(graphIdsData);
        INDArray graphLabelsArr = Nd4j.createFromArray(new int[]{0,1,0,1,0,1});

        INDArray w1Arr  = Nd4j.rand(DataType.DOUBLE, F, H).subi(0.5).muli(0.2);
        INDArray b1Arr  = Nd4j.zeros(DataType.DOUBLE, H);
        INDArray w2Arr  = Nd4j.rand(DataType.DOUBLE, H, H).subi(0.5).muli(0.2);
        INDArray b2Arr  = Nd4j.zeros(DataType.DOUBLE, H);
        INDArray epsArr = Nd4j.scalar(DataType.DOUBLE, 0.0);
        INDArray W3Arr  = Nd4j.rand(DataType.DOUBLE, H, C).subi(0.5).muli(0.2);
        INDArray b3Arr  = Nd4j.zeros(DataType.DOUBLE, C);

        SameDiff sd = SameDiff.create();
        try {
            SDVariable colIdx    = sd.constant("colIdx",    colIdxArr);
            SDVariable rowPtr    = sd.constant("rowPtr",    rowPtrArr);
            SDVariable X         = sd.constant("X",         xArr);
            SDVariable graphIds  = sd.constant("graphIds",  graphIdsArr);
            SDVariable graphLabels = sd.constant("graphLabels", graphLabelsArr);

            SDVariable w1  = sd.var("w1",  w1Arr);
            SDVariable b1  = sd.var("b1",  b1Arr);
            SDVariable w2  = sd.var("w2",  w2Arr);
            SDVariable b2  = sd.var("b2",  b2Arr);
            SDVariable eps = sd.var("eps", epsArr);
            SDVariable W3  = sd.var("W3",  W3Arr);
            SDVariable b3  = sd.var("b3",  b3Arr);

            SDVariable ginOut = sd.gnn().ginConv(X, w1, b1, w2, b2, eps, colIdx, rowPtr, TOTAL_NODES, TOTAL_NODES);
            SDVariable pooled = GraphPooling.globalSumPool(sd, "pooled", ginOut, graphIds, NUM_GRAPHS);
            SDVariable logits = sd.mmul(pooled, W3).add(b3);
            SDVariable loss   = sd.loss().sparseSoftmaxCrossEntropy("loss", logits, graphLabels);

            double initialLoss = loss.eval().getDouble(0);
            System.out.printf("Initial loss: %.6f%n", initialLoss);

            final double LR = 0.02;
            final int EPOCHS = 300;
            String[] params = {"w1","b1","w2","b2","eps","W3","b3"};

            int firstBadGradEpoch = -1;
            // Track current weight max via host-side bookkeeping (avoid eval calls)
            double prevW1Max = absMax(w1Arr.dup());
            double prevW3Norm = W3Arr.dup().norm2Number().doubleValue();

            for (int epoch = 0; epoch < EPOCHS; epoch++) {
                Map<String, INDArray> grads = sd.calculateGradients(null, params);

                INDArray gw1 = grads.get("w1");
                INDArray gW3 = grads.get("W3");

                // Log gradient magnitudes every epoch for the first 60
                if (epoch < 60) {
                    double gw1max = absMax(gw1);
                    double gW3max = absMax(gW3);
                    System.out.printf("Epoch %3d: |gw1|_max=%.6f  |gW3|_max=%.6f  w1_prevMax=%.6f%n",
                            epoch, gw1max, gW3max, prevW1Max);
                }

                // Detect first bad gradient
                boolean gradBad = hasNanOrInf(gw1) || hasNanOrInf(gW3);
                for (String p : params) {
                    if (hasNanOrInf(grads.get(p))) { gradBad = true; break; }
                }

                if (gradBad && firstBadGradEpoch < 0) {
                    firstBadGradEpoch = epoch;
                    System.out.printf("*** First NaN/Inf gradient at epoch %d ***%n", epoch);
                    for (String p : params) {
                        System.out.println("  " + summarize("g_" + p, grads.get(p)));
                    }
                    // Now it's safe to call eval (state is already bad)
                    System.out.println("  " + summarize("ginOut", ginOut.eval()));
                    System.out.println("  " + summarize("pooled", pooled.eval()));
                    System.out.println("  " + summarize("logits", logits.eval()));
                }

                for (String name : params) {
                    INDArray g = grads.get(name);
                    if (g != null) sd.getVariable(name).getArr().subi(g.muli(LR));
                }

                // Update tracking (use the raw gradient to infer weight growth)
                if (gw1 != null && !hasNanOrInf(gw1)) prevW1Max += absMax(gw1) * LR;
            }

            double finalLoss = loss.eval().getDouble(0);
            System.out.printf("Final loss: %.6f (nan=%b, firstBadGradEpoch=%d)%n",
                    finalLoss, Double.isNaN(finalLoss), firstBadGradEpoch);

        } finally {
            sd.close();
        }
    }
}
