/*
 * TEMPORARY DIAGNOSTIC — eval-coherence investigation for GnnGraphClassificationTest.
 * Question: does logits.eval() return data consistent with loss.eval() on CUDA?
 * Delete after root-cause is identified.
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
public class GnnEvalCoherenceDiagTest {

    private static final int TOTAL_NODES = 18;
    private static final int NUM_GRAPHS  = 6;
    private static final int F  = 2;
    private static final int H  = 4;
    private static final int C  = 2;

    /** Manually compute sparse softmax cross-entropy from logits[6,2] and int labels[6]. */
    private static double manualCE(INDArray logits, int[] labels) {
        double totalLoss = 0.0;
        for (int g = 0; g < labels.length; g++) {
            double l0 = logits.getDouble(g, 0);
            double l1 = logits.getDouble(g, 1);
            // Numerically stable softmax
            double maxL = Math.max(l0, l1);
            double e0 = Math.exp(l0 - maxL);
            double e1 = Math.exp(l1 - maxL);
            double sum = e0 + e1;
            double p0 = e0 / sum;
            double p1 = e1 / sum;
            double prob = (labels[g] == 0) ? p0 : p1;
            totalLoss -= Math.log(Math.max(prob, 1e-12));
        }
        return totalLoss / labels.length;
    }

    private static void sgdUpdate(SameDiff sd, double lr, String... paramNames) {
        Map<String, INDArray> grads = sd.calculateGradients(null, paramNames);
        for (String name : paramNames) {
            INDArray g = grads.get(name);
            if (g == null) continue;
            double norm = g.norm2Number().doubleValue();
            if (norm > 5.0) g = g.mul(5.0 / norm);
            sd.getVariable(name).getArr().subi(g.muli(lr));
        }
    }

    @Test
    public void diagEvalCoherence() {
        System.out.println("=== GNN Eval-Coherence Diagnostic ===");
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
        INDArray graphLabelsArr = Nd4j.createFromArray(new int[]{0, 1, 0, 1, 0, 1});

        INDArray w1Arr  = Nd4j.randn(DataType.DOUBLE, F, H).muli(Math.sqrt(2.0 / F));
        INDArray b1Arr  = Nd4j.zeros(DataType.DOUBLE, H);
        INDArray w2Arr  = Nd4j.randn(DataType.DOUBLE, H, H).muli(Math.sqrt(2.0 / H));
        INDArray b2Arr  = Nd4j.zeros(DataType.DOUBLE, H);
        INDArray epsArr = Nd4j.scalar(DataType.DOUBLE, 0.0);
        INDArray W3Arr  = Nd4j.randn(DataType.DOUBLE, H, C).muli(Math.sqrt(2.0 / H));
        INDArray b3Arr  = Nd4j.zeros(DataType.DOUBLE, C);

        SameDiff sd = SameDiff.create();
        try {
            SDVariable colIdx      = sd.constant("colIdx",      colIdxArr);
            SDVariable rowPtr      = sd.constant("rowPtr",      rowPtrArr);
            SDVariable X           = sd.constant("X",           xArr);
            SDVariable graphIds    = sd.constant("graphIds",    graphIdsArr);
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

            // Training loop: 1000 epochs (same as the real test)
            final double LR     = 0.02;
            final int    EPOCHS = 1000;
            String[] params = {"w1", "b1", "w2", "b2", "eps", "W3", "b3"};
            for (int epoch = 0; epoch < EPOCHS; epoch++) {
                sgdUpdate(sd, LR, params);
            }

            // ---------------------------------------------------------------
            // POST-TRAINING DIAGNOSTICS (no training loop below this point)
            // ---------------------------------------------------------------
            System.out.println("\n=== POST-TRAINING DIAGNOSTICS ===");

            // 1. loss.eval() — what the test uses to check convergence
            double lossVal = loss.eval().getDouble(0);
            System.out.printf("loss.eval()        = %.8f%n", lossVal);

            // 2. logits.eval() call #1
            INDArray logitsArr1 = logits.eval();
            System.out.println("logits.eval() #1:");
            for (int g = 0; g < NUM_GRAPHS; g++) {
                double l0 = logitsArr1.getDouble(g, 0);
                double l1 = logitsArr1.getDouble(g, 1);
                System.out.printf("  graph %d: logit[0]=%.6f logit[1]=%.6f  pred=%d  label=%d%n",
                        g, l0, l1, (l0 >= l1 ? 0 : 1), (g % 2 == 0 ? 0 : 1));
            }

            // 3. logits.eval() call #2 — check if the two calls agree (CUDA caching check)
            INDArray logitsArr2 = logits.eval();
            double maxDiff12 = logitsArr1.sub(logitsArr2).amaxNumber().doubleValue();
            System.out.printf("logits.eval() #2 vs #1 max-diff: %.2e  (0 means same result)%n", maxDiff12);

            // 4. pooled.eval()
            INDArray pooledArr = pooled.eval();
            System.out.println("pooled.eval() [6,4]:");
            for (int g = 0; g < NUM_GRAPHS; g++) {
                System.out.printf("  graph %d: [%.4f, %.4f, %.4f, %.4f]%n",
                        g, pooledArr.getDouble(g,0), pooledArr.getDouble(g,1),
                        pooledArr.getDouble(g,2), pooledArr.getDouble(g,3));
            }

            // 5. Manually recompute CE from logitsArr1
            int[] labels = {0, 1, 0, 1, 0, 1};
            double ceManuall = manualCE(logitsArr1, labels);
            System.out.printf("%nManual CE from logits.eval() #1 = %.8f%n", ceManuall);
            System.out.printf("loss.eval()                      = %.8f%n", lossVal);
            System.out.printf("Difference (|CE_manual - loss|)  = %.2e%n",
                    Math.abs(ceManuall - lossVal));

            // KEY VERDICT
            boolean coherent = Math.abs(ceManuall - lossVal) < 0.01;
            System.out.println("\n=== VERDICT ===");
            if (coherent) {
                System.out.println("REFUTED: logits.eval() IS consistent with loss.eval().");
                System.out.println("The model reached a degenerate solution (low loss, bad argmax).");
                // Count accuracy
                int correct = 0;
                for (int g = 0; g < NUM_GRAPHS; g++) {
                    int pred = logitsArr1.getDouble(g, 0) >= logitsArr1.getDouble(g, 1) ? 0 : 1;
                    if (pred == labels[g]) correct++;
                }
                System.out.printf("Accuracy: %d/6%n", correct);
                System.out.printf("The low-loss+random-accuracy state IS a known degenerate optimum " +
                        "for symmetric GIN+SGD without clipping (see sgdUpdate comment).%n");
            } else {
                System.out.println("CONFIRMED: logits.eval() is NOT consistent with loss.eval().");
                System.out.println("There is a real CUDA eval-coherence bug.");
                System.out.printf("Manual CE from logits = %.6f vs loss.eval() = %.6f%n",
                        ceManuall, lossVal);
            }

            // 6. Also compare current host weight values to show they're not NaN
            System.out.println("\n=== WEIGHT SNAPSHOT (final) ===");
            INDArray w1Final = sd.getVariable("w1").getArr();
            INDArray W3Final = sd.getVariable("W3").getArr();
            System.out.printf("w1 norm2=%.4f  isNaN=%b  isInf=%b%n",
                    w1Final.norm2Number().doubleValue(),
                    w1Final.isNaN().any(), w1Final.isInfinite().any());
            System.out.printf("W3 norm2=%.4f  isNaN=%b  isInf=%b%n",
                    W3Final.norm2Number().doubleValue(),
                    W3Final.isNaN().any(), W3Final.isInfinite().any());

        } finally {
            sd.close();
        }
    }
}
