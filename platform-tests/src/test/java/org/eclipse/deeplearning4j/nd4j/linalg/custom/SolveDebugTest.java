package org.eclipse.deeplearning4j.nd4j.linalg.custom;

import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.factory.Nd4j;

public class SolveDebugTest {

    @Test
    public void testSolveBasic() {
        INDArray a = Nd4j.createFromArray(new float[]{
                2.f, -1.f, -2.f, -4.f, 6.f, 3.f, -4.f, -2.f, 8.f
        }).reshape(3, 3);

        INDArray b = Nd4j.createFromArray(new float[]{
                2.f, 4.f, 3.f
        }).reshape(3, 1);

        System.out.println("Input A: " + a);
        System.out.println("Input b: " + b);

        // Test LU decomposition first
        DynamicCustomOp luOp = DynamicCustomOp.builder("lu")
                .addInputs(a)
                .build();
        INDArray[] luResult = Nd4j.exec(luOp);
        System.out.println("LU output: " + luResult[0]);
        System.out.println("LU permutation: " + luResult[1]);

        // Test triangular_solve lower
        INDArray lower = luResult[0].dup();
        // Extract lower triangular part and set diagonal to 1
        for (int r = 0; r < 3; r++) {
            for (int c = r + 1; c < 3; c++) {
                lower.putScalar(r, c, 0);  // clear upper
            }
            lower.putScalar(r, r, 1.0f);  // set diagonal to 1
        }
        System.out.println("Lower: " + lower);

        // Build permutation matrix
        INDArray P = Nd4j.zeros(3, 3);
        for (int r = 0; r < 3; r++) {
            P.putScalar(r, luResult[1].getInt(r), 1.0f);
        }
        System.out.println("P: " + P);

        INDArray Pb = P.mmul(b);
        System.out.println("P*b: " + Pb);

        // Lower triangular solve: L*y = P*b
        DynamicCustomOp triLower = DynamicCustomOp.builder("triangular_solve")
                .addInputs(lower, Pb)
                .addBooleanArguments(true, false)  // lower=true, adjoint=false
                .build();
        INDArray[] yResult = Nd4j.exec(triLower);
        System.out.println("y (lower solve): " + yResult[0]);

        // Extract upper triangular part
        INDArray upper = luResult[0].dup();
        for (int r = 0; r < 3; r++) {
            for (int c = 0; c < r; c++) {
                upper.putScalar(r, c, 0);  // clear lower
            }
        }
        System.out.println("Upper: " + upper);

        // Upper triangular solve: U*x = y
        DynamicCustomOp triUpper = DynamicCustomOp.builder("triangular_solve")
                .addInputs(upper, yResult[0])
                .addBooleanArguments(false, false)  // lower=false, adjoint=false
                .build();
        INDArray[] xResult = Nd4j.exec(triUpper);
        System.out.println("x (upper solve / final result): " + xResult[0]);

        // Now test the actual solve op
        DynamicCustomOp solveOp = DynamicCustomOp.builder("solve")
                .addInputs(a, b)
                .addBooleanArguments(false)  // adjoint=false
                .build();
        INDArray[] solveResult = Nd4j.exec(solveOp);
        System.out.println("solve op result: " + solveResult[0]);

        INDArray expected = Nd4j.createFromArray(new float[]{7.625f, 3.25f, 5.f}).reshape(3, 1);
        System.out.println("Expected: " + expected);
    }
}
