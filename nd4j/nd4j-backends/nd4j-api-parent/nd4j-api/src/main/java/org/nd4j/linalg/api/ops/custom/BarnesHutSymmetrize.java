package org.nd4j.linalg.api.ops.custom;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.factory.Nd4j;

public class BarnesHutSymmetrize extends DynamicCustomOp {

    private INDArray output;

    public BarnesHutSymmetrize(INDArray rowP, INDArray colP, INDArray valP, long N) {

        INDArray rowCounts = Nd4j.create(N);
        for (int n = 0; n < N; n++) {
            int begin = rowP.getInt(n);
            int end = rowP.getInt(n + 1);
            for (int i = begin; i < end; i++) {
                boolean present = false;
                for (int m = rowP.getInt(colP.getInt(i)); m < rowP.getInt(colP.getInt(i) + 1); m++) {
                    if (colP.getInt(m) == n) {
                        present = true;
                    }
                }
                if (present)
                    rowCounts.putScalar(n, rowCounts.getDouble(n) + 1);

                else {
                    rowCounts.putScalar(n, rowCounts.getDouble(n) + 1);
                    rowCounts.putScalar(colP.getInt(i), rowCounts.getDouble(colP.getInt(i)) + 1);
                }
            }
        }
        long outputCols = rowCounts.sum(Integer.MAX_VALUE).getInt(0);
        output = Nd4j.createUninitialized(1, outputCols);

        inputArguments.add(rowP);
        inputArguments.add(colP);
        inputArguments.add(valP);

        outputArguments.add(output);

        iArguments.add(N);
    }

    public INDArray getResult() {
        return output;
    }

    @Override
    public String opName() {
        return "barnes_symmetrized";
    }
}
