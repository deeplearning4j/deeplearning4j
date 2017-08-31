package org.nd4j.autodiff.functions.mmul;

import com.google.common.primitives.Ints;
import lombok.NoArgsConstructor;
import org.nd4j.autodiff.ArrayField;
import org.nd4j.autodiff.Field;
import org.nd4j.autodiff.functions.AbstractBinaryReduceFunction;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.opstate.OpState;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.blas.params.MMulTranspose;

import static org.nd4j.linalg.util.ArrayUtil.*;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Tensor matrix multiply operation
 *
 * @author Adam Gibson
 */
@NoArgsConstructor
public class TensorMmul<X extends Field<ArrayField>> extends AbstractBinaryReduceFunction<X> {
    private int[][] axes;
    protected boolean addedEdges;
    protected MMulTranspose mMulTranspose;

    public TensorMmul(SameDiff sameDiff,
                      DifferentialFunction<ArrayField> i_v1,
                      DifferentialFunction<ArrayField> i_v2,
                      int[][] dimensions) {
        this(sameDiff,i_v1,i_v2,dimensions,MMulTranspose.allFalse());
    }

    public TensorMmul(SameDiff sameDiff,
                      DifferentialFunction<ArrayField> i_v1,
                      DifferentialFunction<ArrayField> i_v2,
                      int[][] dimensions,
                      MMulTranspose mMulTranspose) {
        super(sameDiff);
        this.sameDiff = sameDiff;
        this.mMulTranspose = mMulTranspose;
        this.axes = dimensions;
        this.extraArgs = new Object[] {axes,mMulTranspose};
        this.m_x1 = i_v1;
        this.m_x2 = i_v2;
        if(!addedEdges) {
            ArrayField a = i_v1.getValue(true);
            ArrayField b = i_v2.getValue(true);

            addEdges(sameDiff,
                    i_v1,
                    i_v2,
                    functionName(),
                    OpState.OpType.ACCUMULATION,
                    getTensorMmulShape(a.getInput().getShape(), b.getInput().getShape(), dimensions));
            addedEdges = true;
        }
    }


    @Override
    protected void addEdges(SameDiff sameDiff,
                            DifferentialFunction<ArrayField> i_v1,
                            DifferentialFunction<ArrayField> i_v2,
                            String opName) {
        if(axes != null
                && !addedEdges) {
            addedEdges = true;
            ArrayField arrayField = i_v1.getValue(true);
            ArrayField secondVal = i_v2.getValue(true);

            addEdges(sameDiff,i_v1,i_v2,opName,
                    OpState.OpType.ACCUMULATION,
                    getTensorMmulShape(arrayField.getInput()
                                    .getShape(),
                            secondVal.getInput().getShape(),
                            axes));

        }

    }

    /**
     * Get the value of this function
     *
     * @return
     */
    @Override
    public ArrayField doGetValue() {
        return sameDiff.getArrayFactory().tensorMmul(larg(),rarg(),axes);
    }



    @Override
    public String functionName() {
        return "tensorMmul";
    }



    @Override
    public List<DifferentialFunction<ArrayField>> diff(List<DifferentialFunction<ArrayField>> i_v1) {
        List<DifferentialFunction<ArrayField>> ret = new ArrayList<>();
        int[] bAxes = range(0, rarg().getResultShape().length);
        int[] aAxes = range(0, larg().getResultShape().length);
        int aRank = larg().getResultShape().length;
        int bRank = rarg().getResultShape().length;
        int[][] sumAxes = new int[][]{
                mod(axes[0], aRank), mod(axes[1], bRank)
        };
        int[][] deletedAxes = new int[][]{
                removeIndex(aAxes, sumAxes[0]),
                removeIndex(bAxes, sumAxes[1])};
        int[] gAxes = range(0, i_v1.get(0).getResultShape().length);
        int[][] firstAxes = new int[][]{
                Arrays.copyOfRange(gAxes, deletedAxes[0].length, gAxes.length),
                deletedAxes[1]
        };

        int[][] secondAxes = new int[][]{
                deletedAxes[0],
                Arrays.copyOfRange(gAxes, 0, deletedAxes[0].length)
        };


        //tensor matrix multiply gradient wrt second variable
        int[] firstPerm = argsort(combine(deletedAxes[0],keep(argsort(sumAxes[1]),sumAxes[0])));
        DifferentialFunction<ArrayField> firstResult = doTensorMmul(i_v1.get(0), rarg(), firstAxes);
        DifferentialFunction<ArrayField> permuted = f().permute(firstResult,firstPerm);
        ret.add(permuted);
        larg().setGradient(permuted);

        //tensor matrix multiply gradient wrt first variable
        int[] secondPerm = argsort(combine(keep(argsort(sumAxes[0]),sumAxes[1]),deletedAxes[1]));
        DifferentialFunction<ArrayField> secondResult = doTensorMmul(i_v1.get(0), larg(), secondAxes);
        DifferentialFunction<ArrayField> secondPermuted = f().permute(secondResult,secondPerm);
        ret.add(secondPermuted);
        rarg().setGradient(secondPermuted);
        return ret;
    }



    private DifferentialFunction<ArrayField> doTensorMmul(DifferentialFunction<ArrayField> a,
                                                          DifferentialFunction<ArrayField> b,
                                                          int[][] axes) {

        ArrayField xField = a.getValue(true);
        ArrayField yField = b.getValue(true);
        int validationLength = Math.min(axes[0].length, axes[1].length);
        for (int i = 0; i < validationLength; i++) {
            if (xField.getInput().getShape()[axes[0][i]] != yField.getInput().getShape()[axes[1][i]])
                throw new IllegalArgumentException("Size of the given axes at each dimension must be the same size.");
            if (axes[0][i] < 0)
                axes[0][i] += xField.getInput().getShape().length;
            if (axes[1][i] < 0)
                axes[1][i] += yField.getInput().getShape().length;

        }

        List<Integer> listA = new ArrayList<>();
        for (int i = 0; i < xField.getInput().getShape().length; i++) {
            if (!Ints.contains(axes[0], i))
                listA.add(i);
        }

        int[] newAxesA = Ints.concat(Ints.toArray(listA), axes[0]);


        List<Integer> listB = new ArrayList<>();
        for (int i = 0; i < yField.getInput().getShape().length; i++) {
            if (!Ints.contains(axes[1], i))
                listB.add(i);
        }

        int[] newAxesB = Ints.concat(axes[1], Ints.toArray(listB));

        int n2 = 1;
        int aLength = Math.min(xField.getInput().getShape().length, axes[0].length);
        for (int i = 0; i < aLength; i++) {
            n2 *= xField.getInput().getShape()[axes[0][i]];
        }

        //if listA and listB are empty these do not initialize.
        //so initializing with {1} which will then get overridden if not empty
        int[] newShapeA = {-1, n2};
        int[] oldShapeA;
        if (listA.size() == 0) {
            oldShapeA = new int[] {1};
        } else {
            oldShapeA = Ints.toArray(listA);
            for (int i = 0; i < oldShapeA.length; i++)
                oldShapeA[i] = xField.getInput().getShape()[oldShapeA[i]];
        }

        int n3 = 1;
        int bNax = Math.min(yField.getInput().getShape().length, axes[1].length);
        for (int i = 0; i < bNax; i++) {
            n3 *= yField.getInput().getShape()[axes[1][i]];
        }


        int[] newShapeB = {n3, -1};
        int[] oldShapeB;
        if (listB.size() == 0) {
            oldShapeB = new int[] {1};
        } else {
            oldShapeB = Ints.toArray(listB);
            for (int i = 0; i < oldShapeB.length; i++)
                oldShapeB[i] = yField.getInput().getShape()[oldShapeB[i]];
        }


        DifferentialFunction<ArrayField> at = f()
                .reshape(f().permute
                        (a,newAxesA),newShapeA);
        DifferentialFunction<ArrayField> bt = f()
                .reshape(f()
                        .permute(b,newAxesB),newShapeB);

        DifferentialFunction<ArrayField> ret = f().mmul(at,bt);
        int[] aPlusB = Ints.concat(oldShapeA, oldShapeB);
        return f().reshape(ret,aPlusB);
    }
}
