package org.nd4j.autodiff.functions.mmul;

import com.google.common.primitives.Ints;
import lombok.NoArgsConstructor;
import org.nd4j.autodiff.ArrayField;
import org.nd4j.autodiff.Field;
import org.nd4j.autodiff.functions.AbstractBinaryReduceFunction;
import org.nd4j.autodiff.functions.Differential;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.opstate.OpState;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.util.ArrayUtil;

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

    public TensorMmul(SameDiff sameDiff,
                      DifferentialFunction<ArrayField> i_v1,
                      DifferentialFunction<ArrayField> i_v2,
                      int[][] dimensions) {
        super(sameDiff);
        this.sameDiff = sameDiff;
        this.axes = dimensions;
        this.extraArgs = new Object[] {axes};
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
                    ArrayUtil.getTensorMmulShape(a.getInput().getShape(), b.getInput().getShape(), dimensions));
            addedEdges = true;
        }
    }


    @Override
    protected void addEdges(SameDiff sameDiff,
                            DifferentialFunction<ArrayField> i_v1,
                            DifferentialFunction<ArrayField> i_v2,
                            String opName) {
        if(i_v1.getValue(true) instanceof ArrayField && axes != null
                && !addedEdges) {
            addedEdges = true;
            ArrayField arrayField = i_v1.getValue(true);
            ArrayField secondVal = i_v2.getValue(true);

            addEdges(sameDiff,i_v1,i_v2,opName,
                    OpState.OpType.ACCUMULATION,
                    ArrayUtil.getTensorMmulShape(arrayField.getInput()
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
        int[] bAxes = ArrayUtil.range(0, i_v1.get(1).getResultShape().length);
        int[] aAxes = ArrayUtil.range(0, i_v1.get(0).getResultShape().length);
        int aRank = i_v1.get(0).getResultShape().length;
        int bRank = i_v1.get(1).getResultShape().length;
        int[][] sumAxes = new int[][]{
                ArrayUtil.mod(axes[0], aRank), ArrayUtil.mod(axes[1], bRank)
        };
        int[][] deletedAxes = new int[][]{
                ArrayUtil.removeIndex(aAxes, sumAxes[0]),
                ArrayUtil.removeIndex(bAxes, sumAxes[1])};
        int[] gAxes = ArrayUtil.range(0, i_v1.get(0).getResultShape().length);
        int[][] firstAxes = new int[][]{
                Arrays.copyOfRange(gAxes, deletedAxes[0].length, gAxes.length),
                deletedAxes[1]
        };

        int[][] secondAxes = new int[][]{
                deletedAxes[0],
                Arrays.copyOfRange(gAxes, 0, deletedAxes[0].length)
        };



        /**
         * perm = onp.argsort(onp.concatenate(
         (other_axes[0], summed_axes[0][onp.argsort(summed_axes[1])])))
         */
        int[] firstPerm = ArrayUtil.argsort(ArrayUtil.combine(deletedAxes[0],sumAxes[0][ArrayUtil.k]));
        DifferentialFunction<ArrayField> firstResult = doTensorMmul(i_v1.get(0), rarg(), firstAxes);
        DifferentialFunction<ArrayField> permuted = f().permute(firstResult,firstPerm);
        ret.add(permuted);
        /**
         * perm = onp.argsort(onp.concatenate(
         (summed_axes[1][onp.argsort(summed_axes[0])], other_axes[1])))
         */
        int[] secondPerm = new int[] {};
        DifferentialFunction<ArrayField> secondResult = doTensorMmul(i_v1.get(0), larg(), secondAxes);
        DifferentialFunction<ArrayField> secondPermuted = f().permute(secondResult,secondPerm);
        ret.add(secondPermuted);

/**
 *
 *

 if argnum == 0:

 * out = anp.tensordot(g, B, [g_axes[len(other_axes[0]):], other_axes[1]])
 perm = onp.argsort(onp.concatenate(
 (other_axes[0], summed_axes[0][onp.argsort(summed_axes[1])])))
 return anp.transpose(out, perm)
 else:
 out = anp.tensordot(A, g, [other_axes[0], g_axes[:len(other_axes[0])]])
 perm = onp.argsort(onp.concatenate(
 (summed_axes[1][onp.argsort(summed_axes[0])], other_axes[1])))
 return anp.transpose(out, perm)
 */


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


        DifferentialFunction<ArrayField> at = getSameDiff()
                .getFunctionFactory()
                .reshape(getSameDiff().getFunctionFactory().permute
                        (a,newAxesA),newShapeA);
        DifferentialFunction<ArrayField> bt = getSameDiff().getFunctionFactory()
                .reshape(getSameDiff().getFunctionFactory()
                        .permute(b,newAxesB),newShapeB);

        DifferentialFunction<ArrayField> ret = getSameDiff().getFunctionFactory().mmul(at,bt);
        int[] aPlusB = Ints.concat(oldShapeA, oldShapeB);
        return getSameDiff().getFunctionFactory().reshape(ret,aPlusB);
    }
}
