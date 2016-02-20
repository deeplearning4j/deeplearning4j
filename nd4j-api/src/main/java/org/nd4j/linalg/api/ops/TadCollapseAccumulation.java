package org.nd4j.linalg.api.ops;

import lombok.Data;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.ArrayUtil;

/**
 * This implements a collapsing tad reduction
 * based on different dimensions.
 *
 * The reason we need this is because of the fact that
 * there are certain dimension combinations (usually > 1)
 * that don't have an element wise stride.
 *
 * A way to bypass this problem is to expand the problem
 * in to a 1 dimension reduction problem
 * and then collapsing the results in to the equivalent
 * shape of the multi dimension problem.
 *
 * An example problem would be an array of:
 * linspace(1,24,24).reshape(2,2,3,2)
 *
 * The tad for reduction:
 * 2,3 doesn't have an element wise stride.
 *
 * However, the tad for reduction:
 * 3 does
 *
 * What we can exploit here is the ability
 * to reshape problems of multiple dimensions
 *
 * in to equivalent expanded problems based on smaller tads
 * eg:
 * multiple reductions for each dimension along dimension 3
 * followed by collapsing the problem in to an equivalent state
 * as if we had specified 2,3 for the dimensions instead.
 *
 * This gives us a way of executing an element wise stride based
 * algorithm  that is executable on the gpu.
 *
 * For the GPU, we force each block to process a  tad
 * at the singular dimension level. Eg: dimension 3
 *
 * So for example along dimension 3 of the 2,2,3,2
 * array we have 12 tensors along dimension.
 *
 * We then map those 12 tads to a reduction index.
 *
 * A reduction index is the equivalent value
 * in teh result as if we had specified the reduction dimensions
 * to be 2,3 instead.
 *
 * For example, if we have 12 tads for dimension 3
 * we will only have 4 for dimensions 2,3
 *
 * The goal will be then to generate the equivalent results
 * using dimension 3 but collapsing the results according to
 * the dimension 2,3 space (remember: the reason we are doing this mapping
 * is because we are trying to map the multi dimensional problem on to
 * a problem that allows us to solve it via element wise stride)
 *
 *
 * An example mapping relative to a gpu block is as follows:
 * ([[[[  1.,   2.],
 [  3.,   4.],
 [  5.,   6.]],

 [[  7.,   8.],
 [  9.,  10.],
 [ 11.,  12.]]],


 [[[ 13.,  14.],
 [ 15.,  16.],
 [ 17.,  18.]],

 [[ 19.,  20.],
 [ 21.,  22.],
 [ 23.,  24.]]]])



 * Along dimension 3 we will have tads of length 2
 * and 4 reduction indexes we need to map for the
 * 2,3 dimension problem.
 *
 *
 * The first reduction index will map to the first 3 tads of length 2
 * The next reduction index will map to the next 3, etc.
 *
 * We then process a reduction index per block on the gpu.
 * If any gpu block index is > the number of
 * reduction indexes we skip it.
 *
 * Note here we did this implementation because of
 * race conditions on the block and shared memory.
 *
 * This way of mapping allows us to avoid race conditions.
 *
 */
@Data
public class TadCollapseAccumulation extends BaseOp {
    protected Op accum;
    protected boolean performSmallerDimension;
    protected int[] smallerDimension;
    protected int[] originalDimension;
    protected int tadsForSmallerDimension;
    protected int tadsForLargerDimension;
    public final static String DEFAULT_NAME = "collapseTad";

    public TadCollapseAccumulation() {
    }
    /**
     *
     * @param accum the operation to accumulate
     * @param originalDimension the bigger problem
     * @param smallerDimension the smaller problem
     */
    public TadCollapseAccumulation(Op accum, int[] originalDimension, int[] smallerDimension,boolean performSmallerDimension) {
        this.accum = accum;
        this.performSmallerDimension = performSmallerDimension;
        this.originalDimension = originalDimension;
        this.smallerDimension = smallerDimension;
        tadsForSmallerDimension = accum.x().tensorssAlongDimension(smallerDimension);
        tadsForLargerDimension = accum.x().tensorssAlongDimension(originalDimension);

    }
    /**
     *
     * @param accum the operation to accumulate
     * @param originalDimension the bigger problem
     * @param smallerDimension the smaller problem
     */
    public TadCollapseAccumulation(Op accum, int[] originalDimension, int[] smallerDimension) {
        this(accum, originalDimension, smallerDimension,true);
    }

    public TadCollapseAccumulation(Op accum, int[] originalDimension) {
        this.accum = accum;
        this.originalDimension = originalDimension;
    }

    public TadCollapseAccumulation(Op accum) {
        this.accum = accum;
    }

    public TadCollapseAccumulation(INDArray x, Op accum) {
        super(x);
        this.accum = accum;
    }

    public TadCollapseAccumulation(INDArray x, INDArray y, INDArray z, int n, Op accum) {
        super(x, y, z, n);
        this.accum = accum;
    }

    public TadCollapseAccumulation(INDArray x, INDArray z, Op accum) {
        super(x, z);
        this.accum = accum;
    }

    public TadCollapseAccumulation(INDArray x, INDArray z, int n, Op accum) {
        super(x, z, n);
        this.accum = accum;
    }

    public Op getAccum() {
        return accum;
    }

    @Override
    public boolean isPassThrough() {
        return true;
    }

    @Override
    public void exec() {
        //take the last dimension
        if(smallerDimension == null) {
            smallerDimension = new int[] {originalDimension[originalDimension.length - 1]};
        }

        if(accum instanceof Accumulation && performSmallerDimension) {
            Accumulation acc2 = (Accumulation) accum;
            //avoid the final transform till towards the end
            acc2.setApplyFinalTransform(false);
            Nd4j.getExecutioner().exec(acc2,smallerDimension);
        }
        else if(accum instanceof IndexAccumulation && performSmallerDimension) {
            IndexAccumulation acc2 = (IndexAccumulation) accum;
            Nd4j.getExecutioner().exec(acc2,smallerDimension);
        }


        /**
         * Now combine the results based on
         * the final dimension.
         */
        INDArray aggregated = Nd4j.create(ArrayUtil.removeIndex(accum.x().shape(),originalDimension));
        int smallerProblem = accum.x().tensorssAlongDimension(smallerDimension);
        int biggerProblem = accum.x().tensorssAlongDimension(originalDimension);
        if(accum instanceof Accumulation) {
            int biggerTadLength = accum.x().tensorAlongDimension(0,originalDimension).length();
            Accumulation accumulation = (Accumulation) accum;
            for(int i = 0; i < smallerProblem; i++) {
                int reductionIndex = reductionIndexForTad(i,biggerProblem,smallerProblem);
                aggregated.putScalar(reductionIndex,accumulation.combineSubResults(aggregated.getDouble(reductionIndex),accumulation.z().getDouble(i)));
            }

            accum.setN(biggerTadLength);
            accumulation.setApplyFinalTransform(true);
            for(int i = 0; i < aggregated.length(); i++) {
                aggregated.putScalar(i,accumulation.calculateFinalResult(aggregated.getDouble(i),biggerTadLength));
            }
        }
        else if(accum instanceof IndexAccumulation) {
            IndexAccumulation indexAccumulation = (IndexAccumulation) accum;
            for(int i = 0; i < smallerProblem; i++) {
                int reductionIndex = reductionIndexForTad(i,biggerProblem,smallerProblem);
                aggregated.putScalar(reductionIndex,indexAccumulation.combineSubResults(accum.x().getDouble(i), i, aggregated.getDouble(reductionIndex), reductionIndex));
            }
        }




        //set the new result
        accum.setZ(aggregated);

    }

    @Override
    public INDArray x() {
        return accum.x();
    }

    @Override
    public INDArray y() {
       return accum.y();
    }

    @Override
    public INDArray z() {
        return accum.z();
    }

    @Override
    public void exec(int... dimensions) {
        this.originalDimension = dimensions;
        exec();
    }

    @Override
    public int opNum() {
        return 0;
    }

    @Override
    public String name() {
        if(accum == null) {
            return DEFAULT_NAME;
        }
        return accum.name();
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, double other) {
        return accum.op(origin,other);
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, float other) {
        return accum.op(origin,other);
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, IComplexNumber other) {
        return accum.op(origin, other);
    }

    @Override
    public float op(float origin, float other) {
        return accum.op(origin, other);
    }

    @Override
    public double op(double origin, double other) {
        return accum.op(origin, other);
    }

    @Override
    public double op(double origin) {
        return accum.op(origin);
    }

    @Override
    public float op(float origin) {
        return accum.op(origin);
    }

    @Override
    public IComplexNumber op(IComplexNumber origin) {
        return accum.op(origin);
    }

    @Override
    public Op opForDimension(int index, int dimension) {
        return accum.opForDimension(index, dimension);
    }

    @Override
    public Op opForDimension(int index, int... dimension) {
        return accum.opForDimension(index,dimension);
    }





    /**
     * Given an linear index, element wise stride
     * and the length of each tad
     * map a linear index to a tad
     * @param i the index to map
     * @param elementWiseStride the element wise stride for the tads
     * @param numElementsPerTad the number of elements
     * per tad
     */
    public static int tadIndex(int i,int elementWiseStride,int numElementsPerTad) {
        return i / (numElementsPerTad * elementWiseStride);
    }

    /**
     * Map a tad to a
     * reduction index.
     * @param tadIndexForOriginal the original tad index for the
     * split up problem (eg: split is dimension 3 mapping to a 2,3 problem)
     * @param tadsForReduced the number of tads for the shrunk down problem (eg: 2,3)
     * @param tadsForOriginal the number of tads for the smaller problem (eg: 3)
     */
    public static int reductionIndexForTad(int tadIndexForOriginal,int tadsForReduced,int tadsForOriginal) {
        if(tadIndexForOriginal == 0)
            return 0;
        return tadIndexForOriginal / (tadsForOriginal / tadsForReduced);
    }

    /**
     * Computes the number of tads
     * per reduce index for the
     * reduction tad.
     */
    public static int tadsPerReduceIndex(int tadsForReduce,int tadsForOriginal) {
        return tadsForOriginal / tadsForReduce;
    }


    /**
     * Maps a linear index to a reduction index
     * @param i the linear index to map
     * @param elementWiseStride the element wise stride
     * for the multiple problem
     * @param tadNum the number of tads for the shrunken problem
     * @param originalTadNum the tad number for the reduced version of the problem
     */
    public static int reductionIndexForLinear(
            int i
            ,int elementWiseStride
            ,int numElementsPerTad
            ,int tadNum
            ,int originalTadNum) {
        int tad = tadIndex(i,elementWiseStride,numElementsPerTad);
        return reductionIndexForTad(tad,tadNum,originalTadNum);
    }


}
