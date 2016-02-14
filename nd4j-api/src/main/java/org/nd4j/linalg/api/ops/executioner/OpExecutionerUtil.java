package org.nd4j.linalg.api.ops.executioner;

import lombok.AllArgsConstructor;
import lombok.Data;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.util.ArrayUtil;

import java.util.Arrays;

/**Utility functions for the DefaultOpExecutioner
 * @author Alex Black
 */
public class OpExecutionerUtil {

    /** Can we do the op (X = Op(X)) directly on the arrays without breaking X up into 1d tensors first?
     * In general, this is possible if the elements of X are contiguous in the buffer, OR if every element
     * of X is at position offset+i*elementWiseStride in the buffer
     * */
    public static boolean canDoOpDirectly(INDArray x){
        if(x.isVector()) return true;

        //For a single NDArray all we require is that the elements are contiguous in the buffer or every nth element

        //Full buffer -> implies all elements are contiguous (and match)
        int l1 = x.length();
        int dl1 = x.data().length();
        if(l1 == dl1) return true;

        //Strides are same as a zero offset NDArray -> all elements are contiguous (even if not offset 0)
        int[] shape1 = x.shape();
        int[] stridesAsInit = (x.ordering()=='c' ? ArrayUtil.calcStrides(shape1) : ArrayUtil.calcStridesFortran(shape1));
        boolean stridesSameAsInit = Arrays.equals(x.stride(), stridesAsInit);
        return stridesSameAsInit;
    }

    /** Can we do the transform op (X = Op(X,Y)) directly on the arrays without breaking them up into 1d tensors first? */
    public static boolean canDoOpDirectly(INDArray x, INDArray y){
        if(x.isVector()) return true;
        if(x.ordering() != y.ordering()) return false; //other than vectors, elements in f vs. c NDArrays will never line up

        //Full buffer + matching strides -> implies all elements are contiguous (and match)
        //Need strides to match, otherwise elements in buffer won't line up (i.e., c vs. f order arrays)
        int l1 = x.length();
        int dl1 = x.data().length();
        int l2 = y.length();
        int dl2 = y.data().length();
        int[] strides1 = x.stride();
        int[] strides2 = y.stride();
        boolean equalStrides = Arrays.equals(strides1, strides2);
        if(l1==dl1 && l2==dl2 && equalStrides) return true;

        //Strides match + are same as a zero offset NDArray -> all elements are contiguous (and match)
        if(equalStrides){
            int[] shape1 = x.shape();
            int[] stridesAsInit = (x.ordering()=='c' ? ArrayUtil.calcStrides(shape1) : ArrayUtil.calcStridesFortran(shape1));
            boolean stridesSameAsInit = Arrays.equals(strides1, stridesAsInit);
            return stridesSameAsInit;
        }

        return false;
    }

    /** Can we do the transform op (Z = Op(X,Y)) directly on the arrays without breaking them up into 1d tensors first? */
    public static boolean canDoOpDirectly(INDArray x, INDArray y, INDArray z){
        if(x.isVector()) return true;
        if(x.ordering() != y.ordering() || x.ordering() != z.ordering() ) return false; //other than vectors, elements in f vs. c NDArrays will never line up

        //Full buffer + matching strides -> implies all elements are contiguous (and match)
        int l1 = x.length();
        int dl1 = x.data().length();
        int l2 = y.length();
        int dl2 = y.data().length();
        int l3 = z.length();
        int dl3 = z.data().length();
        int[] strides1 = x.stride();
        int[] strides2 = y.stride();
        int[] strides3 = z.stride();
        boolean equalStrides = Arrays.equals(strides1, strides2) && Arrays.equals(strides1,strides3);
        if(l1==dl1 && l2==dl2 && l3==dl3 && equalStrides) return true;

        //Strides match + are same as a zero offset NDArray -> all elements are contiguous (and match)
        if(equalStrides) {
            int[] shape1 = x.shape();
            int[] stridesAsInit = (x.ordering() == 'c' ? ArrayUtil.calcStrides(shape1) : ArrayUtil.calcStridesFortran(shape1));
            boolean stridesSameAsInit = Arrays.equals(strides1, stridesAsInit);
            return stridesSameAsInit;
        }

        return false;
    }

    /**
     *
     * Choose tensor dimension for operations with one argument: x=Op(x) or similar<br>
     * When doing some operations in parallel, it is necessary to break up
     * operations along a dimension to
     * give a set of 1d tensors. The dimension that this is done on is important for performance reasons;
     * in summary we want to both minimize the number of tensors
     * , but also minimize the separation between
     * elements in the buffer (so the resulting operation is efficient - i.e., avoids cache thrashing).
     * However, achieving both minimal number
     * of tensors and are not always possible.
     * @param x NDArray that we want to split
     * @return The best dimension to split on
     */
    public static int chooseElementWiseTensorDimension(INDArray x) {
        if(x.isVector())
            return ArrayUtil.argMax(x.shape());    //Execute along the vector

        //doing argMin(max(x.stride(i),y.stride(i))) minimizes the maximum
        //separation between elements (helps CPU cache) BUT might result in a huge number
        //of tiny ops - i.e., addi on NDArrays with shape [5,10^6]
        int opAlongDimensionMinStride = ArrayUtil.argMin(x.stride());

        //doing argMax on shape gives us smallest number of largest tensors
        //but may not be optimal in terms of element separation (for CPU cache etc)
        int opAlongDimensionMaxLength = ArrayUtil.argMax(x.shape());

        //Edge cases: shapes with 1s in them can have stride of 1 on the dimensions of length 1
        if(x.isVector() || x.size(opAlongDimensionMinStride) == 1)
            return opAlongDimensionMaxLength;

        //Using a heuristic approach here: basically if we get >= 10x as many tensors using the minimum stride
        //dimension vs. the maximum size dimension, use the maximum size dimension instead
        //The idea is to avoid choosing wrong dimension in cases like shape=[10,10^6]
        //Might be able to do better than this with some additional thought
        int nOpsAlongMinStride = ArrayUtil.prod(ArrayUtil.removeIndex(x.shape(), opAlongDimensionMinStride));
        int nOpsAlongMaxLength = ArrayUtil.prod(ArrayUtil.removeIndex(x.shape(), opAlongDimensionMaxLength));
        if(nOpsAlongMinStride <= 10 * nOpsAlongMaxLength)
            return opAlongDimensionMinStride;
        else
            return opAlongDimensionMaxLength;
    }


    /**
     * Choose tensor dimension for operations with 2 arguments: x=Op(x,y) or similar<br>
     * @see #chooseElementWiseTensorDimension(INDArray)
     */
    public static int chooseElementWiseTensorDimension(INDArray x, INDArray y) {
        if(x.isVector())
            return ArrayUtil.argMax(x.shape());    //Execute along the vector

        //doing argMin(max(x.stride(i),y.stride(i))) minimizes the maximum
        //separation between elements (helps CPU cache) BUT might result in a huge number
        //of tiny ops - i.e., addi on NDArrays with shape [5,10^6]
        int opAlongDimensionMinStride = ArrayUtil.argMinOfMax(x.stride(), y.stride());

        //doing argMax on shape gives us smallest number of largest tensors
        //but may not be optimal in terms of element separation (for CPU cache etc)
        int opAlongDimensionMaxLength = ArrayUtil.argMax(x.shape());

        //Edge case: shapes with 1s in them can have stride of 1 on the dimensions of length 1
        if(opAlongDimensionMinStride == opAlongDimensionMaxLength || x.size(opAlongDimensionMinStride)==1)
            return opAlongDimensionMaxLength;

        //Using a heuristic approach here: basically if we get >= 10x as many tensors using the minimum stride
        //dimension vs. the maximum size dimension, use the maximum size dimension instead
        //The idea is to avoid choosing wrong dimension in cases like shape=[10,10^6]
        //Might be able to do better than this with some additional thought
        int nOpsAlongMinStride = ArrayUtil.prod(ArrayUtil.removeIndex(x.shape(), opAlongDimensionMinStride));
        int nOpsAlongMaxLength = ArrayUtil.prod(ArrayUtil.removeIndex(x.shape(), opAlongDimensionMaxLength));
        if(nOpsAlongMinStride <= 10 * nOpsAlongMaxLength)
            return opAlongDimensionMinStride;
        else return opAlongDimensionMaxLength;
    }

    /**Choose tensor dimension for operations with 3 arguments: z=Op(x,y) or similar<br>
     * @see #chooseElementWiseTensorDimension(INDArray)
     */
    public static int chooseElementWiseTensorDimension(INDArray x, INDArray y, INDArray z){
        if(x.isVector()) return ArrayUtil.argMax(x.shape());

        int opAlongDimensionMinStride = ArrayUtil.argMinOfMax(x.stride(),y.stride(),z.stride());

        int opAlongDimensionMaxLength = ArrayUtil.argMax(x.shape());
        //Edge case: shapes with 1s in them can have stride of 1 on the dimensions of length 1
        if(opAlongDimensionMinStride == opAlongDimensionMaxLength || x.size(opAlongDimensionMinStride)==1)
            return opAlongDimensionMaxLength;

        int nOpsAlongMinStride = ArrayUtil.prod(ArrayUtil.removeIndex(x.shape(), opAlongDimensionMinStride));
        int nOpsAlongMaxLength = ArrayUtil.prod(ArrayUtil.removeIndex(x.shape(), opAlongDimensionMaxLength));
        if(nOpsAlongMinStride <= 10 * nOpsAlongMaxLength) return opAlongDimensionMinStride;
        else return opAlongDimensionMaxLength;
    }


    /** Tensor1DStats, used to efficiently iterate through tensors on a matrix (2d NDArray) for element-wise ops
     * For example, the offset of each 1d tensor can be calculated using only a single tensorAlongDimension method call,
     * hence is potentially faster than approaches requiring multiple tensorAlongDimension calls.<br>
     * Note that this can only (generally) be used for 2d NDArrays. For certain 3+d NDArrays, the tensor starts may not
     * be in increasing order
     */
    public static Tensor1DStats get1DTensorStats(INDArray array, int...dimension) {
        int tensorLength = array.size(dimension[0]);

        //As per tensorssAlongDimension:
        int numTensors = array.tensorssAlongDimension(dimension);

        //First tensor always starts with the first element in the NDArray, regardless of dimension
        int firstTensorOffset = array.offset();

        //Next: Need to work out the separation between the start (first element) of each 1d tensor
        int tensorStartSeparation;
        int elementWiseStride;  //Separation in buffer between elements in the tensor
        if(numTensors == 1) {
            tensorStartSeparation = -1; //Not applicable
            elementWiseStride = array.elementWiseStride();
        } else {
            INDArray secondTensor = array.tensorAlongDimension(1, dimension);
            tensorStartSeparation = secondTensor.offset() - firstTensorOffset;
            elementWiseStride = secondTensor.elementWiseStride();
        }

        return new Tensor1DStats(firstTensorOffset,tensorStartSeparation,
                numTensors,tensorLength,elementWiseStride);
    }

    /** Simple class containing values used for calculating various quantities related to 1d tensors.<br>
     * offset of ith tensor: firstTensorOffset + i * tensorStartSeparation<br>
     * separation between elements in tensor: elementWiseStride<br>
     * number of elements in each 1d tensor: tensorLength<br>
     * number of 1d tensors: numTensors<br>
     */
    @AllArgsConstructor
    @Data
    public static class Tensor1DStats {
        public final int firstTensorOffset;
        public final int tensorStartSeparation;
        public final int numTensors;
        public final int tensorLength;
        public final int elementWiseStride;
    }
}
