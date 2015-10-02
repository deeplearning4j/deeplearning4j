package org.nd4j.linalg.api.ops.executioner;

import lombok.AllArgsConstructor;
import lombok.Data;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.util.ArrayUtil;

import java.util.Arrays;

/**@author Alex Black
 */
public class OpExecutionerUtil {

    /** Can we do the transform op (X = Op(X)) directly on the arrays without breaking X up into 1d tensors first? */
    public static boolean canDoTransformOpDirectly(INDArray x){
        if(x.isVector()) return true;
        //For a single NDArray all we require is that the elements are contiguous in the buffer

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

    /** Can we do the transform op (X = X <op> Y) directly on the arrays without breaking them up into 1d tensors first? */
    public static boolean canDoTransformOpDirectly(INDArray x, INDArray y){
        if(x.isVector()) return true;

        //Full buffer + matching strides -> implies all elements are contiguous (and match)
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

    /** Can we do the transform op (Z = X <op> Y) directly on the arrays without breaking them up into 1d tensors first? */
    public static boolean canDoTransformOpDirectly(INDArray x, INDArray y, INDArray z){
        if(x.isVector()) return true;

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

    public static int chooseElementWiseTensorDimension(INDArray x){
        //doing argMin(max(x.stride(i),y.stride(i))) minimizes the maximum
        //separation between elements (helps CPU cache) BUT might result in a huge number
        //of tiny ops - i.e., addi on NDArrays with shape [5,10^6]
        int opAlongDimensionMinStride = ArrayUtil.argMin(x.stride());

        //doing argMax on shape gives us smallest number of largest tensors
        //but may not be optimal in terms of element separation (for CPU cache etc)
        int opAlongDimensionMaxLength = ArrayUtil.argMax(x.shape());

        //Edge cases: shapes with 1s in them can have stride of 1 on the dimensions of length 1
        if(x.isVector() || x.size(opAlongDimensionMinStride)==1) return opAlongDimensionMaxLength;

        //Using a heuristic approach here: basically if we get >= 10x as many tensors using the minimum stride
        //dimension vs. the maximum size dimension, use the maximum size dimension instead
        //The idea is to avoid choosing wrong dimension in cases like shape=[10,10^6]
        //Might be able to do better than this with some additional thought
        int nOpsAlongMinStride = ArrayUtil.prod(ArrayUtil.keep(x.shape(), opAlongDimensionMinStride));
        int nOpsAlongMaxLength = ArrayUtil.prod(ArrayUtil.keep(x.shape(), opAlongDimensionMaxLength));
        if(nOpsAlongMinStride <= 10*nOpsAlongMaxLength) return opAlongDimensionMinStride;
        else return opAlongDimensionMaxLength;
    }


    public static int chooseElementWiseTensorDimension(INDArray x, INDArray y){
        //doing argMin(max(x.stride(i),y.stride(i))) minimizes the maximum
        //separation between elements (helps CPU cache) BUT might result in a huge number
        //of tiny ops - i.e., addi on NDArrays with shape [5,10^6]
        int opAlongDimensionMinStride = ArrayUtil.argMinOfMax(x.stride(), y.stride());

        //doing argMax on shape gives us smallest number of largest tensors
        //but may not be optimal in terms of element separation (for CPU cache etc)
        int opAlongDimensionMaxLength = ArrayUtil.argMax(x.shape());

        //Edge cases: shapes with 1s in them can have stride of 1 on the dimensions of length 1
        if(x.size(opAlongDimensionMinStride)==1) return opAlongDimensionMaxLength;

        //Using a heuristic approach here: basically if we get >= 10x as many tensors using the minimum stride
        //dimension vs. the maximum size dimension, use the maximum size dimension instead
        //The idea is to avoid choosing wrong dimension in cases like shape=[10,10^6]
        //Might be able to do better than this with some additional thought
        int nOpsAlongMinStride = ArrayUtil.prod(ArrayUtil.keep(x.shape(), opAlongDimensionMinStride));
        int nOpsAlongMaxLength = ArrayUtil.prod(ArrayUtil.keep(x.shape(), opAlongDimensionMaxLength));
        if(nOpsAlongMinStride <= 10*nOpsAlongMaxLength) return opAlongDimensionMinStride;
        else return opAlongDimensionMaxLength;
    }

    public static int chooseElementWiseTensorDimension(INDArray x, INDArray y, INDArray z){
        throw new UnsupportedOperationException("not yet implemented");
    }


    /** Tensor1DStats, used to efficiently iterate through tensors on a matrix (2d NDArray) for element-wise ops
     */
    public static Tensor1DStats get1DTensorStats(INDArray array, int dimension){
        //As per BaseNDArray.tensorAlongDimension
        int tensorLength = ArrayUtil.prod(ArrayUtil.keep(array.shape(), dimension));

        //As per tensorssAlongDimension:
        int numTensors = array.length() / tensorLength;

        //First tensor always starts with the first element in the NDArray, regardless of dimension
        int firstTensorOffset = array.offset();

        //Next: Need to work out the separation between the start (first element) of each 1d tensor
        int tensorStartSeparation;
        int elementWiseStride;  //Separation in buffer between elements in the tensor
        if(numTensors == 1){
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
