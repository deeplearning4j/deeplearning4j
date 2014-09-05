package org.nd4j.linalg.ops.transforms;

import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.convolution.Convolution;
import org.nd4j.linalg.factory.NDArrays;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.ops.ArrayOps;
import org.nd4j.linalg.ops.BaseElementWiseOp;
import org.nd4j.linalg.ops.ElementWiseOp;
import org.nd4j.linalg.util.ArrayUtil;


/**
 * Functional interface for the different transform classes
 */
public class Transforms {



    public static INDArray neg(INDArray ndArray) {
        return exec(ndArray,Negative.class,null);
    }
    public static IComplexNDArray neg(IComplexNDArray ndArray) {
        return exec(ndArray,Negative.class,null);
    }



    /**
     * Down sampling a signal
     * @param d1
     * @param stride
     * @return
     */
    public static INDArray downSample(INDArray d1,int[] stride) {
        INDArray d = NDArrays.ones(stride);
        d.divi(ArrayUtil.prod(stride));
        INDArray ret = Convolution.convn(d1, d, Convolution.Type.VALID);
        ret = ret.get(NDArrayIndex.interval(0, stride[0]), NDArrayIndex.interval(0, stride[1]));
        return ret;
    }


    /**
     * Upsampling a signal
     * @param d
     * @param scale
     * @return
     */
    public static INDArray  upSample(INDArray d, INDArray scale) {

        INDArray idx = NDArrays.create(d.shape().length, 1);


        for (int i = 0; i < d.shape().length; i++) {
            INDArray tmp = NDArrays.zeros(d.size(i) * (int) scale.getScalar(i).element(), 1);
            int[] indices = ArrayUtil.range(0, (int) scale.getScalar(i).element() * d.size(i),(int) scale.getScalar(i).element());
            tmp.putScalar(indices, 1.0f);
            idx.put(i, tmp.cumsum(Integer.MAX_VALUE).sum(Integer.MAX_VALUE));
        }
        return idx;
    }


    /**
     * Consine similarity
     * @param d1
     * @param d2
     * @return
     */
    public static  double cosineSim(INDArray d1, INDArray d2) {
        d1 = unitVec(d1);
        d2 = unitVec(d2);
        double ret = NDArrays.getBlasWrapper().dot(d1,d2);
        return ret;
    }

    public static INDArray normalizeZeroMeanAndUnitVariance(INDArray toNormalize) {
        INDArray columnMeans = toNormalize.mean(1);
        INDArray columnStds = toNormalize.std(1);

        toNormalize.subiRowVector(columnMeans);
        columnStds.addi(1e-6);
        toNormalize.diviRowVector(columnStds);
        return toNormalize;
    }


    /**
     * Scale by 1 / norm2 of the matrix
     * @param toScale the ndarray to scale
     * @return the scaled ndarray
     */
    public static INDArray unitVec(INDArray toScale) {
        float length = (float) toScale.norm2(Integer.MAX_VALUE).element();
        if (length > 0)
            return NDArrays.getBlasWrapper().scal(1.0f / length,toScale);
        return toScale;
    }

    /**
     * Binary matrix of whether the number at a given index is equal
     * @param ndArray
     * @return
     */
    public static INDArray eq(INDArray ndArray) {
        return exec(ndArray,EqualTo.class,null);

    }

    /**
     * Binary matrix of whether the number at a given index is equal
     * @param ndArray
     * @return
     */
    public static IComplexNDArray eq(IComplexNDArray ndArray) {
        return exec(ndArray,EqualTo.class,null);
    }



    /**
     * Binary matrix of whether the number at a given index is greatger than
     * @param ndArray
     * @return
     */
    public static INDArray floor(INDArray ndArray) {
        return exec(ndArray,Floor.class,null);

    }



    /**
     * Binary matrix of whether the number at a given index is equal
     * @param ndArray
     * @return
     */
    public static IComplexNDArray floor(IComplexNDArray ndArray) {
        return exec(ndArray,Floor.class,null);
    }





    /**
     * Binary matrix of whether the number at a given index is equal
     * @param ndArray
     * @return
     */
    public static INDArray gt(INDArray ndArray) {
        return exec(ndArray,GreaterThan.class,null);
    }



    /**
     * Binary matrix of whether the number at a given index is equal
     * @param ndArray
     * @return
     */
    public static IComplexNDArray gt(IComplexNDArray ndArray) {
        return exec(ndArray,GreaterThan.class,null);
    }



    /**
     * Binary matrix of whether the number at a given index is equal
     * @param ndArray
     * @return
     */
    public static INDArray lt(INDArray ndArray) {
        return exec(ndArray,LessThan.class,null);

    }

    /**
     * Binary matrix of whether the number at a given index is equal
     * @param ndArray
     * @return
     */
    public static IComplexNDArray lt(IComplexNDArray ndArray) {
        return exec(ndArray,LessThan.class,null);
    }

    public static INDArray stabilize(INDArray ndArray,float k) {
        return exec(ndArray,Stabilize.class,new Object[]{k});
    }
    public static IComplexNDArray stabilize(IComplexNDArray ndArray,float k) {
        return exec(ndArray,Stabilize.class,new Object[]{k});
    }





    public static INDArray abs(INDArray ndArray) {
        return exec(ndArray,Abs.class,null);

    }
    public static IComplexNDArray abs(IComplexNDArray ndArray) {
        return exec(ndArray,Abs.class,null);
    }

    public static INDArray exp(INDArray ndArray) {
        return exec(ndArray,Exp.class,null);
    }
    public static IComplexNDArray exp(IComplexNDArray ndArray) {
        return exec(ndArray,Exp.class,null);
    }
    public static INDArray hardTanh(INDArray ndArray) {
        return exec(ndArray,HardTanh.class,null);
    }
    public static IComplexNDArray hardTanh(IComplexNDArray ndArray) {
        return exec(ndArray,HardTanh.class,null);
    }
    public static INDArray identity(INDArray ndArray) {
        return exec(ndArray,Identity.class,null);
    }
    public static IComplexNDArray identity(IComplexNDArray ndArray) {
        return exec(ndArray,Identity.class,null);
    }
    public static INDArray max(INDArray ndArray) {
        return exec(ndArray,Max.class,null);
    }
    public static IComplexNDArray max(IComplexNDArray ndArray) {
        return exec(ndArray,Max.class,null);
    }
    public static INDArray pow(INDArray ndArray,Number power) {
        return exec(ndArray,Pow.class,new Object[]{power});
    }
    public static IComplexNDArray pow(IComplexNDArray ndArray,IComplexNumber power) {
        return exec(ndArray,Pow.class,new Object[]{power});
    }
    public static INDArray round(INDArray ndArray) {
        return exec(ndArray,Round.class,null);
    }
    public static IComplexNDArray round(IComplexNDArray ndArray) {
        return exec(ndArray,Round.class,null);
    }
    public static INDArray sigmoid(INDArray ndArray) {
        return exec(ndArray,Sigmoid.class,null);
    }
    public static IComplexNDArray sigmoid(IComplexNDArray ndArray) {
        return exec(ndArray,Sigmoid.class,null);
    }
    public static INDArray sqrt(INDArray ndArray) {
        return exec(ndArray,Sqrt.class,null);
    }
    public static IComplexNDArray sqrt(IComplexNDArray ndArray) {
        return exec(ndArray,Sqrt.class,null);
    }
    public static INDArray tanh(INDArray ndArray) {
        return exec(ndArray,Tanh.class,null);
    }
    public static IComplexNDArray tanh(IComplexNDArray ndArray) {
        return exec(ndArray,Tanh.class,null);
    }
    public static INDArray log(INDArray ndArray) {
        return exec(ndArray,Log.class,null);
    }
    public static IComplexNDArray log(IComplexNDArray ndArray) {
        return exec(ndArray,Log.class,null);
    }


    private static INDArray exec(INDArray indArray,Class<? extends BaseElementWiseOp> clazz,Object[] extraArgs) {

       ElementWiseOp ops = new ArrayOps().
                from(indArray.dup())
                .op(clazz)
                .extraArgs(extraArgs)
                .build();
        ops.exec();

        return ops.from();
    }

    private static IComplexNDArray exec(IComplexNDArray indArray,Class<? extends BaseElementWiseOp> clazz,Object[] extraArgs) {

        ElementWiseOp ops = new ArrayOps().
                from(indArray.dup())
                .op(clazz)
                .extraArgs(extraArgs)
                .build();
        ops.exec();
        IComplexNDArray n = (IComplexNDArray) ops.from();
        return n;
    }
}
