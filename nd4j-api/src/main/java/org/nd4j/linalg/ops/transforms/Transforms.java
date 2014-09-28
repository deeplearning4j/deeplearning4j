package org.nd4j.linalg.ops.transforms;

import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.convolution.Convolution;
import org.nd4j.linalg.factory.Nd4j;
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
     * Max pooling
     * @param input
     * @param ds the strides with which to pool expectations
     * @parma ignoreBorder whether to ignore the borders of images
     * @return
     */
    public static INDArray maxPool(INDArray input,int[] ds,boolean ignoreBorder) {
        assert input.length() >= 2 : "Max pooling requires an ndarray of >= length 2";
        assert ds.length == 2: "Down sampling must be of length 2 (the factors used for each image size";
        assert input.shape().length == 4 : "Only supports 4 dimensional tensors";
        int batchSize = ArrayUtil.prod(new int[]{input.size(0) * input.size(1)});
        //possibly look at a input implementation instead (looping over the outer dimension slice wise with calling input repeatedly)
        //use the given rows and columns if ignoring borders
        int rows = input.size(2);
        int cols = input.size(3);

        INDArray signalNDArray = input.reshape(new int[]{batchSize,1,rows,cols});
        INDArray zz= Nd4j.create(signalNDArray.shape()).assign(Float.MIN_VALUE);

        int rowIter = ignoreBorder ? rows / (int) Math.pow(ds[0],2) : rows;
        int colIter = ignoreBorder ? cols / (int) Math.pow(ds[1],2) : cols;

        for(int i = 0; i < signalNDArray.size(0); i++) {
            for(int j = 0; j < signalNDArray.size(1); j++) {
                for(int k = 0; k < rowIter; k++) {
                    int zk = k / ds[0];
                    for(int l = 0; l < colIter; l++) {
                        int zl = l / ds[1];
                        float num = input.get(new int[]{i, j, k, l});
                        float zzGet = zz.get(new int[]{i, j, zk, zl});
                        zz.putScalar(new int[]{i,j,zk,zl},Math.max(num,zzGet));
                    }
                }
            }
        }

        return zz.reshape(signalNDArray.shape());
    }

    /**
     * Down sampling a signal (specifically the first 2 dimensions)
     * @param d1
     * @param stride
     * @return
     */
    public static INDArray downSample(INDArray d1,int[] stride) {
        INDArray d = Nd4j.ones(stride);
        d.divi(ArrayUtil.prod(stride));
        INDArray ret = Convolution.convn(d1, d, Convolution.Type.VALID);
        ret = ret.get(NDArrayIndex.interval(0, stride[0]), NDArrayIndex.interval(0, stride[1]));
        return ret;
    }


    /**
     * Pooled expectations
     * @param toPool the ndarray to pool
     * @param stride the 2d stride across the ndarray
     * @return
     */
    public static INDArray pool(INDArray toPool,int[] stride) {

        int nDims = toPool.shape().length;
        assert nDims == 3 : "NDArray must have 3 dimensions";
        int nRows = toPool.shape()[nDims - 2];
        int nCols = toPool.shape()[nDims - 1];
        int yStride = stride[0],xStride = stride[1];
        INDArray blocks = Nd4j.create(toPool.shape());
        for(int iR = 0; iR < Math.ceil(nRows / yStride); iR++) {
            NDArrayIndex rows = NDArrayIndex.interval(iR  * yStride,iR * yStride,true);
            for(int jC = 0; jC < Math.ceil(nCols / xStride); jC++) {
                NDArrayIndex cols = NDArrayIndex.interval(jC  * xStride  ,(jC  * xStride) + 1,true);
                INDArray blockVal = toPool.get(rows,cols).sum(toPool.shape().length - 1).sum(toPool.shape().length - 1);
                blocks.put(
                        new NDArrayIndex[]{rows,cols},
                        blockVal.permute(new int[]{1,2,0}))
                        .repmat(new int[]{rows.length(),cols.length()});
            }
        }

        return blocks;
    }

    /**
     * Upsampling a signal (specifically the first 2 dimensions
     * @param d
     * @param scale
     * @return
     */
    public static INDArray  upSample(INDArray d, INDArray scale) {

        INDArray idx = Nd4j.create(d.shape().length, 1);


        for (int i = 0; i < d.shape().length; i++) {
            INDArray tmp = Nd4j.zeros(d.size(i) * (int) scale.get(i), 1);
            int[] indices = ArrayUtil.range(0, (int) scale.get(i) * d.size(i),(int) scale.get(i));
            tmp.putScalar(indices, 1.0f);
            idx.put(i,
                    tmp.cumsum(Integer.MAX_VALUE).sum(Integer.MAX_VALUE));
        }
        return idx;
    }


    /**
     * Cosine similarity
     * @param d1
     * @param d2
     * @return
     */
    public static  double cosineSim(INDArray d1, INDArray d2) {
        d1 = unitVec(d1.dup());
        d2 = unitVec(d2.dup());
        double ret = Nd4j.getBlasWrapper().dot(d1,d2);
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
            return Nd4j.getBlasWrapper().scal(1.0f / length,toScale);
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
