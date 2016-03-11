package org.nd4j.linalg.api.ops.impl.transforms.convolution;

import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseTransformOp;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.convolution.Convolution;
import org.nd4j.linalg.factory.Nd4j;


/**
 * Im 2 col operation
 */
public class Im2col extends BaseTransformOp {

    private int kh,  kw,  sy,  sx,  ph,  pw;
    boolean coverAll;

    public Im2col() {
    }

    public Im2col(INDArray x, int kh, int kw, int sy, int sx, int ph, int pw, boolean coverAll) {
        super(x);
        this.kh = kh;
        this.kw = kw;
        this.sy = sy;
        this.sx = sx;
        this.ph = ph;
        this.pw = pw;
        this.coverAll = coverAll;
        this.z = getNewOutputArray(x,kh,kw,sy,sx,ph,pw,coverAll);
        extraArgs = this.extraArgs();

    }

    @Override
    public boolean isExecSpecial() {
        return true;
    }

    @Override
    public int opNum() {
        return 37;
    }

    @Override
    public String name() {
        return "im2col";
    }

    @Override
    public Object[] extraArgs() {
        return new Object[] {kw,kh,sx,sy,pw,ph,coverAll ? 1.0 : 0.0};
    }

    private  INDArray getNewOutputArray(INDArray img, int kernelHeight, int kernelWidth, int strideY, int strideX,
                                        int padHeight, int padWidth, boolean coverAll) {
        //number of images
        int n = img.size(0);
        //number of channels (depth)
        int c = img.size(1);
        //image height
        int h = img.size(2);
        //image width
        int w = img.size(3);
        int outHeight = Convolution.outSize(h, kernelHeight, strideY, padHeight, coverAll);
        int outWidth = Convolution.outSize(w, kernelWidth, strideX, padWidth, coverAll);

        return Nd4j.create(n, c, kernelHeight, kernelWidth, outHeight, outWidth);
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, double other) {
        return null;
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, float other) {
        return null;
    }

    @Override
    public IComplexNumber op(IComplexNumber origin, IComplexNumber other) {
        return null;
    }

    @Override
    public float op(float origin, float other) {
        return 0;
    }

    @Override
    public double op(double origin, double other) {
        return 0;
    }

    @Override
    public double op(double origin) {
        return 0;
    }

    @Override
    public float op(float origin) {
        return 0;
    }

    @Override
    public IComplexNumber op(IComplexNumber origin) {
        return null;
    }

    @Override
    public Op opForDimension(int index, int dimension) {
        return null;
    }

    @Override
    public Op opForDimension(int index, int... dimension) {
        return null;
    }
}
