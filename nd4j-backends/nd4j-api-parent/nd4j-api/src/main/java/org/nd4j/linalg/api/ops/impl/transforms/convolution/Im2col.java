package org.nd4j.linalg.api.ops.impl.transforms.convolution;

import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseTransformOp;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.convolution.Convolution;
import org.nd4j.linalg.factory.Nd4j;


/**
 * Im2col operation
 */
public class Im2col extends BaseTransformOp {

    private int kh, kw, sy, sx, ph, pw, dh, dw;
    boolean isSameMode;

    public Im2col() {}

    public Im2col(INDArray x, int kh, int kw, int sy, int sx, int ph, int pw, boolean isSameMode) {
        this(x, kh, kw, sy, sx, ph, pw, 1, 1, isSameMode);
    }

    public Im2col(INDArray x, int kh, int kw, int sy, int sx, int ph, int pw, int dh, int dw, boolean isSameMode) {
        this(x, kh, kw, sy, sx, ph, pw, dh, dw, isSameMode, getNewOutputArray(x, kh, kw, sy, sx, ph, pw, dh, dw,false));
    }

    public Im2col(INDArray x, int kh, int kw, int sy, int sx, int ph, int pw, int dh, int dw, boolean isSameMode, INDArray z) {
        super(x);
        this.kh = kh;
        this.kw = kw;
        this.sy = sy;
        this.sx = sx;
        this.ph = ph;
        this.pw = pw;
        this.dh = dh;
        this.dw = dw;
        this.isSameMode = isSameMode;
        this.z = z;
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
        return new Object[] {kh, kw, sy, sx, ph, pw, dh, dw, isSameMode ? 1.0 : 0.0};
    }

    private static INDArray getNewOutputArray(INDArray img, int kernelHeight, int kernelWidth, int strideY, int strideX,
                    int padHeight, int padWidth, int dilationH, int dilationW, boolean coverAll) {
        //number of images
        int n = img.size(0);
        //number of channels (depth)
        int c = img.size(1);
        //image height
        int h = img.size(2);
        //image width
        int w = img.size(3);
        int outHeight = Convolution.outSize(h, kernelHeight, strideY, padHeight, dilationH, coverAll);
        int outWidth = Convolution.outSize(w, kernelWidth, strideX, padWidth, dilationW, coverAll);

        return Nd4j.createUninitialized(new int[] {n, c, kernelHeight, kernelWidth, outHeight, outWidth}, 'c');
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
