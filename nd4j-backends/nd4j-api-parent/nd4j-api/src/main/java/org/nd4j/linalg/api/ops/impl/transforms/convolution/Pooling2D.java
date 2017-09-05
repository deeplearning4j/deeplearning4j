package org.nd4j.linalg.api.ops.impl.transforms.convolution;

import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseTransformOp;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.convolution.Convolution;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;


/**
 * Pooling2D operation
 */
@Slf4j
public class Pooling2D extends BaseTransformOp {
    public enum Pooling2DType {
        MAX, AVG, PNORM,
    }

    private int kh, kw, sy, sx, ph, pw, dh, dw;
    private Pooling2DType type;
    boolean isSameMode;
    double extra;
    @Getter protected DataBuffer im2colShape;

    public Pooling2D() {}

    /*
    public Pooling2D(INDArray x, int kh, int kw, int sy, int sx, int ph, int pw, boolean isSameMode, Pooling2DType type) {
        this(x, kh, kw, sy, sx, ph, pw, isSameMode, type, getNewOutputArray(x, kh, kw, sy, sx, ph, pw, false));
    }
*/
    public Pooling2D(INDArray x, int kh, int kw, int sy, int sx, int ph, int pw, int dh, int dw, boolean isSameMode,
                     Pooling2DType type, double extra,  int virtualHeight, int virtualWidth, INDArray z) {
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
        this.type = type;
        this.z = z;
        this.extra = extra;
        this.im2colShape = getNewOutputShape(x, kh, kw, sy, sx, ph, pw, virtualHeight, virtualWidth, false);
        extraArgs = this.extraArgs();
    }

    @Override
    public boolean isExecSpecial() {
        return true;
    }

    @Override
    public int opNum() {
        return 71;
    }

    @Override
    public String name() {
        return "pooling2d";
    }

    @Override
    public Object[] extraArgs() {
        return new Object[] {kw, kh, sx, sy, pw, ph, dw, dh, isSameMode ? 1.0 : 0.0, type.ordinal(), extra};
    }

    private static DataBuffer getNewOutputShape(INDArray img, int kernelHeight, int kernelWidth, int strideY, int strideX,
                                                int padHeight, int padWidth, int outHeight, int outWidth,  boolean coverAll) {
        //number of images
        int n = img.size(0);
        //number of channels (depth)
        int c = img.size(1);
        //image height
        int h = img.size(2);
        //image width
        int w = img.size(3);

        return Nd4j.getShapeInfoProvider().createShapeInformation(new int[] {n, c,  kernelHeight, kernelWidth, outHeight, outWidth}, 'c').getFirst();
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
