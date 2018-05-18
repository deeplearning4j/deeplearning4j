package org.nd4j.linalg.api.ops.impl.layers.convolution;

import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseTransformOp;
import org.nd4j.linalg.convolution.Convolution;
import org.nd4j.linalg.factory.Nd4j;

import java.util.List;


/**
 * Legacy version of the Pooling2D operation
 * @deprecated Note: This operation will be removed in a future release
 */
@Deprecated
@Slf4j
public class LegacyPooling2D extends BaseTransformOp {

    public enum Pooling2DType {
        MAX, AVG, PNORM,
    }

    private int kh, kw, sy, sx, ph, pw, dh, dw;
    private Pooling2DType type;
    boolean isSameMode;
    double extra;
    @Getter protected DataBuffer im2colShape;

    public LegacyPooling2D() {}

    public LegacyPooling2D(INDArray x, int kh, int kw, int sy, int sx, int ph, int pw, int dh, int dw, boolean isSameMode,
                     Pooling2DType type, double extra, INDArray z) {
        super(x);

        // FIXME: int csast
        int outHeight = Convolution.outputSize((int) x.size(2), kh, sy, ph, dh, isSameMode);
        int outWidth = Convolution.outputSize((int) x.size(3), kw, sx, pw, dw, isSameMode);

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
        this.im2colShape = getIm2ColShape(x, kh, kw, outHeight, outWidth);
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
    public String opName(){
        return "legacypooling2d";
    }

    @Override
    public Object[] extraArgs() {
        return new Object[] {kh, kw, sy, sx, ph, pw, dh, dw, isSameMode ? 1.0 : 0.0, type.ordinal(), extra};
    }

    private static DataBuffer getIm2ColShape(INDArray img, int kernelHeight, int kernelWidth, int outHeight, int outWidth) {
        //number of images
        long n = img.size(0);
        //number of channels (depth)
        long c = img.size(1);

        return Nd4j.getShapeInfoProvider().createShapeInformation(new long[] {n, c,  kernelHeight, kernelWidth, outHeight, outWidth}, 'c').getFirst();
    }

    @Override
    public String onnxName() {
        throw new NoOpNameFoundException("Not supported");
    }

    @Override
    public String tensorflowName() {
        throw new NoOpNameFoundException("Not supported");
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        throw new UnsupportedOperationException("Not supported");
    }

}
