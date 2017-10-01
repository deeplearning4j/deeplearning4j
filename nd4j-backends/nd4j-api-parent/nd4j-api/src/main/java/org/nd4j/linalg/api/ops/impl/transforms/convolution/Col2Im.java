package org.nd4j.linalg.api.ops.impl.transforms.convolution;

import org.nd4j.autodiff.ArrayField;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseTransformOp;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.factory.Nd4j;

import java.util.List;

/**
 * Created by agibsonccc on 3/9/16.
 */
public class Col2Im extends BaseTransformOp {
    private int sy = 0, sx = 0, ph = 0, pw = 0, h = 0, w = 0, dh, dw;
    private boolean isSameMode = false;

    public Col2Im(SameDiff sameDiff, DifferentialFunction i_v1, DifferentialFunction i_v2, int sy, int sx, int ph, int pw, int h, int w, int dh, int dw, boolean isSameMode) {
        super(sameDiff, i_v1, i_v2);
        this.sy = sy;
        this.sx = sx;
        this.ph = ph;
        this.pw = pw;
        this.h = h;
        this.w = w;
        this.dh = dh;
        this.dw = dw;
        this.isSameMode = isSameMode;
    }

    public Col2Im(SameDiff sameDiff, DifferentialFunction i_v1, DifferentialFunction i_v2, boolean inPlace, int sy, int sx, int ph, int pw, int h, int w, int dh, int dw, boolean isSameMode) {
        super(sameDiff, i_v1, i_v2, inPlace);
        this.sy = sy;
        this.sx = sx;
        this.ph = ph;
        this.pw = pw;
        this.h = h;
        this.w = w;
        this.dh = dh;
        this.dw = dw;
        this.isSameMode = isSameMode;
    }

    public Col2Im(SameDiff sameDiff, int sy, int sx, int ph, int pw, int h, int w, int dh, int dw, boolean isSameMode) {
        super(sameDiff);
        this.sy = sy;
        this.sx = sx;
        this.ph = ph;
        this.pw = pw;
        this.h = h;
        this.w = w;
        this.dh = dh;
        this.dw = dw;
        this.isSameMode = isSameMode;
    }

    public Col2Im(SameDiff sameDiff, DifferentialFunction i_v1, DifferentialFunction i_v2, Object[] extraArgs, int sy, int sx, int ph, int pw, int h, int w, int dh, int dw, boolean isSameMode) {
        super(sameDiff, i_v1, i_v2, extraArgs);
        this.sy = sy;
        this.sx = sx;
        this.ph = ph;
        this.pw = pw;
        this.h = h;
        this.w = w;
        this.dh = dh;
        this.dw = dw;
        this.isSameMode = isSameMode;
    }

    public Col2Im(SameDiff sameDiff, DifferentialFunction i_v, boolean inPlace, int sy, int sx, int ph, int pw, int h, int w, int dh, int dw, boolean isSameMode) {
        super(sameDiff, i_v, inPlace);
        this.sy = sy;
        this.sx = sx;
        this.ph = ph;
        this.pw = pw;
        this.h = h;
        this.w = w;
        this.dh = dh;
        this.dw = dw;
        this.isSameMode = isSameMode;
    }

    public Col2Im(SameDiff sameDiff, DifferentialFunction i_v, int[] shape, boolean inPlace, Object[] extraArgs, int sy, int sx, int ph, int pw, int h, int w, int dh, int dw, boolean isSameMode) {
        super(sameDiff, i_v, shape, inPlace, extraArgs);
        this.sy = sy;
        this.sx = sx;
        this.ph = ph;
        this.pw = pw;
        this.h = h;
        this.w = w;
        this.dh = dh;
        this.dw = dw;
        this.isSameMode = isSameMode;
    }

    public Col2Im(SameDiff sameDiff, DifferentialFunction i_v, Object[] extraArgs, int sy, int sx, int ph, int pw, int h, int w, int dh, int dw, boolean isSameMode) {
        super(sameDiff, i_v, extraArgs);
        this.sy = sy;
        this.sx = sx;
        this.ph = ph;
        this.pw = pw;
        this.h = h;
        this.w = w;
        this.dh = dh;
        this.dw = dw;
        this.isSameMode = isSameMode;
    }

    public Col2Im() {}

    public Col2Im(INDArray x, int sy, int sx, int ph, int pw, int h, int w, int dh, int dw) {
        this(x, sy, sx, ph, pw, h, w, dh, dw, false, getNewOutputArray(x, h, w));
    }

    public Col2Im(INDArray x, int sy, int sx, int ph, int pw, int h, int w, int dh, int dw, boolean isSameMode, INDArray z) {
        super(x);
        this.sy = sy;
        this.sx = sx;
        this.ph = ph;
        this.pw = pw;
        this.h = h;
        this.w = w;
        this.dh = dh;
        this.dw = dw;
        this.z = z;
        this.isSameMode = isSameMode;
        extraArgs = this.extraArgs();
    }

    @Override
    public boolean isExecSpecial() {
        return true;
    }

    @Override
    public Object[] extraArgs() {
        return new Object[] {sy, sx, ph, pw, h, w, dh, dw, isSameMode ? 1.0 : 0.0};
    }

    @Override
    public int opNum() {
        return 36;
    }

    @Override
    public String name() {
        return "col2im";
    }

    private static INDArray getNewOutputArray(INDArray x, int imgHeight, int imgWidth) {
        //number of images
        int n = x.size(0);
        //number of columns
        int c = x.size(1);

        return Nd4j.create(n, c, imgHeight, imgWidth);
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

    @Override
    public ArrayField doGetValue() {
        return null;
    }

    @Override
    public List<DifferentialFunction> doDiff(List<DifferentialFunction> f1) {
        return null;
    }
}
