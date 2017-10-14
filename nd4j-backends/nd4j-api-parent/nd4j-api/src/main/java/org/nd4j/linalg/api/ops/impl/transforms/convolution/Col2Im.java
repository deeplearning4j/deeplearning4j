package org.nd4j.linalg.api.ops.impl.transforms.convolution;

import lombok.Builder;
import lombok.Getter;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.factory.Nd4j;

import java.util.List;

/**
 * Created by agibsonccc on 3/9/16.
 */
@Getter
public class Col2Im extends DynamicCustomOp {
    private int sy = 0, sx = 0, ph = 0, pw = 0, h = 0, w = 0, dh, dw;
    private boolean isSameMode = false;



    @Builder(builderMethodName = "sameDiffBuilder")
    public Col2Im(SameDiff sameDiff, DifferentialFunction[] inputs,boolean inPlace, int sy, int sx, int ph, int pw, int h, int w, int dh, int dw, boolean isSameMode) {
        super(null,sameDiff, inputs, inPlace);
        this.sy = sy;
        this.sx = sx;
        this.ph = ph;
        this.pw = pw;
        this.h = h;
        this.w = w;
        this.dh = dh;
        this.dw = dw;
        this.isSameMode = isSameMode;
        getIArguments().add(h);
        getIArguments().add(w);
        getIArguments().add(sy);
        getIArguments().add(sx);
        getIArguments().add(ph);
        getIArguments().add(pw);
        getIArguments().add(dh);
        getIArguments().add(dw);
        getIArguments().add(fromBoolean(isSameMode));
    }


    public Col2Im() {}

    public Col2Im(INDArray[] x, int sy, int sx, int ph, int pw, int h, int w, int dh, int dw) {
        this(x,new INDArray[]{getNewOutputArray(x[0],h,w)}, sy, sx, ph, pw, h, w, dh, dw, false);

    }

    @Builder(builderMethodName = "execBuilder")
    public Col2Im(INDArray[] x, INDArray[] z,int sy, int sx, int ph, int pw, int h, int w, int dh, int dw, boolean isSameMode) {
        super(null,x,z);
        getIArguments().add(h);
        getIArguments().add(w);
        getIArguments().add(sy);
        getIArguments().add(sx);
        getIArguments().add(ph);
        getIArguments().add(pw);
        getIArguments().add(dh);
        getIArguments().add(dw);
        getIArguments().add(fromBoolean(isSameMode));
        this.h = h;
        this.w = w;
        this.sy = sy;
        this.sx = sx;
        this.ph = ph;
        this.pw = pw;
        this.dh = dh;
        this.dw = dw;
        this.isSameMode = isSameMode;
    }

      @Override
    public String opName() {
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
    public List<DifferentialFunction> doDiff(List<DifferentialFunction> f1) {
        return null;
    }
}
