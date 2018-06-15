package org.nd4j.linalg.api.ops.impl.layers.convolution;

import lombok.Builder;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Conv2DConfig;
import org.nd4j.linalg.util.ArrayUtil;

import java.util.Collections;
import java.util.List;
import java.util.Map;


/**
 * Im2col operation
 */
public class Im2col extends DynamicCustomOp {

    protected Conv2DConfig conv2DConfig;

    @Builder(builderMethodName = "builder")
    public Im2col(SameDiff sameDiff, SDVariable[] inputFunctions, INDArray[] inputArrays, INDArray[] outputs, Conv2DConfig conv2DConfig) {
        super(null,inputArrays,outputs);

        if(sameDiff != null) {
            this.sameDiff = sameDiff;
        }

        this.conv2DConfig = conv2DConfig;

        addArgs();
    }

    public Im2col(SameDiff sd, SDVariable input, Conv2DConfig config){
        super(null, sd, new SDVariable[]{input});
        this.conv2DConfig = config;
        addArgs();
    }

    public Im2col() {}

    protected void addArgs() {
        addIArgument(conv2DConfig.getKH());
        addIArgument(conv2DConfig.getKW());
        addIArgument(conv2DConfig.getSH());
        addIArgument(conv2DConfig.getSW());
        addIArgument(conv2DConfig.getPH());
        addIArgument(conv2DConfig.getPW());
        addIArgument(conv2DConfig.getDH());
        addIArgument(conv2DConfig.getDW());
        addIArgument(ArrayUtil.fromBoolean(conv2DConfig.isSameMode()));

    }


    @Override
    public Map<String, Object> propertiesForFunction() {
        return conv2DConfig.toProperties();
    }

    @Override
    public String opName() {
        return "im2col";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        return Collections.singletonList(sameDiff.col2Im(f1.get(0), conv2DConfig));
    }
}
