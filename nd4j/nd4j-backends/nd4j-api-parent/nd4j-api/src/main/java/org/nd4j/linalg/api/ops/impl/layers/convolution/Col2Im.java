package org.nd4j.linalg.api.ops.impl.layers.convolution;

import lombok.Builder;
import lombok.Getter;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Conv2DConfig;
import org.nd4j.linalg.util.ArrayUtil;

import java.util.List;
import java.util.Map;

/**
 * Col2Im operation.
 *
 * Created by agibsonccc on 3/9/16.
 */
@Getter
public class Col2Im extends DynamicCustomOp {

    protected Conv2DConfig conv2DConfig;

    @Builder(builderMethodName = "builder")
    public Col2Im(SameDiff sameDiff, SDVariable[] inputFunctions, INDArray[] inputArrays, INDArray[] outputs, Conv2DConfig conv2DConfig) {
        super(null,inputArrays,outputs);
        if(sameDiff != null) {
            this.sameDiff = sameDiff;
        }

        this.conv2DConfig = conv2DConfig;

        addArgs();
    }

    public Col2Im(SameDiff sd, SDVariable input, Conv2DConfig config){
        super(null, sd, new SDVariable[]{input});
        this.conv2DConfig = config;
        addArgs();
    }

    public Col2Im() {}

    protected void addArgs() {
        addIArgument(conv2DConfig.getsH());
        addIArgument(conv2DConfig.getsW());
        addIArgument(conv2DConfig.getpH());
        addIArgument(conv2DConfig.getpW());
        addIArgument(conv2DConfig.getkH());
        addIArgument(conv2DConfig.getkW());
        addIArgument(conv2DConfig.getdH());
        addIArgument(conv2DConfig.getdW());
        addIArgument(ArrayUtil.fromBoolean(conv2DConfig.isSameMode()));

    }

    @Override
    public Map<String, Object> propertiesForFunction() {
        return conv2DConfig.toProperties();
    }



    @Override
    public String opName() {
        return "col2im";
    }


    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        throw new UnsupportedOperationException("Unable to run derivative op on col2im");
    }
}
