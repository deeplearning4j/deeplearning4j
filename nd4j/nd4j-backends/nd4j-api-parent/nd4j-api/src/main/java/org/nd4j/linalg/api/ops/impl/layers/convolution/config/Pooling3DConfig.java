package org.nd4j.linalg.api.ops.impl.layers.convolution.config;

import lombok.Builder;
import lombok.Data;
import org.nd4j.linalg.api.ops.impl.layers.convolution.Pooling3D;

import java.util.LinkedHashMap;
import java.util.Map;

@Data
@Builder
public class Pooling3DConfig extends BaseConvolutionConfig {
    private long kD, kW, kH; // kernel
    private long sD, sW, sH; // strides
    private long pD, pW, pH; // padding
    private long dD, dW, dH; // dilation
    private Pooling3D.Pooling3DType type;
    private boolean ceilingMode;
    @Builder.Default private boolean isNCDHW = true;

    public Map<String, Object> toProperties() {
        Map<String, Object> ret = new LinkedHashMap<>();
        ret.put("kD", kD);
        ret.put("kW", kW);
        ret.put("kH", kH);
        ret.put("sD", sD);
        ret.put("sW", sW);
        ret.put("sH", sH);
        ret.put("pD", pD);
        ret.put("pW", pW);
        ret.put("pH", pH);
        ret.put("dD", dD);
        ret.put("dW", dW);
        ret.put("dH", dH);
        ret.put("type", type.toString());
        ret.put("ceilingMode", ceilingMode);
        return ret;

    }
}
