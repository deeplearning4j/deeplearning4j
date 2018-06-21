package org.nd4j.linalg.api.ops.impl.layers.convolution.config;

import lombok.Builder;
import lombok.Data;

import java.util.LinkedHashMap;
import java.util.Map;

@Builder
@Data
public class DeConv2DConfig extends BaseConvolutionConfig {
    private long kH, kW, sH, sW, pH, pW, dH, dW;
    private boolean isSameMode;
    @Builder.Default
    private String dataFormat = "NWHC";
    @Builder.Default
    private boolean isNHWC = false;


    public Map<String, Object> toProperties() {
        Map<String, Object> ret = new LinkedHashMap<>();
        ret.put("kH", kH);
        ret.put("kW", kW);
        ret.put("sH", sH);
        ret.put("sW", sW);
        ret.put("pH", pH);
        ret.put("pW", pW);
        ret.put("dH", dH);
        ret.put("dW", dW);
        ret.put("isSameMode", isSameMode);
        ret.put("isNWHC", isNHWC);
        return ret;
    }
}
