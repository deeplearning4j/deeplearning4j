package org.nd4j.linalg.api.ops.impl.layers.convolution.config;

import lombok.Builder;
import lombok.Data;

import java.util.LinkedHashMap;
import java.util.Map;

@Data
@Builder
public class LocalResponseNormalizationConfig extends BaseConvolutionConfig {

    private double alpha, beta, bias;
    private int depth;

    public Map<String, Object> toProperties() {
        Map<String, Object> ret = new LinkedHashMap<>();
        ret.put("alpha", alpha);
        ret.put("beta", beta);
        ret.put("bias", bias);
        ret.put("depth", depth);
        return ret;
    }

}
