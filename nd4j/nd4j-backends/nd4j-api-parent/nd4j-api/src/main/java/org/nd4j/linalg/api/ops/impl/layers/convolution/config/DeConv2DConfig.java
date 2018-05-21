package org.nd4j.linalg.api.ops.impl.layers.convolution.config;

import lombok.Builder;
import lombok.Data;

import java.util.LinkedHashMap;
import java.util.Map;

@Builder
@Data
public class DeConv2DConfig extends BaseConvolutionConfig {
    private long kY,kX,sY,sX,pY,pX,dY,dX;
    private boolean isSameMode;
    @Builder.Default private boolean isNHWC = false;


    public Map<String,Object> toProperties() {
        Map<String,Object> ret = new LinkedHashMap<>();
        ret.put("kY",kY);
        ret.put("kX",kX);
        ret.put("sY",sY);
        ret.put("sX",sX);
        ret.put("pY",pY);
        ret.put("pX",pX);
        ret.put("dY",dY);
        ret.put("dX",dX);
        ret.put("isSameMode",isSameMode);
        ret.put("isNWHC",isNHWC);
        return ret;
    }
}
