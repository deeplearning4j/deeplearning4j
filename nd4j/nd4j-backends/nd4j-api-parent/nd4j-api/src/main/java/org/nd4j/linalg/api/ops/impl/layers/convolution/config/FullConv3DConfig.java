package org.nd4j.linalg.api.ops.impl.layers.convolution.config;

import lombok.Builder;
import lombok.Data;

import java.util.LinkedHashMap;
import java.util.Map;

@Builder
@Data
public class FullConv3DConfig extends BaseConvolutionConfig {
    private long dT,dW,dH,pT,pW,pH,dilationT,dilationW,dilationH,aT,aW,aH;
    private boolean biasUsed;
    private String dataFormat;




    public Map<String,Object> toProperties() {
        Map<String,Object> ret = new LinkedHashMap<>();
        ret.put("dT",dT);
        ret.put("dW",dW);
        ret.put("dH",dH);
        ret.put("pT",pT);
        ret.put("pW",pW);
        ret.put("pH",pH);
        ret.put("dD",dilationT);
        ret.put("dW",dilationW);
        ret.put("dH",dilationH);
        ret.put("aT",aT);
        ret.put("aW",aW);
        ret.put("aH",aH);
        ret.put("biasUsed",biasUsed);
        ret.put("dataFormat",dataFormat);
        return ret;
    }
}
