package org.nd4j.linalg.api.ops.impl.layers.convolution.config;


import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.LinkedHashMap;
import java.util.Map;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class Conv3DConfig extends BaseConvolutionConfig {
    //strides
    private int dT;
    private int dW;
    private int dH;
    //padding
    private int pT;
    private int pW;
    private int pH;
    //dilations
    private int dilationT;
    private int dilationW;
    private int dilationH;
    //output padding
    private int aT;
    private int aW;
    private int aH;
    private boolean biasUsed;
    @Builder.Default
    private String dataFormat = "NDHWC";

    public Map<String,Object> toProperties() {
        Map<String,Object> ret = new LinkedHashMap<>();
        ret.put("dT",dT);
        ret.put("dW",dW);
        ret.put("dH",dH);
        ret.put("pT",pT);
        ret.put("pW",pW);
        ret.put("pH",pH);
        ret.put("dilationT",dilationT);
        ret.put("dilationW",dilationW);
        ret.put("dilationH",dilationH);
        ret.put("aT",aT);
        ret.put("aW",aW);
        ret.put("aH",aH);
        ret.put("biasUsed",biasUsed);
        ret.put("dataFormat",dataFormat);
        return ret;
    }




}
