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
    @Builder.Default private long kT = 1;
    @Builder.Default private long kW = 1;
    @Builder.Default private long kH = 1;

    //strides
    @Builder.Default private long dT = 1;
    @Builder.Default private long dW = 1;
    @Builder.Default private long dH = 1;

    //padding
    @Builder.Default private long pT = 0;
    @Builder.Default private long pW = 0;
    @Builder.Default private int pH = 0;

    //dilations
    @Builder.Default private long dilationT = 1;
    @Builder.Default private long dilationW = 1;
    @Builder.Default private long dilationH = 1;

    //output padding
    @Builder.Default private long aT = 0;
    @Builder.Default private long aW = 0;
    @Builder.Default private long aH = 0;

    private boolean biasUsed = false;
    private boolean isValidMode;
    private boolean isNCDHW;

    @Builder.Default
    private String dataFormat = "NDHWC";

    public Map<String,Object> toProperties() {
        Map<String,Object> ret = new LinkedHashMap<>();
        ret.put("kT",kT);
        ret.put("kW",kW);
        ret.put("kH",kH);
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
        ret.put("isValidMode",isValidMode);

        return ret;
    }




}
