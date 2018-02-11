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
    @Builder.Default private int kT = 1;
    @Builder.Default private int kW = 1;
    @Builder.Default private int kH = 1;

    //strides
    @Builder.Default private int dT = 1;
    @Builder.Default private int dW = 1;
    @Builder.Default private int dH = 1;

    //padding
    @Builder.Default private int pT = 0;
    @Builder.Default private int pW = 0;
    @Builder.Default private int pH = 0;

    //dilations
    @Builder.Default private int dilationT = 1;
    @Builder.Default private int dilationW = 1;
    @Builder.Default private int dilationH = 1;

    //output padding
    @Builder.Default private int aT = 0;
    @Builder.Default private int aW = 0;
    @Builder.Default private int aH = 0;

    private boolean biasUsed;
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
