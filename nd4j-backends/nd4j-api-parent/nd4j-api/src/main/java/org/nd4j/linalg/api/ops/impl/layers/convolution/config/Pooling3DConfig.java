package org.nd4j.linalg.api.ops.impl.layers.convolution.config;

import lombok.Builder;
import lombok.Data;
import org.nd4j.linalg.api.ops.impl.layers.convolution.Pooling3D;

import java.util.LinkedHashMap;
import java.util.Map;

@Data
@Builder
public class Pooling3DConfig {
    private int kT,kW,kH,dT,dW,dH,pT,pW,pH,dilationT,dilationW,dilationH;
    private Pooling3D.Pooling2DType type;
    private boolean ceilingMode;

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
        ret.put("type",type.toString());
        ret.put("ceilingMode",ceilingMode);
        return ret;

    }
}
