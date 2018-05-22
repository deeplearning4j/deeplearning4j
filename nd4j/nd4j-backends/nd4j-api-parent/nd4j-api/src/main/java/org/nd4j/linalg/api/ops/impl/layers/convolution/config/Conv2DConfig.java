package org.nd4j.linalg.api.ops.impl.layers.convolution.config;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.LinkedHashMap;
import java.util.Map;

@Builder
@Data
@AllArgsConstructor
@NoArgsConstructor
public class Conv2DConfig extends BaseConvolutionConfig  {
    private long kh, kw, sy, sx, ph, pw;
    @Builder.Default private long dh = 1;
    @Builder.Default private long dw = 1;
    private boolean isSameMode;
    @Builder.Default
    private String dataFormat = "NWHC";
    @Builder.Default private boolean isNHWC = false;

    public Map<String,Object> toProperties() {
        Map<String,Object> ret = new LinkedHashMap<>();
        ret.put("kh",kh);
        ret.put("kw",kw);
        ret.put("sy",sy);
        ret.put("sx",sx);
        ret.put("ph",ph);
        ret.put("pw",pw);
        ret.put("dh",dh);
        ret.put("dw",dw);
        ret.put("isSameMode",isSameMode);
        ret.put("isNWHC",isNHWC);
        return ret;
    }


}
