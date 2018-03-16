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
public class Conv1DConfig extends BaseConvolutionConfig  {
    private int k, s, p, d;
    @Builder.Default private int d = 1;
    private boolean isSameMode;
    @Builder.Default
    private String dataFormat = "NWHC";
    @Builder.Default private boolean isNHWC = false;

    public Map<String,Object> toProperties() {
        Map<String,Object> ret = new LinkedHashMap<>();
        ret.put("k",k);
        ret.put("s",s);
        ret.put("p",p);
        ret.put("d",d);
        ret.put("isSameMode",isSameMode);
        ret.put("isNWHC",isNHWC);
        return ret;
    }


}
