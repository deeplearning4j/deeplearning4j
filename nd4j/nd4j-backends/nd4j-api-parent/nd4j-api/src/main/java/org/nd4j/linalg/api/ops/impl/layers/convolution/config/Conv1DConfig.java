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
public class Conv1DConfig extends BaseConvolutionConfig {
    @Builder.Default
    private long k = 1;
    @Builder.Default
    private long s = 1;
    @Builder.Default
    private long p = 0;
    @Builder.Default
    private String dataFormat = "NWHC";
    @Builder.Default
    private boolean isNHWC = false;
    private boolean isSameMode;

    public Map<String, Object> toProperties() {
        Map<String, Object> ret = new LinkedHashMap<>();
        ret.put("k", k);
        ret.put("s", s);
        ret.put("p", p);
        ret.put("isSameMode", isSameMode);
        ret.put("dataFormat", dataFormat);
        ret.put("isNWHC", isNHWC);
        return ret;
    }


}
