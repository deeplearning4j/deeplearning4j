package org.nd4j.linalg.api.ops.impl.layers.convolution.config;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Builder
@Data
@AllArgsConstructor
@NoArgsConstructor
public class Conv2DConfig {
    private int kh, kw, sy, sx, ph, pw;
    @Builder.Default private int dh = 1;
    @Builder.Default private int dw = 1;
    private boolean isSameMode;
    @Builder.Default private boolean isNHWC = false;
}
