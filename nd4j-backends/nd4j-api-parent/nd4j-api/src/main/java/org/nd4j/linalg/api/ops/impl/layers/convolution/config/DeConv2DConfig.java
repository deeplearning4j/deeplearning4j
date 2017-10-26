package org.nd4j.linalg.api.ops.impl.layers.convolution.config;

import lombok.Builder;
import lombok.Data;

@Builder
@Data
public class DeConv2DConfig {
    private int kY,kX,sY,sX,pY,pX,dY,dX;
    private boolean isSameMode;



}
