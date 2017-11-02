package org.nd4j.linalg.api.ops.impl.layers.convolution.config;

import lombok.Builder;
import lombok.Data;
import org.nd4j.linalg.api.ops.impl.layers.convolution.Pooling3D;

@Data
@Builder
public class Pooling3DConfig {
    private int kT,kW,kH,dT,dW,dH,pT,pW,pH,dilationT,dilationW,dilationH;
    private Pooling3D.Pooling2DType type;
    private boolean ceilingMode;


}
