package org.nd4j.linalg.api.ops.impl.layers.convolution.config;

import lombok.Builder;
import lombok.Data;

@Data
@Builder
public class LocalResponseNormalizationConfig {

    private double alpha,beta,bias,depth;

}
