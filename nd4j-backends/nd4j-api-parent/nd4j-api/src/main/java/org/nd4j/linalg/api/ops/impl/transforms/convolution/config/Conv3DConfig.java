package org.nd4j.linalg.api.ops.impl.transforms.convolution.config;


import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class Conv3DConfig {
    private int dT;
    private int dW;
    private int dH;
    private int pT;
    private int pW;
    private int pH;
    private int dilationT;
    private int dilationW;
    private int dilationH;
    private int aT;
    private int aW;
    private int aH;
    private boolean biasUsed;

}
