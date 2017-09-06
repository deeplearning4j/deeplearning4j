package org.nd4j.linalg.api.ops;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;

/**
 * This class is simple POJO that contains basic information about CustomOp
 *
 * @author raver119@gmail.com
 */
@AllArgsConstructor
@Builder
public class CustomOpDescriptor {
    @Getter private long hash;
    @Getter private int numInputs;
    @Getter private int numOutputs;
    @Getter private boolean allowsInplace;
    @Getter private int numTArgs;
    @Getter private int numIArgs;
}
