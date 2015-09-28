package org.deeplearning4j.nn.conf.layers;

import lombok.*;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Batch normalization configuration
 *
 * @author Adam Gibson
 */
@Data
@NoArgsConstructor
@ToString(callSuper = true)
@EqualsAndHashCode(callSuper = true)
@Builder
@AllArgsConstructor
public class BatchNormalization extends Layer {
    private int[] shape;
    private double decay = 0.9;
    private double eps = Nd4j.EPS_THRESHOLD;
    private int size;
    private boolean finetune;
    private boolean useBatchMean;
    private int N;



}
