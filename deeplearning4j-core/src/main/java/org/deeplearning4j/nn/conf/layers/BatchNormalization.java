package org.deeplearning4j.nn.conf.layers;

import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.NoArgsConstructor;
import lombok.ToString;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Created by agibsonccc on 9/27/15.
 */
@Data
@NoArgsConstructor
@ToString(callSuper = true)
@EqualsAndHashCode(callSuper = true)
public class BatchNormalization extends Layer {
    private int[] shape;
    private double decay = 0.9;
    private double eps = Nd4j.EPS_THRESHOLD;
    private int size;
    private boolean finetune;
    private boolean useBatchMean;
    private int N;


}
