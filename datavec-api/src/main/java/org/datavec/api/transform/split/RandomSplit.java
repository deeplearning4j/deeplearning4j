package org.datavec.api.transform.split;

import lombok.AllArgsConstructor;
import lombok.Data;

/**
 * Created by Alex on 5/03/2016.
 */
@AllArgsConstructor @Data
public class RandomSplit implements SplitStrategy {

    private double fractionTrain;

}
