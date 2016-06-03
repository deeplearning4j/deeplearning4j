package org.deeplearning4j.spark.impl.common.misc;

import lombok.Data;

import java.io.Serializable;

/**
 * @author raver119@gmail.com
 */
@Data
public class ScoreReport implements Serializable {
    private double s;
    private long m;
    private long c;
}
