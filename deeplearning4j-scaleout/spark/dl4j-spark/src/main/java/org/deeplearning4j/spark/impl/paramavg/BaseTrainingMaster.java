package org.deeplearning4j.spark.impl.paramavg;

import org.deeplearning4j.spark.api.TrainingMaster;
import org.deeplearning4j.spark.api.TrainingResult;
import org.deeplearning4j.spark.api.TrainingWorker;

/**
 * @author raver119@gmail.com
 * @author Alex Black
 */
public abstract class BaseTrainingMaster<R extends TrainingResult, W extends TrainingWorker<R>> implements TrainingMaster<R, W> {
}
