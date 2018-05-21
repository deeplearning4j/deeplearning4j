package org.nd4j.linalg.heartbeat.utils;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.heartbeat.reports.Task;

/**
 * @author raver119@gmail.com
 */
public class TaskUtils {
    private TaskUtils() {}

    public static Task buildTask(INDArray[] array, INDArray[] labels) {
        Task task = new Task();

        return task;
    }

    public static Task buildTask(INDArray array, INDArray labels) {
        return new Task();
    }

    public static Task buildTask(INDArray array) {
        return new Task();
    }

    public static Task buildTask(DataSet dataSet) {
        return new Task();
    }

    public static Task buildTask(org.nd4j.linalg.dataset.api.DataSet dataSet) {
        return new Task();
    }

    public static Task buildTask(DataSetIterator dataSetIterator) {
        return new Task();
    }
}
