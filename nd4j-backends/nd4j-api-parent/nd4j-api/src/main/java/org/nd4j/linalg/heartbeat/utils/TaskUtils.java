package org.nd4j.linalg.heartbeat.utils;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.heartbeat.reports.Task;

/**
 * @author raver119@gmail.com
 */
public class TaskUtils {
    public static Task buildTask(INDArray[] array, INDArray[] labels) {
        Task task = new Task();

        return task;
    }

    public static Task buildTask(INDArray array, INDArray labels) {
        try {
            Task task = new Task();
            task.setNumFeatures(array.columns());

            return task;
        } catch (Exception e) {
            return new Task();
        }
    }

    public static Task buildTask(INDArray array) {
        try {
            Task task = new Task();
            task.setNumFeatures(array.columns());

            return task;
        } catch (Exception e) {
            return new Task();
        }
    }

    public static Task buildTask(DataSet dataSet) {
        try {
            Task task = new Task();
            task.setNumFeatures(dataSet.numInputs());
            task.setNumLabels(dataSet.numOutcomes());

            return task;
        } catch (Exception e) {
            return new Task();
        }
    }

    public static Task buildTask(org.nd4j.linalg.dataset.api.DataSet dataSet) {
        try {
            Task task = new Task();
            task.setNumFeatures(dataSet.numInputs());
            task.setNumLabels(dataSet.numOutcomes());

            return task;
        } catch (Exception e) {
            return new Task();
        }
    }

    public static Task buildTask(DataSetIterator dataSetIterator) {
        try {
            Task task = new Task();
            task.setNumFeatures(dataSetIterator.inputColumns());
            task.setNumLabels(dataSetIterator.totalOutcomes());

            return task;
        } catch (Exception e) {
            return new Task();
        }
    }
}
