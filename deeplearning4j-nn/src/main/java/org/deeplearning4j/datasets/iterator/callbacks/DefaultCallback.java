package org.deeplearning4j.datasets.iterator.callbacks;

import org.nd4j.linalg.api.concurrency.AffinityManager;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.factory.Nd4j;

/**
 * This callback ensures that memory on device is up-to-date with host memory.
 *
 * PLEASE NOTE: This callback is used by default, no need to set it explicitly in AsyncDataSet iterators
 *
 * @author raver119@gmail.com
 */
public class DefaultCallback implements DataSetCallback {
    @Override
    public void call(DataSet dataSet) {
        if (dataSet != null) {
            if (dataSet.getFeatures() != null)
                Nd4j.getAffinityManager().ensureLocation(dataSet.getFeatures(), AffinityManager.Location.DEVICE);

            if (dataSet.getLabels() != null)
                Nd4j.getAffinityManager().ensureLocation(dataSet.getLabels(), AffinityManager.Location.DEVICE);

            if (dataSet.getFeaturesMaskArray() != null)
                Nd4j.getAffinityManager().ensureLocation(dataSet.getFeaturesMaskArray(),
                                AffinityManager.Location.DEVICE);

            if (dataSet.getLabelsMaskArray() != null)
                Nd4j.getAffinityManager().ensureLocation(dataSet.getLabelsMaskArray(), AffinityManager.Location.DEVICE);
        }
    }

    @Override
    public void call(MultiDataSet multiDataSet) {
        if (multiDataSet != null) {
            if (multiDataSet.getFeatures() != null)
                for (int i = 0; i < multiDataSet.getFeatures().length; i++)
                    Nd4j.getAffinityManager().ensureLocation(multiDataSet.getFeatures()[i],
                                    AffinityManager.Location.DEVICE);

            if (multiDataSet.getLabels() != null)
                for (int i = 0; i < multiDataSet.getLabels().length; i++)
                    Nd4j.getAffinityManager().ensureLocation(multiDataSet.getLabels()[i],
                                    AffinityManager.Location.DEVICE);

            if (multiDataSet.getFeaturesMaskArrays() != null)
                for (int i = 0; i < multiDataSet.getFeaturesMaskArrays().length; i++)
                    Nd4j.getAffinityManager().ensureLocation(multiDataSet.getFeaturesMaskArrays()[i],
                                    AffinityManager.Location.DEVICE);

            if (multiDataSet.getLabelsMaskArrays() != null)
                for (int i = 0; i < multiDataSet.getLabelsMaskArrays().length; i++)
                    Nd4j.getAffinityManager().ensureLocation(multiDataSet.getLabelsMaskArrays()[i],
                                    AffinityManager.Location.DEVICE);
        }
    }

    @Override
    public void reset() {
        // do nothing
    }
}
