package org.deeplearning4j.datasets.fetchers;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.image.transform.ImageTransform;

/**
 * Interface for defining a model that can be instantiated and return
 * information about itself.
 *
 * @author Justin Long (crockpotveggies)
 */
interface CacheableDataSet {

    String remoteDataUrl();
    String remoteDataUrl(DataSetType set);
    String localCacheName();
    String dataSetName(DataSetType set);
    long expectedChecksum();
    long expectedChecksum(DataSetType set);
    boolean isCached();
    RecordReader getRecordReader(long rngSeed, int[] imgDim, DataSetType set, ImageTransform imageTransform);

}
