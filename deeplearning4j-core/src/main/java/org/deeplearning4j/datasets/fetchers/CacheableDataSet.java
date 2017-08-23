package org.deeplearning4j.datasets.fetchers;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.image.transform.ImageTransform;

/**
 * Interface for defining a model that can be instantiated and return
 * information about itself.
 *
 * @author Justin Long (crockpotveggies)
 */
public interface CacheableDataSet {

    public String remoteDataUrl();
    public String localCacheName();
    public long expectedChecksum();
    public boolean isCached();
    public RecordReader getRecordReader(long rngSeed, int[] imgDim, DataSetType set, ImageTransform imageTransform);

}
