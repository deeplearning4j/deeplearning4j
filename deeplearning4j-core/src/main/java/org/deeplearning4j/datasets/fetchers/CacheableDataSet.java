package org.deeplearning4j.datasets.fetchers;

/**
 * Interface for defining a model that can be instantiated and return
 * information about itself.
 */
public interface CacheableDataSet {

    public String remoteDataUrl();
    public String localCacheName();
    public long expectedChecksum();

}
