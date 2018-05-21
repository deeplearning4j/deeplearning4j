package org.deeplearning4j.datasets.fetchers;

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.io.FileUtils;
import org.nd4j.util.ArchiveUtils;

import java.io.File;
import java.io.IOException;
import java.net.URL;
import java.util.zip.Adler32;
import java.util.zip.Checksum;

/**
 * Abstract class for enabling dataset downloading and local caching.
 *
 * @author Justin Long (crockpotveggies)
 */
@Slf4j
public abstract class CacheableExtractableDataSetFetcher implements CacheableDataSet {

    public final static String DL4J_DIR = "/.deeplearning4j/";
    public final static File ROOT_CACHE_DIR = new File(System.getProperty("user.home"), DL4J_DIR);
    public File LOCAL_CACHE = new File(ROOT_CACHE_DIR, localCacheName());

    @Override public String dataSetName(DataSetType set) { return ""; }
    @Override public String remoteDataUrl() { return remoteDataUrl(DataSetType.TRAIN); }
    @Override public long expectedChecksum() { return expectedChecksum(DataSetType.TRAIN); }
    public void downloadAndExtract() throws IOException { downloadAndExtract(DataSetType.TRAIN); }

    /**
     * Downloads and extracts the local dataset.
     *
     * @throws IOException
     */
    public void downloadAndExtract(DataSetType set) throws IOException {
        String localFilename = new File(remoteDataUrl(set)).getName();
        File tmpFile = new File(System.getProperty("java.io.tmpdir"), localFilename);
        File cachedFile = new File(ROOT_CACHE_DIR.getAbsolutePath(), localFilename);

        // check empty cache
        if(LOCAL_CACHE.exists()) {
            if(LOCAL_CACHE.listFiles().length<1) LOCAL_CACHE.delete();
        }

        if(!new File(LOCAL_CACHE, dataSetName(set)).exists()) {
            LOCAL_CACHE.mkdirs();
            tmpFile.delete();
            log.info("Downloading dataset to " + tmpFile.getAbsolutePath());
            FileUtils.copyURLToFile(new URL(remoteDataUrl(set)), tmpFile);
        } else {
            log.info("Using cached dataset at " + cachedFile.toString());
        }

        if(expectedChecksum(set) != 0L) {
            log.info("Verifying download...");
            Checksum adler = new Adler32();
            FileUtils.checksum(tmpFile, adler);
            long localChecksum = adler.getValue();
            log.info("Checksum local is " + localChecksum + ", expecting "+expectedChecksum(set));

            if(expectedChecksum(set) != localChecksum) {
                log.error("Checksums do not match. Cleaning up files and failing...");
                cachedFile.delete();
                throw new IllegalStateException(
                        "Dataset file failed checksum. If this error persists, please open an issue at https://github.com/deeplearning4j/deeplearning4j.");
            }
        }

        ArchiveUtils.unzipFileTo(tmpFile.getAbsolutePath(), LOCAL_CACHE.getAbsolutePath());
    }

    /**
     * Returns a boolean indicating if the dataset is already cached locally.
     *
     * @return boolean
     */
    @Override
    public boolean isCached() {
        return LOCAL_CACHE.exists();
    }

}
