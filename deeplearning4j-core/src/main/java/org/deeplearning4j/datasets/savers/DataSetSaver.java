package org.deeplearning4j.datasets.savers;

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.io.FilenameUtils;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.File;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * This class can be helpful when users try to save datasets in parallel manner
 *
 * the default number of threads is cores * 2
 * (but you can set it manually using setNumThreads())
 *
 * Created by kepricon on 17. 4. 19.
 */
@Slf4j
public class DataSetSaver {
    private static int NUM_CORES = Runtime.getRuntime().availableProcessors();
    private static int numThreads = NUM_CORES * 2;
    private static int QUEUE_CAPACITY = 50;
    private static ThreadPoolExecutor taskExecutor = null;

    public static void setNumThreads(int numThreads) {
        DataSetSaver.numThreads = numThreads;
    }

    public static void setQueueCapacity(int qCapacity) {
        DataSetSaver.QUEUE_CAPACITY = qCapacity;
    }

    private static void initExecutors() {
        taskExecutor = new ThreadPoolExecutor(numThreads, numThreads,
                0L, TimeUnit.MILLISECONDS,
                new LinkedBlockingQueue<Runnable>(QUEUE_CAPACITY));

        taskExecutor.setRejectedExecutionHandler(new ThreadPoolExecutor.CallerRunsPolicy());
    }

    public static void saveDataSets(DataSetIterator iter, String savePath, String compressionAlgorithm) {

        initExecutors();

        File dest = new File(savePath);
        if (dest.exists()) {
            log.info(savePath + " exist, delete it and retry");
        } else {
            dest.mkdirs();
            log.info("datasets will be saved at " + savePath);

            log.info("Loading data...");

            AtomicInteger counter = new AtomicInteger(0);

            while (iter.hasNext()) {
                String filePath = FilenameUtils.concat(savePath, "dataset-" + (counter.getAndIncrement()) + ".bin");
                taskExecutor.execute(new Worker(counter.get() % numThreads, iter.next(), new File(filePath), compressionAlgorithm));

                if (counter.get() % 100 == 0) {
                    log.info("{} datasets queued so far...", counter.get());
                }
            }

            taskExecutor.shutdown();

            try {
                taskExecutor.awaitTermination(10, TimeUnit.MINUTES);
            } catch (InterruptedException e) {
                log.warn("Unable to terminate executors", e);
            }
        }
    }

    public static void saveDataSets(DataSetIterator iter, String savePath) {
        saveDataSets(iter, savePath, Worker.DEFAULT_COMPRESSION);
    }

    private static class Worker extends Thread {
        static final String DEFAULT_COMPRESSION = "NOCOMPRESSION";
        int id;
        DataSet ds;
        File path;
        String compressionAlgorithm = DEFAULT_COMPRESSION;

        public Worker(int id, DataSet ds, File path) {
            this(id, ds, path, DEFAULT_COMPRESSION);
        }

        public Worker(int id, DataSet ds, File path, String compressionAlgorithm) {
            this.id = id;
            this.ds = ds;
            this.path = path;
            this.compressionAlgorithm = compressionAlgorithm;
        }

        public void run() {
            if (compressionAlgorithm.equals(DEFAULT_COMPRESSION) == false) {
                ds.save(path, compressionAlgorithm);
            }else {
                ds.save(path);
            }

            log.debug("[" + id + "] thread saved : " + path.getAbsolutePath());
        }
    }
}
