/*-
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */

package org.deeplearning4j.datasets.datavec;

import lombok.*;
import lombok.extern.slf4j.Slf4j;
import org.datavec.api.io.WritableConverter;
import org.datavec.api.io.converters.SelfWritableConverter;
import org.datavec.api.io.converters.WritableConverterException;
import org.datavec.api.records.Record;
import org.datavec.api.records.metadata.RecordMetaData;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.writable.Writable;
import org.datavec.common.data.NDArrayWritable;
import org.deeplearning4j.exception.DL4JInvalidConfigException;
import org.deeplearning4j.exception.DL4JInvalidInputException;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.api.memory.enums.*;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.FeatureUtil;

import java.io.IOException;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.LockSupport;


/**
 * Record reader dataset iterator
 *
 * @author Adam Gibson
 * @author raver119@gmail.com
 */
@Slf4j
public class ParallelRecordReaderDataSetIterator implements DataSetIterator {
    protected RecordReader recordReader;
    protected WritableConverter converter;
    protected int batchSize = 10;
    protected int maxNumBatches = -1;
    protected int batchNum = 0;
    protected int labelIndex = -1;
    protected int labelIndexTo = -1;
    protected int numPossibleLabels = -1;
    protected Iterator<List<Writable>> sequenceIter;
    protected DataSet last;
    protected boolean useCurrent = false;
    protected boolean regression = false;
    @Getter
    protected DataSetPreProcessor preProcessor;
    private AtomicBoolean collectMetaData = new AtomicBoolean(false);
    private volatile int prefetchSize = 8;

    private BlockingQueue<Future<DataSet>> buffer;
    private Future<DataSet> nextElement = null;
    private Future<DataSet> terminator = new DummyFuture();
    private ThreadPoolExecutor executor;
    private AsyncPrefetchThread thread;
    private final String guid = java.util.UUID.randomUUID().toString();
    private AtomicBoolean wasTriggered = new AtomicBoolean(false);
    private boolean useWorkspaces = true;
    private AtomicLong counterGlobal = new AtomicLong(0);
    private int workers = 2;

    // RecordReaderDataSetIterator(recordReader, AppConfig.batchSize, 1, AppConfig.numLabels);

    public ParallelRecordReaderDataSetIterator(RecordReader recordReader, WritableConverter converter, int batchSize) {
        this(recordReader, converter, batchSize, -1,
                recordReader.getLabels() == null ? -1 : recordReader.getLabels().size());
    }

    public ParallelRecordReaderDataSetIterator(RecordReader recordReader, int batchSize) {
        this(recordReader, new SelfWritableConverter(), batchSize, -1,
                recordReader.getLabels() == null ? -1 : recordReader.getLabels().size());
    }

    public ParallelRecordReaderDataSetIterator(RecordReader recordReader, int batchSize, int numThreads) {
        this(recordReader, new SelfWritableConverter(), batchSize, -1,
                recordReader.getLabels() == null ? -1 : recordReader.getLabels().size(), numThreads);
    }

    /**
     * Main constructor for classification. This will convert the input class index (at position labelIndex, with integer
     * values 0 to numPossibleLabels-1 inclusive) to the appropriate one-hot output/labels representation.
     *
     * @param recordReader         RecordReader: provides the source of the data
     * @param batchSize            Batch size (number of examples) for the output DataSet objects
     * @param labelIndex           Index of the label Writable (usually an IntWritable), as obtained by recordReader.next()
     * @param numPossibleLabels    Number of classes (possible labels) for classification
     */
    public ParallelRecordReaderDataSetIterator(RecordReader recordReader, int batchSize, int labelIndex,
                                       int numPossibleLabels) {
        this(recordReader, new SelfWritableConverter(), batchSize, labelIndex, numPossibleLabels);
    }

    public ParallelRecordReaderDataSetIterator(RecordReader recordReader, WritableConverter converter, int batchSize,
                                       int labelIndex, int numPossibleLabels, boolean regression) {
        this(recordReader, converter, batchSize, labelIndex, numPossibleLabels, -1, regression);
    }

    public ParallelRecordReaderDataSetIterator(RecordReader recordReader, WritableConverter converter, int batchSize,
                                       int labelIndex, int numPossibleLabels) {
        this(recordReader, converter, batchSize, labelIndex, numPossibleLabels, -1, false);
    }

    public ParallelRecordReaderDataSetIterator(RecordReader recordReader, WritableConverter converter, int batchSize,
                                       int labelIndex, int numPossibleLabels, int numThreads) {
        this(recordReader, converter, batchSize, labelIndex, numPossibleLabels, -1, false, numThreads);
    }

    public ParallelRecordReaderDataSetIterator(RecordReader recordReader, int batchSize, int labelIndex, int numPossibleLabels,
                                       int maxNumBatches) {
        this(recordReader, new SelfWritableConverter(), batchSize, labelIndex, numPossibleLabels, maxNumBatches, false);
    }

    /**
     * Main constructor for multi-label regression (i.e., regression with multiple outputs)
     *
     * @param recordReader      RecordReader to get data from
     * @param labelIndexFrom    Index of the first regression target
     * @param labelIndexTo      Index of the last regression target, inclusive
     * @param batchSize         Minibatch size
     * @param regression        Require regression = true. Mainly included to avoid clashing with other constructors previously defined :/
     */
    public ParallelRecordReaderDataSetIterator(RecordReader recordReader, int batchSize, int labelIndexFrom, int labelIndexTo,
                                       boolean regression) {
        this(recordReader, new SelfWritableConverter(), batchSize, labelIndexFrom, labelIndexTo, -1, -1, regression);
    }


    public ParallelRecordReaderDataSetIterator(RecordReader recordReader, WritableConverter converter, int batchSize,
                                       int labelIndex, int numPossibleLabels, int maxNumBatches, boolean regression) {
        this(recordReader, converter, batchSize, labelIndex, labelIndex, numPossibleLabels, maxNumBatches, regression);
    }

    public ParallelRecordReaderDataSetIterator(RecordReader recordReader, WritableConverter converter, int batchSize,
                                       int labelIndex, int numPossibleLabels, int maxNumBatches, boolean regression, int numThreads) {
        this(recordReader, converter, batchSize, labelIndex, labelIndex, numPossibleLabels, maxNumBatches, regression, numThreads, 8, false, true);
    }


    /**
     * Main constructor
     *
     * @param recordReader      the recordreader to use
     * @param converter         the batch size
     * @param maxNumBatches     Maximum number of batches to return
     * @param labelIndexFrom    the index of the label (for classification), or the first index of the labels for multi-output regression
     * @param labelIndexTo      only used if regression == true. The last index _inclusive_ of the multi-output regression
     * @param numPossibleLabels the number of possible labels for classification. Not used if regression == true
     * @param regression        if true: regression. If false: classification (assume labelIndexFrom is a
     */
    public ParallelRecordReaderDataSetIterator(RecordReader recordReader, WritableConverter converter, int batchSize,
                                       int labelIndexFrom, int labelIndexTo, int numPossibleLabels, int maxNumBatches,
                                       boolean regression) {
        this(recordReader, converter, batchSize, labelIndexFrom, labelIndexTo, numPossibleLabels, maxNumBatches, regression, 1, 8, false, true);

    }

    public ParallelRecordReaderDataSetIterator(RecordReader recordReader, WritableConverter converter, int batchSize,
                                       int labelIndexFrom, int labelIndexTo, int numPossibleLabels, int maxNumBatches,
                                       boolean regression, int numThreads, int prefetchSize, boolean collectMetaData, boolean useWorkspaces) {


        if (numThreads < 1)
            numThreads = 1;

        this.workers = numThreads;
        this.collectMetaData.set(collectMetaData);
        this.useWorkspaces = useWorkspaces;
        this.prefetchSize = prefetchSize;
        this.buffer = new LinkedBlockingQueue<>(prefetchSize);
        this.recordReader = recordReader;
        this.converter = converter;
        this.batchSize = batchSize;
        this.maxNumBatches = maxNumBatches;
        this.labelIndex = labelIndexFrom;
        this.labelIndexTo = labelIndexTo;
        this.numPossibleLabels = numPossibleLabels;
        this.regression = regression;

        log.info("Starting num threads: {}; prefetchBuffer: {}", numThreads, prefetchSize);

        this.executor = (ThreadPoolExecutor) Executors.newFixedThreadPool(numThreads, new ThreadFactory() {
            @Override
            public Thread newThread(Runnable r) {
                Thread t = Executors.defaultThreadFactory().newThread(r);

                // we enforce the same device probably?
                // TODO: investigate what would be better for perf here
                //Nd4j.getAffinityManager().attachThreadToDevice(t, Nd4j.getAffinityManager().getDeviceForCurrentThread());
                t.setDaemon(true);
                t.setName("RRDSI thread");
                return t;
            }
        });

        // FIXME: fix collectMetaData
        this.thread = new AsyncPrefetchThread(buffer, recordReader, terminator);
        Nd4j.getAffinityManager().attachThreadToDevice(this.thread, Nd4j.getAffinityManager().getDeviceForCurrentThread());
        this.thread.start();
    }


    @Override
    public DataSet next(int num) {
        throw new UnsupportedOperationException();
    }


    private DataSet getDataSet(List<Writable> record) {
        if(record == null)
            return null;

        List<Writable> currList;
        if (record instanceof List)
            currList = record;
        else
            currList = new ArrayList<>(record);

        //allow people to specify label index as -1 and infer the last possible label
        if (numPossibleLabels >= 1 && labelIndex < 0) {
            labelIndex = record.size() - 1;
        }

        INDArray label = null;
        INDArray featureVector = null;
        int featureCount = 0;
        int labelCount = 0;

        //no labels
        if (currList.size() == 2 && currList.get(1) instanceof NDArrayWritable
                && currList.get(0) instanceof NDArrayWritable && currList.get(0) == currList.get(1)) {
            NDArrayWritable writable = (NDArrayWritable) currList.get(0);
            return new DataSet(writable.get(), writable.get());
        }

        if (currList.size() == 2 && currList.get(0) instanceof  NDArrayWritable && currList.get(1) instanceof NDArrayWritable) {
            NDArrayWritable writableF = (NDArrayWritable) currList.get(0);
            NDArrayWritable writableL = (NDArrayWritable) currList.get(1);
            return new DataSet(writableF.get(), writableL.get());
        }

        if (currList.size() == 2 && currList.get(0) instanceof NDArrayWritable) {
            if (!regression) {
                label = FeatureUtil.toOutcomeVector((int) Double.parseDouble(currList.get(1).toString()),
                        numPossibleLabels);
            } else {
                if (currList.get(1) instanceof NDArrayWritable) {
                    label = ((NDArrayWritable) currList.get(1)).get();
                } else {
                    label = Nd4j.scalar(currList.get(1).toDouble());
                }
            }
            NDArrayWritable ndArrayWritable = (NDArrayWritable) currList.get(0);
            featureVector = ndArrayWritable.get();
            return new DataSet(featureVector, label);
        }

        for (int j = 0; j < currList.size(); j++) {
            Writable current = currList.get(j);
            //ndarray writable is an insane slow down herecd
            if (!(current instanceof NDArrayWritable) && current.toString().isEmpty())
                continue;

            if (regression && j == labelIndex && j == labelIndexTo && current instanceof NDArrayWritable) {
                //Case: NDArrayWritable for the labels
                label = ((NDArrayWritable) current).get();
            } else if (regression && j >= labelIndex && j <= labelIndexTo) {
                //This is the multi-label regression case
                if (label == null)
                    label = Nd4j.create(1, (labelIndexTo - labelIndex + 1));
                label.putScalar(labelCount++, current.toDouble());
            } else if (labelIndex >= 0 && j == labelIndex) {
                //single label case (classification, etc)
                if (converter != null)
                    try {
                        current = converter.convert(current);
                    } catch (WritableConverterException e) {
                        e.printStackTrace();
                    }
                if (numPossibleLabels < 1)
                    throw new IllegalStateException("Number of possible labels invalid, must be >= 1");
                if (regression) {
                    label = Nd4j.scalar(current.toDouble());
                } else {
                    int curr = current.toInt();
                    if (curr < 0 || curr >= numPossibleLabels) {
                        throw new DL4JInvalidInputException(
                                "Invalid classification data: expect label value (at label index column = "
                                        + labelIndex + ") to be in range 0 to "
                                        + (numPossibleLabels - 1)
                                        + " inclusive (0 to numClasses-1, with numClasses="
                                        + numPossibleLabels + "); got label value of " + current);
                    }
                    label = FeatureUtil.toOutcomeVector(curr, numPossibleLabels);
                }
            } else {
                try {
                    double value = current.toDouble();
                    if (featureVector == null) {
                        if (regression && labelIndex >= 0) {
                            //Handle the possibly multi-label regression case here:
                            int nLabels = labelIndexTo - labelIndex + 1;
                            featureVector = Nd4j.create(1, currList.size() - nLabels);
                        } else {
                            //Classification case, and also no-labels case
                            featureVector = Nd4j.create(labelIndex >= 0 ? currList.size() - 1 : currList.size());
                        }
                    }
                    featureVector.putScalar(featureCount++, value);
                } catch (UnsupportedOperationException e) {
                    // This isn't a scalar, so check if we got an array already
                    if (current instanceof NDArrayWritable) {
                        assert featureVector == null;
                        featureVector = ((NDArrayWritable) current).get();
                    } else {
                        throw e;
                    }
                }
            }
        }

        return new DataSet(featureVector, labelIndex >= 0 ? label : featureVector);
    }

    @Override
    public int totalExamples() {
        throw new UnsupportedOperationException();
    }

    @Override
    public int inputColumns() {
        if (last == null) {
            DataSet next = next();
            last = next;
            useCurrent = true;
            return next.numInputs();
        } else
            return last.numInputs();

    }

    @Override
    public int totalOutcomes() {
        if (last == null) {
            DataSet next = next();
            last = next;
            useCurrent = true;
            return next.numOutcomes();
        } else
            return last.numOutcomes();


    }

    @Override
    public boolean resetSupported() {
        return true;
    }

    @Override
    public boolean asyncSupported() {
        return true;
    }

    @Override
    public void reset() {
        batchNum = 0;
        thread.shutdown();
        nextElement = null;
        recordReader.reset();
        buffer.clear();
        // TODO: maybe worth shutting down executor as well? or recreate new buffer... or we don't care, Future is gone anyway

        this.thread = new AsyncPrefetchThread(buffer, recordReader, terminator);
        this.thread.start();
    }

    public void setCollectMetaData(boolean reallyCollect) {
        collectMetaData.set(true);
    }

    public boolean getCollectMetaData() {
        return getCollectMetaData();
    }

    @Override
    public int batch() {
        return batchSize;
    }

    @Override
    public int cursor() {
        throw new UnsupportedOperationException();

    }

    @Override
    public int numExamples() {
        throw new UnsupportedOperationException();
    }

    @Override
    public void setPreProcessor(org.nd4j.linalg.dataset.api.DataSetPreProcessor preProcessor) {
        this.preProcessor = preProcessor;
    }

    @Override
    public boolean hasNext() {
        wasTriggered.compareAndSet(false, true);
        try {
            if (nextElement != null && nextElement != terminator) {
                return true;
            } else if (nextElement == terminator)
                return false;

            nextElement = buffer.take();

            if (nextElement == terminator)
                return false;

            return true;
        } catch (InterruptedException e) {
            throw new RuntimeException(e);
        }
        //return (recordReader.hasNext() && (maxNumBatches < 0 || batchNum < maxNumBatches));
    }

    @Override
    public DataSet next() {
        if (!wasTriggered.get() && nextElement == null)
            if (!hasNext())
                throw new NoSuchElementException("No more records below this line");



        Future<DataSet> tmp = nextElement;
        nextElement = null;
        try {
            // yes, we're blocking here, but there are chances Future is complete at this moment after first call
            DataSet ds = tmp.get();

            /*
            if (ds.getFeatures().isAttached()) {
                if (Nd4j.getMemoryManager().getCurrentWorkspace() == null) {
                    ds.detach();
                } else {
                    ds.migrate();
                }
            }
            */

            return ds;
        } catch (Exception e) {
            throw new RuntimeException(e);
        }


        /*
        if (useCurrent) {
            useCurrent = false;
            if (preProcessor != null)
                preProcessor.preProcess(last);
            return last;
        }

        List<DataSet> dataSets = new ArrayList<>();
        List<RecordMetaData> meta = (collectMetaData ? new ArrayList<RecordMetaData>() : null);
        for (int i = 0; i < batchSize; i++) {
            if (!hasNext())
                break;
            if (recordReader instanceof SequenceRecordReader) {
                if (sequenceIter == null || !sequenceIter.hasNext()) {
                    List<List<Writable>> sequenceRecord = ((SequenceRecordReader) recordReader).sequenceRecord();
                    sequenceIter = sequenceRecord.iterator();
                }

                try {
                    List<Writable> record = sequenceIter.next();
                    DataSet d = getDataSet(record);
                    //account for transform process
                    if (d != null)
                        dataSets.add(d);
                }catch(Exception e) {
                    log.warn("Unable to get dataset ...skipping",e);
                }
            } else {
                if (collectMetaData) {
                    Record record = recordReader.nextRecord();
                    DataSet d = getDataSet(record.getRecord());
                    if(d != null) {
                        dataSets.add(d);
                        meta.add(record.getMetaData());
                    }
                } else {
                    try {
                        List<Writable> record = recordReader.next();
                        DataSet d = getDataSet(record);
                        if (d != null)
                            dataSets.add(d);
                    }catch(Exception e) {
                        log.warn("Unable to get dataset ...skipping",e);
                    }
                }
            }
        }
        batchNum++;

        if (dataSets.isEmpty()) {
            return null;
        }

        DataSet ret = DataSet.merge(dataSets);
        if (collectMetaData) {
            ret.setExampleMetaData(meta);
        }
        last = ret;
        if (preProcessor != null)
            preProcessor.preProcess(ret);
        //Add label name values to dataset
        if (recordReader.getLabels() != null)
            ret.setLabelNames(recordReader.getLabels());
        return ret;
        */
    }

    @Override
    public void remove() {
        throw new UnsupportedOperationException();
    }

    @Override
    public List<String> getLabels() {
        return recordReader.getLabels();
    }

    /**
     * Load a single example to a DataSet, using the provided RecordMetaData.
     * Note that it is more efficient to load multiple instances at once, using {@link #loadFromMetaData(List)}
     *
     * @param recordMetaData RecordMetaData to load from. Should have been produced by the given record reader
     * @return DataSet with the specified example
     * @throws IOException If an error occurs during loading of the data
     */
    public DataSet loadFromMetaData(RecordMetaData recordMetaData) throws IOException {
        return loadFromMetaData(Collections.singletonList(recordMetaData));
    }

    /**
     * Load a multiple examples to a DataSet, using the provided RecordMetaData instances.
     *
     * @param list List of RecordMetaData instances to load from. Should have been produced by the record reader provided
     *             to the RecordReaderDataSetIterator constructor
     * @return DataSet with the specified examples
     * @throws IOException If an error occurs during loading of the data
     */
    public DataSet loadFromMetaData(List<RecordMetaData> list) throws IOException {
        List<Record> records = recordReader.loadFromMetaData(list);
        List<DataSet> dataSets = new ArrayList<>();
        List<RecordMetaData> meta = new ArrayList<>();
        for (Record r : records) {
            dataSets.add(getDataSet(r.getRecord()));
            meta.add(r.getMetaData());
        }

        if (dataSets.isEmpty()) {
           return null;
        }

        DataSet ret = DataSet.merge(dataSets);
        ret.setExampleMetaData(meta);
        last = ret;
        if (preProcessor != null)
            preProcessor.preProcess(ret);
        if (recordReader.getLabels() != null)
            ret.setLabelNames(recordReader.getLabels());
        return ret;
    }


    protected class AsyncPrefetchThread extends Thread implements Runnable {
        private BlockingQueue<Future<DataSet>> buffer;
        private Future<DataSet> terminator;
        private RecordReader reader;
        private boolean getMeta;
        private AtomicBoolean isShutdown = new AtomicBoolean(false);
        private AtomicBoolean shouldWork = new AtomicBoolean(true);
        protected RuntimeException exception;
        protected WorkspaceConfiguration configuration;
        private String workspaceId;


        public AsyncPrefetchThread(@NonNull BlockingQueue<Future<DataSet>> buffer, @NonNull RecordReader reader, @NonNull Future<DataSet> terminator) {
            this.buffer = buffer;
            this.terminator = terminator;
            this.reader = reader;
            this.workspaceId = "APT_LOOP-" + guid;



            this.setName("RRDSI prefetch thread");
            this.setDaemon(true);

            if (reader.batchesSupported()) {
                configuration = WorkspaceConfiguration.builder()
                        .overallocationLimit(prefetchSize * Math.max(2, workers))
                        .minSize(10 * 1024L * 1024L)
                        .policyMirroring(MirroringPolicy.FULL)
                        .policySpill(SpillPolicy.EXTERNAL)
                        .policyLearning(LearningPolicy.FIRST_LOOP)
                        .policyReset(ResetPolicy.ENDOFBUFFER_REACHED)
                        .policyAllocation(AllocationPolicy.OVERALLOCATE)
                        .build();
            } else {
                configuration = WorkspaceConfiguration.builder()
                        .overallocationLimit(batchSize * prefetchSize * Math.max(2, workers))
                        .minSize(10 * 1024L * 1024L)
                        .policyMirroring(MirroringPolicy.FULL)
                        .policySpill(SpillPolicy.EXTERNAL)
                        .policyLearning(LearningPolicy.FIRST_LOOP)
                        .policyReset(ResetPolicy.ENDOFBUFFER_REACHED)
                        .policyAllocation(AllocationPolicy.OVERALLOCATE)
                        .build();
            }

            log.info("Workspace overallocation ratio: {}", configuration.getOverallocationLimit());
        }


        @Override
        public void run() {
            while (shouldWork.get()) {
                try {
                    AtomicLong counterExampes = new AtomicLong(0);
                    AtomicLong counterOrder = new AtomicLong(0);
                    getMeta = collectMetaData.get();
                    boolean limitHit = maxNumBatches > 0 && maxNumBatches < counterOrder.get();
                    if (!reader.batchesSupported() || getMeta) {
                        OrderedBatch currentBatch = new OrderedBatch(counterOrder.getAndIncrement());
                        while (!limitHit && reader.hasNext()) {
                            if (useWorkspaces) {
                                try (MemoryWorkspace workspace = Nd4j.getWorkspaceManager().getAndActivateWorkspace(configuration, workspaceId)) {
                                    if (!getMeta)
                                        currentBatch.addWritable(reader.next());
                                    else
                                        currentBatch.addRecord(reader.nextRecord());
                                }
                            } else {
                                if (!getMeta)
                                    currentBatch.addWritable(reader.next());
                                else
                                    currentBatch.addRecord(reader.nextRecord());
                            }

                            // if we've built our batch size - we send it to processing
                            if (currentBatch.size() == batchSize) {
                                // we should put callable here
                                currentBatch.commit();
                                Future<DataSet> future = executor.submit(new DataSetCallable(currentBatch));
                                getMeta = collectMetaData.get();
                                buffer.put(future);
                                currentBatch = new OrderedBatch(counterOrder.getAndIncrement());
                                limitHit = maxNumBatches > 0 && maxNumBatches < counterOrder.get();
                            }
                        }
                        if (currentBatch.size() > 0 && !limitHit) {
                            // process last batch
                            currentBatch.commit();
                            Future<DataSet> future = executor.submit(new DataSetCallable(currentBatch));
                            buffer.put(future);
                        }

                        buffer.put(terminator);
                    } else {
                        while (!limitHit && reader.hasNext()) {
                            if (useWorkspaces) {
                                try (MemoryWorkspace workspace = Nd4j.getWorkspaceManager().getAndActivateWorkspace(configuration, workspaceId)) {
                                    long time1 = System.currentTimeMillis();
                                    List<Writable> batch = reader.next(batchSize);
                                    DataSet ds = getDataSet(batch);
                                    long time2 = System.currentTimeMillis();

                                    log.info("Compilation time: {} ms; Footprint: {} bytes", time2 - time1, ds.getMemoryFootprint());
                                    buffer.put(new DummyFuture(ds));
                                }
                            } else {
                                List<Writable> batch = reader.next(batchSize);
                                buffer.put(new DummyFuture(getDataSet(batch)));
                            }
                            counterOrder.getAndIncrement();

                            limitHit = maxNumBatches > 0 && maxNumBatches < counterOrder.get();
                        }

                        buffer.put(terminator);
                    }
                } catch (InterruptedException e) {
                    shouldWork.set(false);
                    isShutdown.set(true);
                } catch (RuntimeException e) {
                    e.printStackTrace();
                    this.exception = e;
                    isShutdown.set(true);
                    shouldWork.set(false);
                }
            }
            isShutdown.set(true);
        }


        protected void shutdown() {
            shouldWork.set(false);
            thread.interrupt();
            while (!isShutdown.get()) {
                LockSupport.parkNanos(100);
                thread.interrupt();
            }
        }
    }


    protected static class OrderedBatch {
        protected long order;
        @Getter protected List<List<Writable>> writables = new ArrayList<>();
        @Getter protected List<Record> records = new ArrayList<>();
        @Getter protected volatile boolean isRecord = false;
        protected AtomicInteger counter = new AtomicInteger(0);
        @Getter protected long timeStart;
        @Getter protected long timeCommit;
        @Getter protected long timeCompilation;

        protected OrderedBatch(long order) {
            this.order = order;
            timeStart = System.currentTimeMillis();
        }

        protected void addWritable(List<Writable> writables) {
            this.writables.add(writables);
            counter.incrementAndGet();
        }

        protected void addRecord(Record record) {
            this.records.add(record);
            counter.incrementAndGet();
            isRecord = true;
        }

        protected int size() {
            return counter.get();
        }


        protected void commit(){
            timeCommit = System.currentTimeMillis();
            timeCompilation = timeCommit - timeStart;
        }
    }


    protected class DataSetCallable implements Callable<DataSet> {
        private OrderedBatch batch;
        private WorkspaceConfiguration configuration;
        private String workspaceId;
        protected AtomicBoolean firstLoop = new AtomicBoolean(true);

        public DataSetCallable(OrderedBatch batch) {
            this.batch = batch;

            configuration = WorkspaceConfiguration.builder()
                    // FIXME: overalloc limit is wrong here obviously. We should do (divide prefetch size by number of threads) + 1 probably
                    .overallocationLimit((prefetchSize *2)+1)
                    .minSize(10 * 1024L * 1024L)
                    .policyMirroring(MirroringPolicy.FULL)
                    .policySpill(SpillPolicy.EXTERNAL)
                    .policyLearning(LearningPolicy.OVER_TIME)
                    .policyReset(ResetPolicy.ENDOFBUFFER_REACHED)
                    .policyAllocation(AllocationPolicy.OVERALLOCATE)
                    .build();

            this.workspaceId = "RRDSI_LOOP-" + guid;
        }

        /**
         * This method does OrderedBatch -> DataSet conversion
         * @return
         * @throws Exception
         */
        @Override
        public DataSet call() throws Exception {

            DataSet ret = null;

            if (useWorkspaces) {
                if (Nd4j.getWorkspaceManager().checkIfWorkspaceExists(workspaceId))
                    firstLoop.set(false);

                // we need to initialize workspace first. so we'll do 2 loops for
                for (int l = firstLoop.get() ? 0 : 1; l < 2; l++) {
                    try (MemoryWorkspace workspace = Nd4j.getWorkspaceManager().getAndActivateWorkspace(configuration, workspaceId)) {
                        // here we create our DataSet
                        ret = process(batch);
                    }

                    if (firstLoop.get()) {
                        Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread(workspaceId).initializeWorkspace();
                        log.info("Workspace size on initialization: {}", Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread(workspaceId).getCurrentSize());
                        firstLoop.set(false);
                    }
                }
            } else {
                ret = process(batch);
            }

            return ret;
        }


        public DataSet process(OrderedBatch batch) {
            long timeResume = System.currentTimeMillis();
            DataSet ret = null;
            List<DataSet> dataSets = new ArrayList<>();
            List<RecordMetaData> meta = (batch.isRecord() ? new ArrayList<RecordMetaData>() : null);

            if (batch.isRecord()) {
                List<Record> records = batch.getRecords();
                for (int i = 0; i < records.size(); i++) {
                    DataSet d = getDataSet(records.get(i).getRecord());
                    dataSets.add(d);
                    meta.add(records.get(i).getMetaData());
                }
            } else {
                List<List<Writable>> writables = batch.getWritables();
                for (int i = 0; i < writables.size(); i++) {
                    DataSet d = getDataSet(writables.get(i));
                    dataSets.add(d);
                }
            }

            long timePull = System.currentTimeMillis();

            // should NOT ever happen
            if (dataSets.isEmpty()) {
                return null;
            }

            ret = DataSet.merge(dataSets);

            if (batch.isRecord()) {
                ret.setExampleMetaData(meta);
            }

            if (preProcessor != null)
                preProcessor.preProcess(ret);

            //Add label name values to dataset
            if (recordReader.getLabels() != null)
                ret.setLabelNames(recordReader.getLabels());

            long timeMerge = System.currentTimeMillis();

            log.info("Compilation: {} ms; Pull: {} ms; Merge: {} ms; Travel: {} ms;", batch.getTimeCompilation(), timePull - timeResume, timeMerge - timePull, timeMerge - batch.getTimeStart());

            return ret;
        }
    }


    protected static class DummyFuture implements Future<DataSet> {
        DataSet dataSet;

        public DummyFuture() {

        }

        public DummyFuture(DataSet ds) {
            this.dataSet = ds;
        }

        @Override
        public boolean cancel(boolean mayInterruptIfRunning) {
            return false;
        }

        @Override
        public boolean isCancelled() {
            return false;
        }

        @Override
        public boolean isDone() {
            return true;
        }

        @Override
        public DataSet get() throws InterruptedException, ExecutionException {
            return dataSet;
        }

        @Override
        public DataSet get(long timeout, TimeUnit unit) throws InterruptedException, ExecutionException, TimeoutException {
            return dataSet;
        }
    }

    public static class Builder {
        protected RecordReader recordReader;
        protected WritableConverter converter = new SelfWritableConverter();
        protected int batchSize = 10;
        protected int maxNumBatches = -1;
        protected int labelIndex = -1;
        protected int labelIndexTo = -1;
        protected int numPossibleLabels = -1;
        protected boolean regression = false;
        protected DataSetPreProcessor preProcessor;
        private boolean collectMetaData = false;
        private int prefetchSize = 2;
        private int workers = 1;
        private boolean useWorkspaces = true;

        public Builder(@NonNull RecordReader reader) {
            this.recordReader = reader;
            if (reader.getLabels()!=null)
                numPossibleLabels = reader.getLabels().size();
        }

        public Builder setWritableConverter(@NonNull WritableConverter converter) {
            this.converter = converter;
            return this;
        }

        public Builder setBatchSize(int batchSize) {
            if (batchSize < 1)
                throw new DL4JInvalidConfigException("batchSize can't be negative value");

            this.batchSize = batchSize;
            return this;
        }

        public Builder setMaxNumberOfDataSets(int size) {
            this.maxNumBatches = size;
            return this;
        }

        public Builder setRegressionMode(boolean reallySet) {
            this.regression = reallySet;
            return this;
        }

        public Builder setLabelIndex(int idx) {
            this.labelIndex = idx;
            return this;
        }

        public Builder setLabelIndexTo(int idx) {
            this.labelIndexTo = idx;
            return this;
        }


        public Builder setNumberOfPossibleLabels(int number) {
            this.numPossibleLabels = number;
            return this;
        }

        public Builder collectMetaData(boolean reallyCollect) {
            this.collectMetaData = reallyCollect;
            return this;
        }

        public Builder useWorkspaces(boolean reallyUse) {
            this.useWorkspaces = reallyUse;
            return this;
        }

        public Builder setDataSetPreProcessor(DataSetPreProcessor preProcessor) {
            this.preProcessor = preProcessor;
            return this;
        }

        public Builder prefetchBufferSize(int size) {
            if (size < 2)
                size = 2;

            this.prefetchSize = size;
            return this;
        }

        public Builder numberOfWorkers(int workers) {
            if (workers < 1)
                workers = 1;

            this.workers = workers;
            return this;
        }


        public ParallelRecordReaderDataSetIterator build() {
            ParallelRecordReaderDataSetIterator rrdsi = new ParallelRecordReaderDataSetIterator(recordReader, converter, batchSize, labelIndex, labelIndexTo, numPossibleLabels, maxNumBatches, regression, workers, prefetchSize, collectMetaData, useWorkspaces);
            return rrdsi;
        }
    }
}
