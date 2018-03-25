package org.datavec.arrow.recordreader;

import org.datavec.api.conf.Configuration;
import org.datavec.api.records.writer.RecordWriter;
import org.datavec.api.split.InputSplit;
import org.datavec.api.split.partition.PartitionMetaData;
import org.datavec.api.split.partition.Partitioner;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.Writable;
import org.datavec.arrow.ArrowConverter;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;

/**
 * Output arrow records to an output stream.
 *
 * @author Adam Gibson
 */
public class ArrowRecordWriter implements RecordWriter {

    private Configuration configuration;
    private Schema schema;
    private Partitioner partitioner;

    public ArrowRecordWriter(Schema schema) {
        this.schema = schema;
    }

    @Override
    public boolean supportsBatch() {
        return true;
    }

    @Override
    public void initialize(InputSplit inputSplit, Partitioner partitioner) throws Exception {
        this.partitioner = partitioner;
        partitioner.init(inputSplit);

    }

    @Override
    public void initialize(Configuration configuration, InputSplit split, Partitioner partitioner) throws Exception {
        setConf(configuration);
        this.partitioner = partitioner;
    }

    @Override
    public PartitionMetaData write(List<Writable> record) throws IOException {
        return writeBatch(Arrays.asList(record));
    }

    @Override
    public PartitionMetaData writeBatch(List<List<Writable>> batch) throws IOException {
        if(partitioner.needsNewPartition()) {
            partitioner.currentOutputStream().flush();
            partitioner.currentOutputStream().close();
            partitioner.openNewStream();
        }

        if(batch instanceof ArrowWritableRecordBatch) {
            ArrowWritableRecordBatch arrowWritableRecordBatch = (ArrowWritableRecordBatch) batch;
            ArrowConverter.writeRecordBatchTo(arrowWritableRecordBatch,schema,partitioner.currentOutputStream());
        }
        else {
            ArrowConverter.writeRecordBatchTo(batch, schema, partitioner.currentOutputStream());
        }

        partitioner.currentOutputStream().flush();
        return PartitionMetaData.builder().numRecordsUpdated(batch.size()).build();
    }

    @Override
    public void close() {
    }

    @Override
    public void setConf(Configuration conf) {
        this.configuration = conf;
    }

    @Override
    public Configuration getConf() {
        return configuration;
    }
}
