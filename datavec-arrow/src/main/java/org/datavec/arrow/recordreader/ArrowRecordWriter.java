package org.datavec.arrow.recordreader;

import org.datavec.api.conf.Configuration;
import org.datavec.api.records.writer.RecordWriter;
import org.datavec.api.split.InputSplit;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.Writable;
import org.datavec.arrow.ArrowConverter;

import java.io.IOException;
import java.io.OutputStream;
import java.util.Arrays;
import java.util.List;

public class ArrowRecordWriter implements RecordWriter {

    private Configuration configuration;
    private Schema schema;
    private OutputStream to;

    public ArrowRecordWriter(Configuration configuration, Schema schema) {
        this.configuration = configuration;
        this.schema = schema;
    }

    @Override
    public void initialize(InputSplit inputSplit) throws Exception {

    }

    @Override
    public void initialize(Configuration configuration, InputSplit split) throws Exception {
        setConf(configuration);
    }

    @Override
    public void write(List<Writable> record) throws IOException {
        writeBatch(Arrays.asList(record));
    }

    @Override
    public void writeBatch(List<List<Writable>> batch) throws IOException {
        if(batch instanceof ArrowWritableRecordBatch) {
            ArrowWritableRecordBatch arrowWritableRecordBatch = (ArrowWritableRecordBatch) batch;

        }
        else {
            ArrowConverter.writeRecordBatchTo(batch, schema, to);
        }

        to.flush();
    }

    @Override
    public void close() {
        if(to != null) {
            try {
                to.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
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
