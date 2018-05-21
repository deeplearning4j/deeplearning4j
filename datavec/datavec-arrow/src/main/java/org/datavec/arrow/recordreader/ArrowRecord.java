package org.datavec.arrow.recordreader;

import lombok.AllArgsConstructor;
import org.datavec.api.records.Record;
import org.datavec.api.records.metadata.RecordMetaData;
import org.datavec.api.records.metadata.RecordMetaDataIndex;
import org.datavec.api.writable.Writable;

import java.net.URI;
import java.util.List;

/**
 * An {@link ArrowRecord} is a {@link Record}
 * wrapper around {@link ArrowWritableRecordBatch}
 * containing an index to the individual row.
 *
 * @author Adam Gibson
 */
@AllArgsConstructor
public class ArrowRecord implements Record {
    private ArrowWritableRecordBatch arrowWritableRecordBatch;
    private  int index;
    private URI recordUri;

    @Override
    public List<Writable> getRecord() {
        return arrowWritableRecordBatch.get(index);
    }

    @Override
    public void setRecord(List<Writable> record) {
        arrowWritableRecordBatch.set(index,record);
    }

    @Override
    public RecordMetaData getMetaData() {
        RecordMetaData ret = new RecordMetaDataIndex(index,recordUri,ArrowRecordReader.class);
        return ret;
    }

    @Override
    public void setMetaData(RecordMetaData recordMetaData) {

    }
}
