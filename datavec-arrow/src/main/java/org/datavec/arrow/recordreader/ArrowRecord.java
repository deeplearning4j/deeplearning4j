package org.datavec.arrow.recordreader;

import lombok.AllArgsConstructor;
import org.datavec.api.records.Record;
import org.datavec.api.records.metadata.RecordMetaData;
import org.datavec.api.records.metadata.RecordMetaDataIndex;
import org.datavec.api.writable.Writable;

import java.net.URI;
import java.util.List;

@AllArgsConstructor
public class ArrowRecord implements Record {
    private ArrowListWritable arrowListWritable;
    private  int index;
    private URI recordUri;

    @Override
    public List<Writable> getRecord() {
        return arrowListWritable.get(index);
    }

    @Override
    public void setRecord(List<Writable> record) {
        arrowListWritable.set(index,record);
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
