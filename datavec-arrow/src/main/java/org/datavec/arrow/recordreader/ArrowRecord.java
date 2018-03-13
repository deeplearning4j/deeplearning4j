package org.datavec.arrow.recordreader;

import org.datavec.api.records.Record;
import org.datavec.api.records.metadata.RecordMetaData;
import org.datavec.api.writable.Writable;

import java.util.List;

public class ArrowRecord implements Record {


    @Override
    public List<Writable> getRecord() {
        return null;
    }

    @Override
    public void setRecord(List<Writable> record) {

    }

    @Override
    public RecordMetaData getMetaData() {
        return null;
    }

    @Override
    public void setMetaData(RecordMetaData recordMetaData) {

    }
}
