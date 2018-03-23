package org.datavec.api.records.mapper;

import org.datavec.api.conf.Configuration;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.writer.RecordWriter;
import org.datavec.api.split.InputSplit;
import org.datavec.api.split.partition.Partitioner;
import org.datavec.api.writable.Writable;

import java.util.List;

public class RecordMapper {
    private RecordReader recordReader;
    private RecordWriter recordWriter;
    private InputSplit inputUrl;
    private InputSplit outputUrl;
    private Configuration configuration;
    private Partitioner partitioner;

    public void copy() throws Exception {
        recordReader.initialize(configuration,inputUrl);
        partitioner.init(configuration,outputUrl);
        recordWriter.initialize(configuration,outputUrl,partitioner);


        while(recordReader.hasNext()) {
            List<Writable> next = recordReader.next();
            recordWriter.write(next);
            //update records written
            partitioner.updatePartitionInfo(1);
            if(partitioner.needsNewPartition()) {
                partitioner.openNewStream();
            }
        }
    }


}
