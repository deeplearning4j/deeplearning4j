package org.canova.image.format;

import java.io.IOException;

import org.etl4j.api.conf.Configuration;
import org.etl4j.api.formats.input.BaseInputFormat;
import org.etl4j.api.records.reader.RecordReader;
import org.etl4j.api.split.InputSplit;
import org.canova.image.recordreader.MNISTRecordReader;

public class MNISTInputFormat extends BaseInputFormat {
	
    @Override
    public RecordReader createReader(InputSplit split, Configuration conf) throws IOException, InterruptedException {
        return createReader(split);
    }

    @Override
    public RecordReader createReader(InputSplit split) throws IOException, InterruptedException {
    	MNISTRecordReader reader = new MNISTRecordReader();
        reader.initialize(split);
        return reader;
    }


}
