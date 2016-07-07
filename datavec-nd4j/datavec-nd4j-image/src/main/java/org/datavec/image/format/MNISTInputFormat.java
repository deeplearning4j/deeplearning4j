package org.datavec.image.format;

import java.io.IOException;

import org.datavec.api.conf.Configuration;
import org.datavec.api.formats.input.BaseInputFormat;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.split.InputSplit;
import org.datavec.image.recordreader.MNISTRecordReader;

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
