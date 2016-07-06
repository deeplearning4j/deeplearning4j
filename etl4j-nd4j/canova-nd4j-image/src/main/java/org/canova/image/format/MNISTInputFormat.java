package org.canova.image.format;

import java.io.IOException;

import org.canova.api.conf.Configuration;
import org.canova.api.formats.input.BaseInputFormat;
import org.canova.api.records.reader.RecordReader;
import org.canova.api.split.InputSplit;
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
