package org.deeplearning4j.cli.output;

import org.canova.api.records.reader.RecordReader;
import org.canova.api.records.reader.impl.CSVRecordReader;
import org.canova.api.records.reader.impl.SVMLightRecordReader;
import org.canova.api.records.writer.RecordWriter;
import org.canova.api.records.writer.impl.CSVRecordWriter;
import org.canova.api.records.writer.impl.SVMLightRecordWriter;
import org.canova.api.split.FileSplit;
import org.canova.api.split.InputSplit;
import org.canova.api.writable.Writable;
import org.deeplearning4j.cli.api.flags.Output;

import java.io.*;
import java.net.URI;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

/**
 * CSV Output class saves model to a CSV using Canova CSVRecordWriter
 *
 * @author sonali
 */
public class CSVOutput extends Output {
    @Override
    protected RecordWriter createWriter(URI uri)  {

        //Stream in model
        File out = new File(uri.toString());
        try {
            RecordWriter writer = new CSVRecordWriter(out);
            List<Collection<Writable>> records = new ArrayList<>();
            for (Collection<Writable> record : records)
                writer.write(record);
            writer.close();
            return writer;
        }catch(Exception e       ) {
            throw new RuntimeException(e);
        }

    }



}
