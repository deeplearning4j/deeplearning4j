package org.deeplearning4j.cli.input;

import org.canova.api.records.reader.RecordReader;
import org.canova.api.records.reader.impl.CSVRecordReader;
import org.canova.api.split.FileSplit;
import org.canova.api.split.InputSplit;
import org.deeplearning4j.cli.api.flags.Input;

import java.io.File;
import java.net.URI;

/**
 * CSV Input class uses Canova CSVRecordReader to read in data
 *
 * @author sonali
 */
public class CSVInput extends Input {
    @Override
    protected RecordReader create(URI uri) throws Exception {
        File file = new File(uri.toString());
        InputSplit split = new FileSplit(file);
        RecordReader reader = new CSVRecordReader();
        reader.initialize(split);

        while(reader.hasNext()) {
            //do something?
        }
        return reader;
    }
}
