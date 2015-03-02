package org.deeplearning4j.cli.api.flags;

import org.canova.api.records.reader.RecordReader;
import org.canova.api.records.writer.RecordWriter;
import org.deeplearning4j.cli.subcommands.SubCommand;

import java.io.FileNotFoundException;
import java.net.URI;

/**
 * Interface for saving the model
 * @author sonali
 */
public abstract class Output extends BaseIOFlag {

    /**
     * Parse URI first to identify destination
     * Then save data to this location
     * @return
     */


    @Override
    public <E> E value(String value) throws Exception {
        URI uri = URI.create(value);
        String path = uri.getPath();
        String extension = path.substring(path.lastIndexOf(".") + 1);

        return (E) createWriter(uri);
    }

}
