package org.deeplearning4j.cli.api.flags;

import java.net.URI;
import org.canova.api.records.reader.RecordReader;
import org.canova.api.records.writer.RecordWriter;

/**
 * Input flag for loading input data for the model
 *
 * @author sonali
 */
public abstract class Input extends BaseIOFlag {

    @Override
    public <E> E value(String value) throws Exception {
        URI uri = URI.create(value);
        String path = uri.getPath();
        String extension = path.substring(path.lastIndexOf(".") + 1);

        return (E) createReader(uri);
    }

    @Override
    protected RecordWriter createWriter(URI uri) {
        return null;
    }

    @Override
    protected RecordReader createReader(URI uri) {
        return null;
    }
}
