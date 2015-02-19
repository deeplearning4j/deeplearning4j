package org.deeplearning4j.cli.api.flags;

import org.canova.api.records.reader.RecordReader;
import org.canova.api.records.writer.RecordWriter;

import java.net.URI;

/**
 * Base Input/Output Flag class provides extra URI parsing utilities
 *
 * @author sonali
 */
public abstract class BaseIOFlag implements Flag {
    //URI parsing utilities

    protected RecordReader createReader(URI uri) {
        return null;
    }

    protected RecordWriter createWriter(URI uri) {
        return null;
    }

}
