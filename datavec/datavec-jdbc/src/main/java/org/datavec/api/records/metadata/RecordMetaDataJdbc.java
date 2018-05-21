package org.datavec.api.records.metadata;

import java.net.URI;
import java.util.Collections;
import java.util.List;
import lombok.Getter;

/**
 * Record metadata to use with JDBCRecordReader. To uniquely identify and recover a record, we use a parameterized
 * request which will be prepared with the values stored in the params attribute.
 *
 * @author Adrien Plagnol
 */
public class RecordMetaDataJdbc implements RecordMetaData {

    private final URI uri;
    @Getter
    private final String request;
    @Getter
    private final List<Object> params;
    private final Class<?> readerClass;

    public RecordMetaDataJdbc(URI uri, String request, List<? extends Object> params, Class<?> readerClass) {
        this.uri = uri;
        this.request = request;
        this.params = Collections.unmodifiableList(params);
        this.readerClass = readerClass;
    }

    @Override
    public String getLocation() {
        return this.toString();
    }

    @Override
    public URI getURI() {
        return uri;
    }

    @Override
    public Class<?> getReaderClass() {
        return readerClass;
    }

    @Override
    public String toString() {
        return "jdbcRecord(uri=" + uri +
            ", request='" + request + '\'' +
            ", parameters='" + params.toString() + '\'' +
            ", readerClass=" + readerClass +
            ')';
    }
}
