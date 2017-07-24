package org.datavec.api.records.reader.impl.csv;

import au.com.bytecode.opencsv.CSVParser;
import lombok.AllArgsConstructor;
import lombok.NoArgsConstructor;

import java.io.Serializable;

/**
 * Allows CSVParser to be serialized for spark.
 */
public class SerializableCSVParser extends CSVParser implements Serializable {

    public SerializableCSVParser() {
        super();
    }

    public SerializableCSVParser(char delimiter) {
        super(delimiter);
    }

    public SerializableCSVParser(char delimiter, char quote) {
        super(delimiter, quote);
    }

    public SerializableCSVParser(char delimiter, char quote, char escape, boolean strictQuotes, boolean ignoreLeadingWhiteSpace) {
        super(delimiter, quote, escape, strictQuotes, ignoreLeadingWhiteSpace);
    }
}
