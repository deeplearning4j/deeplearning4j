package org.datavec.api.records.reader.impl.csv;

import au.com.bytecode.opencsv.CSVParser;
import lombok.AllArgsConstructor;
import lombok.NoArgsConstructor;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;

/**
 * Allows CSVParser to be serialized for spark.
 * Underlying implementation uses {@link CSVParser}
 */
public class SerializableCSVParser implements Serializable {

    private final char delimiter;
    private final char quote;
    private final char escape;
    private final boolean strictQuotes;
    private final boolean ignoreLeadingWhiteSpace;
    private transient CSVParser parser;

    /**
     * Constructs CSVParser using a comma for the separator.
     */
    public SerializableCSVParser() {
        this(CSVParser.DEFAULT_SEPARATOR);
    }

    /**
     * Constructs CSVParser with supplied separator.
     *
     * @param delimiter the delimiter to use for separating entries.
     */
    public SerializableCSVParser(char delimiter) {
        this(delimiter, CSVParser.DEFAULT_QUOTE_CHARACTER);
    }

    /**
     * Constructs CSVParser with supplied separator and quote char.
     *
     * @param delimiter the delimiter to use for separating entries
     * @param quote     the character to use for quoted elements
     */
    public SerializableCSVParser(char delimiter, char quote) {
        this(delimiter, quote, CSVParser.DEFAULT_ESCAPE_CHARACTER);
    }

    /**
     * Constructs CSVReader with supplied separator and quote char.
     *
     * @param separator the delimiter to use for separating entries
     * @param quotechar the character to use for quoted elements
     * @param escape    the character to use for escaping a separator or quote
     */
    public SerializableCSVParser(char separator, char quotechar, char escape) {
        this(separator, quotechar, escape, CSVParser.DEFAULT_STRICT_QUOTES, CSVParser.DEFAULT_IGNORE_LEADING_WHITESPACE);
    }

    /**
     * Constructs CSVReader with supplied separator and quote char.
     * Allows setting the "strict quotes" and "ignore leading whitespace" flags
     *
     * @param delimiter               the delimiter to use for separating entries
     * @param quote                   the character to use for quoted elements
     * @param escape                  the character to use for escaping a separator or quote
     * @param strictQuotes            if true, characters outside the quotes are ignored
     * @param ignoreLeadingWhiteSpace if true, white space in front of a quote in a field is ignored
     */
    public SerializableCSVParser(char delimiter, char quote, char escape, boolean strictQuotes, boolean ignoreLeadingWhiteSpace) {
        this.delimiter = delimiter;
        this.quote = quote;
        this.escape = escape;
        this.strictQuotes = strictQuotes;
        this.ignoreLeadingWhiteSpace = ignoreLeadingWhiteSpace;

        parser = new CSVParser(delimiter, quote, escape, strictQuotes, ignoreLeadingWhiteSpace);
    }

    /**
     * @return true if something was left over from last call(s)
     */
    public boolean isPending() {
        return parser.isPending();
    }

    public String[] parseLineMulti(String nextLine) throws IOException {
        return parser.parseLineMulti(nextLine);
    }

    public String[] parseLine(String nextLine) throws IOException {
        return parser.parseLine(nextLine);
    }

    //custom Java deserialization
    private void readObject(ObjectInputStream is) throws IOException, ClassNotFoundException {
        is.defaultReadObject();
        parser = new CSVParser(delimiter, quote, escape, strictQuotes, ignoreLeadingWhiteSpace);
    }

}
