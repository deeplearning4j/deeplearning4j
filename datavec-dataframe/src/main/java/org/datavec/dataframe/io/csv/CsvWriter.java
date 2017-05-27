package org.datavec.dataframe.io.csv;

import org.datavec.dataframe.api.Table;
import org.datavec.dataframe.columns.Column;
import com.opencsv.CSVWriter;

import net.jcip.annotations.Immutable;
import java.io.FileWriter;
import java.io.IOException;

/**
 * Static utility class that writes tables and individual columns to CSV files
 * <p>
 * TODO(lwhite): Do something with the missing indicator param in write() method
 * TODO(lwhite): Add a missing indicator to the column write method, plus a method defining a default missing indicator
 */
@Immutable
final public class CsvWriter {

    /**
     * Private constructor to prevent instantiation
     */
    private CsvWriter() {}

    /**
     * Writes the given table to a file with the given filename
     *
     * @throws IOException
     */
    public static void write(String fileName, Table table) throws IOException {
        write(fileName, table, null);
    }

    /**
     * Writes the given table to a file with the given filename, using the given string to represent missing data
     *
     * @throws IOException
     */
    public static void write(String fileName, Table table, String missing) throws IOException {
        try (CSVWriter writer = new CSVWriter(new FileWriter(fileName))) {
            String[] header = new String[table.columnCount()];
            for (int c = 0; c < table.columnCount(); c++) {
                header[c] = table.column(c).name();
            }
            writer.writeNext(header);
            for (int r = 0; r < table.rowCount(); r++) {
                String[] entries = new String[table.columnCount()];
                for (int c = 0; c < table.columnCount(); c++) {
                    table.get(c, r);
                    entries[c] = table.get(c, r);
                }
                writer.writeNext(entries);
            }
        }
    }

    /**
     * Writes the given column to a file with the given fileName as a single column CSV file
     *
     * @throws IOException
     */
    public static void write(String fileName, Column column) throws IOException {
        try (CSVWriter writer = new CSVWriter(new FileWriter(fileName))) {
            String[] header = {column.name()};
            writer.writeNext(header);

            for (int r = 0; r < column.size(); r++) {
                String[] entries = {column.getString(r)};
                writer.writeNext(entries);
            }
        }
    }
}
