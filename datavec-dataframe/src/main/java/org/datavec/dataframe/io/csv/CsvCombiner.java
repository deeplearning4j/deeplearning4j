package org.datavec.dataframe.io.csv;


import com.opencsv.CSVReader;
import com.opencsv.CSVWriter;

import net.jcip.annotations.Immutable;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;

/**
 * Utility that takes all CSV files in a folder and combines them into a single file.  The files should all have the
 * same structure (number and type of columns) and be consistent in having or not having header lines at the beginning
 */
@Immutable
public class CsvCombiner {

    private CsvCombiner() {}

    public static void combineAll(String foldername, String newFileName, char columnSeparator, boolean headers)
                    throws IOException {

        FileWriter fileWriter = new FileWriter(newFileName);
        try (CSVWriter writer = new CSVWriter(fileWriter, ',')) {
            final boolean[] skipHeader = {false};
            Files.walk(Paths.get(foldername)).forEach(filePath -> {

                if (Files.isRegularFile(filePath) && filePath.toString().endsWith(".csv")) {
                    CsvCombiner.append(filePath.toString(), writer, columnSeparator, headers && skipHeader[0]);
                    skipHeader[0] = true;
                }
            });
            writer.flush();
        }
    }

    public static void append(String fileName, final CSVWriter writer, char columnSeparator, boolean skipHeader) {

        CSVReader reader = null;
        try {
            reader = new CSVReader(new FileReader(fileName), columnSeparator);
            if (skipHeader) { // skip the header
                reader.readNext();
            }
            String[] nextLine;
            while ((nextLine = reader.readNext()) != null) {
                writer.writeNext(nextLine);
            }
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            if (reader != null) {
                try {
                    reader.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
    }
}
