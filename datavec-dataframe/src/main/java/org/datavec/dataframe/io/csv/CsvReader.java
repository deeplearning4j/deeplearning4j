package org.datavec.dataframe.io.csv;

import org.datavec.dataframe.api.ColumnType;
import org.datavec.dataframe.api.Table;
import org.datavec.dataframe.columns.Column;
import org.datavec.dataframe.io.TypeUtils;
import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Strings;
import com.google.common.collect.Lists;
import com.opencsv.CSVReader;
import org.apache.commons.lang3.StringUtils;

import edu.umd.cs.findbugs.annotations.Nullable;
import net.jcip.annotations.Immutable;
import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.LocalTime;
import java.time.format.DateTimeParseException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.function.Predicate;

import static org.datavec.dataframe.api.ColumnType.*;

/**
 *
 */
@Immutable
public class CsvReader {

    /**
     * Constructs and returns a table from one or more CSV files, all containing the same column types
     * <p>
     * This constructor assumes the files have a one-line header, which is used to populate the column names,
     * and that they use a comma to separate between columns.
     *
     * @throws IOException If there is an issue reading any of the files
     */
    public static Table read(ColumnType types[], String... fileNames) throws IOException {
        if (fileNames.length == 1) {
            return read(types, true, ',', fileNames[0]);
        } else {

            Table table = read(types, true, ',', fileNames[0]);
            for (int i = 1; i < fileNames.length; i++) {
                String fileName = fileNames[i];
                table.append(read(types, true, ',', fileName));
            }
            return table;
        }
    }

    /**
     * Returns a Table constructed from a CSV File with the given file name
     * <p>
     * The @code{fileName} is used as the initial table name for the new table
     *
     * @param types           An array of the types of columns in the file, in the order they appear
     * @param header          Is the first row in the file a header?
     * @param columnSeparator the delimiter
     * @param fileName        The fully specified file name. It is used to provide a default name for the table
     * @return A Table containing the data in the csv file.
     * @throws IOException
     */
    public static Table read(ColumnType types[], boolean header, char columnSeparator, String fileName)
                    throws IOException {
        InputStream stream = new FileInputStream(fileName);
        return read(fileName, types, header, columnSeparator, stream);
    }

    /**
     * Retuns the given file after autodetecting the column types, or trying to
     *
     * @param fileName The name of the file to load
     * @param header      True if the file has a single header row. False if it has no header row.
     *                    Multi-line headers are not supported
     * @param delimiter   a char that divides the columns in the source file, often a comma or tab
     * @return A table containing the data from the file
     * @throws IOException
     */
    public static Table read(String fileName, boolean header, char delimiter) throws IOException {
        ColumnType[] columnTypes = detectColumnTypes(fileName, header, delimiter);
        return read(columnTypes, true, delimiter, fileName);
    }


    public static Table read(String tableName, ColumnType types[], boolean header, char columnSeparator,
                    InputStream stream) throws IOException {

        BufferedReader streamReader = new BufferedReader(new InputStreamReader(stream));

        Table table;
        try (CSVReader reader = new CSVReader(streamReader, columnSeparator, '"')) {

            String[] nextLine;
            String[] columnNames;
            List<String> headerRow;
            if (header) {
                nextLine = reader.readNext();
                headerRow = Lists.newArrayList(nextLine);
                columnNames = selectColumnNames(headerRow, types);
            } else {
                columnNames = makeColumnNames(types);
                headerRow = Lists.newArrayList(columnNames);
            }

            table = Table.create(nameMaker(tableName));
            for (int x = 0; x < types.length; x++) {
                if (types[x] != ColumnType.SKIP) {
                    String columnName = headerRow.get(x);
                    if (Strings.isNullOrEmpty(columnName)) {
                        columnName = "Column " + table.columnCount();
                    }
                    Column newColumn = TypeUtils.newColumn(columnName.trim(), types[x]);
                    table.addColumn(newColumn);
                }
            }
            int[] columnIndexes = new int[columnNames.length];
            for (int i = 0; i < columnIndexes.length; i++) {
                // get the index in the original table, which includes skipped fields
                columnIndexes[i] = headerRow.indexOf(columnNames[i]);
            }
            // Add the rows
            long rowNumber = header ? 1L : 0L;
            while ((nextLine = reader.readNext()) != null) {
                // for each column that we're including (not skipping)
                int cellIndex = 0;
                for (int columnIndex : columnIndexes) {
                    Column column = table.column(cellIndex);
                    try {
                        column.addCell(nextLine[columnIndex]);
                    } catch (Exception e) {
                        throw new AddCellToColumnException(e, columnIndex, rowNumber, columnNames, nextLine);
                    }
                    cellIndex++;
                }
                rowNumber++;
            }
        }
        return table;
    }

    /**
     * Returns a Table constructed from a CSV File with the given file name
     * <p>
     * The @code{fileName} is used as the initial table name for the new table
     *
     * @param types           An array of the types of columns in the file, in the order they appear
     * @param header          Is the first row in the file a header?
     * @param columnSeparator the delimiter
     * @param name            The fully specified file name. It is used to provide a default name for the table
     * @return A Table containing the data in the csv file.
     * @throws IOException
     */
    public static Table headerOnly(String name, ColumnType types[], boolean header, char columnSeparator,
                    InputStream stream) throws IOException {

        BufferedReader streamReader = new BufferedReader(new InputStreamReader(stream));

        Table table;
        try (CSVReader reader = new CSVReader(streamReader, columnSeparator, '"')) {

            String[] nextLine;
            String[] columnNames;
            List<String> headerRow;
            if (header) {
                nextLine = reader.readNext();
                headerRow = Lists.newArrayList(nextLine);
                columnNames = selectColumnNames(headerRow, types);
            } else {
                columnNames = makeColumnNames(types);
                headerRow = Lists.newArrayList(columnNames);
            }

            table = Table.create(nameMaker(name));
            for (int x = 0; x < types.length; x++) {
                if (types[x] != ColumnType.SKIP) {
                    Column newColumn = TypeUtils.newColumn(headerRow.get(x).trim(), types[x]);
                    table.addColumn(newColumn);
                }
            }
            int[] columnIndexes = new int[columnNames.length];
            for (int i = 0; i < columnIndexes.length; i++) {
                // get the index in the original table, which includes skipped fields
                columnIndexes[i] = headerRow.indexOf(columnNames[i]);
            }
            // Add the rows
            while ((nextLine = reader.readNext()) != null) {
                // for each column that we're including (not skipping)
                int cellIndex = 0;
                for (int columnIndex : columnIndexes) {
                    Column column = table.column(cellIndex);
                    column.addCell(nextLine[columnIndex]);
                    cellIndex++;
                }
            }
        }
        return table;
    }

    /**
     * Returns a Table constructed from a CSV File with the given file name
     * <p>
     * The @code{fileName} is used as the initial table name for the new table
     *
     * @param types           An array of the types of columns in the file, in the order they appear
     * @param header          Is the first row in the file a header?
     * @param columnSeparator the delimiter
     * @param fileName        The fully specified file name. It is used to provide a default name for the table
     * @return A Relation containing the data in the csv file.
     * @throws IOException
     */
    public static Table headerOnly(ColumnType types[], boolean header, char columnSeparator, String fileName)
                    throws IOException {

        InputStream stream = new FileInputStream(fileName);
        return headerOnly(fileName, types, header, columnSeparator, stream);
    }

    /**
     * Returns the structure of the table given by {@code csvFileName} as detected by analysis of a sample of the data
     *
     * @throws IOException
     */
    public static Table detectedColumnTypes(String csvFileName, boolean header, char delimiter) throws IOException {
        ColumnType[] types = detectColumnTypes(csvFileName, header, delimiter);
        Table t = headerOnly(types, header, delimiter, csvFileName);
        return t.structure();
    }

    /**
     * Returns a string representation of the file types in file {@code csvFilename},
     * as determined by the type-detection algorithm
     * <p>
     * This method is intended to help analysts quickly fix any erroneous types, by printing out the types in a format
     * such that they can be edited to correct any mistakes, and used in an array literal
     * <p>
     * For example:
     * <p>
     * LOCAL_DATE, // 0     date
     * SHORT_INT,  // 1     approval
     * CATEGORY,   // 2     who
     * <p>
     * Note that the types are array separated, and that the index position and the column name are printed such that
     * they
     * would be interpreted as comments if you paste the output into an array:
     * <p>
     * ColumnType[] types = {
     * LOCAL_DATE, // 0     date
     * SHORT_INT,  // 1     approval
     * CATEGORY,   // 2     who
     * }
     *
     * @throws IOException
     */
    public static String printColumnTypes(String csvFileName, boolean header, char delimiter) throws IOException {

        Table structure = detectedColumnTypes(csvFileName, header, delimiter);

        StringBuilder buf = new StringBuilder();
        buf.append("ColumnType[] columnTypes = {");
        buf.append('\n');

        Column typeCol = structure.column("Column Type");
        Column indxCol = structure.column("Index");
        Column nameCol = structure.column("Column Name");

        // add the column headers
        int typeColIndex = structure.columnIndex(typeCol);
        int indxColIndex = structure.columnIndex(indxCol);
        int nameColIndex = structure.columnIndex(nameCol);

        int typeColWidth = typeCol.columnWidth();
        int indxColWidth = indxCol.columnWidth();
        int nameColWidth = nameCol.columnWidth();

        for (int r = 0; r < structure.rowCount(); r++) {
            String cell = StringUtils.rightPad(structure.get(typeColIndex, r) + ",", typeColWidth);
            buf.append(cell);
            buf.append(" // ");

            cell = StringUtils.rightPad(structure.get(indxColIndex, r), indxColWidth);
            buf.append(cell);
            buf.append(' ');

            cell = StringUtils.rightPad(structure.get(nameColIndex, r), nameColWidth);
            buf.append(cell);
            buf.append(' ');

            buf.append('\n');
        }
        buf.append("}");
        buf.append('\n');
        return buf.toString();
    }

    /**
     * Reads column names from header, skipping any for which the type == SKIP
     */
    private static String[] selectColumnNames(List<String> names, ColumnType types[]) {
        List<String> header = new ArrayList<>();
        for (int i = 0; i < types.length; i++) {
            if (types[i] != ColumnType.SKIP) {
                header.add(names.get(i));
            }
        }
        String[] result = new String[header.size()];
        return header.toArray(result);
    }

    private static String nameMaker(String path) {
        Path p = Paths.get(path);
        return p.getFileName().toString();
    }

    /**
     * Provides placeholder column names for when the file read has no header
     */
    private static String[] makeColumnNames(ColumnType types[]) {
        String[] header = new String[types.length];
        for (int i = 0; i < types.length; i++) {
            header[i] = "C" + i;
        }
        return header;
    }

    @VisibleForTesting
    /**
     * Estimates and returns the type for each column in the delimited text file {@code file}
     *
     * The type is determined by checking a sample of the data in the file. Because only a sample of the data is checked,
     * the types may be incorrect. If that is the case a Parse Exception will be thrown.
     *
     * The method {@code printColumnTypes()} can be used to print a list of the detected columns that can be corrected and
     * used to explicitely specify the correct column types.
     */
    static ColumnType[] detectColumnTypes(String file, boolean header, char delimiter) throws IOException {

        int linesToSkip = header ? 1 : 0;

        // to hold the results
        List<ColumnType> columnTypes = new ArrayList<>();

        // to hold the data read from the file
        List<List<String>> columnData = new ArrayList<>();

        int rowCount = 0; // make sure we don't go over maxRows
        try (CSVReader reader = new CSVReader(new FileReader(file), delimiter, '"', linesToSkip)) {
            String[] nextLine;
            int nextRow = 0;
            while ((nextLine = reader.readNext()) != null) {

                // initialize the arrays to hold the strings. we don't know how many we need until we read the first row
                if (rowCount == 0) {
                    for (String aNextLine : nextLine) {
                        columnData.add(new ArrayList<>());
                    }
                }
                int columnNumber = 0;
                if (rowCount == nextRow) {
                    for (String field : nextLine) {
                        columnData.get(columnNumber).add(field);
                        columnNumber++;
                    }
                }
                if (rowCount == nextRow) {
                    nextRow = nextRow(nextRow);
                }
                rowCount++;
            }
        }

        // now detect
        for (List<String> valuesList : columnData) {
            ColumnType detectedType = detectType(valuesList);
            columnTypes.add(detectedType);
        }

        return columnTypes.toArray(new ColumnType[columnTypes.size()]);
    }

    private static int nextRow(int nextRow) {
        if (nextRow < 100) {
            return nextRow + 1;
        }
        if (nextRow < 1000) {
            return nextRow + 10;
        }
        if (nextRow < 10_000) {
            return nextRow + 100;
        }
        if (nextRow < 100_000) {
            return nextRow + 1000;
        }
        if (nextRow < 1_000_000) {
            return nextRow + 10_000;
        }
        if (nextRow < 10_000_000) {
            return nextRow + 100_000;
        }
        if (nextRow < 100_000_000) {
            return nextRow + 1_000_000;
        }
        return nextRow + 10_000_000;
    }

    private static ColumnType detectType(List<String> valuesList) {

        // Types to choose from. When more than one would work, we pick the first of the options
        ColumnType[] typeArray = // we leave out category, as that is the default type
                        {LOCAL_DATE_TIME, LOCAL_TIME, LOCAL_DATE, BOOLEAN, SHORT_INT, INTEGER, LONG_INT, FLOAT};

        CopyOnWriteArrayList<ColumnType> typeCandidates = new CopyOnWriteArrayList<>(typeArray);


        for (String s : valuesList) {
            if (Strings.isNullOrEmpty(s) || TypeUtils.MISSING_INDICATORS.contains(s)) {
                continue;
            }
            if (typeCandidates.contains(LOCAL_DATE_TIME)) {
                if (!isLocalDateTime.test(s)) {
                    typeCandidates.remove(LOCAL_DATE_TIME);
                }
            }
            if (typeCandidates.contains(LOCAL_TIME)) {
                if (!isLocalTime.test(s)) {
                    typeCandidates.remove(LOCAL_TIME);
                }
            }
            if (typeCandidates.contains(LOCAL_DATE)) {
                if (!isLocalDate.test(s)) {
                    typeCandidates.remove(LOCAL_DATE);
                }
            }
            if (typeCandidates.contains(BOOLEAN)) {
                if (!isBoolean.test(s)) {
                    typeCandidates.remove(BOOLEAN);
                }
            }
            if (typeCandidates.contains(SHORT_INT)) {
                if (!isShort.test(s)) {
                    typeCandidates.remove(SHORT_INT);
                }
            }
            if (typeCandidates.contains(INTEGER)) {
                if (!isInteger.test(s)) {
                    typeCandidates.remove(INTEGER);
                }
            }
            if (typeCandidates.contains(LONG_INT)) {
                if (!isLong.test(s)) {
                    typeCandidates.remove(LONG_INT);
                }
            }
            if (typeCandidates.contains(FLOAT)) {
                if (!isFloat.test(s)) {
                    typeCandidates.remove(FLOAT);
                }
            }
        }
        return selectType(typeCandidates);
    }

    /**
     * Returns the selected candidate for a column of data, by picking the first value in the given list
     *
     * @param typeCandidates a possibly empty list of candidates. This list should be sorted in order of preference
     */
    private static ColumnType selectType(List<ColumnType> typeCandidates) {
        if (typeCandidates.isEmpty()) {
            return CATEGORY;
        } else {
            return typeCandidates.get(0);
        }
    }

    private static java.util.function.Predicate<String> isBoolean =
                    s -> TypeUtils.TRUE_STRINGS_FOR_DETECTION.contains(s)
                                    || TypeUtils.FALSE_STRINGS_FOR_DETECTION.contains(s);

    private static Predicate<String> isLong = new Predicate<String>() {

        @Override
        public boolean test(@Nullable String s) {
            try {
                Long.parseLong(s);
                return true;
            } catch (NumberFormatException e) {
                // it's all part of the plan
                return false;
            }
        }
    };

    private static Predicate<String> isInteger = s -> {
        try {
            Integer.parseInt(s);
            return true;
        } catch (NumberFormatException e) {
            // it's all part of the plan
            return false;
        }
    };

    private static Predicate<String> isFloat = s -> {
        try {
            Float.parseFloat(s);
            return true;
        } catch (NumberFormatException e) {
            // it's all part of the plan
            return false;
        }
    };

    private static Predicate<String> isShort = s -> {
        try {
            Short.parseShort(s);
            return true;
        } catch (NumberFormatException e) {
            // it's all part of the plan
            return false;
        }
    };

    private static Predicate<String> isLocalDate = s -> {
        try {
            LocalDate.parse(s, TypeUtils.DATE_FORMATTER);
            return true;
        } catch (DateTimeParseException e) {
            // it's all part of the plan
            return false;
        }
    };

    private static Predicate<String> isLocalTime = s -> {
        try {
            LocalTime.parse(s, TypeUtils.TIME_DETECTION_FORMATTER);
            return true;
        } catch (DateTimeParseException e) {
            // it's all part of the plan
            return false;
        }
    };

    private static Predicate<String> isLocalDateTime = s -> {
        try {
            LocalDateTime.parse(s, TypeUtils.DATE_TIME_FORMATTER);
            return true;
        } catch (DateTimeParseException e) {
            // it's all part of the plan
            return false;
        }
    };

    /**
     * Private constructor to prevent instantiation
     */
    private CsvReader() {}
}
