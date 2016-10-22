package org.datavec.dataframe;

import org.datavec.dataframe.api.ColumnType;
import org.datavec.dataframe.api.Table;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * This class setup tablesaw Table from test data sources.
 * It purpose is to make easy for tests or example code get data to work with.
 */
public enum TestData {
    SIMPLE_DATA_WITH_CANONICAL_DATE_FORMAT(new String[]{"Name", "IQ", "City", "DOB"},
            new ColumnType[]{CATEGORY, INTEGER, CATEGORY, LOCAL_DATE},
            "data/simple-data-with-canonical-date-format.csv"),

    SIMPLE_UNSORTED_DATA(new String[]{"Name", "IQ", "City", "DOB"},
            new ColumnType[]{CATEGORY, INTEGER, CATEGORY, LOCAL_DATE}, "data/unsorted-simple-data.csv"),

    SIMPLE_SORTED_DATA_BY_INTEGER_ASCENDING(new String[]{"Name", "IQ", "City", "DOB"},
            new ColumnType[]{CATEGORY, INTEGER, CATEGORY, LOCAL_DATE}, "data/simple-data-sort_by_int_ascending.csv"),

    SIMPLE_SORTED_DATA_BY_INTEGER_DESCENDING(new String[]{"Name", "IQ", "City", "DOB"},
            new ColumnType[]{CATEGORY, INTEGER, CATEGORY, LOCAL_DATE}, "data/simple-data-sort_by_int_descending.csv"),

    SIMPLE_SORTED_DATA_BY_INT_ASCENDING_AND_THEN_DATE_DESCENDING(new String[]{"Name", "IQ", "City", "DOB"},
            new ColumnType[]{CATEGORY, INTEGER, CATEGORY, LOCAL_DATE}, "data/simple-data-sort_by_int_ascending.csv"),

    SIMPLE_SORTED_DATA_BY_INTEGER_AND_DATE_ASCENDING(new String[]{"Name", "IQ", "City", "DOB"},
            new ColumnType[]{CATEGORY, INTEGER, CATEGORY, LOCAL_DATE},
            "data/simple-data-sort_by_int_and_date_ascending.csv"),

    SIMPLE_SORTED_DATA_BY_INTEGER_AND_DATE_DESCENDING(
            new String[]{"Name", "IQ", "City", "DOB"}, new ColumnType[]{CATEGORY, INTEGER, CATEGORY, LOCAL_DATE},
            "data/simple-data-sort_by_int_and_date_descending.csv"),

    BUSH_APPROVAL(new String[]{"date", "approval", "who"}, new ColumnType[]{LOCAL_DATE, INTEGER, CATEGORY},
            "data/BushApproval.csv"),

    TORNADOES(new String[]{"Number", "Year", "Month", "Day", "Date", "Time", "Zone", "State", "State FIPS", "State No",
            "Scale", "Injuries", "Fatalities", "Loss", "Crop Loss", "Start Lat", "Start Lon", "End Lat", "End Lon",
            "Length", "Width", "NS", "SN", "SG", "FIPS 1", "FIPS 2", "FIPS 3", "FIPS 4"},
            new ColumnType[]{INTEGER, INTEGER, INTEGER, INTEGER, LOCAL_DATE, LOCAL_TIME, CATEGORY, CATEGORY, CATEGORY,
                    INTEGER, INTEGER, INTEGER, INTEGER, FLOAT, FLOAT, FLOAT, FLOAT, FLOAT, FLOAT, FLOAT, FLOAT, FLOAT,
                    FLOAT, FLOAT, CATEGORY, CATEGORY, CATEGORY, CATEGORY}, "data/1950-2014_torn.csv");




    private Table table;
    private ColumnType[] columnTypes;
    private Path source;
    private String[] columnNames;

    /**
     * Creates a Table from the specified daa.
     *
     * @param columnNames the first row of data which should be tge column labels
     * @param columnTypes the data type for each column
     * @param csvSource   the raw source for this data.
     */
    TestData(String[] columnNames, ColumnType[] columnTypes, String csvSource) {
        this.columnNames = columnNames;
      try {
        this.table = Table.createFromCsv(columnTypes, csvSource);
      } catch (IOException e) {
        e.printStackTrace();
        throw new RuntimeException("Unable to read from CSV file");
      }
      this.columnTypes = columnTypes;
        this.source = Paths.get(csvSource);
    }

    /**
     * @return The TableSaw instance for a specific data set
     */
    public Table getTable() {
        return table;
    }

    /**
     *
     * @return the column types for the data set.
     */
    public ColumnType[] getColumnTypes() {
        return columnTypes;
    }

    /**
     * @return The path to the raw data for this data set
     */
    public Path getSource() {
        return source;
    }

    /*
     * @return the column names (i.e. header, labels) for each column.
     */
    public String[] getColumnNames() {
        return columnNames;
    }

}
