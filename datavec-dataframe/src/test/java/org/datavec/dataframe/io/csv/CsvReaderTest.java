package org.datavec.dataframe.io.csv;

import org.datavec.dataframe.api.ColumnType;
import org.datavec.dataframe.api.Table;
import org.datavec.dataframe.columns.Column;
import org.datavec.dataframe.api.QueryHelper;
import org.junit.Ignore;
import org.junit.Test;

import java.io.InputStream;
import java.net.URL;
import java.util.Arrays;

import static org.junit.Assert.*;

/**
 * Tests for CSV Reading
 */
public class CsvReaderTest {

  private final ColumnType[] bus_types = {ColumnType.SHORT_INT, ColumnType.CATEGORY, ColumnType.CATEGORY, ColumnType.FLOAT, ColumnType.FLOAT};

  @Test
  public void testWithBusData() throws Exception {
    // Read the CSV file
    Table table = CsvReader.read(bus_types, true, ',', "data/bus_stop_test.csv");

    // Look at the column names
    assertEquals("[stop_id, stop_name, stop_desc, stop_lat, stop_lon]", table.columnNames().toString());

    table = table.sortDescendingOn("stop_id");
    table.removeColumns("stop_desc");

    Column c = table.floatColumn("stop_lat");
    Table v = table.selectWhere(QueryHelper.column("stop_lon").isGreaterThan(-0.1f));
  }

  @Test
  public void testWithBushData() throws Exception {

    // Read the CSV file
    ColumnType[] types = {ColumnType.LOCAL_DATE, ColumnType.SHORT_INT, ColumnType.CATEGORY};
    Table table = CsvReader.read(types, "data/BushApproval.csv");

    assertEquals(323, table.rowCount());

    // Look at the column names
    assertEquals("[date, approval, who]", table.columnNames().toString());
  }

  @Test
  public void testDataTypeDetection() throws Exception {
    ColumnType[] columnTypes = CsvReader.detectColumnTypes("data/bus_stop_test.csv", true, ',');
    assertTrue(Arrays.equals(bus_types, columnTypes));
  }

  @Test
  public void testPrintStructure() throws Exception {
    String output =
        "ColumnType[] columnTypes = {\n" +
        "LOCAL_DATE, // 0     date        \n" +
        "SHORT_INT,  // 1     approval    \n" +
        "CATEGORY,   // 2     who         \n" +
        "}\n";
    assertEquals(output, CsvReader.printColumnTypes("data/BushApproval.csv", true, ','));
  }

  @Test
  public void testDataTypeDetection2() throws Exception {
    ColumnType[] columnTypes = CsvReader.detectColumnTypes("data/BushApproval.csv", true, ',');
    assertEquals(ColumnType.LOCAL_DATE, columnTypes[0]);
    assertEquals(ColumnType.SHORT_INT, columnTypes[1]);
    assertEquals(ColumnType.CATEGORY, columnTypes[2]);
  }

  @Ignore
  @Test
  public void testLoadFromUrl() throws Exception {
    ColumnType[] types = {ColumnType.LOCAL_DATE, ColumnType.SHORT_INT, ColumnType.CATEGORY};
    String location = "https://raw.githubusercontent.com/lwhite1/tablesaw/master/data/BushApproval.csv";
    Table table;
    try (InputStream input = new URL(location).openStream()) {
      table = Table.createFromStream(types, true, ',', input, "Bush approval ratings");
    }
    assertNotNull(table);
  }

  @Test
  public void testBoundary1() throws Exception {
    Table table1 = Table.createFromCsv("data/boundaryTest1.csv");
    table1.structure();  // just make sure the import completed
  }

  @Test
  public void testBoundary2() throws Exception {
    Table table1 = Table.createFromCsv("data/boundaryTest2.csv");
    table1.structure(); // just make sure the import completed
  }
}
