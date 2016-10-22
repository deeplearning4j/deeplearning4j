package org.datavec.dataframe.store;

import org.datavec.dataframe.api.CategoryColumn;
import org.datavec.dataframe.api.FloatColumn;
import org.datavec.dataframe.api.DateColumn;
import org.datavec.dataframe.api.LongColumn;
import org.datavec.dataframe.api.Table;
import org.datavec.dataframe.table.Relation;
import org.datavec.dataframe.api.ColumnType;
import org.datavec.dataframe.io.csv.CsvReader;
import com.google.common.base.Stopwatch;
import org.junit.Before;
import org.junit.Test;

import java.io.IOException;
import java.time.LocalDate;
import java.util.concurrent.TimeUnit;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;

/**
 * Tests for StorageManager
 */
public class StorageManagerTest {

  private static final int COUNT = 5;

  private Relation table = Table.create("t");
  private FloatColumn floatColumn = FloatColumn.create("float");
  private CategoryColumn categoryColumn = CategoryColumn.create("cat");
  private DateColumn localDateColumn = DateColumn.create("date");
  private LongColumn longColumn = LongColumn.create("long");

  @Before
  public void setUp() throws Exception {

    for (int i = 0; i < COUNT; i++) {
      floatColumn.add((float) i);
      localDateColumn.add(LocalDate.now());
      categoryColumn.add("Category " + i);
      longColumn.add(i);
    }
    table.addColumn(floatColumn);
    table.addColumn(localDateColumn);
    table.addColumn(categoryColumn);
    table.addColumn(longColumn);
  }

  @Test
  public void testCatStorage() throws Exception {
    StorageManager.writeColumn("/tmp/cat_dogs", categoryColumn);
    CategoryColumn readCat = StorageManager.readCategoryColumn("/tmp/cat_dogs", categoryColumn.columnMetadata());
    for (int i = 0; i < categoryColumn.size(); i++) {
      assertEquals(categoryColumn.get(i), readCat.get(i));
    }
  }

  @Test
  public void testWriteTable() throws IOException {
    StorageManager.saveTable("/tmp/zeta", table);
    Table t = StorageManager.readTable("/tmp/zeta/t.saw");
    assertEquals(table.columnCount(), t.columnCount());
    assertEquals(table.rowCount(), t.rowCount());
    for (int i = 0; i < table.rowCount(); i++) {
      assertEquals(categoryColumn.get(i), t.categoryColumn("cat").get(i));
    }
    t.sortOn("cat"); // exercise the column a bit
  }

  @Test
  public void testWriteTableTwice() throws IOException {

    StorageManager.saveTable("/tmp/mytables2", table);
    Table t = StorageManager.readTable("/tmp/mytables2/t.saw");
    t.floatColumn("float").setName("a float column");

    StorageManager.saveTable("/tmp/mytables2", table);
    t = StorageManager.readTable("/tmp/mytables2/t.saw");

    assertEquals(table.name(), t.name());
    assertEquals(table.rowCount(), t.rowCount());
    assertEquals(table.columnCount(), t.columnCount());
  }

  @Test
  public void testSeparator() {
    assertNotNull(StorageManager.separator());
  }

  public static void main(String[] args) throws Exception {

    Stopwatch stopwatch = Stopwatch.createStarted();
    System.out.println("loading");
    Table tornados = CsvReader.read(COLUMN_TYPES, "data/1950-2014_torn.csv");
    tornados.setName("tornados");
    System.out.println(String.format("loaded %d records in %d seconds",
        tornados.rowCount(),
        stopwatch.elapsed(TimeUnit.SECONDS)));
    System.out.println(tornados.shape());
    System.out.println(tornados.columnNames().toString());
    System.out.println(tornados.first(10).print());
    stopwatch.reset().start();
    StorageManager.saveTable("/tmp/tablesaw/testdata", tornados);
    stopwatch.reset().start();
    tornados = StorageManager.readTable("/tmp/tablesaw/testdata/tornados.saw");
    System.out.println(tornados.first(5).print());
  }

  // column types for the tornado table
  private static final ColumnType[] COLUMN_TYPES = {
      FLOAT,   // number by year
      FLOAT,   // year
      FLOAT,   // month
      FLOAT,   // day
      LOCAL_DATE,  // date
      LOCAL_TIME,  // time
      CATEGORY, // tz
      CATEGORY, // st
      CATEGORY, // state fips
      FLOAT,    // state torn number
      FLOAT,    // scale
      FLOAT,    // injuries
      FLOAT,    // fatalities
      CATEGORY, // loss
      FLOAT,   // crop loss
      FLOAT,   // St. Lat
      FLOAT,   // St. Lon
      FLOAT,   // End Lat
      FLOAT,   // End Lon
      FLOAT,   // length
      FLOAT,   // width
      FLOAT,   // NS
      FLOAT,   // SN
      FLOAT,   // SG
      CATEGORY,  // Count FIPS 1-4
      CATEGORY,
      CATEGORY,
      CATEGORY};
}