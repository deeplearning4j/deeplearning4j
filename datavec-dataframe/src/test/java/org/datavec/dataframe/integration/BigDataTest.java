package org.datavec.dataframe.integration;

import org.datavec.dataframe.api.ColumnType;
import org.datavec.dataframe.api.Table;
import org.datavec.dataframe.store.StorageManager;
import com.google.common.base.Stopwatch;

import java.util.concurrent.TimeUnit;

import static java.lang.System.exit;

/**
 *
 */
public class BigDataTest {

  static final ColumnType types[] = {
      ColumnType.LOCAL_DATE_TIME,
      ColumnType.LOCAL_DATE_TIME,
      ColumnType.CATEGORY,
      ColumnType.CATEGORY,
      ColumnType.CATEGORY,
      ColumnType.CATEGORY,
      ColumnType.CATEGORY
  };

  static int[] wanted = {1, 2, 3, 5, 6, 16, 19};
  static String file = "bigdata/311_Service_Requests_from_2010_to_Present.csv";

  public static void main(String[] args) throws Exception {


    Stopwatch stopwatch = Stopwatch.createStarted();
/*
    Relation table = CsvReader.read(file, types, wanted, ',', true);
    out(String.format("Loaded %d rows in %d seconds", table.rowCount(), stopwatch.elapsed(TimeUnit.SECONDS)));

    table.first(3).print();
    stopwatch.reset();

    stopwatch.start();
    CsvWriter.saveTable("testfolder/BigData.csv", table);
    out(String.format("Relation written as csv file in %d seconds", stopwatch.elapsed(TimeUnit.SECONDS)));
    stopwatch.reset();

    stopwatch.start();
    StorageManager.saveTable("bigdata", table);
    out(String.format("Relation written to column store in %d seconds", stopwatch.elapsed(TimeUnit.SECONDS)));
    stopwatch.reset();
*/

    //stopwatch.start();
    Table table = StorageManager.readTable("bigdata/" + "3f07b9bf-053f-4f9b-9dff-9d354835b276");
    out(String.format("Table read from column store in %d seconds", stopwatch.elapsed(TimeUnit.SECONDS)));
    out(table.first(3).print());

    out(table.columnNames());

    out(table.floatColumn("DEP_DELAY").summary().print());
    exit(1);

  }

  //@Test
  public void readCsvTest() throws Exception {
    Stopwatch stopwatch = Stopwatch.createStarted();
    Table table = Table.createFromCsv(types, "data/BigData.csv", true);
    table.first(3).print();
    out(table.rowCount());
    out("Table read from csv file");
    out(stopwatch.elapsed(TimeUnit.SECONDS));
  }

  private static void out(Object str) {
    System.out.println(String.valueOf(str));
  }
}
