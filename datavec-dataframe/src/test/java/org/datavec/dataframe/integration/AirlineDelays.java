package org.datavec.dataframe.integration;

import org.datavec.dataframe.api.Table;
import org.datavec.dataframe.api.ColumnType;
import org.datavec.dataframe.io.csv.CsvReader;
import org.datavec.dataframe.store.StorageManager;
import com.google.common.base.Stopwatch;

import java.util.concurrent.TimeUnit;

/**
 *
 */
public class AirlineDelays {

  private static Table flights2008;

  public static void main(String[] args) throws Exception {

    new AirlineDelays();

    Stopwatch stopwatch = Stopwatch.createStarted();
    flights2008.sortAscendingOn("Origin", "UniqueCarrier");
    System.out.println("Sorting " + stopwatch.elapsed(TimeUnit.SECONDS));
  }

  private AirlineDelays() throws Exception {
    Stopwatch stopwatch = Stopwatch.createStarted();
    System.out.println("loading");
    flights2008 = CsvReader.read(reduced_set, "bigdata/2015.csv");
    System.out.println(String.format("loaded %d records in %d seconds",
        flights2008.rowCount(),
        stopwatch.elapsed(TimeUnit.SECONDS)));
    out(flights2008.shape());
    out(flights2008.columnNames().toString());
    flights2008.first(10).print();
    StorageManager.saveTable("bigdata", flights2008);
    stopwatch.reset().start();
  }

  private static void out(Object obj) {
    System.out.println(String.valueOf(obj));
  }

  // The full set of all available columns in tbe dataset
  static ColumnType[] heading = {
      ColumnType.INTEGER, // year
      ColumnType.INTEGER, // month
      ColumnType.INTEGER, // day
      ColumnType.CATEGORY,  // dow
      ColumnType.LOCAL_TIME, // DepTime
      ColumnType.LOCAL_TIME, // CRSDepTime
      ColumnType.LOCAL_TIME, // ArrTime
      ColumnType.LOCAL_TIME, // CRSArrTime
      ColumnType.CATEGORY, // Carrier
      ColumnType.CATEGORY, // FlightNum
      ColumnType.CATEGORY, // TailNum
      ColumnType.INTEGER, // ActualElapsedTime
      ColumnType.INTEGER, // CRSElapsedTime
      ColumnType.INTEGER, // AirTime
      ColumnType.INTEGER, // ArrDelay
      ColumnType.INTEGER, // DepDelay
      ColumnType.CATEGORY, // Origin
      ColumnType.CATEGORY, // Dest
      ColumnType.INTEGER, // Distance
      ColumnType.INTEGER, // TaxiIn
      ColumnType.INTEGER, // TaxiOut
      ColumnType.BOOLEAN, // Cancelled
      ColumnType.CATEGORY, // CancellationCode
      ColumnType.BOOLEAN, // Diverted
      ColumnType.FLOAT, // CarrierDelay
      ColumnType.FLOAT, // WeatherDelay
      ColumnType.FLOAT, // NASDelay
      ColumnType.FLOAT, // SecurityDelay
      ColumnType.FLOAT  // LateAircraftDelay
  };

  // A filtered set of columns
  private static ColumnType[] reduced_set = {
      ColumnType.SKIP, // year
      ColumnType.INTEGER, // month
      ColumnType.INTEGER, // day
      ColumnType.CATEGORY,  // dow
      ColumnType.SKIP, // DepTime
      ColumnType.LOCAL_TIME, // CRSDepTime
      ColumnType.SKIP, // ArrTime
      ColumnType.SKIP, // CRSArrTime
      ColumnType.CATEGORY, // Carrier
      ColumnType.SKIP, // FlightNum
      ColumnType.SKIP, // TailNum
      ColumnType.SKIP, // ActualElapsedTime
      ColumnType.SKIP, // CRSElapsedTime
      ColumnType.SKIP, // AirTime
      ColumnType.SKIP, // ArrDelay
      ColumnType.INTEGER, // DepDelay
      ColumnType.CATEGORY, // Origin
      ColumnType.CATEGORY, // Dest
      ColumnType.INTEGER, // Distance
      ColumnType.SKIP, // TaxiIn
      ColumnType.SKIP, // TaxiOut
      ColumnType.BOOLEAN, // Cancelled
      ColumnType.SKIP, // CancellationCode
      ColumnType.SKIP, // Diverted
      ColumnType.SKIP, // CarrierDelay
      ColumnType.SKIP, // WeatherDelay
      ColumnType.SKIP, // NASDelay
      ColumnType.SKIP, // SecurityDelay
      ColumnType.SKIP  // LateAircraftDelay
  };
}
