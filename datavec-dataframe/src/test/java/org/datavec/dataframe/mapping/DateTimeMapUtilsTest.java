package org.datavec.dataframe.mapping;

import org.datavec.dataframe.api.LongColumn;
import org.datavec.dataframe.api.DateTimeColumn;
import org.datavec.dataframe.columns.packeddata.PackedLocalDateTime;
import org.datavec.dataframe.api.ShortColumn;
import org.junit.Test;

import java.time.LocalDateTime;
import java.time.temporal.ChronoUnit;

import static org.junit.Assert.assertEquals;

/**
 * Tests for DateTimeMapUtils
 */
public class DateTimeMapUtilsTest {

  DateTimeColumn startCol = DateTimeColumn.create("start");
  DateTimeColumn stopCol = DateTimeColumn.create("stop");
  LocalDateTime start = LocalDateTime.now();


  @Test
  public void testDifferenceInMilliseconds() throws Exception {
    long pStart = PackedLocalDateTime.pack(start);
    LocalDateTime stop = start.plus(100_000L, ChronoUnit.MILLIS);
    long pStop = PackedLocalDateTime.pack(stop);

    startCol.add(start);
    stopCol.add(stop);

    assertEquals(100_000L, startCol.difference(pStart, pStop, ChronoUnit.MILLIS));
    LongColumn result = startCol.differenceInMilliseconds(stopCol);
    assertEquals(100_000L, result.firstElement());
  }

  @Test
  public void testDifferenceInSeconds() throws Exception {
    LocalDateTime stop = start.plus(100_000L, ChronoUnit.SECONDS);

    startCol.add(start);
    stopCol.add(stop);

    LongColumn result = startCol.differenceInSeconds(stopCol);
    assertEquals(100_000L, result.firstElement());
  }

  @Test
  public void testDifferenceInMinutes() throws Exception {
    LocalDateTime stop = start.plus(100_000L, ChronoUnit.MINUTES);

    startCol.add(start);
    stopCol.add(stop);

    LongColumn result = startCol.differenceInMinutes(stopCol);
    assertEquals(100_000L, result.firstElement());
  }

  @Test
  public void testDifferenceInHours() throws Exception {
    LocalDateTime stop = start.plus(100_000L, ChronoUnit.HOURS);

    startCol.add(start);
    stopCol.add(stop);

    LongColumn result = startCol.differenceInHours(stopCol);
    assertEquals(100_000L, result.firstElement());

  }

  @Test
  public void testDifferenceInDays() throws Exception {
    LocalDateTime stop = start.plus(100_000L, ChronoUnit.DAYS);

    startCol.add(start);
    stopCol.add(stop);

    LongColumn result = startCol.differenceInDays(stopCol);
    assertEquals(100_000L, result.firstElement());
  }

  @Test
  public void testDifferenceInYears() throws Exception {

    LocalDateTime stop = start.plus(10_000L, ChronoUnit.YEARS);
    startCol.add(start);
    stopCol.add(stop);

    LongColumn result = startCol.differenceInYears(stopCol);
    assertEquals(10_000L, result.firstElement());
  }

  @Test
  public void testHour() throws Exception {
    startCol.add(LocalDateTime.of(1984, 12, 12, 7, 30));
    ShortColumn hour = startCol.hour();
    assertEquals(7, hour.firstElement());
  }
}