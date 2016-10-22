package org.datavec.dataframe.columns;

import org.datavec.dataframe.columns.packeddata.PackedLocalDate;
import org.datavec.dataframe.columns.packeddata.PackedLocalTime;
import org.junit.Test;

import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.LocalTime;
import java.time.temporal.ChronoField;

import static org.junit.Assert.*;

/**
 *
 */
public class PackedLocalDateTimeTest {

  @Test
  public void testGetDayOfMonth() {
    LocalDateTime today = LocalDateTime.now();
    assertEquals(today.getDayOfMonth(),
        getDayOfMonth(pack(today)));
  }

  @Test
  public void testGetYear() {
    LocalDateTime today = LocalDateTime.now();
    assertEquals(today.getYear(), getYear(pack(today)));
  }

  @Test
  public void testAsLocalDateTime() {
    LocalDateTime dateTime = LocalDateTime.now();
    long packed = pack(dateTime.toLocalDate(), dateTime.toLocalTime());
    LocalDateTime upacked = asLocalDateTime(packed);
    assertEquals(dateTime.getDayOfYear(), upacked.getDayOfYear());
    assertEquals(dateTime.getHour(), upacked.getHour());
    assertEquals(dateTime.getMinute(), upacked.getMinute());
    assertEquals(dateTime.getSecond(), upacked.getSecond());
  }

  @Test
  public void testGetMonthValue() {
    long dateTime = pack(LocalDate.of(2015, 12, 25), LocalTime.now());
    assertEquals(12, getMonthValue(dateTime));
  }

  @Test
  public void testPack() {
    LocalDate date = LocalDate.now();
    LocalTime time = LocalTime.now();

    long packed = pack(date, time);

    LocalDate d1 = PackedLocalDate.asLocalDate(date(packed));
    LocalTime t1 = PackedLocalTime.asLocalTime(time(packed));
    assertNotNull(d1);
    assertNotNull(t1);
    assertEquals(date.toString(), d1.toString());
  }

  @Test
  public void testGetHour() {
    LocalDateTime now = LocalDateTime.now();
    assertEquals(now.getHour(), getHour(pack(now)));
  }

  @Test
  public void testGetMinute() {
    LocalDateTime now = LocalDateTime.now();
    assertEquals(now.getMinute(), getMinute(pack(now)));
  }

  @Test
  public void testGetSecond() {
    LocalDateTime now = LocalDateTime.now();
    assertEquals(now.getSecond(), getSecond(pack(now)));
  }

  @Test
  public void testGetSecondOfDay() {
    LocalDateTime now = LocalDateTime.now();
    assertEquals(now.get(ChronoField.SECOND_OF_DAY), getSecondOfDay(pack(now)));
  }

  @Test
  public void testGetMinuteOfDay() {
    LocalDateTime now = LocalDateTime.now();
    assertEquals(now.get(ChronoField.MINUTE_OF_DAY), getMinuteOfDay(pack(now)));
  }

  @Test
  public void testGetMillisecondOfDay() {
    LocalDateTime now = LocalDateTime.now();
    assertEquals(now.get(ChronoField.MILLI_OF_DAY), getMillisecondOfDay(pack(now)));
  }

  @Test
  public void testGetDayOfWeek() {
    LocalDateTime now = LocalDateTime.now();
    assertEquals(now.get(ChronoField.DAY_OF_WEEK), getDayOfWeek(pack(now)).getValue());
  }
}