package org.datavec.dataframe.columns;

import org.datavec.dataframe.columns.packeddata.PackedLocalDateTime;
import org.datavec.dataframe.columns.packeddata.PackedLocalTime;
import org.junit.Assert;
import org.junit.Test;

import java.time.LocalTime;
import java.time.temporal.ChronoField;

import static org.junit.Assert.*;

/**
 * Tests for PackedLocalTime
 */
public class PackedLocalTimeTest {

  @Test
  public void testGetHour() {
    LocalTime now = LocalTime.now();
    Assert.assertEquals(now.getHour(), PackedLocalTime.getHour(PackedLocalTime.pack(now)));
  }

  @Test
  public void testGetMinute() {
    LocalTime now = LocalTime.now();
    assertEquals(now.getMinute(), PackedLocalTime.getMinute(PackedLocalTime.pack(now)));
  }

  @Test
  public void testGetSecond() {
    LocalTime now = LocalTime.now();
    assertEquals(now.getSecond(), PackedLocalTime.getSecond(PackedLocalTime.pack(now)));
  }

  @Test
  public void testGetSecondOfDay() {
    LocalTime now = LocalTime.now();
    assertEquals(now.get(ChronoField.SECOND_OF_DAY), PackedLocalTime.getSecondOfDay(PackedLocalTime.pack(now)));
  }

  @Test
  public void testGetMinuteOfDay() {
    LocalTime now = LocalTime.now();
    assertEquals(now.get(ChronoField.MINUTE_OF_DAY), PackedLocalTime.getMinuteOfDay(PackedLocalTime.pack(now)));
  }

  @Test
  public void testGetMillisecondOfDay() {
    LocalTime now = LocalTime.now();
    assertEquals(now.get(ChronoField.MILLI_OF_DAY), PackedLocalTime.getMillisecondOfDay(PackedLocalTime.pack(now)));
  }

  @Test
  public void testPack() {
    LocalTime time = LocalTime.now();
    int packed = PackedLocalTime.pack(time);

    LocalTime t1 = PackedLocalTime.asLocalTime(PackedLocalDateTime.time(packed));
    assertNotNull(t1);
    assertEquals(time.getHour(), t1.getHour());
    assertEquals(time.getMinute(), t1.getMinute());
    assertEquals(time.getSecond(), t1.getSecond());
    assertEquals(time.get(ChronoField.MILLI_OF_SECOND), t1.get(ChronoField.MILLI_OF_SECOND));
  }
}