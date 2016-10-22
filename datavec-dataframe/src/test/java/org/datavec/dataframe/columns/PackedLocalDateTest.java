package org.datavec.dataframe.columns;

import org.datavec.dataframe.columns.packeddata.PackedLocalDate;
import org.junit.Test;

import java.time.LocalDate;

import static org.junit.Assert.*;

/**
 *
 */
public class PackedLocalDateTest {

  @Test
  public void testGetDayOfMonth() {
    LocalDate today = LocalDate.now();
    assertEquals(today.getDayOfMonth(),
        PackedLocalDate.getDayOfMonth(PackedLocalDate.pack(today)));
  }

  @Test
  public void testGetYear() {
    LocalDate today = LocalDate.now();
    assertEquals(today.getYear(), PackedLocalDate.getYear(PackedLocalDate.pack(today)));
  }

  @Test
  public void testGetMonthValue() {
    int dateTime = PackedLocalDate.pack(LocalDate.of(2015, 12, 25));
    assertEquals(12, PackedLocalDate.getMonthValue(dateTime));
  }

  @Test
  public void testGetDayOfWeek() {
    LocalDate date = LocalDate.of(2015, 12, 25);
    int dateTime = PackedLocalDate.pack(date);
    assertEquals(date.getDayOfWeek(), PackedLocalDate.getDayOfWeek(dateTime));
  }
}