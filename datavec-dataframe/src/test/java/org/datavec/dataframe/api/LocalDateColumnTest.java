package org.datavec.dataframe.api;

import org.junit.Before;
import org.junit.Test;

import java.time.LocalDate;

import static org.junit.Assert.assertEquals;

/**
 *  Tests for LocalDate Column
 */
public class LocalDateColumnTest {

  private DateColumn column1;

  @Before
  public void setUp() throws Exception {
    Table table = Table.create("Test");
    column1 = DateColumn.create("Game date");
    table.addColumn(column1);
  }

  @Test
  public void testAddCell() throws Exception {
    column1.addCell("2013-10-23");
    column1.addCell("12/23/1924");
    column1.addCell("12-May-2015");
    column1.addCell("12-Jan-2015");
    assertEquals(4, column1.size());
    LocalDate date = LocalDate.now();
    column1.add(date);
    assertEquals(5, column1.size());
  }

  @Test
  public void testDayOfMonth() throws Exception {
    column1.addCell("2013-10-23");
    column1.addCell("12/24/1924");
    column1.addCell("12-May-2015");
    column1.addCell("14-Jan-2015");
    ShortColumn c2 = column1.dayOfMonth();
    assertEquals(23, c2.get(0));
    assertEquals(24, c2.get(1));
    assertEquals(12, c2.get(2));
    assertEquals(14, c2.get(3));
  }

  @Test
  public void testMonth() throws Exception {
    column1.addCell("2013-10-23");
    column1.addCell("12/24/1924");
    column1.addCell("12-May-2015");
    column1.addCell("14-Jan-2015");
    ShortColumn c2 = column1.monthValue();
    assertEquals(10, c2.get(0));
    assertEquals(12, c2.get(1));
    assertEquals(5, c2.get(2));
    assertEquals(1, c2.get(3));
  }

  @Test
  public void testSummary() throws Exception {
    column1.addCell("2013-10-23");
    column1.addCell("12/24/1924");
    column1.addCell("12-May-2015");
    column1.addCell("14-Jan-2015");
    Table summary = column1.summary();
    assertEquals(4, summary.rowCount());
    assertEquals(2, summary.columnCount());
    assertEquals("Measure", summary.column(0).name());
    assertEquals("Value", summary.column(1).name());
  }


}
