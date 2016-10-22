package org.datavec.dataframe.filters;

import org.datavec.dataframe.api.DateColumn;
import org.datavec.dataframe.api.Table;
import org.datavec.dataframe.columns.ColumnReference;
import org.datavec.dataframe.filtering.LocalDatePredicate;
import org.datavec.dataframe.filtering.datetimes.IsFirstDayOfTheMonth;
import org.datavec.dataframe.filtering.datetimes.IsInFebruary;
import org.datavec.dataframe.filtering.datetimes.IsInMarch;
import org.datavec.dataframe.filtering.datetimes.IsInYear;
import org.datavec.dataframe.filtering.datetimes.IsLastDayOfTheMonth;
import org.datavec.dataframe.filtering.datetimes.IsMonday;
import org.datavec.dataframe.filtering.datetimes.IsSunday;
import org.datavec.dataframe.util.Selection;
import org.junit.Before;
import org.junit.Test;

import java.time.LocalDate;

import static org.junit.Assert.*;

/**
 *
 */
public class LocalDateFilterTest {

  DateColumn localDateColumn = DateColumn.create("testing");
  Table table = Table.create("test");

  @Before
  public void setUp() throws Exception {
    localDateColumn.add(LocalDate.of(2016, 2, 28));
    localDateColumn.add(LocalDate.of(2016, 2, 29));
    localDateColumn.add(LocalDate.of(2016, 3, 1));
    table.addColumn(localDateColumn);
  }

  @Test
  public void testIsSunday() {
    ColumnReference reference = new ColumnReference("testing");
    IsSunday isSunday = new IsSunday(reference);
    Selection selection = isSunday.apply(table);
    assertTrue(selection.contains(0));
    assertFalse(selection.contains(1));
    assertFalse(selection.contains(2));
  }

  @Test
  public void testIsMonday() {
    ColumnReference reference = new ColumnReference("testing");
    IsMonday isSunday = new IsMonday(reference);
    Selection selection = isSunday.apply(table);
    assertFalse(selection.contains(0));
    assertTrue(selection.contains(1));
    assertFalse(selection.contains(2));
  }

  @Test
  public void testIsFebruary() {
    ColumnReference reference = new ColumnReference("testing");
    IsInFebruary isFebruary = new IsInFebruary(reference);
    Selection selection = isFebruary.apply(table);
    assertTrue(selection.contains(0));
    assertTrue(selection.contains(1));
    assertFalse(selection.contains(2));
  }

  @Test
  public void testIsMarch() {
    ColumnReference reference = new ColumnReference("testing");
    IsInMarch result = new IsInMarch(reference);
    Selection selection = result.apply(table);
    assertFalse(selection.contains(0));
    assertFalse(selection.contains(1));
    assertTrue(selection.contains(2));
  }

  @Test
  public void testIsFirstDayOfTheMonth() {
    ColumnReference reference = new ColumnReference("testing");
    IsFirstDayOfTheMonth result = new IsFirstDayOfTheMonth(reference);
    Selection selection = result.apply(table);
    assertFalse(selection.contains(0));
    assertFalse(selection.contains(1));
    assertTrue(selection.contains(2));
  }

  @Test
  public void testIsLastDayOfTheMonth() {
    ColumnReference reference = new ColumnReference("testing");
    IsLastDayOfTheMonth result = new IsLastDayOfTheMonth(reference);
    Selection selection = result.apply(table);
    assertFalse(selection.contains(0));
    assertTrue(selection.contains(1));
    assertFalse(selection.contains(2));
  }

  @Test
  public void testIsInYear() {
    ColumnReference reference = new ColumnReference("testing");
    IsInYear result = new IsInYear(reference, 2016);
    Selection selection = result.apply(table);
    assertTrue(selection.contains(0));
    assertTrue(selection.contains(1));
    assertTrue(selection.contains(2));
    result = new IsInYear(reference, 2015);
    selection = result.apply(table);
    assertFalse(selection.contains(0));
    assertFalse(selection.contains(1));
    assertFalse(selection.contains(2));
  }

  @Test
  public void testColumnFilters() {

    LocalDatePredicate after_2_28 = new LocalDatePredicate() {
      LocalDate date = LocalDate.of(2016, 2, 28);
      @Override
      public boolean test(LocalDate i) {
        return i.isAfter(date);
      }
    };

    DateColumn filtered = localDateColumn.selectIf(after_2_28);

  }

  private void print(Object o) {
    System.out.println(o);
  }
}
