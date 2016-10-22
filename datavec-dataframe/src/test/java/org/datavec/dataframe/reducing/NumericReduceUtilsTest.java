package org.datavec.dataframe.reducing;

import org.datavec.dataframe.api.ColumnType;
import org.datavec.dataframe.api.Table;
import org.datavec.dataframe.columns.Column;
import org.datavec.dataframe.io.csv.CsvReader;
import org.datavec.dataframe.table.ViewGroup;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

/**
 *
 */
public class NumericReduceUtilsTest {

  private static ColumnType[] types = {
      ColumnType.LOCAL_DATE,     // date of poll
      ColumnType.INTEGER,        // approval rating (pct)
      ColumnType.CATEGORY        // polling org
  };

  private Table table;

  @Before
  public void setUp() throws Exception {
    table = CsvReader.read(types, "data/BushApproval.csv");
  }

  @Test
  public void testMean() {
    double result = table.reduce("approval", NumericReduceUtils.mean);
    assertEquals(64.88235294117646, result, 0.01);
  }

  @Test
  public void testGroupMean() {
    Column byColumn = table.column("who");
    ViewGroup group = new ViewGroup(table, byColumn);
    Table result = group.reduce("approval", NumericReduceUtils.mean);
    assertEquals(2, result.columnCount());
    Assert.assertEquals("who", result.column(0).name());
    assertEquals(6, result.rowCount());
    assertEquals("65.671875", result.get(1, 0));
  }

  @Test
  public void test2ColumnGroupMean() {
    Column byColumn1 = table.column("who");
    ViewGroup group = new ViewGroup(table, byColumn1);
    Table result = group.reduce("approval", NumericReduceUtils.mean);
    assertEquals(2, result.columnCount());
    Assert.assertEquals("who", result.column(0).name());
    assertEquals(6, result.rowCount());
    assertEquals("65.671875", result.get(1, 0));
  }
}