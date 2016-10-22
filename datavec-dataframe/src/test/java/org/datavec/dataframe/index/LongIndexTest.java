package org.datavec.dataframe.index;

import org.datavec.dataframe.api.ColumnType;
import org.datavec.dataframe.api.Table;
import org.datavec.dataframe.columns.LongColumnUtils;
import org.datavec.dataframe.io.csv.CsvReader;
import org.datavec.dataframe.util.Selection;

import org.junit.Before;
import org.junit.Test;

import static org.datavec.dataframe.api.ColumnType.CATEGORY;
import static org.datavec.dataframe.api.ColumnType.LOCAL_DATE;
import static org.datavec.dataframe.api.ColumnType.LONG_INT;
import static org.junit.Assert.assertEquals;

/**
 *
 */
public class LongIndexTest {
  private ColumnType[] types = {
      LOCAL_DATE,     // date of poll
      LONG_INT,       // approval rating (pct)
      CATEGORY        // polling org
  };

  private LongIndex index;
  private Table table;

  @Before
  public void setUp() throws Exception {
    table = CsvReader.read(types, "data/BushApproval.csv");
    index = new LongIndex(table.longColumn("approval"));
  }

  @Test
  public void testGet() {
    Selection fromCol = table.longColumn("approval").select(LongColumnUtils.isEqualTo, 71);
    Selection fromIdx = index.get(71);
    assertEquals(fromCol, fromIdx);
  }

  @Test
  public void testGTE() {
    Selection fromCol = table.longColumn("approval").select(LongColumnUtils.isGreaterThanOrEqualTo, 71);
    Selection fromIdx = index.atLeast(71);
    assertEquals(fromCol, fromIdx);
  }

  @Test
  public void testLTE() {
    Selection fromCol = table.longColumn("approval").select(LongColumnUtils.isLessThanOrEqualTo, 71);
    Selection fromIdx = index.atMost(71);
    assertEquals(fromCol, fromIdx);
  }

  @Test
  public void testLT() {
    Selection fromCol = table.longColumn("approval").select(LongColumnUtils.isLessThan, 71);
    Selection fromIdx = index.lessThan(71);
    assertEquals(fromCol, fromIdx);
  }

  @Test
  public void testGT() {
    Selection fromCol = table.longColumn("approval").select(LongColumnUtils.isGreaterThan, 71);
    Selection fromIdx = index.greaterThan(71);
    assertEquals(fromCol, fromIdx);
  }
}