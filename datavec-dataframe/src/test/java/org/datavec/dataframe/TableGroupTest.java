package org.datavec.dataframe;

import org.datavec.dataframe.api.ColumnType;
import org.datavec.dataframe.api.Table;
import org.datavec.dataframe.api.CategoryColumn;
import org.datavec.dataframe.io.csv.CsvReader;
import org.datavec.dataframe.table.SubTable;
import org.datavec.dataframe.table.TableGroup;
import org.junit.Before;
import org.junit.Test;

import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * Tests grouping and aggregation operations on tables
 */
public class TableGroupTest {

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
  public void testGetSubTables() {
    TableGroup tableGroup = new TableGroup(table, table.column("who"));
    List<SubTable> tables = tableGroup.getSubTables();
    assertEquals(6, tables.size());
  }

  @Test
  public void testWith2GroupingCols() {
    CategoryColumn month = table.dateColumn(0).month();
    month.setName("month");
    table.addColumn(month);
    String[] splitColumnNames = {table.column(2).name(), "month"};
    TableGroup tableGroup = new TableGroup(table, splitColumnNames);
    List<SubTable> tables = tableGroup.getSubTables();
    Table t = table.sum("approval").by(splitColumnNames);

    // compare the sum of the original column with the sum of the sums of the group table
    assertEquals(table.intColumn(1).sum(), Math.round(t.floatColumn(2).sum()));
    assertEquals(65, tables.size());
  }

  @Test
  public void testCountByGroup() {
    Table groups = table.count("approval").by("who");
    assertEquals(2, groups.columnCount());
    assertEquals(6, groups.rowCount());
    CategoryColumn group = groups.categoryColumn(0);
    assertTrue(group.contains("fox"));
  }

  @Test
  public void testSumGroup() {
    Table groups = table.sum("approval").by("who");
    // compare the sum of the original column with the sum of the sums of the group table
    assertEquals(table.intColumn(1).sum(), Math.round(groups.floatColumn(1).sum()));
  }
}