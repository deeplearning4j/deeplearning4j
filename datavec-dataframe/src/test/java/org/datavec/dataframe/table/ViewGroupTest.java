package org.datavec.dataframe.table;

import org.datavec.dataframe.api.CategoryColumn;
import org.datavec.dataframe.api.ColumnType;
import org.datavec.dataframe.api.Table;
import org.datavec.dataframe.io.csv.CsvReader;
import org.datavec.dataframe.reducing.NumericReduceFunction;
import org.apache.commons.math3.stat.StatUtils;
import org.junit.Before;
import org.junit.Test;

import java.util.List;

import static org.junit.Assert.*;

/**
 *
 */
public class ViewGroupTest {

  private final ColumnType[] types = {
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
  public void testViewGroupCreation() {

    ViewGroup group = new ViewGroup(table, table.column("who"));
    assertEquals(6, group.size());
    List<TemporaryView> viewList = group.getSubTables();

    int count = 0;
    for (TemporaryView view : viewList) {
      count += view.rowCount();
    }
    assertEquals(table.rowCount(), count);
  }

  @Test
  public void testViewTwoColumn() {

    ViewGroup group = new ViewGroup(table, table.column("who"), table.column("approval"));
    List<TemporaryView> viewList = group.getSubTables();

    int count = 0;
    for (TemporaryView view : viewList) {
      count += view.rowCount();
    }
    assertEquals(table.rowCount(), count);
  }

  @Test
  public void testWith2GroupingCols() {
    CategoryColumn month = table.dateColumn(0).month();
    month.setName("month");
    table.addColumn(month);
    String[] splitColumnNames = {table.column(2).name(), "month"};
    ViewGroup tableGroup = ViewGroup.create(table, splitColumnNames);
    List<TemporaryView> tables = tableGroup.getSubTables();
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
  public void testCustomFunction() {
    Table exaggeration = table.summarize("approval", exaggerate).by("who");
    CategoryColumn group = exaggeration.categoryColumn(0);
    assertTrue(group.contains("fox"));
  }

  @Test
  public void testSumGroup() {
    Table groups = table.sum("approval").by("who");
    // compare the sum of the original column with the sum of the sums of the group table
    assertEquals(table.intColumn(1).sum(), Math.round(groups.floatColumn(1).sum()));
  }

  static NumericReduceFunction exaggerate = new NumericReduceFunction() {
    @Override
    public String functionName() {
      return "exaggeration";
    }

    @Override
    public double reduce(double[] data) {
      return StatUtils.max(data) + 1000;
    }
  };
}