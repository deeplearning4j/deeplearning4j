package org.datavec.dataframe;

import org.datavec.dataframe.api.ColumnType;
import org.datavec.dataframe.api.Table;
import org.datavec.dataframe.api.IntColumn;
import org.datavec.dataframe.api.DateColumn;
import org.datavec.dataframe.columns.packeddata.PackedLocalDate;
import org.datavec.dataframe.io.csv.CsvReader;
import org.junit.Before;
import org.junit.Test;

import java.time.LocalDate;

import static org.datavec.dataframe.api.QueryHelper.and;
import static org.datavec.dataframe.api.QueryHelper.both;
import static org.datavec.dataframe.api.QueryHelper.column;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * Tests for filtering on the T class
 */
public class TableFilteringTest {

  private final ColumnType[] types = {
      ColumnType.LOCAL_DATE,     // date of poll
      ColumnType.INTEGER,        // approval rating (pct)
      ColumnType.CATEGORY             // polling org
  };

  private Table table;

  @Before
  public void setUp() throws Exception {
    table = CsvReader.read(types, "data/BushApproval.csv");
  }

  @Test
  public void testFilter1() {
    Table result = table.selectWhere(column("approval").isLessThan(70));
    IntColumn a = result.intColumn("approval");
    for (int v : a) {
      assertTrue(v < 70);
    }
  }

  @Test
  public void testFilter2() {
    Table result = table.selectWhere(column("date").isInApril());
    DateColumn d = result.dateColumn("date");
    for (LocalDate v : d) {
      assertTrue(PackedLocalDate.isInApril(PackedLocalDate.pack(v)));
    }
  }

  @Test
  public void testFilter3() {
    Table result = table.selectWhere(
        both(column("date").isInApril(),
             column("approval").isGreaterThan(70)));

    DateColumn dates = result.dateColumn("date");
    IntColumn approval = result.intColumn("approval");
    for (int row : result) {
      assertTrue(PackedLocalDate.isInApril(dates.getInt(row)));
      assertTrue(approval.get(row) > 70);
    }
  }

  @Test
  public void testFilter4() {
    Table result =
        table.select("who", "approval")
            .where(
                and(column("date").isInApril(),
                     column("approval").isGreaterThan(70)));
    assertEquals(2, result.columnCount());
    assertTrue(result.columnNames().contains("who"));
    assertTrue(result.columnNames().contains("approval"));
  }
}
