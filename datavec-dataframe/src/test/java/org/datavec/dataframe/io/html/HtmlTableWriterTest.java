package org.datavec.dataframe.io.html;

import org.datavec.dataframe.api.ColumnType;
import org.datavec.dataframe.api.Table;
import org.datavec.dataframe.columns.Column;
import org.datavec.dataframe.io.csv.CsvReader;
import org.datavec.dataframe.reducing.NumericReduceUtils;
import org.datavec.dataframe.table.ViewGroup;
import org.junit.Before;
import org.junit.Test;

/**
 *
 */
public class HtmlTableWriterTest {

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
  public void testWrite() {
    Column byColumn = table.column("who");
    ViewGroup group = new ViewGroup(table, byColumn);
    Table result = group.reduce("approval", NumericReduceUtils.mean);
    String str = HtmlTableWriter.write(result, "NA");
  }

}