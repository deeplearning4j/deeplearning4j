package org.datavec.dataframe.examples;

import org.datavec.dataframe.columns.Column;
import org.datavec.dataframe.api.ColumnType;
import org.datavec.dataframe.api.Table;
import org.datavec.dataframe.io.csv.CsvReader;
import org.junit.Before;
import org.junit.Test;

/**
 * Basic example code
 */
public class GettingStarted {

  private ColumnType[] types = {
      LOCAL_DATE,     // date of poll
      INTEGER,        // approval rating (pct)
      CATEGORY        // polling org
  };

  private Table table;

  @Before
  public void setUp() throws Exception {
    table = CsvReader.read(types, "data/BushApproval.csv");
  }

  @Test
  public void printStructure() throws Exception {
    out(table.structure().print());

    out(table.first(10).print());

    out(table.summary());

    out(table.columnNames());

    Column approval = table.column("approval");
    out(approval.summary().print());

    Column who = table.column("who");
    out(who.summary().print());

    Column date = table.column("date");
    out(date.summary().print());
  }

  private synchronized void out(Object obj) {
    System.out.println(String.valueOf(obj));
  }

}
