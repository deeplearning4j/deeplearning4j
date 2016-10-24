package org.datavec.dataframe.reducing;

import org.datavec.dataframe.api.Table;
import org.junit.Test;

/**
 * Tests for Cross Tabs
 */
public class CrossTabTest {

  @Test
  public void testXCount() throws Exception {

    Table t = Table.createFromCsv("data/tornadoes_1950-2014.csv");

    Table xtab = CrossTab.xCount(t, t.column("Scale"), t.column("Scale"));
    //System.out.println(xtab.print());

    Table rPct = CrossTab.rowPercents(xtab);
    //System.out.println(rPct.print());

    Table tPct = CrossTab.tablePercents(xtab);
    //System.out.println(tPct.print());

    Table cPct = CrossTab.columnPercents(xtab);
    //System.out.println(cPct.print());

    //TODO(lwhite): Real tests go here
  }
}