package org.datavec.dataframe.util.collections;

import com.google.common.collect.Range;
import com.google.common.collect.RangeSet;
import com.google.common.collect.TreeRangeSet;
import org.junit.Test;

import java.util.Set;

import static org.junit.Assert.*;

/**
 *  Tests for primitive int rangesets
 */
public class IntTreeRangeSetTest {

  private IntTreeRangeSet intTreeRangeSet = IntTreeRangeSet.create();
  private TreeRangeSet<Integer> treeRangeSet = TreeRangeSet.create();

  @Test
  public void testAdd() {
    intTreeRangeSet.add(IntRange.closed(0, 4));
    treeRangeSet.add(Range.closed(0, 4));
   // System.out.println(intTreeRangeSet);
   // System.out.println(treeRangeSet);

    intTreeRangeSet.add(IntRange.open(5, 7));
    treeRangeSet.add(Range.open(5, 7));
    //System.out.println(intTreeRangeSet);
    //System.out.println(treeRangeSet);

    intTreeRangeSet.add(IntRange.closedOpen(4, 5));
    treeRangeSet.add(Range.closedOpen(4, 5));
    //System.out.println(intTreeRangeSet);
    //System.out.println(treeRangeSet);

    IntRange intSpan = intTreeRangeSet.span();
    Range<Integer> span = treeRangeSet.span();
    //System.out.println(intSpan);
    //System.out.println(span);

    assertEquals(IntRange.closedOpen(0, 7), intSpan);
    Set<IntRange> ranges = intTreeRangeSet.asRanges();
    assertEquals(2, ranges.size());

    IntRangeSet intComplement = intTreeRangeSet.complement();
    RangeSet<Integer> complement = treeRangeSet.complement();
    //System.out.println(intComplement);
    //System.out.println(complement);
    //System.out.println(intTreeRangeSet);
  }
}