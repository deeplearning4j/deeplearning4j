package org.datavec.dataframe.testutil;

import it.unimi.dsi.fastutil.ints.IntArrayList;
import it.unimi.dsi.fastutil.ints.IntRBTreeSet;
import me.lemire.integercompression.differential.IntegratedIntCompressor;
import org.apache.commons.lang3.RandomUtils;
import org.github.jamm.MemoryMeter;
import org.roaringbitmap.RoaringBitmap;

/**
 * Benchmarks on representations of integers using a variety of approaches and scenarios,
 * where the scenarios vary the number of ints represented as well as the size of the population
 * that they are drawn from.
 *
 * Note that we use jamm to measures the number of bytes. To run jamm, you need to execute this test
 * with the following jvm arguments
 * -javaagent:jamm-0.3.1.jar
 *
 * Jamm needs to be in the pom to compile and needs to be in the java path to execute
 */
public class IntegerBenchmarks {

  public static void main(String[] args) {

    MemoryMeter meter = new MemoryMeter();

    // test sets of ints ranging in size from 0 to CYCLES
    int CYCLES = 5000;
    int POPULATION_SIZE = 50_000_000;

    IntArrayList list = new IntArrayList();
    IntArrayList list2 = new IntArrayList();
    RoaringBitmap bitmap = new RoaringBitmap();

    // In this step we sort the ints
    IntRBTreeSet testData = new IntRBTreeSet();
    for (int i = 0; i < CYCLES; i++) {
      int x = RandomUtils.nextInt(0, POPULATION_SIZE);
      testData.add(x);
    }

    IntegratedIntCompressor iic = new IntegratedIntCompressor();
    int[] compressed;
    int count = 1;
    for (int i : testData) {
      list.add(i);
      list2.add(i);
      list2.trim();
      bitmap.add(i);

      compressed = iic.compress(list.elements());
      System.out.println();
      System.out.println(count);
      System.out.println("IntArrayList:         " + meter.measureDeep(list));
      System.out.println("Trimmed IntArrayList: " + meter.measureDeep(list2));
      System.out.println("Trimmed int[]:        " + meter.measureDeep(list2.elements()));
      System.out.println("Array:                " + meter.measureDeep(list2.elements()));
      System.out.println("Bitmap:               " + meter.measureDeep(bitmap));
      System.out.println("FastPfor:             " + meter.measureDeep(compressed));
      count++;
    }
  }
}