package org.datavec.dataframe;

import org.apache.commons.lang3.RandomUtils;

import java.util.Arrays;
import java.util.List;

/**
 *
 */
public class TestDataUtil {

  public static String[] usStateArray = {"Alabama","Alaska","Arizona","Arkansas","California","Colorado","Connecticut",
      "Delaware","District Of Columbia","Florida","Georgia","Hawaii","Idaho","Illinois","Indiana","Iowa","Kansas",
      "Kentucky","Louisiana","Maine","Maryland","Massachusetts","Michigan","Minnesota","Mississippi","Missouri",
      "Montana","Nebraska","Nevada","New Hampshire","New Jersey","New Mexico","New York","North Carolina",
      "North Dakota","Ohio","Oklahoma","Oregon","Pennsylvania","Rhode Island","South Carolina","South Dakota",
      "Tennessee","Texas","Utah","Vermont","Virginia","Washington","West Virginia","Wisconsin","Wyoming"};

  public static List<String> usStates() {
    return Arrays.asList(usStateArray);
  }

  public static String randomUsState() {
    return usStateArray[RandomUtils.nextInt(0, usStateArray.length)];
  }
}
