package com.ccc.sendalyzeit.textanalytics.util;

public class ArrayUtil {

	public static int[] flatten(int[][] arr) {
		int[] ret = new int[arr.length * arr[0].length];
		int count = 0;
		for(int i = 0; i < arr.length; i++)
			for(int j = 0; j < arr[i].length; j++)
				ret[count++] = arr[i][j];
		return ret;
	}
	
	public static int[] toOutcomeArray(int outcome,int numOutcomes) {
		int[] nums = new int[numOutcomes];
		nums[outcome] = 1;
		return nums;
	}

}
