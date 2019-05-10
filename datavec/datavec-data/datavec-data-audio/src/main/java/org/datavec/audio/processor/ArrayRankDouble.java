/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.datavec.audio.processor;

public class ArrayRankDouble {

    /**
     * Get the index position of maximum value the given array 
     * @param array an array
     * @return	index of the max value in array
     */
    public int getMaxValueIndex(double[] array) {

        int index = 0;
        double max = Integer.MIN_VALUE;

        for (int i = 0; i < array.length; i++) {
            if (array[i] > max) {
                max = array[i];
                index = i;
            }
        }

        return index;
    }

    /**
     * Get the index position of minimum value in the given array 
     * @param array an array
     * @return	index of the min value in array
     */
    public int getMinValueIndex(double[] array) {

        int index = 0;
        double min = Integer.MAX_VALUE;

        for (int i = 0; i < array.length; i++) {
            if (array[i] < min) {
                min = array[i];
                index = i;
            }
        }

        return index;
    }

    /**
     * Get the n-th value in the array after sorted
     * @param array an array
     * @param n position in array
     * @param ascending	is ascending order or not
     * @return value at nth position of array
     */
    public double getNthOrderedValue(double[] array, int n, boolean ascending) {

        if (n > array.length) {
            n = array.length;
        }

        int targetindex;
        if (ascending) {
            targetindex = n;
        } else {
            targetindex = array.length - n;
        }

        // this value is the value of the numKey-th element

        return getOrderedValue(array, targetindex);
    }

    private double getOrderedValue(double[] array, int index) {
        locate(array, 0, array.length - 1, index);
        return array[index];
    }

    // sort the partitions by quick sort, and locate the target index
    private void locate(double[] array, int left, int right, int index) {

        int mid = (left + right) / 2;
        // System.out.println(left+" to "+right+" ("+mid+")");

        if (right == left) {
            // System.out.println("* "+array[targetIndex]);
            // result=array[targetIndex];
            return;
        }

        if (left < right) {
            double s = array[mid];
            int i = left - 1;
            int j = right + 1;

            while (true) {
                while (array[++i] < s);
                while (array[--j] > s);
                if (i >= j)
                    break;
                swap(array, i, j);
            }

            // System.out.println("2 parts: "+left+"-"+(i-1)+" and "+(j+1)+"-"+right);

            if (i > index) {
                // the target index in the left partition
                // System.out.println("left partition");
                locate(array, left, i - 1, index);
            } else {
                // the target index in the right partition
                // System.out.println("right partition");
                locate(array, j + 1, right, index);
            }
        }
    }

    private void swap(double[] array, int i, int j) {
        double t = array[i];
        array[i] = array[j];
        array[j] = t;
    }
}
