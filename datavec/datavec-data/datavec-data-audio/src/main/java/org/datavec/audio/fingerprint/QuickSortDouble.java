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

package org.datavec.audio.fingerprint;

public class QuickSortDouble extends QuickSort {

    private int[] indexes;
    private double[] array;

    public QuickSortDouble(double[] array) {
        this.array = array;
        indexes = new int[array.length];
        for (int i = 0; i < indexes.length; i++) {
            indexes[i] = i;
        }
    }

    public int[] getSortIndexes() {
        sort();
        return indexes;
    }

    private void sort() {
        quicksort(array, indexes, 0, indexes.length - 1);
    }

    // quicksort a[left] to a[right]
    private void quicksort(double[] a, int[] indexes, int left, int right) {
        if (right <= left)
            return;
        int i = partition(a, indexes, left, right);
        quicksort(a, indexes, left, i - 1);
        quicksort(a, indexes, i + 1, right);
    }

    // partition a[left] to a[right], assumes left < right
    private int partition(double[] a, int[] indexes, int left, int right) {
        int i = left - 1;
        int j = right;
        while (true) {
            while (a[indexes[++i]] < a[indexes[right]]); // find item on left to swap, a[right] acts as sentinel
            while (a[indexes[right]] < a[indexes[--j]]) { // find item on right to swap
                if (j == left)
                    break; // don't go out-of-bounds
            }
            if (i >= j)
                break; // check if pointers cross
            swap(a, indexes, i, j); // swap two elements into place
        }
        swap(a, indexes, i, right); // swap with partition element
        return i;
    }

    // exchange a[i] and a[j]
    private void swap(double[] a, int[] indexes, int i, int j) {
        int swap = indexes[i];
        indexes[i] = indexes[j];
        indexes[j] = swap;
    }

}
