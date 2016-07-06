/*
 *
 *  *
 *  *  * Copyright 2015 Skymind,Inc.
 *  *  *
 *  *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *  *    you may not use this file except in compliance with the License.
 *  *  *    You may obtain a copy of the License at
 *  *  *
 *  *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *  *
 *  *  *    Unless required by applicable law or agreed to in writing, software
 *  *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *  *    See the License for the specific language governing permissions and
 *  *  *    limitations under the License.
 *  *
 *
 */

package org.canova.cli.csv.transforms;

/*

	purpose: represent column transforms of CSV data to vectorize

*/
public class Transforms {

	public double copy(String inputColumnValue) {
		return Double.valueOf(inputColumnValue);
	}

	/*
	 * Needed Statistics for binarize()
	 * - range of values (min, max)
	 * - similar to normalize, but we threshold on 0.5 after normalize
	 */
	public double binarize(String inputColumnValue) {
		double d = copy(inputColumnValue);
		return d > 0.5 ? 1 : 0.0;
	}

	/*
     * Needed Statistics for normalize()
     * - range of values (min, max)
     * -
     */
	public double normalize(String inputColumnValue) {
		throw new UnsupportedOperationException();
	}

	/*
     * Needed Statistics for label()
     * - count of distinct labels
     * - index of labels to IDs (hashtable?)
     */
	public double label(String inputColumnValue) {
		throw new UnsupportedOperationException();
	}

}
