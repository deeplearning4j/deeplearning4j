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

package org.deeplearning4j.datasets.fetchers;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.image.transform.ImageTransform;

/**
 * Interface for defining a model that can be instantiated and return
 * information about itself.
 *
 * @author Justin Long (crockpotveggies)
 */
interface CacheableDataSet {

    String remoteDataUrl();
    String remoteDataUrl(DataSetType set);
    String localCacheName();
    String dataSetName(DataSetType set);
    long expectedChecksum();
    long expectedChecksum(DataSetType set);
    boolean isCached();
    RecordReader getRecordReader(long rngSeed, int[] imgDim, DataSetType set, ImageTransform imageTransform);

}
