/*-
 *  * Copyright 2016 Skymind, Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 */

package org.deeplearning4j.datasets.fetchers;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;

public class TinyImageNetFetcher extends CacheableExtractableDataSetFetcher {

    private File fileDir;
    private static Logger log = LoggerFactory.getLogger(TinyImageNetFetcher.class);

    @Override
    public String remoteDataUrl() { return "http://blob.deeplearning4j.org/datasets/tinyimagenet_200_dl4j.v1.zip"; }
    @Override
    public String localCacheName(){ return "TINYIMAGENET_200"; }
    @Override
    public long expectedChecksum() { return 1L; }


}
