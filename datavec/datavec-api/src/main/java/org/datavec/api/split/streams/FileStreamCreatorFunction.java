/*
 *  ******************************************************************************
 *  * Copyright (c) 2021 Deeplearning4j Contributors
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.datavec.api.split.streams;

import org.nd4j.common.base.Preconditions;
import org.nd4j.common.function.Function;

import java.io.*;
import java.net.URI;

public class FileStreamCreatorFunction implements Function<URI,InputStream>, Serializable {

    @Override
    public InputStream apply(URI uri) {
        Preconditions.checkState(uri.getScheme() == null || uri.getScheme().equalsIgnoreCase("file"),
                "Attempting to open URI that is not a File URI; for other stream types, you must use an appropriate stream loader function. URI: %s", uri);
        try {
            return new FileInputStream(new File(uri));
        } catch (IOException e){
            throw new RuntimeException("Error loading stream for file: " + uri, e);
        }
    }

}
