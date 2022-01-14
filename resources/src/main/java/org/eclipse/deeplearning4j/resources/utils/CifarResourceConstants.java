/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */
package org.eclipse.deeplearning4j.resources.utils;

import org.apache.commons.io.FilenameUtils;
import org.eclipse.deeplearning4j.resources.ResourceDataSets;

import java.io.File;

public class CifarResourceConstants {
    public static final String CIFAR_ROOT_URL = "https://www.cs.toronto.edu/~kriz";
    public final static String CIFAR_ARCHIVE_FILE = "cifar-10-binary.tar.gz";
    public final static File CIFAR_DEFAULT_DIR =  new File(ResourceDataSets.topLevelResourceDir(),
            FilenameUtils.concat("cifar", "cifar-10-batches-bin"));
}
