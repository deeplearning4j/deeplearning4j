/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * See the NOTICE file distributed with this work for additional
 *  * information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.codegen.impl.java;
import org.apache.commons.lang3.StringUtils;
import org.nd4j.codegen.api.Language;
import org.nd4j.codegen.api.NamespaceOps;
import org.nd4j.codegen.api.generator.Generator;
import org.nd4j.codegen.api.generator.GeneratorConfig;

import java.io.File;

public class JavaPoetGenerator implements Generator {


    @Override
    public Language language() {
        return Language.JAVA;
    }

    @Override
    public void generateNamespaceNd4j(NamespaceOps namespace, GeneratorConfig config, File directory, String className) throws java.io.IOException {
        Nd4jNamespaceGenerator.generate(namespace, config, directory, className, "org.nd4j.linalg.factory", StringUtils.EMPTY);
    }

    @Override
    public void generateNamespaceSameDiff(NamespaceOps namespace, GeneratorConfig config, File directory, String className) throws java.io.IOException {
        //throw new UnsupportedOperationException("Not yet implemented");
        Nd4jNamespaceGenerator.generate(namespace, config, directory, className, "org.nd4j.autodiff.samediff", StringUtils.EMPTY);
    }
}
