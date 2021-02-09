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

package org.nd4j.codegen.dsl;

import org.apache.commons.io.FileUtils;
import org.apache.commons.lang3.StringUtils;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.nd4j.codegen.api.NamespaceOps;
import org.nd4j.codegen.impl.java.Nd4jNamespaceGenerator;
import org.nd4j.codegen.ops.RNNKt;

import java.io.File;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;

class TestGeneration {

    @SuppressWarnings("unused")
    @TempDir
    public File testDir;

    @Test
    void test() throws Exception {
        File f = testDir;

//        List<NamespaceOps> list = Arrays.asList(BitwiseKt.Bitwise(), RandomKt.Random());
        List<NamespaceOps> list = Arrays.asList(RNNKt.SDRNN());

        for(NamespaceOps ops : list) {
            Nd4jNamespaceGenerator.generate(ops, null, f, ops.getName() + ".java", "org.nd4j.linalg.factory", StringUtils.EMPTY);
        }

        File[] files = f.listFiles();
        Iterator<File> iter = FileUtils.iterateFiles(f, null, true);
        if(files != null) {
            while(iter.hasNext()){
                File file = iter.next();
                if(file.isDirectory())
                    continue;
                System.out.println(FileUtils.readFileToString(file, StandardCharsets.UTF_8));
                System.out.println("\n\n================\n\n");
            }
        }
    }

}
