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

package org.nd4j.codegen.impl.python;

import org.apache.commons.io.FileUtils;
import org.nd4j.codegen.api.*;
import org.nd4j.codegen.api.doc.DocTokens;
import org.nd4j.codegen.api.generator.Generator;
import org.nd4j.codegen.api.generator.GeneratorConfig;
import org.nd4j.codegen.util.GenUtil;

import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

/**
 * This is a very simple, manual namespace generator
 * We could of course use a templating library such as Freemarker, which woudl work fine - but:
 * (a) on the one hand, it's overkill/unnecessary
 * (b) on the other hand, may provide less flexibility than a manual implementation
 *
 */
public class PythonGenerator implements Generator {

    private static final String I4 = "    ";

    @Override
    public Language language() {
        return Language.PYTHON;
    }

    @Override
    public void generateNamespaceNd4j(NamespaceOps namespace, GeneratorConfig config, File directory, String fileName) throws IOException {


        StringBuilder sb = new StringBuilder();
        sb.append("class ").append(GenUtil.ensureFirstIsCap(namespace.getName())).append(":\n\n");

        List<Op> ops = new ArrayList<>();
        for(Op o : namespace.getOps()){
            if(o.isAbstract())
                continue;
            ops.add(o);
        }

        //TODO: handle includes

        for(Op o : ops){
            String s = generateMethod(o);
            sb.append(GenUtil.addIndent(s, 4));
            sb.append("\n\n");
        }

        File f = new File(directory, GenUtil.ensureFirstIsCap(namespace.getName()) + ".py");
        String content = sb.toString();

        FileUtils.writeStringToFile(f, content, StandardCharsets.UTF_8);
    }

    protected static String generateMethod(Op op){
        StringBuilder sb = new StringBuilder();
        sb.append("@staticmethod\n")
                .append("def ").append(GenUtil.ensureFirstIsNotCap(op.getOpName())).append("(");

        //Add inputs to signature
        boolean firstArg = true;
        if(op.getInputs() != null){
            for(Input i : op.getInputs()){
                if(!firstArg)
                    sb.append(", ");

                sb.append(i.getName());

                firstArg = false;
            }
        }


        //Add arguments and default args to signature

        sb.append("):\n");

        String docString = genDocString(op);
        sb.append(GenUtil.addIndent(docString, 4));

        sb.append(I4).append("# Execution code here\n");


        return sb.toString();
    }

    protected static String genDocString(Op op){
        //Following roughly: https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html
        StringBuilder sb = new StringBuilder();
        sb.append("\"\"\"")
                .append(op.getOpName())
                .append(" operation")
                .append("\n\n");
        if(op.getInputs() != null){
            sb.append("Args:");
            sb.append("\n");
            for(Input i : op.getInputs()){
                sb.append(I4).append(i.getName()).append(" (ndarray): ");
                if(i.getDescription() != null)
                    sb.append(DocTokens.processDocText(i.getDescription(), op, DocTokens.GenerationType.ND4J));
                sb.append("\n");
            }
        }

        sb.append("\n");

        if(op.getOutputs() != null){
            sb.append("Returns:\n");
            List<Output> o = op.getOutputs();

            if(o.size() == 1){
                sb.append(I4).append("ndarray: ").append(o.get(0).getName());
                String d = o.get(0).getDescription();
                if(d != null){
                    sb.append(" - ").append(DocTokens.processDocText(d, op, DocTokens.GenerationType.ND4J));
                }
                sb.append("\n");
            } else {
                throw new UnsupportedOperationException("Not yet implemented: Python docstring generation for multiple output ops");
            }
        }

        if(op.getArgs() != null){
            //Args and default args
            throw new UnsupportedOperationException("Generating method with args not yet implemented");
        }

        sb.append("\"\"\"\n");

        return sb.toString();
    }



    @Override
    public void generateNamespaceSameDiff(NamespaceOps namespace, GeneratorConfig config, File directory, String fileName) throws IOException {
        throw new UnsupportedOperationException("Not yet implemented");
    }
}
