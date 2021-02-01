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

package org.deeplearning4j.optimize.listeners.callbacks;

import lombok.NonNull;
import org.apache.commons.io.FilenameUtils;
import org.deeplearning4j.exception.DL4JInvalidConfigException;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.optimize.listeners.EvaluativeListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.IEvaluation;

import java.io.File;
import java.io.IOException;

public class ModelSavingCallback implements EvaluationCallback {
    protected File rootFolder;
    protected String template;

    /**
     * This constructor will create ModelSavingCallback instance that will save models in current folder
     *
     * PLEASE NOTE: Make sure you have write access to the current folder
     *
     * @param fileNameTemplate
     */
    public ModelSavingCallback(@NonNull String fileNameTemplate) {
        this(new File("./"), fileNameTemplate);
    }

    /**
     * This constructor will create ModelSavingCallback instance that will save models in specified folder
     *
     * PLEASE NOTE: Make sure you have write access to the target folder
     *
     * @param rootFolder File object referring to target folder
     * @param fileNameTemplate
     */
    public ModelSavingCallback(@NonNull File rootFolder, @NonNull String fileNameTemplate) {
        if (!rootFolder.isDirectory())
            throw new DL4JInvalidConfigException("rootFolder argument should point to valid folder");

        if (fileNameTemplate.isEmpty())
            throw new DL4JInvalidConfigException("Filename template can't be empty String");

        this.rootFolder = rootFolder;
        this.template = fileNameTemplate;
    }

    @Override
    public void call(EvaluativeListener listener, Model model, long invocationsCount, IEvaluation[] evaluations) {

        String temp = template.replaceAll("%d", "" + invocationsCount);

        String finalName = FilenameUtils.concat(rootFolder.getAbsolutePath(), temp);
        save(model, finalName);
    }


    /**
     * This method saves model
     *
     * @param model
     * @param filename
     */
    protected void save(Model model, String filename) {
        try {
            ModelSerializer.writeModel(model, filename, true);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }
}
