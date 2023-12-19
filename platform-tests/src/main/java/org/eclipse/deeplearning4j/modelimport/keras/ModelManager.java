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

package org.eclipse.deeplearning4j.modelimport.keras;

import org.apache.commons.io.FileUtils;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.io.IOException;
import java.util.*;

public class ModelManager {
    protected String directoryPath;
    protected Map<String, List<INDArray>> inputs = new HashMap<>();
    protected Map<String, List<INDArray>> outputs = new HashMap<>();
    protected Map<String, List<INDArray>> gradients = new HashMap<>();
    protected Map<String,Model> models = new HashMap<>();
    protected Map<String, List<INDArray>> computedOutputs = new HashMap<>();
    protected Map<String, List<INDArray>> computedGradients = new HashMap<>();

    public ModelManager(String directoryPath) {
        this.directoryPath = directoryPath;
        File dir = new File(directoryPath);
        if (!dir.exists() || !dir.isDirectory()) {
            throw new IllegalArgumentException("Invalid directoryPath: " + directoryPath);
        }
        File[] subDirs = dir.listFiles(File::isDirectory);
        for (File modelDir : Objects.requireNonNull(subDirs)) {
            String modelName = modelDir.getName(); // Directory name is used as modelName
            File[] npyFiles = modelDir.listFiles((d, name) -> name.endsWith(".npy"));
            List<String> names = new ArrayList<>();
            for (File npyFile : Objects.requireNonNull(npyFiles)) {
                names.add(getFormattedFileName(npyFile.getName(), modelName));
            }
            try {
                loadInputsOutputs(names.toArray(new String[0]));
            } catch (IOException e) {
                throw new RuntimeException("Unable to load inputs/outputs for model: " + modelName, e);
            }
        }

    }


    private String getFormattedFileName(String fileName, String modelName) {
        fileName = fileName.replace(".npy", "");
        String[] splits = fileName.split("_");
        return modelName + "_" + splits[0] + "_" + splits[1];
    }

    public void loadModels() throws IOException, InvalidKerasConfigurationException, UnsupportedKerasConfigurationException, Exception {
        for(File modelDir : new File(directoryPath).listFiles()) {
            String modelType = FileUtils.readFileToString(new File(modelDir, "model_type.txt"), "UTF-8");
            switch (modelType) {
                case "Sequential":
                    models.put(modelDir.getName(), KerasModelImport.importKerasSequentialModelAndWeights(modelDir.getAbsolutePath(), true));
                    break;
                case "Model":
                    models.put(modelDir.getName(), KerasModelImport.importKerasModelAndWeights(modelDir.getAbsolutePath()));
                    break;
                case "Functional":
                    models.put(modelDir.getName(), KerasModelImport.importKerasModelAndWeights(modelDir.getAbsolutePath()));
                    break;
                default:
                    throw new Exception("Unknown model type: " + modelType);
            }
        }
    }

    public void loadInputsOutputs(String... names) throws IOException {
        Map<String, List<INDArray>> inputs = new LinkedHashMap<>();
        Map<String, List<INDArray>> outputs = new LinkedHashMap<>();
        Map<String, List<INDArray>> gradients = new LinkedHashMap<>();
        for (String name : names) {
            String modelName = name.split("_")[0];
            String type = name.split("_")[1];
            int index = Integer.parseInt(name.split("_")[2]);
            INDArray array = Nd4j.getNDArrayFactory().createFromNpyFile(new File(directoryPath, name + ".npy"));
            switch (type) {
                case "input":
                    if (!inputs.containsKey(modelName)) {
                        inputs.put(modelName, new ArrayList<>());
                    }
                    fillListAtIndex(inputs.get(modelName), index, array);
                    break;
                case "output":
                    if (!outputs.containsKey(modelName)) {
                        outputs.put(modelName, new ArrayList<>());
                    }
                    fillListAtIndex(outputs.get(modelName), index, array);
                    break;
                case "gradient":
                    if (!gradients.containsKey(modelName)) {
                        gradients.put(modelName, new ArrayList<>());
                    }
                    fillListAtIndex(gradients.get(modelName), index, array);
                    break;
            }
        }
    }

    private void fillListAtIndex(List<INDArray> list, int index, INDArray element) {
        while (index >= list.size()) {
            list.add(null);
        }
        list.set(index, element);
    }

    public Map<String, List<INDArray>> getInputs() {
        return inputs;
    }

    public Map<String, List<INDArray>> getOutputs() {
        return outputs;
    }

    public Map<String, List<INDArray>> getGradients() {
        return gradients;
    }

    public void computeOutputAndGradient(Model model) throws Exception {
        for (String modelName : inputs.keySet()) {
            List<INDArray> modelInputs = inputs.get(modelName);
            for (int i = 0; i < modelInputs.size(); i++) {
                // Set input and label then compute derivative and scores
                model.computeGradientAndScore();

                // Cache model's gradients
                Gradient gradients = model.gradient();
                List<INDArray> grads = new ArrayList<>();

                for (String varName : gradients.gradientForVariable().keySet()) {
                    grads.add(gradients.getGradientFor(varName));
                }

                computedGradients.put(modelName + "_gradient_" + i, grads);

            }
        }
    }


    public boolean compareOutputs(Model model) throws Exception {
        double epsilon = 1e-6;

        for (String modelName : outputs.keySet()) {
            List<INDArray> modelOutputs = outputs.get(modelName);
            List<INDArray> computedOutputs = this.computedOutputs.get(modelName);
            for (int i = 0; i < modelOutputs.size(); i++) {
                INDArray loadedOutput = modelOutputs.get(i);
                INDArray computedOutput = computedOutputs.get(i);
                INDArray diff = loadedOutput.sub(computedOutput).mul(loadedOutput.sub(computedOutput));
                if (diff.sumNumber().doubleValue() > epsilon)
                    return false;
            }
        }
        return true;
    }

    public boolean compareGradients() throws Exception {
        double epsilon = 1e-6;

        for (String modelName : gradients.keySet()) {
            List<INDArray> modelGradients = gradients.get(modelName);
            List<INDArray> computedGradients = this.computedGradients.get(modelName);
            for (int i = 0; i < modelGradients.size(); i++) {
                INDArray loadedGradient = modelGradients.get(i);
                INDArray computedGradient = computedGradients.get(i);
                INDArray diff = loadedGradient.sub(computedGradient).mul(loadedGradient.sub(computedGradient));
                if (diff.sumNumber().doubleValue() > epsilon)
                    return false;
            }
        }
        return true;
    }
}