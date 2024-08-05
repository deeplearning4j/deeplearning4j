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

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.apache.commons.io.FileUtils;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.io.IOException;
import java.util.*;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class ModelTestCase {

    protected Map<String, List<INDArray>> inputs = new HashMap<>();
    protected Map<String, List<INDArray>> outputs = new HashMap<>();
    protected Map<String, Model> models = new HashMap<>();
    protected Map<String, List<INDArray>> computedOutputs = new HashMap<>();
    protected Map<String,List<Boolean>> outputsEquals = new HashMap<>();
    private String testName;
    protected File testDirectory;



    public ModelTestCase(File testDirectory,String testName) {
        this.testDirectory = testDirectory;
        File[] subDirs = testDirectory.listFiles();
        for (File modelDir : Objects.requireNonNull(subDirs)) {
            String modelName = modelDir.getName(); // Directory name is used as modelName
            File[] npyFiles = modelDir.listFiles((d, name) -> name.endsWith(".npy"));
            List<String> names = new ArrayList<>();
            for (File npyFile : Objects.requireNonNull(npyFiles)) {
                names.add(npyFile.getName());
            }
            try {
                loadInputsOutputs(names.toArray(new String[0]));
            } catch (IOException e) {
                throw new RuntimeException("Unable to load inputs/outputs for model: " + modelName, e);
            }
        }
    }

    /**
     * Loads all models in the test directory
     * @param names
     * @throws IOException
     */
    public void loadInputsOutputs(String... names) throws IOException {
        for (String name : names) {
            String modelName = name.split("_")[0];
            String type = name.split("_")[1];
            int index = Integer.parseInt(name.replace(".npy","").split("_")[2]);
            File modelDir = new File(testDirectory, modelName);
            INDArray array = Nd4j.createFromNpyFile(new File(modelDir, name));
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

    /**
     * Compute outputs for a given model.
     * @param model the model to compute.
     * @throws Exception
     */
    public void computeOutputAndGradient(Model model) throws Exception {
        for (String modelName : models.keySet()) {
            List<INDArray> modelInputs = inputs.get(modelName);
            List<INDArray> expectedOutputs = outputs.get(modelName);
            if(!inputs.containsKey(modelName)) {
                throw new Exception("No inputs found for model: " + modelName);
            }

            for(int i = 0; i < modelInputs.size(); i++) {
                model.setInput(i, modelInputs.get(i));
            }

            //this is relative to what python does when we compute gradients
            for(int i = 0; i < expectedOutputs.size(); i++) {
                model.setLabels(i, Nd4j.zerosLike(expectedOutputs.get(i)));
            }

            // Set input and label then compute derivative and scores
            INDArray[] outputs = model.output(modelInputs.toArray(new INDArray[0]));
            this.computedOutputs.put(modelName, Arrays.asList(outputs));
        }
    }


    /**
     * Run all models.
     * @throws Exception
     */
    public void runModels() throws Exception {
        for (String modelName : models.keySet()) {
            Model model = models.get(modelName);
            computeOutputAndGradient(model);
        }
    }

    public void compareOutputs() throws  Exception {
        for (String modelName : models.keySet()) {
            Model model = this.models.get(modelName);
            computeOutputAndGradient(model);
            List<INDArray> modelOutputs = outputs.get(modelName);
            List<INDArray> computedOutputs = this.computedOutputs.get(modelName);
            outputsEquals.put(modelName, new ArrayList<>());
            for (int i = 0; i < modelOutputs.size(); i++) {
                INDArray loadedOutput = modelOutputs.get(i);
                INDArray computedOutput = computedOutputs.get(i);
                outputsEquals.get(modelName).add(loadedOutput.equalsWithEps(computedOutput,1e-6));
                System.out.println("Loaded output: " + loadedOutput);
                System.out.println("Computed output: " + computedOutput);
            }
        }
    }


    public void loadModels() throws Exception {
        for(File modelDir : testDirectory.listFiles()) {
            String modelType = FileUtils.readFileToString(new File(modelDir, "model_type.txt"), "UTF-8");
            File modelFile = new File(modelDir, "model.h5");
            Map<String,Model> models = new HashMap<>();
            switch (modelType) {
                case "Sequential":
                    models.put(modelDir.getName(), KerasModelImport.importKerasSequentialModelAndWeights(modelFile.getAbsolutePath(), false));
                    break;
                case "Model":
                    models.put(modelDir.getName(), KerasModelImport.importKerasModelAndWeights(modelFile.getAbsolutePath()));
                    break;
                case "Functional":
                    models.put(modelDir.getName(), KerasModelImport.importKerasModelAndWeights(modelFile.getAbsolutePath()));
                    break;
                default:
                    throw new Exception("Unknown model type: " + modelType);
            }
        }
    }


}
