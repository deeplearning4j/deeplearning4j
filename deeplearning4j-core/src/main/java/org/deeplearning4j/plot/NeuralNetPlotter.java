/*
 *
 *  * Copyright 2015 Skymind,Inc.
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
 *
 */

package org.deeplearning4j.plot;

import java.io.*;
import java.util.*;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.IOUtils;
import org.apache.commons.lang3.StringUtils;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.nn.params.PretrainParamInitializer;
import org.nd4j.linalg.api.ndarray.INDArray;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.ClassPathResource;


/**
 * Credit to :
 * http://yosinski.com/media/papers/Yosinski2012VisuallyDebuggingRestrictedBoltzmannMachine.pdf
 *
 *
 * for visualizations
 * @author Adam Gibson
 *
 */
public class NeuralNetPlotter implements Serializable {

    private static 	ClassPathResource script = new ClassPathResource("scripts/plot.py");
    private static final Logger log = LoggerFactory.getLogger(NeuralNetPlotter.class);
    private static String localPath = "graph-tmp/";
    private static String localPlotPath = loadIntoTmp();

    private static String loadIntoTmp() {

        File plotPath = new File(localPath+"plot.py");

        try {
            List<String> lines = IOUtils.readLines(script.getInputStream());
            FileUtils.writeLines(plotPath, lines);

        } catch (IOException e) {
            throw new IllegalStateException("Unable to load python file");

        }

        return plotPath.getAbsolutePath();
    }

    protected String writeMatrix(INDArray matrix)  {
        try {
            String filePath = System.getProperty("java.io.tmpdir") + File.separator +  UUID.randomUUID().toString();
            File write = new File(filePath);
            BufferedOutputStream bos = new BufferedOutputStream(new FileOutputStream(write,true));
            write.deleteOnExit();
            for(int i = 0; i < matrix.rows(); i++) {
                INDArray row = matrix.getRow(i);
                StringBuilder sb = new StringBuilder();
                for(int j = 0; j < row.length(); j++) {
                    sb.append(String.format("%.10f", row.getDouble(j)));
                    if(j < row.length() - 1)
                        sb.append(",");
                }
                sb.append("\n");
                String line = sb.toString();
                    bos.write(line.getBytes());
                    bos.flush();
            }
            bos.close();
            return filePath;

        } catch(IOException e){
            throw new RuntimeException(e);
        }
    }

    public void renderGraph(String action, String dataPath, String saveFilePath) {

        try {
            log.info("Rendering " + action + " graphs for data analysis... ");
            Process is = Runtime.getRuntime().exec("python " + localPlotPath + " " + action + " " + dataPath + " " + saveFilePath);
            log.info("Std out " + IOUtils.readLines(is.getInputStream()).toString());
            log.error("Std error " + IOUtils.readLines(is.getErrorStream()).toString());
        }catch(IOException e) {
            log.warn("Image closed");
            throw new RuntimeException(e);
    }
    }

    /**
     * graphPlotType sets up data to pass to scripts that render graphs
     * @param plotType sets which plot to call whether "multi" for multiple matrices histograms,
     *                 "hist" for a histogram of one matrix, or "scatter" for scatter plot
     * @param titles the titles of the plots
     * @param matrices the matrices to plot
     */
    public void graphPlotType(String plotType, List<String> titles, INDArray[] matrices, String saveFilePath) {
        String[] path = new String[matrices.length * 2];

        if(titles.size() != matrices.length)
            throw new IllegalArgumentException("Titles and matrix lengths must be equal");

        for(int i = 0; i < path.length - 1; i+=2) {
            path[i] = writeMatrix(matrices[i / 2].ravel());
            path[i + 1] = titles.get(i / 2);
        }

        String dataPath = StringUtils.join(path, ",");
        renderGraph(plotType, dataPath, saveFilePath);

    }

    /**
     * plotWeightHistograms graphs values of vBias, W, and hBias on aggregate and
     *          most recent mini-batch updates (-gradient)
     * @param network the trained neural net model
     * @param gradient latest updates to weights and biases
     */
    public void plotWeightHistograms(Layer network, Gradient gradient) {
        Set<String> vars = new TreeSet<>(gradient.gradientForVariable().keySet());
        List<String> titles = new ArrayList<>(vars);
        for(String s : vars) {
            titles.add(s + "-gradient");
        }
        graphPlotType(
                "histogram",
                titles,
                new INDArray[]{
                        network.getParam(DefaultParamInitializer.WEIGHT_KEY),
                        network.getParam(PretrainParamInitializer.BIAS_KEY),
                        network.getParam(PretrainParamInitializer.VISIBLE_BIAS_KEY),
                        gradient.gradientForVariable().get(DefaultParamInitializer.WEIGHT_KEY),
                        gradient.gradientForVariable().get(DefaultParamInitializer.BIAS_KEY),
                        gradient.gradientForVariable().get(PretrainParamInitializer.VISIBLE_BIAS_KEY)
                },
                localPath + "weightHistograms.png"
                );
    }


    public void plotWeightHistograms(Layer network) {
        plotWeightHistograms(network, network.gradient());
    }

    /**
    * plotActivations show how hidden neurons are used, how often on vs. off and correlation
     * @param network the trained neural net model
     **/
    public void plotActivations(Layer network) {

        if(network.input() == null)
            throw new IllegalStateException("Unable to plot; missing input");

        // TODO hidden_mean coming back with only 4 values - need further digging to understand issue
        INDArray hbiasMean = network.activationMean();
        String dataPath = writeMatrix(hbiasMean);

        renderGraph("activations", dataPath, localPath + "activationPlot.png");

    }

    /**
     * renderFilter plot learned filter for each hidden neuron
     * @param weight the trained neural net model
     **/
    public void renderFilter(INDArray weight, int patchesPerRow) {
        INDArray w = weight.dup();
        FilterRenderer render = new FilterRenderer();

        try {
            if(w.shape().length > 2) {
                INDArray render2 = w.transpose();
                render.renderFilters(render2,
                        localPath + "renderFilter.png",
                        w.columns(),
                        w.rows(),
                        w.slices());

            }
            else {
                render.renderFilters(w,
                        localPath + "renderFilter.png",
                        (int) Math.sqrt(w.rows()),
                        (int) Math.sqrt(w.columns()),
                        patchesPerRow);

                //Alternative python approach
//                String dataPath = writeMatrix(w);
//                renderGraph("filter", dataPath, nRows, nCols);

            }
        } catch (Exception e) {
            log.error("Unable to plot filter, continuing...", e);
            e.printStackTrace();

        }

    }


    /**
     * plotNetworkGradient used for debugging gradients with different data visualizations
     * top layer is
     * @param network the trained neural net model
     * @param gradient latest updates to weights and biases
     **/
    public void plotNetworkGradient(Layer network,Gradient gradient,int patchesPerRow) {
        INDArray weight = network.getParam(DefaultParamInitializer.WEIGHT_KEY);
        plotWeightHistograms(network, gradient);
        plotActivations(network);
        renderFilter(weight, patchesPerRow);

    }

    public void plotNetworkGradient(Layer network,INDArray gradient,int patchesPerRow) {
        INDArray weight =  network.getParam(DefaultParamInitializer.WEIGHT_KEY);
        graphPlotType(
                "histogram",
                Arrays.asList("W", "w-gradient"),
                new INDArray[]{
                        network.getParam(DefaultParamInitializer.WEIGHT_KEY),
                        gradient
                },
                localPath + "weightHistograms.png"
        );
        plotActivations(network);
        renderFilter(weight, patchesPerRow);
    }


}
