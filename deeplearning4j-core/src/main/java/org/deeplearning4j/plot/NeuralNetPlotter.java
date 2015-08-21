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
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.factory.*;
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
 * for visualizations
 * @author Adam Gibson
 *
 */
public class NeuralNetPlotter implements Serializable {

    private static 	ClassPathResource script = new ClassPathResource("scripts" + File.separator + "plot.py");
    private static final Logger log = LoggerFactory.getLogger(NeuralNetPlotter.class);
    private static String ID_FOR_SESSION = UUID.randomUUID().toString();
    private static String localPath = System.getProperty("java.io.tmpdir") + File.separator;
    private static String dataFilePath = localPath + "data" + File.separator;
    private static String graphPath = localPath + "graphs" + File.separator;
    private static String graphFilePath = graphPath + ID_FOR_SESSION + File.separator;
    private static String localPlotPath = loadIntoTmp();
    private static String layerGraphFilePath = graphFilePath;


    public String getLayerGraphFilePath() { return layerGraphFilePath; }

    public void setLayerGraphFilePath(String newPath) { this.layerGraphFilePath=newPath; }

    public static void printDataFilePath() { log.info("Data stored at " + dataFilePath); }

    public static void printGraphFilePath() { log.warn("Graphs stored at " + graphFilePath + ". " +
            "Warning: You must manually delete the folder when you are done."); }

    private static String loadIntoTmp() {
        setupDirectory(dataFilePath);
        setupDirectory(graphFilePath);
        printDataFilePath();
        printGraphFilePath();

        File plotPath = new File(graphPath,"plot.py");
        plotPath.deleteOnExit();
        if (!plotPath.exists()) {
            try {
                List<String> lines = IOUtils.readLines(script.getInputStream());
                FileUtils.writeLines(plotPath, lines);

            } catch (IOException e) {
                throw new IllegalStateException("Unable to load python file");

            }
        }
        return plotPath.getAbsolutePath();
    }

    protected static void setupDirectory(String path){
        File newPath = new File(path);
        if (!newPath.isDirectory())
            newPath.mkdir();
    }

    public void updateGraphDirectory(Layer layer){
        String layerType = layer.getClass().toString();
        String[] layerPath = layerType.split("\\.");
        String layerName = Integer.toString(layer.getIndex()) + layerPath[layerPath.length - 1] ;
        String newPath = graphFilePath + File.separator + layerName + File.separator;
        if (!new File(newPath).exists()) {
            setupDirectory(newPath);
            setLayerGraphFilePath(newPath);
        }

    }

    protected String writeMatrix(INDArray matrix)  {
        try {
            String tmpFilePath = dataFilePath + UUID.randomUUID().toString();
            File write = new File(tmpFilePath);
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
            return tmpFilePath;

        } catch(IOException e){
            throw new RuntimeException(e);
        }
    }

    public String writeArray(ArrayList data)  {
        try {
            String tmpFilePath = dataFilePath + UUID.randomUUID().toString();
            File write = new File(tmpFilePath);
            BufferedOutputStream bos = new BufferedOutputStream(new FileOutputStream(write,true));
            write.deleteOnExit();
            StringBuilder sb = new StringBuilder();
            for(Object value : data) {
                sb.append(String.format("%.10f", (Double) value));
                sb.append(",");
            }
            String line = sb.toString();
            line = line.substring(0, line.length()-1);
            bos.write(line.getBytes());
            bos.flush();
            bos.close();
            return tmpFilePath;

        } catch(IOException e){
            throw new RuntimeException(e);
        }
    }


    /**
     * Calls out to python for rendering charts
     * @param action the action to take
     * @param dataPath the path to the data
     * @param saveFilePath the saved file path for output of graphs
     */
    public void renderGraph(String action, String dataPath, String saveFilePath) {

        try {
            log.info("Rendering " + action + " graphs for data analysis...");
            Process is = Runtime.getRuntime().exec("python " + localPlotPath + " " + action + " " + dataPath + " " + saveFilePath);
            log.info("Std out " + IOUtils.readLines(is.getInputStream()).toString());
            log.error("Std error " + IOUtils.readLines(is.getErrorStream()).toString());
        }catch(IOException e) {
            log.warn("Image closed");
            throw new RuntimeException(e);
        }
    }

    /**
     * Calls out to python for rendering charts
     * @param action the action to take
     * @param dataPath the path to the data
     * @param saveFilePath the saved file path for output of graphs
     * @param feature_width width of feature
     * @param feature_height height of feature
     */
    public void renderGraph(String action, String dataPath, String saveFilePath, int feature_width, int feature_height) {

        try {
            log.info("Rendering " + action + " graphs for data analysis...");
            Process is = Runtime.getRuntime().exec("python " + localPlotPath + " " + action + " " + dataPath + " " + saveFilePath
            + " " + feature_width + " " + feature_height);
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

        INDArray[] variablesAndGradients = new INDArray[network.conf().variables().size() * 2];
        int count = 0;
        for(int i = 0; i < network.conf().variables().size(); i++) {
            String variable = network.conf().variables().get(i);
            variablesAndGradients[count++] = network.getParam(variable);
        }

        for(int i = 0; i < network.conf().variables().size(); i++) {
            String variable = network.conf().variables().get(i);
            variablesAndGradients[count++] = gradient.getGradientFor(variable);

        }

        graphPlotType(
                "histogram",
                titles,
                variablesAndGradients,
                layerGraphFilePath + "weightHistograms.png"
                );
    }


    public void plotWeightHistograms(Layer network) {
        plotWeightHistograms(network, network.gradient());
    }

    /**
    * plotActivations show how hidden neurons are used, how often on vs. off and correlation
     * @param layer the trained neural net layer
     **/
    public void plotActivations(Layer layer) {

        if(layer.input() == null)
            throw new IllegalStateException("Unable to plot; missing input");

        // TODO simplify hbiasMean as a sample of the data vs all examples (cut by % if over 40 examples & 100 neurons)
        INDArray hbiasMean = layer.activationMean();
        String dataPath = writeMatrix(hbiasMean);

        renderGraph("activations", dataPath, layerGraphFilePath + "activationPlot.png");

    }

    /**
     * renderFilter plot learned filter for each hidden neuron
     * @param layer the trained neural net layer in the model
     **/
    public void renderFilter(Layer layer, int patchesPerRow) {
        INDArray weight = layer.getParam(DefaultParamInitializer.WEIGHT_KEY);
        INDArray w = weight.dup();
        FilterRenderer render = new FilterRenderer();

        try {
            if(w.shape().length > 2) {
                INDArray render2 = w.transpose();
                render.renderFilters(render2,
                        layerGraphFilePath + "renderFilter.png",
                        w.columns(),
                        w.rows(),
                        w.slices());

            }
            else {
                render.renderFilters(w,
                        layerGraphFilePath + "renderFilter.png",
                        (int) Math.sqrt(w.rows()),
                        (int) Math.sqrt(w.columns()),
                        patchesPerRow);
        }
//        Alternative python approach - work in progress
//            String dataPath = writeMatrix(w);
//            renderGraph("filter", dataPath, layerGraphFilePath + "renderFilter.png");

        } catch (Exception e) {
            log.error("Unable to plot filter, continuing...", e);
            e.printStackTrace();
        }
    }

    /**
     * plotNetworkGradient used for debugging RBM gradients with different data visualizations
     *
     * @param layer the neural net layer
     * @param gradient latest updates to weights and biases
     **/
    public void plotNetworkGradient(Layer layer, Gradient gradient) {
        plotWeightHistograms(layer, gradient);
        plotActivations(layer);
    }

    public void plotNetworkGradient(Layer layer,INDArray gradient) {
        graphPlotType(
                "histogram",
                Arrays.asList("W", "w-gradient"),
                new INDArray[]{
                        layer.getParam(DefaultParamInitializer.WEIGHT_KEY),
                        gradient
                },
                layerGraphFilePath + "weightHistograms.png"
        );
        plotActivations(layer);
    }


}
