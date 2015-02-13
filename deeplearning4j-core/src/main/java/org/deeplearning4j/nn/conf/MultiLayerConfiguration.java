package org.deeplearning4j.nn.conf;

import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.IOException;
import java.io.Serializable;
import java.util.*;

/**
 * Configuration for a multi layer network
 *
 * @author Adam Gibson
 */
public class MultiLayerConfiguration implements Serializable {

    private int[] hiddenLayerSizes;
    private List<NeuralNetConfiguration> confs;
    private boolean useDropConnect = false;
    private boolean useGaussNewtonVectorProductBackProp = false;
    protected boolean pretrain = true;
    /* Sample if true, otherwise use the straight activation function */
    private boolean useRBMPropUpAsActivations = true;
    private double dampingFactor = 100;
    private Map<Integer,OutputPreProcessor> processors = new HashMap<>();
    private boolean backward = false;

    private MultiLayerConfiguration() {}

    public MultiLayerConfiguration(MultiLayerConfiguration multiLayerConfiguration) {
        this.hiddenLayerSizes = multiLayerConfiguration.hiddenLayerSizes;
        this.confs = new ArrayList<>(multiLayerConfiguration.confs);
        this.useDropConnect = multiLayerConfiguration.useDropConnect;
        this.useGaussNewtonVectorProductBackProp = multiLayerConfiguration.useGaussNewtonVectorProductBackProp;
        this.pretrain = multiLayerConfiguration.pretrain;
        this.useRBMPropUpAsActivations = multiLayerConfiguration.useRBMPropUpAsActivations;
        this.dampingFactor = multiLayerConfiguration.dampingFactor;
        this.processors = new HashMap<>(multiLayerConfiguration.processors);
        this.backward = multiLayerConfiguration.backward;

    }

    public NeuralNetConfiguration getConf(int i) {
        return confs.get(i);
    }

    public OutputPreProcessor getPreProcessor(int layer) {
        return processors.get(layer);
    }

    public double getDampingFactor() {
        return dampingFactor;
    }

    public void setDampingFactor(double dampingFactor) {
        this.dampingFactor = dampingFactor;
    }

    public boolean isUseRBMPropUpAsActivations() {
        return useRBMPropUpAsActivations;
    }

    public void setUseRBMPropUpAsActivations(boolean useRBMPropUpAsActivations) {
        this.useRBMPropUpAsActivations = useRBMPropUpAsActivations;
    }

    public boolean isUseDropConnect() {
        return useDropConnect;
    }

    public void setUseDropConnect(boolean useDropConnect) {
        this.useDropConnect = useDropConnect;
    }

    public boolean isUseGaussNewtonVectorProductBackProp() {
        return useGaussNewtonVectorProductBackProp;
    }

    public void setUseGaussNewtonVectorProductBackProp(boolean useGaussNewtonVectorProductBackProp) {
        this.useGaussNewtonVectorProductBackProp = useGaussNewtonVectorProductBackProp;
    }

    public boolean isPretrain() {
        return pretrain;
    }

    public void setPretrain(boolean pretrain) {
        this.pretrain = pretrain;
    }

    public int[] getHiddenLayerSizes() {
        return hiddenLayerSizes;
    }

    public void setHiddenLayerSizes(int[] hiddenLayerSizes) {
        this.hiddenLayerSizes = hiddenLayerSizes;
    }

    public List<NeuralNetConfiguration> getConfs() {
        return confs;
    }

    public void setConfs(List<NeuralNetConfiguration> confs) {
        this.confs = confs;
    }

    public Map<Integer, OutputPreProcessor> getProcessors() {
        return processors;
    }

    public void setProcessors(Map<Integer, OutputPreProcessor> processors) {
        this.processors = processors;
    }

    public boolean isBackward() {
        return backward;
    }

    public void setBackward(boolean backward) {
        this.backward = backward;
    }

    /**
     *
     * @return  JSON representation of NN configuration
     */
    public String toJson() {
        ObjectMapper mapper = NeuralNetConfiguration.mapper();
        try {
            return mapper.writeValueAsString(this).replaceAll("\"activationFunction\",", "")
                .replaceAll("\"rng\",","").replaceAll("\"dist\",", "").replaceAll("\"stepFunction\",","")
                .replaceAll("\"layerFactory\",","");
        } catch (com.fasterxml.jackson.core.JsonProcessingException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Create a neural net configuration from json
     * @param json the neural net configuration from json
     * @return {@link org.deeplearning4j.nn.conf.MultiLayerConfiguration}
     */
    public static MultiLayerConfiguration fromJson(String json) {
        ObjectMapper mapper = NeuralNetConfiguration.mapper();
        try {
            return mapper.readValue(json, MultiLayerConfiguration.class);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public String toString() {
        return "MultiLayerConfiguration{" +
                "hiddenLayerSizes=" + Arrays.toString(hiddenLayerSizes) +
                ", confs=" + confs +
                ", useDropConnect=" + useDropConnect +
                ", useGaussNewtonVectorProductBackProp=" + useGaussNewtonVectorProductBackProp +
                ", pretrain=" + pretrain +
                ", useRBMPropUpAsActivations=" + useRBMPropUpAsActivations +
                ", dampingFactor=" + dampingFactor +
                ", processors=" + processors +
                ", backward=" + backward +
                '}';
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof MultiLayerConfiguration)) return false;

        MultiLayerConfiguration that = (MultiLayerConfiguration) o;

        if (backward != that.backward) return false;
        if (Double.compare(that.dampingFactor, dampingFactor) != 0) return false;
        if (pretrain != that.pretrain) return false;
        if (useDropConnect != that.useDropConnect) return false;
        if (useGaussNewtonVectorProductBackProp != that.useGaussNewtonVectorProductBackProp) return false;
        if (useRBMPropUpAsActivations != that.useRBMPropUpAsActivations) return false;
        if (confs != null ? !confs.equals(that.confs) : that.confs != null) return false;
        if (!Arrays.equals(hiddenLayerSizes, that.hiddenLayerSizes)) return false;

        return true;
    }

    @Override
    public int hashCode() {
        int result;
        long temp;
        result = hiddenLayerSizes != null ? Arrays.hashCode(hiddenLayerSizes) : 0;
        result = 31 * result + (confs != null ? confs.hashCode() : 0);
        result = 31 * result + (useDropConnect ? 1 : 0);
        result = 31 * result + (useGaussNewtonVectorProductBackProp ? 1 : 0);
        result = 31 * result + (pretrain ? 1 : 0);
        result = 31 * result + (useRBMPropUpAsActivations ? 1 : 0);
        temp = Double.doubleToLongBits(dampingFactor);
        result = 31 * result + (int) (temp ^ (temp >>> 32));
        result = 31 * result + (processors != null ? processors.hashCode() : 0);
        result = 31 * result + (backward ? 1 : 0);
        return result;
    }

    public MultiLayerConfiguration clone() {
        return new MultiLayerConfiguration(this);
    }

    public static class Builder {

        private List<NeuralNetConfiguration> confs = new ArrayList<>();
        private int[] hiddenLayerSizes;
        private boolean useDropConnect = false;
        protected boolean pretrain = true;
        protected boolean useRBMPropUpAsActivations = false;
        protected double dampingFactor = 100;
        protected Map<Integer,OutputPreProcessor> preProcessors = new HashMap<>();



        public Builder preProcessor(Integer layer,OutputPreProcessor preProcessor) {
            preProcessors.put(layer,preProcessor);
            return this;
        }

        public Builder preProcessors(Map<Integer,OutputPreProcessor> preProcessors) {
            this.preProcessors = preProcessors;
            return this;
        }

        public Builder dampingFactor(double dampingFactor) {
            this.dampingFactor = dampingFactor;
            return this;
        }

        public Builder useRBMPropUpAsActivations(boolean useRBMPropUpAsActivations) {
            this.useRBMPropUpAsActivations = useRBMPropUpAsActivations;
            return this;
        }




        public Builder pretrain(boolean pretrain) {
            this.pretrain = pretrain;
            return this;
        }

        public Builder useDropConnect(boolean useDropConnect) {
            this.useDropConnect = useDropConnect;
            return this;
        }



        public Builder confs(List<NeuralNetConfiguration> confs) {
            this.confs = confs;
            return this;

        }


        public Builder hiddenLayerSizes(int[] hiddenLayerSizes) {
            this.hiddenLayerSizes = hiddenLayerSizes;
            return this;
        }

        public MultiLayerConfiguration build() {
            MultiLayerConfiguration conf = new MultiLayerConfiguration();
            conf.confs = this.confs;
            if(hiddenLayerSizes == null)
                throw new IllegalStateException("Please specify hidden layer sizes");
            conf.hiddenLayerSizes = this.hiddenLayerSizes;
            conf.useDropConnect = useDropConnect;
            conf.pretrain = pretrain;
            conf.useRBMPropUpAsActivations = useRBMPropUpAsActivations;
            conf.dampingFactor = dampingFactor;
            conf.processors = preProcessors;
            return conf;

        }

        @Override
        public String toString() {
            return "Builder{" +
                    "confs=" + confs +
                    ", hiddenLayerSizes=" + Arrays.toString(hiddenLayerSizes) +
                    ", useDropConnect=" + useDropConnect +
                    ", pretrain=" + pretrain +
                    ", useRBMPropUpAsActivations=" + useRBMPropUpAsActivations +
                    ", dampingFactor=" + dampingFactor +
                    ", preProcessors=" + preProcessors +
                    '}';
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (!(o instanceof Builder)) return false;

            Builder builder = (Builder) o;

            return Double.compare(builder.dampingFactor, dampingFactor) == 0
                && pretrain == builder.pretrain && useDropConnect == builder.useDropConnect
                && useRBMPropUpAsActivations == builder.useRBMPropUpAsActivations
                && !(confs != null ? !confs.equals(builder.confs) : builder.confs != null)
                && Arrays.equals(hiddenLayerSizes, builder.hiddenLayerSizes);

        }

        @Override
        public int hashCode() {
            int result;
            long temp;
            result = confs != null ? confs.hashCode() : 0;
            result = 31 * result + (hiddenLayerSizes != null ? Arrays.hashCode(hiddenLayerSizes) : 0);
            result = 31 * result + (useDropConnect ? 1 : 0);
            result = 31 * result + (pretrain ? 1 : 0);
            result = 31 * result + (useRBMPropUpAsActivations ? 1 : 0);
            temp = Double.doubleToLongBits(dampingFactor);
            result = 31 * result + (int) (temp ^ (temp >>> 32));
            result = 31 * result + (preProcessors != null ? preProcessors.hashCode() : 0);
            return result;
        }
    }




}
