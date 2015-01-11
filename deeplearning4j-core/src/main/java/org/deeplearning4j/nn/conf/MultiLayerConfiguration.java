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

    private MultiLayerConfiguration() {}





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

    /**
     *
     * @return
     */
    public String toJson() {
        ObjectMapper mapper = NeuralNetConfiguration.mapper();
        try {
            return mapper.writeValueAsString(this).replaceAll("\"activationFunction\",", "").replaceAll("\"rng\",","").replaceAll("\"dist\",", "").replaceAll("\"stepFunction\",","").replaceAll("\"layerFactory\",","");
        } catch (com.fasterxml.jackson.core.JsonProcessingException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Create a neural net configuration from json
     * @param json the neural net configuration from json
     * @return
     */
    public static MultiLayerConfiguration fromJson(String json) {
        ObjectMapper mapper = NeuralNetConfiguration.mapper();
        try {
            return mapper.readValue(json, MultiLayerConfiguration.class);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
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
        public boolean equals(Object o) {
            if (this == o) return true;
            if (!(o instanceof Builder)) return false;

            Builder builder = (Builder) o;

            if (Double.compare(builder.dampingFactor, dampingFactor) != 0) return false;
            if (pretrain != builder.pretrain) return false;
            if (useDropConnect != builder.useDropConnect) return false;
            if (useRBMPropUpAsActivations != builder.useRBMPropUpAsActivations) return false;
            if (confs != null ? !confs.equals(builder.confs) : builder.confs != null) return false;
            if (!Arrays.equals(hiddenLayerSizes, builder.hiddenLayerSizes)) return false;
            if (preProcessors != null ? !preProcessors.equals(builder.preProcessors) : builder.preProcessors != null)
                return false;

            return true;
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
