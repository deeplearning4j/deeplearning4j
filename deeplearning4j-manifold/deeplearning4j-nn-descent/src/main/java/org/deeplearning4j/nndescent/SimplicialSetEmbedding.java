package org.deeplearning4j.nndescent;

import lombok.Builder;
import org.nd4j.linalg.api.ndarray.INDArray;

@Builder
public class SimplicialSetEmbedding {
   private INDArray graph;
   private int nComponents;
   private double initialAlpha;
   private double alpha;
   private double a;
   private double b;
   private double gamma;
   private double negativeSampleRate;
   private int nEpochs;

}
