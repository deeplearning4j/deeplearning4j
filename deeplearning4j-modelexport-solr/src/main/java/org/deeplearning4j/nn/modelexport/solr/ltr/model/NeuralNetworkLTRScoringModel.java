package org.deeplearning4j.nn.modelexport.solr.ltr.model;

import java.io.InputStream;
import java.io.IOException;
import java.util.List;
import java.util.Map;

import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.search.Explanation;
import org.apache.solr.core.SolrResourceLoader;
import org.apache.solr.ltr.feature.Feature;
import org.apache.solr.ltr.model.AdapterModel;
import org.apache.solr.ltr.model.ModelException;
import org.apache.solr.ltr.norm.Normalizer;
import org.deeplearning4j.nn.api.NeuralNetwork;

// TODO: change 7_2 to 7_3 once that is released
/**
 * An abstract <a href="https://lucene.apache.org/solr/7_2_0/solr-ltr/org/apache/solr/ltr/model/LTRScoringModel.html">
 * org.apache.solr.ltr.model.LTRScoringModel</a> that computes scores using a {@link NeuralNetwork}.
 * Concrete classes must implement the {@link #restoreNeuralNetwork(InputStream)} and {@link #score(float[])} methods.
 * <p>
 */
abstract public class NeuralNetworkLTRScoringModel extends AdapterModel {

  private String serializedModelFileName;
  protected NeuralNetwork neuralNetwork;

  public NeuralNetworkLTRScoringModel(String name, List<Feature> features, List<Normalizer> norms, String featureStoreName,
      List<Feature> allFeatures, Map<String,Object> params) {
    super(name, features, norms, featureStoreName, allFeatures, params);
  }

  public void setSerializedModelFileName(String serializedModelFileName) {
    this.serializedModelFileName = serializedModelFileName;
  }

  @Override
  public void init(SolrResourceLoader solrResourceLoader) throws ModelException {
    super.init(solrResourceLoader);
    try {
      neuralNetwork = restoreNeuralNetwork(openInputStream());
    } catch (IOException e) {
      throw new ModelException("Failed to restore model from given file (" + serializedModelFileName + ")", e);
    }
    validate();
  }

  protected InputStream openInputStream() throws IOException {
    return solrResourceLoader.openResource(serializedModelFileName);
  }

  abstract protected NeuralNetwork restoreNeuralNetwork(InputStream inputStream) throws IOException;

  @Override
  protected void validate() throws ModelException {
    super.validate();
    if (serializedModelFileName == null) {
      throw new ModelException("no serializedModelFileName configured for model "+name);
    }
    if (neuralNetwork != null) {
      validateNeuralNetwork();
    }
  }

  protected void validateNeuralNetwork() throws ModelException {
    try {
      final float[] mockModelFeatureValuesNormalized = new float[features.size()];
      score(mockModelFeatureValuesNormalized);
    } catch (Exception exception) {
      throw new ModelException("score(...) test failed for model "+name, exception);
    }
  }

  abstract public float score(float[] modelFeatureValuesNormalized);

  @Override
  public Explanation explain(LeafReaderContext context, int doc, float finalScore,
      List<Explanation> featureExplanations) {

    final StringBuilder sb = new StringBuilder();

    sb.append("(name=").append(getName());
    sb.append(",class=").append(getClass().getSimpleName());
    sb.append(",featureValues=[");
    for (int i = 0; i < featureExplanations.size(); i++) {
      Explanation featureExplain = featureExplanations.get(i);
      if (i > 0) {
        sb.append(',');
      }
      final String key = features.get(i).getName();
      sb.append(key).append('=').append(featureExplain.getValue());
    }
    sb.append("])");

    return Explanation.match(finalScore, sb.toString());
  }

}
