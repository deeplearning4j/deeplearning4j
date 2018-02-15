package org.deeplearning4j.nn.modelexport.solr.ltr.model;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import org.apache.lucene.search.Explanation;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.solr.core.SolrResourceLoader;
import org.apache.solr.ltr.feature.Feature;
import org.apache.solr.ltr.feature.FeatureException;
import org.apache.solr.ltr.model.AdapterModel;
import org.apache.solr.ltr.model.ModelException;
import org.apache.solr.ltr.norm.IdentityNormalizer;
import org.apache.solr.ltr.norm.Normalizer;
import org.apache.solr.request.SolrQueryRequest;
import org.deeplearning4j.nn.api.NeuralNetwork;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;
import org.junit.Test;

abstract public class NeuralNetworkLTRScoringModelTest {

  protected static class DummyFeature extends Feature {

    public DummyFeature(String name) {
      super(name, Collections.EMPTY_MAP);
    }

    @Override
    protected void validate() throws FeatureException {
    }

    @Override
    public FeatureWeight createWeight(IndexSearcher searcher, boolean needsScores, SolrQueryRequest request,
        Query originalQuery, Map<String,String[]> efi) throws IOException {
      return null;
    }

    @Override
    public LinkedHashMap<String,Object> paramsToMap() {
      return null;
    }

  };

  protected List<Feature> featuresList(int numFeatures) throws Exception {
    final ArrayList<Feature> features = new ArrayList<Feature>();
    for (int ii=1; ii<=numFeatures; ++ii)
    {
      features.add(new DummyFeature("dummy"+ii));
    }
    return features;
  }

  protected List<Normalizer> normalizersList(int numNormalizers) throws Exception {
    final ArrayList<Normalizer> normalizers = new ArrayList<Normalizer>();
    for (int ii=1; ii<=numNormalizers; ++ii)
    {
      normalizers.add(IdentityNormalizer.INSTANCE);
    }
    return normalizers;
  }

  protected List<float[]> floatsList(int numFloats) {
    final List<float[]> floatsList = new ArrayList<float[]>();
    final float[] floats0 = new float[numFloats];
    final float[] floats1 = new float[numFloats];
    for (int ii=0; ii<numFloats; ++ii) {
      floats0[ii] = 0f;
      floats1[ii] = 1f;
    }
    floatsList.add(floats0);
    floatsList.add(floats1);
    return floatsList;
  }

  abstract protected NeuralNetwork buildAndSaveModel(int numFeatures, String serializedModelFileName) throws Exception;

  abstract protected NeuralNetwork restoreModel(String serializedModelFileName) throws Exception;

  abstract protected AdapterModel newAdapterModel(String name, List<Feature> features, List<Normalizer> norms, String serializedModelFileName) throws Exception;

  abstract protected float score(NeuralNetwork neuralNetwork, float[] vals) throws Exception;

  @Test
  public void test() throws Exception {

    final int numFeatures = 3;

    final Path tempDirPath = Files.createTempDirectory(null);
    final File tempDirFile = tempDirPath.toFile();
    tempDirFile.deleteOnExit();

    final SolrResourceLoader solrResourceLoader = new SolrResourceLoader(tempDirPath);

    final File tempFile = File.createTempFile("prefix", "suffix", tempDirFile);
    tempFile.deleteOnExit();

    final String serializedModelFileName = tempFile.getPath();

    final NeuralNetwork originalModel = buildAndSaveModel(numFeatures, serializedModelFileName);

    final NeuralNetwork restoredModel = restoreModel(serializedModelFileName);

    final AdapterModel ltrModel = newAdapterModel("myModel", featuresList(numFeatures), normalizersList(numFeatures), serializedModelFileName);
    ltrModel.init(solrResourceLoader);

    for (final float[] floats : floatsList(numFeatures)) {

      final float originalScore = score(originalModel, floats);
      final float restoredScore = score(restoredModel, floats);
      final float ltrScore = ltrModel.score(floats);

      assertEquals(originalScore, restoredScore, 0f);
      assertEquals(originalScore, ltrScore, 0f);

      final List<Explanation> explanations = new ArrayList<Explanation>();
      explanations.add(Explanation.match(floats[0], ""));
      explanations.add(Explanation.match(floats[1], ""));
      explanations.add(Explanation.match(floats[2], ""));

      final Explanation explanation = ltrModel.explain(null, 0, ltrScore, explanations);
      assertEquals(ltrScore+" = (name=myModel"+
          ",class="+ltrModel.getClass().getSimpleName()+
          ",featureValues="+
          "[dummy1="+Float.toString(floats[0])+
          ",dummy2="+Float.toString(floats[1])+
          ",dummy3="+Float.toString(floats[2])+
          "])\n",
          explanation.toString());
    }

    final AdapterModel invalidLtrModel = newAdapterModel("invalidModel", featuresList(numFeatures+1), normalizersList(numFeatures+1), serializedModelFileName);
    try {
      invalidLtrModel.init(solrResourceLoader);
      fail("expected to exception from invalid model init");
    } catch (ModelException me) {
      assertTrue(me.getMessage().startsWith("score(...) test failed for model "));
    }

  }

}
