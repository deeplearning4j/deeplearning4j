package org.deeplearning4j.nn.modelexport.solr.handler;

import java.io.File;
import java.io.InputStream;
import java.io.IOException;
import java.util.List;
import org.apache.solr.client.solrj.io.Tuple;
import org.apache.solr.client.solrj.io.comp.StreamComparator;
import org.apache.solr.client.solrj.io.stream.StreamContext;
import org.apache.solr.client.solrj.io.stream.TupleStream;
import org.apache.solr.client.solrj.io.stream.expr.Explanation;
import org.apache.solr.client.solrj.io.stream.expr.Expressible;
import org.apache.solr.client.solrj.io.stream.expr.StreamExpression;
import org.apache.solr.client.solrj.io.stream.expr.StreamExpressionNamedParameter;
import org.apache.solr.client.solrj.io.stream.expr.StreamExpressionParameter;
import org.apache.solr.client.solrj.io.stream.expr.StreamExpressionValue;
import org.apache.solr.client.solrj.io.stream.expr.StreamFactory;
import org.apache.solr.core.SolrResourceLoader;
import org.apache.solr.handler.SolrDefaultStreamFactory; // TODO: https://issues.apache.org/jira/browse/SOLR-12402
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.modelexport.solr.ltr.model.ScoringModel; // TODO: temporary only
import org.deeplearning4j.util.ModelGuesser;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * A <a href="https://lucene.apache.org/solr/7_3_1/solr-solrj/org/apache/solr/client/solrj/io/stream/TupleStream.html">
 * org.apache.solr.client.solrj.io.stream.TupleStream</a> that uses a {@link Model} to compute output scores.
 * <a href="https://lucene.apache.org/solr/7_3_1/solr-solrj/org/apache/solr/client/solrj/io/Tuple.html">Tuple</a>
 * fields are the model inputs and the model output(s) are added to the returned tuple.
 * <p>
 * Illustrative configuration snippet:
 * <pre>
  &lt;lst name="streamFunctions"&gt;
    &lt;str name="emailModel"&gt;org.deeplearning4j.nn.modelexport.solr.handler.ModelTupleStream&lt;/str&gt;
  &lt;/lst&gt;
</pre>
 * <p>
 * Illustrative expression snippet:
 * <pre>
  emailModel(search(myCollection,
                    q="*:*",
                    fl="id,fieldX,fieldY,fieldZ",
                    sort="id asc",
                    qt="/export"),
             serializedModelFileName="mySerializedModel",
             inputKeys="fieldX,fieldY,fieldZ",
             outputKeys="modelScoreField1,modelScoreField2")
</pre>
 * <p>
 * Apache Solr Reference Guide:
 * <ul>
 * <li> <a href="https://lucene.apache.org/solr/guide/7_3/streaming-expressions.html">Streaming Expressions</a>
 * </ul>
 */
public class ModelTupleStream extends TupleStream implements Expressible {

  final private TupleStream tupleStream;
  final private String serializedModelFileName;
  final private String[] inputKeys;
  final private String[] outputKeys;
  final private SolrResourceLoader solrResourceLoader;
  final private Model model;

  public ModelTupleStream(StreamExpression streamExpression, StreamFactory streamFactory) throws IOException {

    final List<StreamExpression> streamExpressions = streamFactory.getExpressionOperandsRepresentingTypes(streamExpression, Expressible.class, TupleStream.class);
    if (streamExpressions.size() == 1) {
      this.tupleStream = streamFactory.constructStream(streamExpressions.get(0));
    } else {
      throw new IOException("TODO");
    }

    this.serializedModelFileName = getOperandValue(streamExpression, streamFactory, "serializedModelFileName");

    this.inputKeys = getOperandValue(streamExpression, streamFactory, "inputKeys").split(",");
    this.outputKeys = getOperandValue(streamExpression, streamFactory, "outputKeys").split(",");

    if (!(streamFactory instanceof SolrDefaultStreamFactory)) {
      throw new IOException("TODO");
    }
    this.solrResourceLoader = ((SolrDefaultStreamFactory)streamFactory).getSolrResourceLoader();

    this.model = restoreModel(openInputStream());
  }

  private static String getOperandValue(StreamExpression streamExpression, StreamFactory streamFactory, String operandName) throws IOException {
    final StreamExpressionNamedParameter namedParameter = streamFactory.getNamedOperand(streamExpression, operandName);
    if (namedParameter != null && namedParameter.getParameter() instanceof StreamExpressionValue) {
      return ((StreamExpressionValue)namedParameter.getParameter()).getValue();
    } else {
      throw new IOException("TODO");
    }
  }

  public void setStreamContext(StreamContext streamContext) {
    tupleStream.setStreamContext(streamContext);
  }

  public List<TupleStream> children() {
    return tupleStream.children();
  }

  public void open() throws IOException {
    tupleStream.open();
  }

  public void close() throws IOException {
    tupleStream.close();
  }

  public Tuple read() throws IOException {
    final Tuple tuple = tupleStream.read();
    if (tuple.EOF) {
      return tuple;
    } else {
      final INDArray inputs = getInputsFromTuple(tuple);
      final INDArray outputs = ScoringModel.output(model, inputs);
      return applyOutputsToTuple(tuple, outputs);
    }
  }

  public StreamComparator getStreamSort() {
    return tupleStream.getStreamSort();
  }

  public Explanation toExplanation(StreamFactory streamFactory) throws IOException {
    return null; // TODO
  }

  public StreamExpressionParameter toExpression(StreamFactory streamFactory) throws IOException {
    return null; // TODO
  }

  protected InputStream openInputStream() throws IOException {
    return solrResourceLoader.openResource(serializedModelFileName);
  }

  /**
   * Uses the {@link ModelGuesser#loadModelGuess(InputStream)} method.
   */
  protected Model restoreModel(InputStream inputStream) throws IOException {
    final File instanceDir = solrResourceLoader.getInstancePath().toFile();
    try {
      return ModelGuesser.loadModelGuess(inputStream, instanceDir);
    } catch (Exception e) {
      throw new IOException("Failed to restore model from given file (" + serializedModelFileName + ")", e);
    }
  }

  protected INDArray getInputsFromTuple(Tuple tuple) {
    final double[] inputs = new double[inputKeys.length];
    for (int ii=0; ii<inputKeys.length; ++ii)
    {
      inputs[ii] = tuple.getDouble(inputKeys[ii]).doubleValue();
    }
    return Nd4j.create(inputs);
  }

  protected Tuple applyOutputsToTuple(Tuple tuple, INDArray output) {
    for (int ii=0; ii<outputKeys.length; ++ii)
    {
      tuple.put(outputKeys[ii], output.getFloat(ii));
    }
    return tuple;
  }

}
