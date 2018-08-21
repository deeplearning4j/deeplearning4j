package org.deeplearning4j.nn.modelexport.solr.handler;

import java.io.File;
import java.io.InputStream;
import java.io.IOException;
import java.util.List;
import java.util.Map;
import org.apache.solr.client.solrj.io.Tuple;
import org.apache.solr.client.solrj.io.comp.StreamComparator;
import org.apache.solr.client.solrj.io.stream.StreamContext;
import org.apache.solr.client.solrj.io.stream.TupleStream;
import org.apache.solr.client.solrj.io.stream.expr.Explanation.ExpressionType;
import org.apache.solr.client.solrj.io.stream.expr.Explanation;
import org.apache.solr.client.solrj.io.stream.expr.Expressible;
import org.apache.solr.client.solrj.io.stream.expr.StreamExplanation;
import org.apache.solr.client.solrj.io.stream.expr.StreamExpression;
import org.apache.solr.client.solrj.io.stream.expr.StreamExpressionNamedParameter;
import org.apache.solr.client.solrj.io.stream.expr.StreamExpressionParameter;
import org.apache.solr.client.solrj.io.stream.expr.StreamExpressionValue;
import org.apache.solr.client.solrj.io.stream.expr.StreamFactory;
import org.apache.solr.core.SolrResourceLoader;
import org.apache.solr.handler.SolrDefaultStreamFactory;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.util.ModelGuesser;
import org.deeplearning4j.util.NetworkUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * A <a href="https://lucene.apache.org/solr/7_5_0/solr-solrj/org/apache/solr/client/solrj/io/stream/TupleStream.html">
 * org.apache.solr.client.solrj.io.stream.TupleStream</a> that uses a {@link Model} to compute output scores.
 * <a href="https://lucene.apache.org/solr/7_5_0/solr-solrj/org/apache/solr/client/solrj/io/Tuple.html">Tuple</a>
 * fields are the model inputs and the model output(s) are added to the returned tuple.
 * <p>
 * Illustrative configuration snippet:
 * <pre>
  &lt;expressible name="emailModel" class="org.deeplearning4j.nn.modelexport.solr.handler.ModelTupleStream"/&gt;
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
 * <li> <a href="https://lucene.apache.org/solr/guide/7_5/streaming-expressions.html">Streaming Expressions</a>
 * </ul>
 */
public class ModelTupleStream extends TupleStream implements Expressible {

  public Map toMap(Map<String, Object> map) {
    // We (ModelTupleStream) extend TupleStream which implements MapWriter which extends MapSerializable.
    // MapSerializable says to have a toMap method.
    // org.apache.solr.common.MapWriter has a toMap method which has 'default' visibility.
    // So MapWriter.toMap here is not 'visible' but it is 'callable' it seems.
    return super.toMap(map);
  }

  final private static String SERIALIZED_MODEL_FILE_NAME_PARAM = "serializedModelFileName";
  final private static String INPUT_KEYS_PARAM = "inputKeys";
  final private static String OUTPUT_KEYS_PARAM = "outputKeys";

  final private TupleStream tupleStream;
  final private String serializedModelFileName;
  final private String inputKeysParam;
  final private String outputKeysParam;
  final private String[] inputKeys;
  final private String[] outputKeys;
  final private SolrResourceLoader solrResourceLoader;
  final private Model model;

  public ModelTupleStream(StreamExpression streamExpression, StreamFactory streamFactory) throws IOException {

    final List<StreamExpression> streamExpressions = streamFactory.getExpressionOperandsRepresentingTypes(streamExpression, Expressible.class, TupleStream.class);
    if (streamExpressions.size() == 1) {
      this.tupleStream = streamFactory.constructStream(streamExpressions.get(0));
    } else {
      throw new IOException("Expected exactly one stream in expression: "+streamExpression);
    }

    this.serializedModelFileName = getOperandValue(streamExpression, streamFactory, SERIALIZED_MODEL_FILE_NAME_PARAM);

    this.inputKeysParam = getOperandValue(streamExpression, streamFactory, INPUT_KEYS_PARAM);
    this.inputKeys = inputKeysParam.split(",");

    this.outputKeysParam = getOperandValue(streamExpression, streamFactory, OUTPUT_KEYS_PARAM);
    this.outputKeys = outputKeysParam.split(",");

    if (!(streamFactory instanceof SolrDefaultStreamFactory)) {
      throw new IOException(this.getClass().getName()+" requires a "+SolrDefaultStreamFactory.class.getName()+" StreamFactory");
    }
    this.solrResourceLoader = ((SolrDefaultStreamFactory)streamFactory).getSolrResourceLoader();

    this.model = restoreModel(openInputStream());
  }

  private static String getOperandValue(StreamExpression streamExpression, StreamFactory streamFactory, String operandName) throws IOException {
    final StreamExpressionNamedParameter namedParameter = streamFactory.getNamedOperand(streamExpression, operandName);
    String operandValue = null;
    if (namedParameter != null && namedParameter.getParameter() instanceof StreamExpressionValue) {
      operandValue = ((StreamExpressionValue)namedParameter.getParameter()).getValue();
    }
    if (operandValue == null) {
      throw new IOException("Expected '"+operandName+"' in expression: "+streamExpression);
    } else {
      return operandValue;
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
      final INDArray outputs = NetworkUtils.output(model, inputs);
      return applyOutputsToTuple(tuple, outputs);
    }
  }

  public StreamComparator getStreamSort() {
    return tupleStream.getStreamSort();
  }

  public Explanation toExplanation(StreamFactory streamFactory) throws IOException {
    return new StreamExplanation(getStreamNodeId().toString())
      .withChildren(new Explanation[]{
        tupleStream.toExplanation(streamFactory)
      })
      .withExpressionType(ExpressionType.STREAM_DECORATOR)
      .withFunctionName(streamFactory.getFunctionName(this.getClass()))
      .withImplementingClass(this.getClass().getName())
      .withExpression(toExpression(streamFactory, false).toString());
  }

  public StreamExpressionParameter toExpression(StreamFactory streamFactory) throws IOException {
    return toExpression(streamFactory, true /* includeStreams */);
  }

  private StreamExpression toExpression(StreamFactory streamFactory, boolean includeStreams) throws IOException {
    final String functionName = streamFactory.getFunctionName(this.getClass());
    final StreamExpression streamExpression = new StreamExpression(functionName);

    if (includeStreams) {
      if (this.tupleStream instanceof Expressible) {
        streamExpression.addParameter(((Expressible)this.tupleStream).toExpression(streamFactory));
      } else {
        throw new IOException("This "+this.getClass().getName()+" contains a non-Expressible TupleStream "+this.tupleStream.getClass().getName());
      }
    } else {
      streamExpression.addParameter("<stream>");
    }

    streamExpression.addParameter(new StreamExpressionNamedParameter(SERIALIZED_MODEL_FILE_NAME_PARAM, this.serializedModelFileName));
    streamExpression.addParameter(new StreamExpressionNamedParameter(INPUT_KEYS_PARAM, this.inputKeysParam));
    streamExpression.addParameter(new StreamExpressionNamedParameter(OUTPUT_KEYS_PARAM, this.outputKeysParam));

    return streamExpression;
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
