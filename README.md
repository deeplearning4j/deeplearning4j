#gym-java-client

A java http client for [gym-http-api](https://github.com/openai/gym-http-api).

# quickstart

An example agent is provided:
```java
import org.deeplearning4j.ExampleAgent;
...
ExampleAgent.run();
```

To create a new Client, use the ClientFactory. If the url is not localhost:5000, provide it as a second argument

```java
        Client<Box, Integer, BoxSpace, DiscreteSpace> client = ClientFactory.build("CartPole-v0");
```

The type parameters of a client are the Observation, the Action, the Observation Space and the ActionSpace. It is a bit cumbersome to both declare an ActionSpace and an Action since an ActionSpace knows what type is an Action but unfortunately java does't support type member and path dependant types.
### Warning
Unfortunately because of java's limitation (type erasure), if you set the wrong type for Observation and/or Action corresponding to the Environment Id, since it is retrieved from the server at runtime, the code will fail at runtime only when you cast an Observation or Action (when you actually retrieve one). If you get a cast error like that, it is the reason.


The methods nomenclature follows closely the api interface of gym-http-api, O is Observation an A is Action:

```java

//static methods
static Set<String> listAll(String url)
static void serverShutdown(String url)


//methods accessible from a client instance (no need for instanceId or url, how convenient :)
String getInstanceId()
String getEnvId()
String getUrl()
OS getObservationSpace()
AS getActionSpace()
Set<String> listAll()
O reset()
void monitorStart(String directory, boolean force, boolean resume)
void monitorClose()
void close()
void upload(String trainingDir, String apiKey, String algorithmId)
void upload(String trainingDir, String apiKey)
void ServerShutdown() {
StepReply<O> step(A action)
```

## TODO

* Add all ObservationSpace and ActionSpace.
* Having les cumbersome type parameters.
