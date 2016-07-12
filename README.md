#gym-java-client

A java http client for [gym-http-api](https://github.com/openai/gym-http-api).

# quickstart

An example agent is provided:
```java
import org.deeplearning4j.gym.ExampleAgent;
...
ExampleAgent.run();
```

To create a new Client, use the ClientFactory. If the url is not localhost:5000, provide it as a second argument

```java
Client<Box, Integer> client = ClientFactory.build("CartPole-v0");
```

The type parameters of a client are an Observation and an Action. It enables to statically check the type of an observation for the user.

### Warning
Unfortunately because of java's limitation (type erasure), if you set the wrong type for Observation and/or Action corresponding to the Environment Id, since it is retrieved from the server at runtime, the code will fail at runtime only when you cast an Observation or Action (when you actually retrieve one). If you get a cast error like that, it is the reason.


The methods nomenclature follows closely the api interface of gym-http-api:

```java

//static methods
static Set<String> listAll(String url)
static void serverShutdown(String url)


//methods accessible from a client instance (no need for instanceId or url, how convenient :)
String getInstanceId()
String getEnvId()
String getUrl()
ObservationSpace<O> getObservationSpace()
ActionSpace<A> getActionSpace() {
Set<String> listAll()
void reset()
void monitorStart(String directory, boolean force, boolean resume)
void monitorStart(String directory)
void monitorClose()
void close()
void upload(String trainingDir, String apiKey, String algorithmId)
void upload(String trainingDir, String apiKey)
void ServerShutdown() {
StepReply<O> step(A action)
```
