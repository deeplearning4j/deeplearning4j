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
Client<Integer, Integer> client = ClientFactory.build("CartPole-v0");
```

The type parameters of a client are an Observation and an Action. It enables to statically check the type of an observation and an action later in the code.


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
