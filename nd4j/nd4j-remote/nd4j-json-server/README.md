## SameDiff model serving

This modules provides JSON-based serving of SameDiff models

## Example

First of all we'll create server instance. Most probably you'll do it in application that will be running in container
```java
val server = SameDiffJsonModelServer.<String, Sentiment>builder()
                .adapter(new StringToSentimentAdapter())
                .model(mySameDiffModel)
                .port(8080)
                .serializer(new SentimentSerializer())
                .deserializer(new StringDeserializer())
                .build();

server.start();
server.join();
```

Now, presumably in some other container, we'll set up remote inference client:
```java
val client = JsonRemoteInference.<String, Sentiment>builder()
                .endpointAddress("http://youraddress:8080/v1/serving")
                .serializer(new StringSerializer())
                .deserializer(new SentimentDeserializer())
                .build();

Sentiment result = client.predict(myText);
```
 On top of that, there's async call available, for cases when you need to chain multiple requests to one or multiple remote model servers.
 
```java
Future<Sentiment> result = client.predictAsync(myText);
```