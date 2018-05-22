# gym-java-client

A java http client for [gym-http-api](https://github.com/openai/gym-http-api).

Note: If you are encountering errors as reported in [issue #13](https://github.com/deeplearning4j/gym-java-client/issues/13), please execute the following command before launching `python gym_http_server.py`:

```bash
$ sudo sysctl -w net.ipv4.tcp_tw_recycle=1
```

# Quickstart

To create a new Client, use the ClientFactory. If the url is not localhost:5000, provide it as a second argument

```java
Client<Box, Integer, DiscreteSpace> client = ClientFactory.build("CartPole-v0");
```

"CartPole-v0" is the name of the gym environment.

The type parameters of a client are the Observation type, the Action type, the Observation Space type and the ActionSpace type.

It is a bit cumbersome to both declare an ActionSpace and an Action since an ActionSpace knows what type is an Action but unfortunately java does't support type member and path dependant types.

Here we use Box and BoxSpace for the environment and Integer and Discrete Space because it is how [CartPole-v0](https://gym.openai.com/envs/CartPole-v0) is specified.

The methods nomenclature follows closely the api interface of [gym-http-api](https://github.com/openai/gym-http-api#api-specification), O is Observation an A is Action:

```java
//Static methods

/**
 * @param url url of the server
 * @return set of all environments running on the server at the url
 */
public static Set<String> listAll(String url);

/**
 * Shutdown the server at the url
 *
 * @param url url of the server
 */
public static void serverShutdown(String url);



//Methods accessible from a Client
/**
 * @return set of all environments running on the same server than this client
 */
public Set<String> listAll();

/**
 * Step the environment by one action
 *
 * @param action action to step the environment with
 * @return the StepReply containing the next observation, the reward, if it is a terminal state and optional information.
 */
public StepReply<O> step(A action);
/**
 * Reset the state of the environment and return an initial observation.
 *
 * @return initial observation
 */
public O reset();

/**
 * Start monitoring.
 *
 * @param directory path to directory in which store the monitoring file
 * @param force     clear out existing training data from this directory (by deleting every file prefixed with "openaigym.")
 * @param resume    retain the training data already in this directory, which will be merged with our new data
 */
public void monitorStart(String directory, boolean force, boolean resume);

/**
 * Flush all monitor data to disk
 */
public void monitorClose();

/**
 * Upload monitoring data to OpenAI servers.
 *
 * @param trainingDir directory that contains the monitoring data
 * @param apiKey      personal OpenAI API key
 * @param algorithmId an arbitrary string indicating the paricular version of the algorithm (including choices of parameters) you are running.
 **/
public void upload(String trainingDir, String apiKey, String algorithmId);

/**
 * Upload monitoring data to OpenAI servers.
 *
 * @param trainingDir directory that contains the monitoring data
 * @param apiKey      personal OpenAI API key
 */
public void upload(String trainingDir, String apiKey);


/**
 * Shutdown the server at the same url than this client
 */
public void serverShutdown()

```

## TODO

* Add all ObservationSpace and ActionSpace when they will be available.
