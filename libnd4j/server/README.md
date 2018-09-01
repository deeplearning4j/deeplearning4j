## GraphServing

We provide minimalistic binary capable of serving inference requests to SameDiff graphs via gRPC.
Idea is simple: you start GraphServer, optionally providing path to serialized graph, and you can immediately start sending inference requests.
You can also update graphs in runtime (i.e. if you've got new model version, or want to do some A/B testing).

## Configuration

There's not too much to configure:

```
-p 40123 // port number to be used
-f filename.fb // path to flatbuffers file with serialized SameDiff graph
```

## gRPC endpoints

GraphServer at this moment has 4 endpoints:
- RegisterGraph(FlatGraph)
- ReplaceGraph(FlatGraph)
- ForgetGraph(FlatDropRequest)
- InferenceRequest(FlatInferenceRequest)

#### RegisterGraph(FlatGraph)
This endpoint must be used if you want to add graph to the serving process. GraphServer instance can easily handle more then 1 graph for inference requests.

#### ReplaceGraph(FlatGraph)
This endpoint must be used if you want to update model used for serving in safe way. However, keep in mind, if new graph expects different structure of inputs/outputs - you might want to add it with new ID instead.

#### ForgetGraph(FlatDropRequest)
This endpoint must be used if you want to remove graph from serving for any reason.

#### InferenceRequest(FlatInferenceRequest)
This endpoint must be used for actual inference requests. You send inputs in, and get outputs back. Simple as that.


## Docker & K8S

We provide basic Dockerfile, which allows to build Docker image with GraphServer. Image is based on Ubuntu 18.04, and has pretty small footprint.