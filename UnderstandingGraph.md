# Graph

### Basic idea
libnd4j contains Directed Acyclic Graph execution engine, suited for both local and remote execution. However, main goal here is execution of externally originated graphs, serialized into FlatBuffers and provided either via pointer, or file.  


This basic example shows execution of graph loaded from file:
```c++
auto graph = GraphExecutioner<float>::importFromFlatBuffers("./some_file.fb");
GraphExecutioner<float>::execute(graph);
...
delete graph;
```

### FlatBuffers schemas
You can find scheme files [here](https://github.com/deeplearning4j/libnd4j/tree/master/include/graph/scheme).

At this moment libnd4j repo contains compiled definitions for C++, Python, Java, and JSON, but FlatBuffers can be compiled for PHP, C#, JavaScript, TypeScript and Go as well. Please refer to `flatc` instructions to do that.

Such bindings allow you to build FlatBuffers files/buffers suitable for remote execution of your graph and obtaining results back. I.e. you can use JavaScript to build graph (or just update variables/placeholders), send them to remote RPC server powered by libnd4j, and get results back.

### Graph execution logic
No matter how graph is represented on the front-end, on backend it's rather simple: topologically sorted list of operations executed sequentially if there's shared dependencies, or (optionally) in parallel, if there's no shared dependencies for current graph nodes.

Each node in the graph represents single linear algebra operation applied to input(s) of the node. For example: `z = Add(x, y)` is operation that takes 2 NDArrays as input, and produes 1 NDArray as output. So, graph is built of such primitive operations, which are executed sequentially. 

### Memory management within graph
Everything that happens within graph during execution, stays within VariableSpace. It acts as storage for Variables and NDArrays produced during graph execution. On top of that, there's an option to use pre-allocated Workspaces for allocation of NDArrays.


### Current graph limitations
There are some limitations. Some of them will be lifted eventually, others won't be. Here's the list:
- Graph has single data type. I.e. Graph&lt;float&gt; or Graph&lt;float16&gt; or Graph&lt;double&gt; _This limitation will be lifted soon._
- On some platforms, like Java, single Variable/Placeholder size is limited to 2GB buffer size. However, on libnd4j side there's no such limitation.
- Variable size/dimensionality has limitations: max NDArray rank is limited to 32 at this moment, and any single dimension is limited to MAX_INT size. 
- Recursion isn't supported at this moment.
- CUDA isn't supported at this moment. _This limitation will be lifted soon._

### Minified Graph binaries
There's an option to build minified binaries suited for execution of ***specific graphs***. Idea is quite simple: you feed your existing Graph(s) in FlatBuffers format into special app, which extracts operations used in your Graph(s) and excludes all other operations from target binary.
```bash
# building full libnd4j copy AND minfier app
./buildnativeoperations.sh -a native -m 
...
# building libnd4j for 2 specific graphs
./minifier -l -a native -o libnd4j_special ../some_path/some_graph1.fb ../some_path/some_graph2.fb
Option 'l': Build library
Option 'a': Target arch: native
Option 'o': Output file name is libnd4j_special
Total available operations: 423

Retrieving ops from the Graph and collect them...

Collecting out Scopes...
Operations found so far:
rank
range
subtract
transpose
matmul
biasadd
TRANSFORM{15}

Building minified library...
``` 

Once `minifier` finishes - you'll have `libnd4j_special.so` and `libnd4j_special.h` files ready, and they'll contain only those operations used in 2 graphs provided at compilation time + basic primitives used to work with Graph. Things like NDArray, GraphExecutioner etc will be included as well.

This library can be used in your application as any other shared libray out there: you'll include headers file and you'll be able to call for things you need. 

### Documentation 
Documentation for individual operations, and basic classes (like NDArray, Graph etc) is available as part of Nd4j javadoc: https://nd4j.org/doc/



