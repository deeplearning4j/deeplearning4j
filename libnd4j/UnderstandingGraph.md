# Graph

### Basic idea
libnd4j contains Directed Acyclic Graph execution engine, suited for both local and remote execution. However, main goal here is execution of externally originated graphs, serialized into FlatBuffers and provided either via pointer, or file.  


This basic example shows execution of graph loaded from file:
```c++
auto graph = GraphExecutioner<float>::importFromFlatBuffers("./some_file.fb");
GraphExecutioner<float>::execute(graph);
// ... do something with results ...
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
- Recursion isn't directly supported at this moment.
- CUDA isn't supported at this moment. _This limitation will be lifted soon._
- When used from C++, Graph only supports FeedForward mode. _This limitation will be lifted soon._

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

### Embedded profiling
If you're adding new ops, and want to make sure they run ok on your specific device - you might want to give a shot to embedded Graph profiling helper.
Despite being simple - it still provides you with time spent in various parts of Graph.

```c++
Environment::getInstance()->setProfiling(true);
auto graph = GraphExecutioner<float>::importFromFlatBuffers("./resources/ae_00.fb");

auto profile = GraphProfilingHelper<float>::profile(graph, 1000);
profile->printOut();

delete graph;
```

1000 iterations laterm you'll get statistics printed out. Statistics basically includes time spent in various parts of code and memory allocation details. 

Here's how it'll look like:
```
Printing out Graph...
8. matmul; Inputs: [{1:0}, {2:0}]; 
9. biasadd; Inputs: [{8:0}, {3:0}]; 
10. TRANSFORM:{15}; Inputs: [{9:0}]; 
11. rank; Inputs: [{2:0}]; 
12. subtract; Inputs: [{11:0}, {4:0}]; 
13. range; Inputs: [{5:0}, {11:0}, {6:0}]; 
14. subtract; Inputs: [{12:0}, {13:0}]; 
15. transpose; Inputs: [{2:0}, {14:0}]; 
16. matmul; Inputs: [{10:0}, {15:0}]; 
17. biasadd; Inputs: [{16:0}, {7:0}]; 
18. TRANSFORM:{15}; Inputs: [{17:0}]; 

Printing out Scopes...
Graph profile: 1000 executions

Memory:
ACT: 0; TMP: 0; OBJ: 0; TTL: 1788;

Time:
Construction time: 2135 ns;
Execution time: 41820 ns;

Per-node reports:
Node: <8:MatMul>
      Memory: ACT: 0; TMP: 0; OBJ: 0; TTL: 200;
      Time: PREP: 1160 ns; EXEC: 3167 ns; TTL: 5929 ns;
      PREP: INPUT: 251 ns; SHAPE: 382 ns; ARRAY: 217 ns;
Node: <9:BiasAdd>
      Memory: ACT: 0; TMP: 0; OBJ: 0; TTL: 104;
      Time: PREP: 917 ns; EXEC: 3580 ns; TTL: 5957 ns;
      PREP: INPUT: 220 ns; SHAPE: 213 ns; ARRAY: 217 ns;
Node: <10:Tanh>
      Memory: ACT: 0; TMP: 0; OBJ: 0; TTL: 104;
      Time: PREP: 756 ns; EXEC: 241 ns; TTL: 1927 ns;
      PREP: INPUT: 140 ns; SHAPE: 195 ns; ARRAY: 205 ns;
Node: <11:transpose/Rank>
      Memory: ACT: 0; TMP: 0; OBJ: 0; TTL: 36;
      Time: PREP: 522 ns; EXEC: 119 ns; TTL: 1403 ns;
      PREP: INPUT: 109 ns; SHAPE: 69 ns; ARRAY: 171 ns;
Node: <12:transpose/sub>
      Memory: ACT: 0; TMP: 0; OBJ: 0; TTL: 36;
      Time: PREP: 666 ns; EXEC: 185 ns; TTL: 1684 ns;
      PREP: INPUT: 192 ns; SHAPE: 94 ns; ARRAY: 168 ns;
Node: <13:transpose/Range>
      Memory: ACT: 0; TMP: 0; OBJ: 0; TTL: 556;
      Time: PREP: 808 ns; EXEC: 647 ns; TTL: 2416 ns;
      PREP: INPUT: 297 ns; SHAPE: 228 ns; ARRAY: 181 ns;
Node: <14:transpose/sub_1>
      Memory: ACT: 0; TMP: 0; OBJ: 0; TTL: 56;
      Time: PREP: 721 ns; EXEC: 541 ns; TTL: 2205 ns;
      PREP: INPUT: 23 ns; SHAPE: 92 ns; ARRAY: 165 ns;
Node: <15:transpose>
      Memory: ACT: 0; TMP: 0; OBJ: 0; TTL: 96;
      Time: PREP: 3936 ns; EXEC: 602 ns; TTL: 5811 ns;
      PREP: INPUT: 194 ns; SHAPE: 3241 ns; ARRAY: 257 ns;
Node: <16:MatMul_1>
      Memory: ACT: 0; TMP: 0; OBJ: 0; TTL: 312;
      Time: PREP: 970 ns; EXEC: 3565 ns; TTL: 6066 ns;
      PREP: INPUT: 203 ns; SHAPE: 320 ns; ARRAY: 193 ns;
Node: <17:BiasAdd_1>
      Memory: ACT: 0; TMP: 0; OBJ: 0; TTL: 144;
      Time: PREP: 914 ns; EXEC: 3528 ns; TTL: 5870 ns;
      PREP: INPUT: 231 ns; SHAPE: 191 ns; ARRAY: 223 ns;
Node: <18:output>
      Memory: ACT: 0; TMP: 0; OBJ: 0; TTL: 144;
      Time: PREP: 805 ns; EXEC: 285 ns; TTL: 1928 ns;
      PREP: INPUT: 157 ns; SHAPE: 192 ns; ARRAY: 232 ns;

Special timers:
No special timers were set
```


### Roadmap
In short-to-medium term following improvements are expected:
- CUDA support for all new ops
- Additional data types support: int, long long, q types, bool
- Sparse tensors support


