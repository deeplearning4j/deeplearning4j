There's multiple different Ops designs supported in libND4j, and in this guide we'll try to explain how to build your very own operation.

## XYZ operations

This kind of operations is actually split into multiple subtypes, based on element-access and result type:
- Transform operations: These operations typically take some NDArray in, and change each element independent of others.
- Reduction operations: These operations take some NDArray and dimensions, and return reduced NDArray (or scalar) back. I.e. sum along dimension(s).
- Scalar operations: These operations are similar to transforms, but they only do arithmetic operations, and second operand is scalar. I.e. each element in given NDArray will add given scalar value.
- Pairwise operations:  These operations are between regular transform opeartions and scalar operations. I.e. element-wise addition of two NDArrays.
- Random operations: Most of these operations related to random numbers distributions: Uniform, Gauss, Bernoulli etc.

Despite differences between these operations, they are all using XZ/XYZ three-operand design, where X and Y are inputs, and Z is output.
Data access in these operations is usually trivial, and loop based. I.e. most trivial loop for scalar transform will look like this:
```c++
for (Nd4jLong i = start; i < end; i++) {
    result[i] = OpType::op(x[i], scalar, extraParams);
}
```

Operation used in this loop will be template-driven, and compiled statically. There are another loops implementation, depending on op group or strides within NDArrays, but idea will be the same all the time: each element of the NDArray will be accessed within loop.

Now, let's take a look into typical XYZ op implementation. Here's how `Add` operation will look like:

```c++

template<typename T>
class Add {
public:
    op_def static T op(T d1, T d2) {
	    return d1 + d2;
	}

    // this signature will be used in Scalar loops
	op_def static T op(T d1, T d2, T *params) {
		return d1 + d2;
	}

    // this signature will be used in reductions
	op_def static T op(T d1) {
		return d1;
	}

	// op for MetaOps
	op_def static T op(T d1, T *params) {
		return d1 + params[0];
	}
};
```

This particular operation is used in different XYZ op groups, but you see the idea: element-wise operation, which is invoked on each element in given NDArray.
So, if you want to add new XYZ operation to libnd4j, you should just add operation implementation to file `includes/ops/ops.h`, and assign it to specific ops group in file `includes/loops/legacy_ops.h` together with some number unique to this ops group, i.e.: `(21, simdOps::Add)`

After libnd4j is recompiled, this op will become available for legacy execution mechanism, NDArray wrappers, and `LegacyOp` wrappers (those are made to map legacy operations to CustomOps design for Graph).


## Custom operations

Custom operations is a new concept, added recently and mostly suits SameDiff/Graph needs.
For CustomOps we defined universal signature, with variable number of input/output NDArrays, and variable number of floating-point and integer arguments.
However, there are some minor difference between various CustomOp declarations:
- **DECLARE_OP**(string, int, int, bool): these operations take no fp/int arguments, and output shape equals to input shape.
- **DECLARE_CONFIGURABLE_OP**(string, int, int, bool, int, int): these operations do take fp/int output arguments, and output shape equals to input shape.
- **DECLARE_REDUCTION_OP**(string, int, int, bool, int, int): these operations do take fp/int output arguments, and output shape is calculated as Reduction.
- **DECLARE_CUSTOM_OP**(string, int, int, bool, int, int): these operations return NDArray with custom shape, that usually depends on input and arguments.
- **DECLARE_BOOLEAN_OP**(string, int, bool): these operations take some NDArrays and return scalar, where 0 is **False**, and other values are treated as **True**.

Let's take a look at example CustomOp:

```c++

CUSTOM_OP_IMPL(tear, 1, -1, false, 0, -1) {
    auto input = INPUT_VARIABLE(0);

    REQUIRE_TRUE(!block.getIArguments()->empty(), 0, "At least 1 dimension should be specified for Tear");

    std::vector<int> dims(*block.getIArguments());

    for (auto &v: dims)
        REQUIRE_TRUE(v >= 0 && v < input->rankOf(), 0, "Tear dimensions should be non-negative values, and lower then input rank. Got %i instead", v);

    auto tads = input->allTensorsAlongDimension(dims);
    for (int e = 0; e < tads->size(); e++) {
        auto outE = OUTPUT_VARIABLE(e);
        outE->assign(tads->at(e));

        this->storeResult(block, e, *outE);
    }

    delete tads;

    return ND4J_STATUS_OK;
}

DECLARE_SHAPE_FN(tear) {
    auto inShape = inputShape->at(0);

    std::vector<int> dims(*block.getIArguments());
    
    if (dims.size() > 1)
        std::sort(dims.begin(), dims.end());

    shape::TAD tad(inShape, dims.data(), (int) dims.size());
    tad.createTadOnlyShapeInfo();
    Nd4jLong numTads = shape::tadLength(inShape, dims.data(), (int) dims.size());

    auto result = SHAPELIST();
    for (int e = 0; e < numTads; e++) {
        int *newShape;
        COPY_SHAPE(tad.tadOnlyShapeInfo, newShape);
        result->push_back(newShape);
    }

    return result;
}
```

In the example above, we declare `tear` CustomOp implementation, and shape function for this op.
So, at the moment of op execution, we assume that we will either have output array(s) provided by end-user, or they will be generated with shape function.

You can also see number of macros used, we'll cover those later as well. Beyond that - op execution logic is fairly simple & linear:
Each new op implements protected member function `DeclarableOp<T>::validateAndExecute(Block<T>& block)`, and this method is eventually called either from GraphExecutioner, or via direct call, like `DeclarableOp<T>::execute(Block<T>& block)`.

Important part of op declaration is input/output description for the op. I.e. as shown above: `CUSTOM_OP_IMPL(tear, 1, -1, false, 0, -1)`.
This declaration means: 
- Op name: `tear`
- Op expects at least 1 NDArray as input
- Op returns unknown positive number of NDArrays as output
- Op can't be run in-place, so under any circumstances original NDArray will stay intact
- Op doesn't expect any T (aka floating point) arguments
- Op expects unknown positive number of integer arguments. In case of this op it's dimensions to split input NDArray.

Here's another example: `DECLARE_CUSTOM_OP(permute, 1, 1, true, 0, -2);`
This declaration means:
- Op name: `permute`
- Op expects at least 1 NDArray as input
- Op returns 1 NDArray as output
- Op can be run in-place if needed (it means: input == output, and input is modified and returned as output)
- Op doesn't expect any T arguments
- Op expects unknown number of integer arguments OR no integer arguments at all.

## c++11 syntactic sugar

In ops you can easily use c++11 features, including lambdas. In some cases it might be easiest way to build your custom op (or some part of it) via `NDArray::applyLambda` or `NDArray::applyPairwiseLambda`:
```c++
auto lambda = LAMBDA_TT(_x, _y) {
    return (_x + _y) * 2;
};

x.applyPairwiseLambda(&y, lambda);
``` 

In this simple example, each element of NDArray `x` will get values set to `x[e] = (x[e] + y[e]) * 2`.

## Tests

For tests libnd4j uses Google Tests suit. All tests are located at `tests_cpu/layers_tests` folder. Here's simple way to run those from command line:
```
cd tests_cpu
cmake -G "Unix Makefiles"
make -j 4
./layers_tests/runtests
```

You can also use your IDE (i.e. Jetbrains CLion) to run tests via GUI.

**PLEASE NOTE:** if you're considering submitting your new op to libnd4j repository via pull request - consider adding tests for it. Ops without tests won't be approved.

## Backend-specific operation

GPU/MPI/whatever to be added soon.


## Utility macros
We have number of utility macros, suitable for custom ops. Here they are:
- **INPUT_VARIABLE**(int): this macro returns you NDArray at specified input index.
- **OUTPUT_VARIABLE**(int): this macro returns you NDArray at specified output index.
- **STORE_RESULT**(NDArray<T>): this macro stores result to VariableSpace.
- **STORE_2_RESULTS**(NDArray<T>, NDArray<T>): this macro stores results accordingly to VariableSpace.
- **INT_ARG**(int): this macro returns you specific Integer argument passed to the given op.
- **T_ARG**(int): this macro returns you specific T argument passed to the given op.
- **ALLOCATE**(...): this macro check if Workspace is available, and either uses Workspace or direct memory allocation if Workspace isn't available.
- **RELEASE**(...): this macro is made to release memory allocated with **ALLOCATE()** macro.
- **REQUIRE_TRUE**(...): this macro takes condition, and evaluates it. If evaluation doesn't end up as True - exception is raised, and specified message is printed out.
- **LAMBDA_T**(X) and **LAMBDA_TT**(X, Y): lambda declaration for `NDArray::applyLambda` and `NDArray::applyPairwiseLambda`
- **COPY_SHAPE**(SRC, TGT): this macro allocates memory for TGT pointer and copies shape from SRC pointer 
- **ILAMBDA_T**(X) and **ILAMBDA_TT**(X, Y): lambda declaration for indexed lambdas, index argument is passed in as Nd4jLong (aka **long long**)
- **FORCEINLINE**: platform-specific definition for functions inlining
