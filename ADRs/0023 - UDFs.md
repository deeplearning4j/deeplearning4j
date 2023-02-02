# UDFs

## Status

Implemented

Proposed by: Adam Gibson (31 Jan 2023)

Discussed with: Paul Dubs

Finalized by: Adam Gibson (2nd Feb 2023)


## Context

Users should be able to define their own custom operations in SameDiff, including custom gradients. 
Currently, defining a User-Defined Function (UDF) is not properly integrated into SameDiff and requires handling multiple aspects of the system, such as:

1. Registering the operation with the ImportClassMappings operation registry
2. Implementing special code paths in the operation executioners to support calling user-defined code
3. Implementing special code paths for serialization to correctly load a SameDiff graph with the operation
4. Making a direct function call in SameDiff to properly register the operation as part of the graph.

## Proposal 
To support custom UDFs in SameDiff, the following components will be created:

1. An operation type for the FlatBuffersMapper (used for saving graphs) to know how to save and load UDFs
2. A base class extending DynamicCustom with clear method and constructor overrides for defining a custom operation
3.  An execution method where the user passes in relevant inputs, which will be used by the operation executioner (instead of accessing the low-level code)
4. A hook in SameDiff to register the UDF, such as sd.udf(...)
5. An annotation or subclass scanner to discover and register user-defined operations in relevant areas, such as the ImportClassMappings.

These components work together to allow for the following:

1. Scanning for annotations to discover and register user-defined operations with the operation registry
2. Users extending the base class to create their custom UDFs
3. Integrating UDFs into SameDiff by registering the operation with the graph, for example:
java
```java
SameDiff sd = SameDiff.create();
UserDefinedCustomOp userDefinedCustomOp = ...;
SDVariable[] opOutputs = sd.doUdf(userDefinedCustomOp);
```

When an operation is registered, it is saved and loaded with the graph like any other operation.
Dynamic creation of operations via reflection when a graph is loaded, using the annotation scanning.

Below is an example:
```java
@UserDefinedOp // Annotation for discovering custom ops to register
public class TestAddUdf extends UserDefinedCustomOp { // Class to extend
    

    // Empty constructor. Used when creating a graph from flatbuffers in the underlying { org.nd4j.autodiff.samediff.serde.FlatBuffersMapper}.
    public TestAddUdf() {
        super();
    }

// Other constructors can be whatever the user wishes. Custom ops usually take in a
// SameDiff instance and one or more SDVariable args. These are the minimum components to instantiate an op.
// Each of these calls super(...) to properly configure the op to be used within the SameDiff graph passed in.

    public TestAddUdf(SameDiff sameDiff, SDVariable arg) {
        super(sameDiff, arg);
    }

    public TestAddUdf(SameDiff sameDiff, SDVariable[] args) {
        super(sameDiff, args);
    }

    // Used to calculate output variables when registering an op with a graph.
    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes) {
        // A user must implement this method. It is used in SameDiff to determine the number of 
        // output variables needed when it can't be determined from getNumOutputs().
        return Arrays.asList(dataTypes.get(0));
    }

    @Override
    public void setPropertiesForFunction(Map<String, Object> properties) {
        // A user can define properties as fields. If so, they must implement this method and propertiesForFunction().
        // These are used to create an op from scratch when saving/loading a model.
    }

    @Override
    public Map<String, Object> propertiesForFunction() {
        // Returns properties (fields on the java class) as a map. Properties can be any value that
        // is a field on the op itself. These properties are optional and may not be needed, 
        // depending on the op. All properties will end up being passed to the underlying iArguments, 
        // tArguments, and other associated data structures inherited from DynamicCustomOp.
        return Collections.emptyMap();
    }

    @Override
    public int getNumOutputs() {
        // Returns the number of outputs for the op. If an op has a variable number of outputs, 
        // a user will need to use an SDVariable.eval() call to return an int to determine the number of outputs.
        return 1;
    }

    @Override
    public String opName() {
        // The op name, required for proper registration with the registry.
        return "test_add_udf";
    }

    @Override
    public void configureFromArguments() {
        // A hook for configuring the op after creation. Used for configuration from specified arguments,
        // such as ints, floats/doubles, and input variables. The arguments referenced are the underlying 
        // arguments that get passed to every c/c++ ops, including iArguments, tArguments, dArguments,
        // inputArguments, and outputArguments.
    }

    @Override
    public void configureWithSameDiff(SameDiff sameDiff) {
        this.sameDiff = sameDiff;
        // Implemented this method for handling initialization after the op is created. It initiates values using relevant 
        // SameDiff metadata, such as obtaining input and output argument metadata from SDVariable found as args().
    }


    @Override
    public boolean isInplaceCall() {
        // Indicates whether the inputs are also the outputs.
        // Note that extra care should be taken to avoid bugs when an operation is in-place.
        // This is particularly important when an input to an operation is a view.
        return false;
    }

 

    @Override
    public List<LongShapeDescriptor> calculateOutputShape() {
        // Describes how to calculate the output shape based on the inputs. 
        // Note that calculateOutputShape is called when dynamically creating output arrays to store the result 
        // of an operation's execution. 
        // It is not called when an operation is in-place.
        return Arrays.asList(inputArguments.get(0).shapeDescriptor());
    }

    @Override
    public List<LongShapeDescriptor> calculateOutputShape(OpContext oc) {
        // Describes how to calculate the output shape based on the inputs from the operation context. 
        // Note that calculateOutputShape is called when dynamically creating output arrays to store 
        // the result of an operation's execution. 
        // This is different from the above method as the inputs are obtained from the operation
        // context instead of the operation itself. 
        // It is not called when an operation is in-place.
        return Arrays.asList(oc.getInputArrays().get(0).shapeDescriptor());
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        // The doDiff method must be implemented by the user if the operation is to be used for training. 
        // It should return one gradient for each input.
        return new AddBpOp(sameDiff, larg(), rarg(), f1.get(0)).outputs();
    }

    @Override
    public void exec() {
        // The exec method for the operation itself, consisting of operation execution 
        // and setting the outputs for the operation.
        AddOp addOp = new AddOp();
        addOp.addInputArgument(inputArguments.get(0), inputArguments.get(1));
        Nd4j.getExecutioner().exec(addOp);
        this.outputArguments.addAll(addOp.outputArguments);
    }

    @Override
    public void exec(OpContext opContext) {
        // The exec method for the operation itself, consisting of operation execution and 
        // setting the outputs for the operation context.
        Nd4j.getExecutioner().exec(new AddOp(), opContext);
    }
}

```

With the above definition, a user just has to pass in a created op as an instantiated object.
As long as an op is annotated it is properly integrated with the samediff graph.

When executing, the special code paths in the op executioners will call exec() or exec(opContext)



## Consequences

### Advantages

* Allows users to define their own ops to make optimizations or to introduce custom ops
for use within samediff
* Augments model import by combining annotation scanning from model import with 
annotation registration of udfs with similar annotations

### Disadvantages
* A bit lower level which means users can misuse the api or encounter bugs they might not 
otherwise with ops maintained in the core framework.