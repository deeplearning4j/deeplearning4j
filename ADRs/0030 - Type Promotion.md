# Publish a Smaller Artifact with Limited Type Support

## Status

**Proposed**

Proposed by: [Adam Gibson] Oct 22, 2024

## Context

The current C++ library published via Java Maven supports multi-type arithmetic, achieved through extensive template usage in C++ and built using CMake. While this approach provides flexibility by accommodating various data types, it results in a higher binary size. For many users, the full spectrum of type support may be unnecessary, leading to increased storage requirements and longer download times. To address these concerns, there is consideration to publish a smaller artifact that supports only specific types, thereby reducing the binary size.

## Proposal

We propose to publish a smaller Maven artifact of the C++ library that supports a limited set of data types. This specialized artifact will cater to users who do not require multi-type arithmetic, offering a more lightweight alternative. The key aspects of this proposal include:

1. **Create a Limited Type Support Artifact:**
    - Develop a separate build configuration using CMake that includes only the necessary type specializations.
    - Ensure that the limited artifact excludes type support beyond the specified types to minimize binary size.

2. **Maintain the Existing Multi-Type Artifact:**
    - Continue to offer the current multi-type arithmetic artifact for users who need comprehensive type support.

3. **Clear Naming and Versioning:**
    - Use distinct naming conventions (e.g., `mylib-core`, `mylib-lite`) to differentiate between the full and limited artifacts.
    - Align versioning strategies to ensure compatibility and ease of maintenance.

4. **Update Documentation and Support Materials:**
    - Clearly document the differences between the full and limited artifacts.
    - Provide guidelines on selecting the appropriate artifact based on user needs.

5. **Implement Testing for Both Artifacts:**
    - Develop separate test suites to validate the functionality and performance of each artifact.
    - Ensure that the limited artifact maintains the core functionality required by its target users.

### Example Test

As a sample test to ensure the limited artifact functions correctly, consider the following parameterized test:

```java
@ParameterizedTest
@MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
public void testMixedDataTypeViews(Nd4jBackend backend) {
    INDArray arrFloat = Nd4j.arange(24).reshape(4, 6).castTo(DataType.FLOAT);
    INDArray arrDouble = Nd4j.arange(24).reshape(4, 6).castTo(DataType.DOUBLE);
    INDArray arrLong = Nd4j.arange(24).reshape(4, 6).castTo(DataType.LONG);

    INDArray viewFloat = arrFloat.get(NDArrayIndex.interval(1, 3), NDArrayIndex.interval(2, 5));
    INDArray viewDouble = arrDouble.get(NDArrayIndex.interval(1, 3), NDArrayIndex.interval(2, 5));
    INDArray viewLong = arrLong.get(NDArrayIndex.interval(1, 3), NDArrayIndex.interval(2, 5));

    assertEquals(8.0f, viewFloat.getFloat(0, 0), 1e-5);
    assertEquals(16.0f, viewFloat.getFloat(1, 2), 1e-5);
    assertEquals(8.0, viewDouble.getDouble(0, 0), 1e-5);
    assertEquals(16.0, viewDouble.getDouble(1, 2), 1e-5);
    assertEquals(8L, viewLong.getLong(0, 0));
    assertEquals(16L, viewLong.getLong(1, 2));
}
``` 
```cpp
Key Macros for Type Promotion Approach
To manage type promotion effectively within the limited artifact, the following macros and templates are utilized:


/*
 * Type Ranking System:
 * type_rank template and its specializations assign an integer rank to each supported type.
 * This ranking helps in determining the "promoted" type when combining different types.
 * Type Promotion Traits:
 * promote_type and promote_type3 templates determine the promoted type between two or three types based on their ranks.
 * Type Name System:
 * type_name template and its specializations provide a string representation for each supported type.
 * Helper Functions and Macros:
 * promote function template converts a value to the promoted type.
 * Macros like INSTANTIATE_PROMOTE and CALLBACK_INSTANTIATE_PROMOTE help in instantiating the promote function for different type combinations.
 * PROMOTE_ARGS macro handles function arguments correctly.
 */

// Type ranking system
template<typename T> struct type_rank;

#if defined(HAS_BOOL)
template<> struct type_rank<bool>        : std::integral_constant<int, 0> {};
#endif

#if defined(HAS_INT8)
template<> struct type_rank<int8_t>      : std::integral_constant<int, 1> {};
#endif

#if defined(HAS_UINT8)
template<> struct type_rank<uint8_t>     : std::integral_constant<int, 1> {};
#endif

#if defined(HAS_INT16)
template<> struct type_rank<int16_t>     : std::integral_constant<int, 2> {};
#endif

#if defined(HAS_UINT16)
template<> struct type_rank<uint16_t>    : std::integral_constant<int, 2> {};
#endif

#if defined(HAS_INT32)
template<> struct type_rank<int32_t>     : std::integral_constant<int, 3> {};
#endif

#if defined(HAS_UINT32)
template<> struct type_rank<uint32_t>    : std::integral_constant<int, 3> {};
#endif

template<> struct type_rank<int64_t>     : std::integral_constant<int, 4> {};
template<> struct type_rank<long long int>   : std::integral_constant<int, 4> {};
template<> struct type_rank<uint64_t>    : std::integral_constant<int, 4> {};

#if defined(HAS_FLOAT16)
template<> struct type_rank<float16>     : std::integral_constant<int, 5> {};
#endif

#if defined(HAS_BFLOAT16)
template<> struct type_rank<bfloat16>    : std::integral_constant<int, 5> {};
#endif

#if defined(HAS_FLOAT32)
template<> struct type_rank<float>       : std::integral_constant<int, 6> {};
#endif

#if defined(HAS_DOUBLE)
template<> struct type_rank<double>      : std::integral_constant<int, 7> {};
#endif

// promote_type trait
template<typename T1, typename T2>
struct promote_type {
  using type = typename std::conditional<
      (type_rank<T1>::value >= type_rank<T2>::value),
      T1,
      T2
      >::type;
};

// promote function template
template <typename Type1, typename Type2, typename ValueType>
typename promote_type<Type1, Type2>::type promote(ValueType value) {
  return static_cast<typename promote_type<Type1, Type2>::type>(value);
}

// promote_type3 trait for three types
template<typename T1, typename T2, typename T3>
struct promote_type3 {
  using type = typename promote_type<
      typename promote_type<T1, T2>::type,
      T3
      >::type;
};

// Primary template for type_name - undefined to trigger a compile-time error for unsupported types
template<typename T>
struct type_name;

#if defined(HAS_BOOL)
template<> struct type_name<bool>        { static const char* get() { return "bool"; } };
#endif

#if defined(HAS_INT8)
template<> struct type_name<int8_t>      { static const char* get() { return "int8_t"; } };
#endif

#if defined(HAS_UINT8)
template<> struct type_name<uint8_t>     { static const char* get() { return "uint8_t"; } };
#endif

#if defined(HAS_INT16)
template<> struct type_name<int16_t>     { static const char* get() { return "int16_t"; } };
#endif

#if defined(HAS_UINT16)
template<> struct type_name<uint16_t>    { static const char* get() { return "uint16_t"; } };
#endif

#if defined(HAS_INT32)
template<> struct type_name<int32_t>     { static const char* get() { return "int32_t"; } };
#endif

#if defined(HAS_UINT32)
template<> struct type_name<uint32_t>    { static const char* get() { return "uint32_t"; } };
#endif

#if defined(HAS_INT64)
template<> struct type_name<int64_t>     { static const char* get() { return "int64_t"; } };
template<> struct type_name<long long int> { static const char* get() { return "long long int"; } };
#endif

#if defined(HAS_UINT64)
template<> struct type_name<uint64_t>    { static const char* get() { return "uint64_t"; } };
#endif

#if defined(HAS_FLOAT16)
template<> struct type_name<float16>     { static const char* get() { return "float16"; } };
#endif

#if defined(HAS_BFLOAT16)
template<> struct type_name<bfloat16>    { static const char* get() { return "bfloat16"; } };
#endif

#if defined(HAS_FLOAT32)
template<> struct type_name<float>       { static const char* get() { return "float"; } };
#endif

#if defined(HAS_DOUBLE)
template<> struct type_name<double>      { static const char* get() { return "double"; } };
#endif

// Helper function to get type name
template<typename T>
const char* get_type_name() {
  return type_name<T>::get();
}

// Macro to instantiate the promote function
#define INSTANTIATE_PROMOTE(a1, b1, FUNC_NAME, ARGS) \
    template promote_type<GET_SECOND(a1), GET_SECOND(b1)>::type \
    promote<GET_SECOND(a1), GET_SECOND(b1), GET_SECOND(a1)>(GET_SECOND(a1));

// Callback macro
#define CALLBACK_INSTANTIATE_PROMOTE(a1, b1, FUNC_NAME, ARGS) \
    INSTANTIATE_PROMOTE(a1, b1, FUNC_NAME, ARGS)

// Macro to define functions with advanced type promotion and debugging
#define SD_PROMOTE_FUNC(FUNC_NAME, BODY)                                \
template<typename T, typename U = T, typename Z = T>                    \
Z FUNC_NAME(T val1, U val2) {                                         \
    using calc_type = typename promote_type3<T, U, Z>::type;           \
    calc_type promoted_val1 = static_cast<calc_type>(val1);            \
    calc_type promoted_val2 = static_cast<calc_type>(val2);            \
    calc_type result = BODY;                                           \
    return static_cast<Z>(result);                                     \
}

template <typename T, typename Z>
SD_HOST_DEVICE SD_INLINE Z sd_abs(T value);

template <typename T, typename Z>
SD_HOST_DEVICE SD_INLINE Z sd_eq(T value, T value2, double eps) {
  return sd_abs<T, Z>(value - value2) <= eps;
}

template <typename T>
SD_HOST_DEVICE SD_INLINE void sd_swap(T& val1, T& val2);

SD_PROMOTE_FUNC(sd_max, (promoted_val1 > promoted_val2 ? promoted_val1 : promoted_val2))

SD_PROMOTE_FUNC(sd_min, (promoted_val1 < promoted_val2 ? promoted_val1 : promoted_val2))

SD_PROMOTE_FUNC(sd_add, (promoted_val1 + promoted_val2))
SD_PROMOTE_FUNC(sd_subtract, (promoted_val1 - promoted_val2))
SD_PROMOTE_FUNC(sd_multiply, (promoted_val1 * promoted_val2))
SD_PROMOTE_FUNC(sd_divide, (promoted_val1 / promoted_val2))
```

## Consequences

### Advantages

#### Reduced Binary Size
The smaller artifact consumes less storage and reduces download times, making it suitable for environments with limited resources or slow network connections.

#### Faster Compilation and Build Times
Limiting the supported types decreases the complexity of the codebase, leading to quicker compilation and faster build processes.

#### Simpler Maintenance
A streamlined codebase with fewer type specializations simplifies maintenance, allowing for easier bug fixes and feature enhancements.

#### Specific Target Audience
Tailoring the artifact to specific user needs ensures optimal performance and usability for those requiring only certain data types.

### Disadvantages

#### Fragmentation
Offering multiple artifacts can lead to user confusion regarding which version to use, complicating documentation and support.

#### Increased Maintenance Overhead
Maintaining separate artifacts requires additional effort to ensure consistency and compatibility across versions.

#### User Flexibility
Users may need to switch to the larger artifact in the future if their requirements evolve, potentially leading to compatibility issues.

#### Inconsistent Performance
Differences in optimization between artifacts may result in varying performance characteristics, causing confusion among users.

#### Dependency Management Complexity
Managing dependencies for multiple artifacts increases the risk of conflicts and integration issues within user projects.

### Risks

#### Introduction of Bugs
Refactoring to create a limited artifact may inadvertently introduce bugs, especially if the transition is not meticulously managed.

#### Performance Impacts
If the limited artifact is not properly optimized, it could underperform compared to expectations.

#### Developer Confusion
Developers accustomed to the multi-type system may find the limited artifact restrictive, leading to potential misuse or frustration.

#### Breaking Existing Code
Users relying on the full type support may experience disruptions if they transition to the limited artifact without proper migration.


## Conclusion

Publishing a smaller artifact with limited type support offers significant benefits 
in terms of reduced binary size, faster build times, and simplified maintenance. 


However, it also introduces challenges related to fragmentation, increased maintenance 
overhead, and potential user confusion. By carefully planning the implementation,
maintaining clear documentation, and providing robust support, the advantages can be 
leveraged while mitigating the disadvantages. 

This strategic approach ensures that the  library remains flexible and user-friendly, 
catering to a broader range of use cases 
without compromising on performance or usability.
