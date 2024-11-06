# ADR-0031: Handling Type Combinations and Template Instantiations with Macros

## Status

**Proposed**

Proposed by: Adam Gibson Oct 22, 2024

## Context

In the current C++ library published via Java Maven, multi-type arithmetic is supported through extensive use of templates and built using CMake. While this approach offers flexibility by accommodating various data types, it leads to an increased binary size. Many users do not require the full spectrum of type support, resulting in unnecessary storage consumption and longer download times. To optimize performance and reduce the binary footprint, there is a need to streamline type combinations and template instantiations.

## Decision

To efficiently manage type combinations and template instantiations within the limited artifact, a series of preprocessor macros are employed. These macros automate the retrieval and processing of type lists, enabling systematic instantiation of template functions based on various type combinations. Below is a detailed explanation of the key macros and their functionalities:

### GET Macro

```cpp
#define GET(n, list) CAT(GET_, n) list

#define GET_0(t1, ...) t1
#define GET_1(t1, t2, ...) t2
#define GET_2(t1, t2, t3, ...) t3
#define GET_3(t1, t2, t3, t4, ...) t4
#define GET_4(t1, t2, t3, t4, t5, ...) t5
#define GET_5(t1, t2, t3, t4, t5, t6, ...) t6
#define GET_6(t1, t2, t3, t4, t5, t6, t7, ...) t7
#define GET_7(t1, t2, t3, t4, t5, t6, t7, t8, ...) t8
#define GET_8(t1, t2, t3, t4, t5, t6, t7, t8, t9, ...) t9
#define GET_9(t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, ...) t10
#define GET_10(t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, ...) t11
#define GET_11(t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, ...) t12
#define GET_12(t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, ...) t13
#define GET_13(t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, ...) t14
#define GET_14(t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, ...) t15
#define GET_15(t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, t16, ...) t16
Explanation:

The GET macro retrieves the nth element from a provided list by concatenating GET_ with the index n and applying it to the list.
Each GET_n macro is defined to extract the nth element from a variadic list of parameters.
GET_FIRST and GET_SECOND Macros
```cpp
#define GET_FIRST(tuple) GET_FIRST_IMPL tuple
#define GET_FIRST_IMPL(a, b) a

#define GET_SECOND(tuple) GET_SECOND_IMPL tuple
#define GET_SECOND_IMPL(a, b) b
```
Explanation:

GET_FIRST and GET_SECOND macros extract the first and second elements from a tuple, respectively.
They expand the tuple and apply the corresponding implementation macros to retrieve the desired element.
PROCESS_COMBINATION and CALLBACK_PROCESS_COMBINATION Macros
```cpp
#define PROCESS_COMBINATION(a1, b1, a2, b2, FUNC_NAME, ARGS) \
    std::cout << "(" << a1 << ", " << b1 << ", " << a2 << ", " << b2 << ")\n";

#define CALLBACK_PROCESS_COMBINATION(outer, inner, FUNC_NAME, ARGS) \
    PROCESS_COMBINATION(GET_FIRST(outer), GET_SECOND(outer), GET_FIRST(inner), GET_SECOND(inner), FUNC_NAME, ARGS)
```


Explanation:

PROCESS_COMBINATION defines how to process a combination of types. In this example, it simply prints the combination.
CALLBACK_PROCESS_COMBINATION retrieves elements from the outer and inner lists and passes them to PROCESS_COMBINATION.
INNER_LOOP Macros
```cpp

#define INNER_LOOP_1(OUTER_ELEMENT, INNER_LIST, CALLBACK, FUNC_NAME, ARGS) \
    CALLBACK(OUTER_ELEMENT, GET(0, INNER_LIST), FUNC_NAME, ARGS)

#define INNER_LOOP_2(OUTER_ELEMENT, INNER_LIST, CALLBACK, FUNC_NAME, ARGS) \
    INNER_LOOP_1(OUTER_ELEMENT, INNER_LIST, CALLBACK, FUNC_NAME, ARGS) \
    CALLBACK(OUTER_ELEMENT, GET(1, INNER_LIST), FUNC_NAME, ARGS)

#define INNER_LOOP_3(OUTER_ELEMENT, INNER_LIST, CALLBACK, FUNC_NAME, ARGS) \
    INNER_LOOP_2(OUTER_ELEMENT, INNER_LIST, CALLBACK, FUNC_NAME, ARGS) \
    CALLBACK(OUTER_ELEMENT, GET(2, INNER_LIST), FUNC_NAME, ARGS)

#define INNER_LOOP_4(OUTER_ELEMENT, INNER_LIST, CALLBACK, FUNC_NAME, ARGS) \
    INNER_LOOP_3(OUTER_ELEMENT, INNER_LIST, CALLBACK, FUNC_NAME, ARGS) \
    CALLBACK(OUTER_ELEMENT, GET(3, INNER_LIST), FUNC_NAME, ARGS)

#define INNER_LOOP_5(OUTER_ELEMENT, INNER_LIST, CALLBACK, FUNC_NAME, ARGS) \
    INNER_LOOP_4(OUTER_ELEMENT, INNER_LIST, CALLBACK, FUNC_NAME, ARGS) \
    CALLBACK(OUTER_ELEMENT, GET(4, INNER_LIST), FUNC_NAME, ARGS)
```


// ... Similarly up to INNER_LOOP_16
Explanation:

INNER_LOOP_n macros iterate over the INNER_LIST up to the nth element.
Each iteration applies the CALLBACK macro to process the current combination.
OUTER_LOOP Macros
```cpp
#define OUTER_LOOP_1(OUTER_LIST, INNER_LIST, INNER_SIZE, CALLBACK, FUNC_NAME, ARGS) \
    CAT(INNER_LOOP_, INNER_SIZE)(GET(0, OUTER_LIST), INNER_LIST, CALLBACK, FUNC_NAME, ARGS)

#define OUTER_LOOP_2(OUTER_LIST, INNER_LIST, INNER_SIZE, CALLBACK, FUNC_NAME, ARGS) \
    OUTER_LOOP_1(OUTER_LIST, INNER_LIST, INNER_SIZE, CALLBACK, FUNC_NAME, ARGS) \
    CAT(INNER_LOOP_, INNER_SIZE)(GET(1, OUTER_LIST), INNER_LIST, CALLBACK, FUNC_NAME, ARGS)

#define OUTER_LOOP_3(OUTER_LIST, INNER_LIST, INNER_SIZE, CALLBACK, FUNC_NAME, ARGS) \
    OUTER_LOOP_2(OUTER_LIST, INNER_LIST, INNER_SIZE, CALLBACK, FUNC_NAME, ARGS) \
    CAT(INNER_LOOP_, INNER_SIZE)(GET(2, OUTER_LIST), INNER_LIST, CALLBACK, FUNC_NAME, ARGS)
```
// ... Similarly up to OUTER_LOOP_16
Explanation:

OUTER_LOOP_n macros iterate over the OUTER_LIST up to the nth element.
For each element in the OUTER_LIST, they invoke the corresponding INNER_LOOP_n macro to process combinations with the INNER_LIST.
ITERATE_COMBINATIONS Macro
```cpp
#define ITERATE_COMBINATIONS(OUTER_LIST, INNER_LIST, CALLBACK, FUNC_NAME, ARGS) \
    CAT(OUTER_LOOP_, PP_NARGS(EXPAND OUTER_LIST))(OUTER_LIST, INNER_LIST, PP_NARGS(EXPAND INNER_LIST), CALLBACK, FUNC_NAME, ARGS)
```
Explanation:

ITERATE_COMBINATIONS initiates the nested iteration over OUTER_LIST and INNER_LIST.
It determines the number of elements in the OUTER_LIST and INNER_LIST using PP_NARGS (a macro to count arguments) and dispatches to the appropriate OUTER_LOOP_n macro.
Template Instantiation Macros

```cpp

#define INSTANT_PROCESS_COMBINATION(a1, b1, FUNC_NAME, ARGS) \
    template void FUNC_NAME<GET_SECOND(a1), GET_SECOND(b1)>ARGS;

#define INSTANT_PROCESS_COMBINATION_3(a1, b1, c1, FUNC_NAME, ARGS) \
    template void FUNC_NAME<GET_SECOND(a1), GET_SECOND(b1), GET_SECOND(c1)>ARGS;

#define INSTANT_PROCESS_COMBINATION_CLASS(a1, b1, FUNC_NAME, ARGS) \
    template class FUNC_NAME<GET_SECOND(a1), GET_SECOND(b1)>ARGS;

#define INSTANT_PROCESS_COMBINATION_CLASS_3(a1, b1, c1, FUNC_NAME, ARGS) \
    extern template class FUNC_NAME<GET_SECOND(a1), GET_SECOND(b1), GET_SECOND(c1)>ARGS;
```

Explanation:

These macros instantiate template functions with specific type combinations extracted from type lists.
INSTANT_PROCESS_COMBINATION handles two-type combinations, while INSTANT_PROCESS_COMBINATION_3 handles three-type combinations.
Similarly, class templates can be instantiated or declared extern using the corresponding macros.
ITERATE_COMBINATIONS_3 Macro
cpp
Copy code
#define ITERATE_COMBINATIONS_3(OUTER_LIST, MIDDLE_LIST, INNER_LIST, CALLBACK, FUNC_NAME, ARGS) \
    CAT(OUTER_LOOP_, CAT(PP_NARGS(EXPAND OUTER_LIST), _3))(OUTER_LIST, MIDDLE_LIST, INNER_LIST, PP_NARGS(EXPAND MIDDLE_LIST), PP_NARGS(EXPAND INNER_LIST), CALLBACK, FUNC_NAME, ARGS)
Explanation:

ITERATE_COMBINATIONS_3 iterates over three type lists (SD_COMMON_TYPES) and applies the INSTANT_PROCESS_COMBINATION_3 macro to each combination.
This results in the instantiation of the PairWiseTransform::exec function for each combination of types.
Usage Example
The following example demonstrates how the macros are used to instantiate template functions based on combinations of types:

```cpp
/*
 *
 *
ITERATE_COMBINATIONS_3: This macro iterates over three lists of data types (SD_COMMON_TYPES) and applies the INSTANT_PROCESS_COMBINATION_3 macro to each combination. This results in the instantiation of the PairWiseTransform::exec function for each combination of data types.
ITERATE_COMBINATIONS: This macro iterates over two lists of data types (SD_COMMON_TYPES) and applies the CALLBACK_INSTANTIATE_PROMOTE macro to each combination. This is likely used for promoting data types.
Function Instantiation:
The PairWiseTransform::exec function is instantiated for various combinations of data types. The function signature includes parameters for operation number (opNum), input arrays (x, y), their shape information (xShapeInfo, yShapeInfo), output array (z), its shape information (zShapeInfo), extra parameters (extraParams), and the range of elements to process (start, stop).
 */
ITERATE_COMBINATIONS_3(
    (SD_COMMON_TYPES),
    (SD_COMMON_TYPES),
    (SD_COMMON_TYPES),
    INSTANT_PROCESS_COMBINATION_3,
    functions::pairwise_transforms::PairWiseTransform,
    ::exec(int opNum, const void *x, const sd::LongType *xShapeInfo, const void *y,
           const sd::LongType *yShapeInfo, void *z, const sd::LongType *zShapeInfo,
           void *extraParams, sd::LongType start, sd::LongType stop)
)

ITERATE_COMBINATIONS(
    (SD_COMMON_TYPES),
    (SD_COMMON_TYPES),
    CALLBACK_INSTANTIATE_PROMOTE,
    promote,
    ;
)
```

Explanation:

ITERATE_COMBINATIONS_3 iterates over three type lists (SD_COMMON_TYPES) and instantiates the PairWiseTransform::exec function for each combination of types.
ITERATE_COMBINATIONS iterates over two type lists and applies the CALLBACK_INSTANTIATE_PROMOTE macro to handle type promotion.
Summary
The macros defined above provide a systematic approach to handling multiple type combinations and template instantiations. By automating the retrieval and processing of type lists, the codebase ensures that all necessary template instances are generated without manual intervention. This not only reduces the potential for errors but also streamlines the maintenance and scalability of the library.

Benefits:

Automation: Reduces the need for repetitive code by automating template instantiations.
Scalability: Easily handles a large number of type combinations without increasing code complexity.
Maintainability: Simplifies updates and maintenance by centralizing type handling logic within macros.
Considerations:

Complexity: Macros can be difficult to debug and understand, especially for those unfamiliar with advanced preprocessor techniques.
Compilation Time: Extensive use of templates and macros may lead to longer compilation times.
By incorporating these macros into the limited artifact, the library achieves a balance between flexibility and efficiency, ensuring that only necessary type combinations are included, thereby reducing binary size and optimizing performance.

# Consequences

## Advantages

### Flexibility in Compile-Time Code Generation
- **Automated Template Instantiation:** The combination macros enable the automatic generation of multiple template instantiations during compile time. This reduces the need for repetitive manual code, ensuring that all necessary type combinations are systematically covered.
- **Dynamic Type Support:** By leveraging macros, the library can easily support new data types without significant changes to the core codebase. Adding or modifying type combinations becomes a matter of updating macro definitions, enhancing the library's adaptability.
- **Consistent Code Patterns:** Macros ensure that template instantiations follow a uniform pattern, minimizing discrepancies and maintaining consistency across different type combinations. This uniformity simplifies the integration of new functionalities and type supports.

## Disadvantages

### Increased Code Complexity
- **Complex Macro Definitions:** The use of advanced preprocessor macros introduces a layer of complexity that can be challenging to understand and manage. Developers unfamiliar with intricate macro techniques may find the codebase harder to navigate and modify.
- **Obfuscated Code Flow:** Macros can obscure the actual code being generated, making it difficult to trace the flow of template instantiations. This obscurity can complicate debugging efforts and hinder the comprehension of how different type combinations are handled.
- **Maintenance Challenges:** Updating or extending macros to accommodate new type combinations requires careful adjustments to prevent introducing bugs. The intricate nature of macro-based code generation demands meticulous attention, increasing the maintenance overhead.
- **Limited Tooling Support:** Many development tools and IDEs offer limited support for macro-heavy code, reducing the effectiveness of features like code completion, refactoring, and error highlighting. This limitation can slow down development and increase the likelihood of unnoticed issues.


## Conclusion

Publishing a smaller artifact with limited type support offers significant benefits,
including reduced binary size, faster build times, and simplified maintenance.
However, it also brings challenges such as fragmentation, increased maintenance
overhead, and potential user confusion. By carefully planning the implementation,
maintaining clear documentation, and providing robust support, the advantages can
be maximized while mitigating the disadvantages. This strategic approach ensures
that the library remains flexible and user-friendly, catering to a broader range
of use cases without compromising on performance or usability.
