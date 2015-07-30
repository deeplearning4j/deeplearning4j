#ND4S: Scala bindings for ND4J

ND4S is an Apache2 Licensed open-sourced Scala bindings for [ND4J](https://github.com/deeplearning4j/nd4j).

#Main Features
* NDArray manipulation syntax sugar with safer type.
* NDArray slicing syntax, similar with NumPy.

#Installation

##Clone from the GitHub Repo

ND4S is actively developed. You can clone the repository, compile it, and reference it in your project.

Clone the repository:

```
$ git clone https://github.com/deeplearning4j/nd4s.git
```

Compile the project:

```
$ cd nd4s
$ sbt +publish-local
```

#CheatSheet

| ND4S syntax                                | Equivalent NumPy syntax           | Result                                               | Explanation                                                                                             |
|--------------------------------------------|-----------------------------------|------------------------------------------------------|---------------------------------------------------------------------------------------------------------|
| Array(Array(1,2,3),Array(4,5,6)).toNDArray | np.array([[1, 2 , 3], [4, 5, 6]]) | [[1.0, 2.0, 3.0] [4.0, 5.0, 6.0]]                    |                                                                                                         |
| val arr = (1 to 9).asNDArray(3,3)          |                                   | [[1.0, 2.0, 3.0] [4.0, 5.0, 6.0] ,[7.0, 8.0, 9.0]]   | NDArray can be built from Range.                                                                        |
| arr(0,0)                                   | arr[0,0]                          | 1.0                                                  | Value at the specified cell                                                                             |
| arr(0,->)                                  | arr[0,:]                          | [1.0, 2.0, 3.0]                                      | -> means all indices in specified dimension.                                                            |
| arr(--->)                                  | arr[...]                          | [[1.0, 2.0, 3.0] [4.0, 5.0, 6.0] ,[7.0, 8.0, 9.0]]   | ---> means ellipsis of dimensions                                                                       |
| arr(0 -> 3 by 2, ->)                       | arr[0:3:2,:]                      | [[1.0, 2.0, 3.0] [7.0, 8.0, 9.0]]                    | start -> end(exclusive) syntax shows indices to be sliced. `by` is optional to specify step of indices. |
| arr(0 to 2 by 2, ->)                       |                                   | [[1.0, 2.0, 3.0] [7.0, 8.0, 9.0]]                    | Range can be used as well.                                                                              |
| arr.filter(_ > 3)                          |                                   | [[0.0, 0.0, 0.0] [4.0, 5.0, 6.0] ,[7.0, 8.0, 9.0]]   |                                                                                                         |
| arr.map(_ % 3)                             |                                   | [[1.0, 2.0, 0.0] [1.0, 2.0, 0.0] ,[1.0, 2.0, 0.0]]   |                                                                                                         |
| arr.filterBit(_ < 4)                       |                                   | [[1.0, 1.0, 1.0] [0.0, 0.0, 0.0] ,[0.0, 0.0, 0.0]]   |                                                                                                         |