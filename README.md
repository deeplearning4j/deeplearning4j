#ND4S: Scala bindings for ND4J

[![Join the chat at https://gitter.im/deeplearning4j/nd4s](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/deeplearning4j/nd4s?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

ND4S is open-source Scala bindings for [ND4J](https://github.com/deeplearning4j/nd4j). Released under an Apache 2.0 license. 

# Main Features
* NDArray manipulation syntax sugar with safer type.
* NDArray slicing syntax, similar with NumPy.

# Installation

## Install via Maven
ND4S is already included in official Maven repositories.

With IntelliJ, incorporation of ND4S is easy: just create a new Scala project, go to "Project Settings"/Libraries, add "From Maven...", and search for nd4s.

No need for git-cloning & compiling!

## Clone from the GitHub Repo
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

## Try ND4S in REPL
The easiest way to play ND4S around is cloning this repository and run the following command.

```
$ cd nd4s
$ sbt test:console
```

It starts REPL with importing `org.nd4s.Implicits._` and `org.nd4j.linalg.factory.Nd4j` automatically. It uses jblas backend at default.

```scala
scala> val arr = (1 to 9).asNDArray(3,3) 
arr: org.nd4j.linalg.api.ndarray.INDArray =
[[1.00,2.00,3.00]
 [4.00,5.00,6.00]
 [7.00,8.00,9.00]]

scala> val sub = arr(0->2,1->3)
sub: org.nd4j.linalg.api.ndarray.INDArray =
[[2.00,3.00]
 [5.00,6.00]]
```

#CheatSheet(WIP)

| ND4S syntax                                | Equivalent NumPy syntax                     | Result                                                         |
|--------------------------------------------|---------------------------------------------|----------------------------------------------------------------|
| Array(Array(1,2,3),Array(4,5,6)).toNDArray | np.array([[1, 2 , 3], [4, 5, 6]])           | [[1.0, 2.0, 3.0]  [4.0, 5.0, 6.0]]                             |
| val arr = (1 to 9).asNDArray(3,3)          | arr = np.array([[1, 2 , 3], [4, 5, 6],[7, 8, 9]]) | [[1.0, 2.0, 3.0]  [4.0, 5.0, 6.0] ,[7.0, 8.0, 9.0]]            |
| arr(0,0)                                   | arr[0,0]                                    | 1.0                                                            |
| arr(0,->)                                  | arr[0,:]                                    | [1.0, 2.0, 3.0]                                                |
| arr(--->)                                  | arr[...]                                    | [[1.0, 2.0, 3.0]   [4.0, 5.0, 6.0] ,[7.0, 8.0, 9.0]]           |
| arr(0 -> 3 by 2, ->)                       | arr[0:3:2,:]                                | [[1.0, 2.0, 3.0]  [7.0, 8.0, 9.0]]                             |
| arr(0 to 2 by 2, ->)                       | arr[0:3:2,:]                                | [[1.0, 2.0, 3.0] [7.0, 8.0, 9.0]]                              |
| arr.filter(_ > 3)                          |                                             | [[0.0, 0.0, 0.0]  [4.0, 5.0, 6.0] ,[7.0, 8.0, 9.0]]            |
| arr.map(_ % 3)                             |                                             | [[1.0, 2.0, 0.0] [1.0, 2.0, 0.0] ,[1.0, 2.0, 0.0]]             |
| arr.filterBit(_ < 4)                       |                                             | [[1.0, 1.0, 1.0] [0.0, 0.0, 0.0] ,[0.0, 0.0, 0.0]]             |
| arr + arr                                  | arr + arr                                   | [[2.0, 4.0, 6.0] [8.0, 10.0, 12.0] ,[14.0, 16.0, 18.0]]        |
| arr * arr                                  | arr * arr                                   | [[1.0, 4.0, 9.0] [16.0, 25.0, 36.0] ,[49.0, 64.0, 81.0]]       |
| arr dot arr                                | np.dot(arr, arr)                            | [[30.0, 36.0, 42.0] [66.0, 81.0, 96.0] ,[102.0, 126.0, 150.0]] |
| arr.sumT                                   | np.sum(arr)                                 | 45.0  //returns Double value                                   |
| val comp = Array(1 + i, 1 + 2 * i).toNDArray | comp = np.array([1 + 1j, 1 + 2j])           | [1.0 + 1.0i ,1.0 + 2.0i]                                       |
| comp.sumT                                  | np.sum(comp)                                | 2.0 + 3.0i //returns IComplexNumber value                      |
| for(row <- arr.rowP if row.get(0) > 1) yield row*2 |   | [[8.00,10.00,12.00] [14.00,16.00,18.00]] |
| val tensor = (1 to 8).asNDArray(2,2,2) | tensor = np.array([[[1, 2], [3, 4]],[[5,6],[7,8]]]) | [[[1.00,2.00] [3.00,4.00]] [[5.00,6.00] [7.00,8.00]]] |
| for(slice <- tensor.sliceP if slice.get(0) > 1) yield slice*2 |                           |[[[10.00,12.00][14.00,16.00]]] |
|arr(0 -> 3 by 2, ->) = 0                  |                                                | [[0.00,0.00,0.00] [4.00,5.00,6.00] [0.00,0.00,0.00]] |
