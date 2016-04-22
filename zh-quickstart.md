---
title: "必备项"
layout: zh-default
---

快速入门指南
=========================================

## 必备项

本快速入门指南假设您已将下述各项安装就绪：

1. Java 7或以上
2. IntelliJ（或其他IDE）
3. Maven（自动化生成工具）
4. Github

若您需要安装上述任意一项，请阅读[ND4J入门指南](http://nd4j.org/zh-getstarted.html)。（ND4J是我们用于深度学习的科学计算引擎，其入门指南中的内容对两个项目均适用。）本页中的样例只需安装上面列出的四项，无需安装“ND4J入门指南”中的所有软件。

建议您加入我们的[Gitter线上交流群](https://gitter.im/deeplearning4j/deeplearning4j)以便提问或反馈。即使您加入后无意发言，也可默默潜水学习。此外，如果您对深度学习刚刚入门，我们还准备了一份[上手时需要学习的清单](../deeplearningforbeginners.html)。

Deeplearning4j是开源项目，意在吸引熟悉应用程序部署、IntelliJ等IDE，以及Maven等生成工具的专业Java开发者。如果您已熟悉上述工具，则更能对我们提供的工具驾轻就熟。

## 几步部署DL4J

必备项安装完成后，只需根据下面的步骤即可（Windows系统用户请见下面[流程](#walk)部分）：

* 在命令行中输入`git clone https://github.com/deeplearning4j/dl4j-0.4-examples.git`。（目前示例版本为0.0.4.x。）
* 在IntelliJ中使用Maven创建新项目，并指向上述示例的根目录。
* 复制、粘贴下列代码，确保您的POM.xml文件与[此文件](https://github.com/deeplearning4j/dl4j-0.4-examples/blob/master/pom.xml)一致。
* 附加的[Windows系统说明可通过此链接访问](../zh-gettingstarted.html#windows)。
* 从左侧文件列表中选择`DBNIrisExample.java`。
* 单击运行！（也即右键单击源文件时出现的绿色按钮……）

### 提示

* 请确保您未将其他存储库克隆至本地。鉴于对Deeplearning4j主存储库的改进持续进行，最新版可能尚未对示例进行充分测试。
* 请确保示例的所有依赖项均从Maven下载得到，而非本地文件。`(rm -rf  ls ~/.m2/repository/org/deeplearning4j)`
* 在dl4j-0.4-examples目录运行`mvn clean install -DskipTests=true -Dmaven.javadoc.skip=true`，确保正确安装。
* 运行TSNE或其他示例请输入`mvn exec:java -Dexec.mainClass="org.deeplearning4j.examples.tsne.TSNEStandardExample" -Dexec.cleanupDaemonThreads=false`。如果执行失败，或Maven在退出时无法终止守护进程，则可能需要上述参数。
* 1000次迭代后，`tsne-standard-coords.csv`应被置于`dl4j-0.4-examples/target/archive-tmp/`目录。

您应获得0.66上下、对Iris这类小型数据组来说较好的F1值。示例的逐行解释请见[Iris DBN教程](../iris-flower-dataset-tutorial.html)。

如发生问题，应首先检查POM.xml文件。该文件应与[此文件](https://github.com/deeplearning4j/dl4j-0.4-examples/blob/master/pom.xml)一致。

## 依赖项和后端

后端用于驱动DL4J神经网络背后的线性代数运算。后端根据芯片而不同。就CPU而言，x86速度最快；对GPU而言则是Jcublas。可在[Maven中心](https://search.maven.org)找到所有后端。点击“最新版本”下的版本号；复制下一页左侧的依赖项代码；将之粘贴至IntelliJ中项目根目录下的POM.xml中。

nd4j-x86后端应如下所示：

     <dependency>
       <groupId>org.nd4j</groupId>
       <artifactId>nd4j-x86</artifactId>
       <version>${nd4j.version}</version>
     </dependency>

*nd4j-x86*可用于所有示例。OpenBlas、Windows和Linux用户如欲安装附加依赖项，请参见[Deepelearining4j入门页](../zh-gettingstarted.html#open)。

## 高级：在AWS上使用命令行

如果Linux操作系统的在AWS服务器上安装Deeplearning4j，可能需要使用命令行而非IDE来运行示例。在此情况下，请根据上述说明运行*git clone*和*mvn clean install*。安装完成后，可在命令行中使用一行代码运行实际示例。该行代码因存储库和所选的特定示例而各不相同。

以下为样板：

    java -cp target/nameofjar.jar fully.qualified.class.name

以下为展示命令大致形态的实例：

    java -cp target/dl4j-0.4-examples.jar org.deeplearning4j.MLPBackpropIrisExample

也就是说，根据版本和所选示例，存在两个通配符：

    java -cp target/*.jar org.deeplearning4j.*

若需从命令行更改并运行示例，可以对*src/main/java/org/deeplearning4j/multilayer*下的*MLPBackpropIrisExample*进行修改，而后再次通过Maven生成示例。

## Scala

示例的Scala版本可[在此获取](https://github.com/kogecoo/dl4j-0.4-examples-scala)。

## 后续步骤

运行示例后，请访问[完全安装页面](../gettingstarted.html)作进一步了解。

## <a name="walk">逐步说明</a>

* 在命令行中输入下列代码，检查是否已经安装Git。

		git --version

* 若无，则[点此安装Git](https://git-scm.herokuapp.com/book/en/v2/Getting-Started-Installing-Git)。
* 此外，还可创建[Github账户](https://github.com/join)、下载[Mac版](https://mac.github.com/)或[Windows版](https://windows.github.com/)Github。
* Windows用户请在开始菜单中找到并打开“Git Bash”。Git Bash终端应与cmd.exe类似。
* 在需要安装DL4J示例的目录输入`cd`。可通过`mkdir dl4j-examples`创建新目录，而后输入`cd`。然后运行：

    `git clone https://github.com/deeplearning4j/dl4j-0.4-examples`
* 输入`ls`，检查以确保文件已完整下载。
* 现在可打开IntelliJ。
* 点击“文件”菜单，然后点击“导入项目”或“通过已有来源创建新项目”。如此便会显示本地文件菜单。
* 选择包含DL4J示例的目录。
* 在下一个窗口中，将可以选择生成工具。选择Maven。
* 勾选“递归搜索项目”和“自动导入Maven项目”框，点击“下一步”。
* 确保JDK/SDK已得到部署。若无，请点击SDK窗口底部加号进行添加。
* 继续单击下一步，直至出现项目命名的提示。使用缺省项目名称即可。点击“完成”。
