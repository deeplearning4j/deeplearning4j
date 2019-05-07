
package org.deeplearning4j.ui.views.html.convolutional

import play.twirl.api._
import play.twirl.api.TemplateMagic._


     object Activations_Scope0 {
import models._
import controllers._
import play.api.i18n._
import views.html._
import play.api.templates.PlayMagic._
import play.api.mvc._
import play.api.data._

class Activations extends BaseScalaTemplate[play.twirl.api.HtmlFormat.Appendable,Format[play.twirl.api.HtmlFormat.Appendable]](play.twirl.api.HtmlFormat) with play.twirl.api.Template0[play.twirl.api.HtmlFormat.Appendable] {

  /**/
  def apply():play.twirl.api.HtmlFormat.Appendable = {
    _display_ {
      {


Seq[Any](format.raw/*1.1*/("""<!DOCTYPE html>

<!--~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  ~ Copyright (c) 2015-2018 Skymind, Inc.
  ~
  ~ This program and the accompanying materials are made available under the
  ~ terms of the Apache License, Version 2.0 which is available at
  ~ https://www.apache.org/licenses/LICENSE-2.0.
  ~
  ~ Unless required by applicable law or agreed to in writing, software
  ~ distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
  ~ WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
  ~ License for the specific language governing permissions and limitations
  ~ under the License.
  ~
  ~ SPDX-License-Identifier: Apache-2.0
  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~-->

<html lang="en">
    <head>
        <meta charset="utf-8" />
        <title>Neural Network activations</title>
            <!-- jQuery -->
        <script src="/assets/webjars/jquery/2.2.0/jquery.min.js"></script>
        """),format.raw/*25.68*/("""
            """),format.raw/*26.13*/("""<!-- Latest compiled and minified CSS -->
        <link id="bootstrap-style" href="/assets/webjars/bootstrap/2.3.2/css/bootstrap.min.css" rel="stylesheet">

            <!-- Latest compiled and minified JavaScript -->
        <script src="/assets/webjars/bootstrap/2.3.2/js/bootstrap.min.js"></script>

        <style>
        body """),format.raw/*33.14*/("""{"""),format.raw/*33.15*/("""
            """),format.raw/*34.13*/("""font-family: 'Roboto', sans-serif;
            color: #333;
            font-weight: 300;
            font-size: 16px;
        """),format.raw/*38.9*/("""}"""),format.raw/*38.10*/("""

        """),format.raw/*40.9*/(""".hd """),format.raw/*40.13*/("""{"""),format.raw/*40.14*/("""
            """),format.raw/*41.13*/("""background-color: #000000;
            font-size: 18px;
            color: #FFFFFF;
        """),format.raw/*44.9*/("""}"""),format.raw/*44.10*/("""

        """),format.raw/*46.9*/(""".block """),format.raw/*46.16*/("""{"""),format.raw/*46.17*/("""
            """),format.raw/*47.13*/("""width: 250px;
            height: 350px;
            display: inline-block;
            border: 1px solid #DEDEDE;
            margin-right: 64px;
        """),format.raw/*52.9*/("""}"""),format.raw/*52.10*/("""

        """),format.raw/*54.9*/(""".hd-small """),format.raw/*54.19*/("""{"""),format.raw/*54.20*/("""
            """),format.raw/*55.13*/("""background-color: #000000;
            font-size: 14px;
            color: #FFFFFF;
        """),format.raw/*58.9*/("""}"""),format.raw/*58.10*/("""
        """),format.raw/*59.9*/("""</style>

        <script type="text/javascript">
        setInterval(function () """),format.raw/*62.33*/("""{"""),format.raw/*62.34*/("""
            """),format.raw/*63.13*/("""var d = new Date();
            $("#pic").removeAttr("src").attr("src", "/activations/data?timestamp=" + new Date().getTime());
        """),format.raw/*65.9*/("""}"""),format.raw/*65.10*/(""", 3000);
        </script>

    </head>



    <body>
        <table style="width: 100%;
            padding: 5px;" class="hd">
            <tbody>
                <tr>
                    <td style="width: 48px;"><a href="/"><img src="/assets/legacy/deeplearning4j.img" border="0"/></a></td>
                    <td>DeepLearning4j UI</td>
                    <td style="width: 128px;">&nbsp; <!-- placeholder for future use --></td>
                </tr>
            </tbody>
        </table>
        <br /> <br />
        <div style="width: 100%;
            text-align: center">
            <div id="embed" style="display: inline-block;"> <!-- style="border: 1px solid #CECECE;" -->
                <img src="/activations/data" id="pic" />
            </div>
        </div>
    </body>

</html>"""))
      }
    }
  }

  def render(): play.twirl.api.HtmlFormat.Appendable = apply()

  def f:(() => play.twirl.api.HtmlFormat.Appendable) = () => apply()

  def ref: this.type = this

}


}

/**/
object Activations extends Activations_Scope0.Activations
              /*
                  -- GENERATED --
                  DATE: Tue May 07 13:23:16 AEST 2019
                  SOURCE: c:/DL4J/Git/deeplearning4j/deeplearning4j/deeplearning4j-ui-parent/deeplearning4j-play/src/main/views/org/deeplearning4j/ui/views/convolutional/Activations.scala.html
                  HASH: fb7b9cdfdd31f76d21208213b839aeebac3dabe2
                  MATRIX: 657->0|1724->1098|1766->1112|2133->1451|2162->1452|2204->1466|2362->1597|2391->1598|2430->1610|2462->1614|2491->1615|2533->1629|2655->1724|2684->1725|2723->1737|2758->1744|2787->1745|2829->1759|3016->1919|3045->1920|3084->1932|3122->1942|3151->1943|3193->1957|3315->2052|3344->2053|3381->2063|3494->2148|3523->2149|3565->2163|3730->2301|3759->2302
                  LINES: 25->1|49->25|50->26|57->33|57->33|58->34|62->38|62->38|64->40|64->40|64->40|65->41|68->44|68->44|70->46|70->46|70->46|71->47|76->52|76->52|78->54|78->54|78->54|79->55|82->58|82->58|83->59|86->62|86->62|87->63|89->65|89->65
                  -- GENERATED --
              */
          