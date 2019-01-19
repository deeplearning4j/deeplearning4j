
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
        <script src="/assets/legacy/jquery-2.2.0.min.js"></script>
        """),format.raw/*25.68*/("""
            """),format.raw/*26.13*/("""<!-- Latest compiled and minified CSS -->
        <link rel="stylesheet" href="/assets/legacy/bootstrap.min.css" />

            <!-- Optional theme -->
        <link rel="stylesheet" href="/assets/legacy/bootstrap-theme.min.css" />

            <!-- Latest compiled and minified JavaScript -->
        <script src="/assets/legacy/bootstrap.min.js" ></script>

        <style>
        body """),format.raw/*36.14*/("""{"""),format.raw/*36.15*/("""
            """),format.raw/*37.13*/("""font-family: 'Roboto', sans-serif;
            color: #333;
            font-weight: 300;
            font-size: 16px;
        """),format.raw/*41.9*/("""}"""),format.raw/*41.10*/("""

        """),format.raw/*43.9*/(""".hd """),format.raw/*43.13*/("""{"""),format.raw/*43.14*/("""
            """),format.raw/*44.13*/("""background-color: #000000;
            font-size: 18px;
            color: #FFFFFF;
        """),format.raw/*47.9*/("""}"""),format.raw/*47.10*/("""

        """),format.raw/*49.9*/(""".block """),format.raw/*49.16*/("""{"""),format.raw/*49.17*/("""
            """),format.raw/*50.13*/("""width: 250px;
            height: 350px;
            display: inline-block;
            border: 1px solid #DEDEDE;
            margin-right: 64px;
        """),format.raw/*55.9*/("""}"""),format.raw/*55.10*/("""

        """),format.raw/*57.9*/(""".hd-small """),format.raw/*57.19*/("""{"""),format.raw/*57.20*/("""
            """),format.raw/*58.13*/("""background-color: #000000;
            font-size: 14px;
            color: #FFFFFF;
        """),format.raw/*61.9*/("""}"""),format.raw/*61.10*/("""
        """),format.raw/*62.9*/("""</style>

        <script type="text/javascript">
        setInterval(function () """),format.raw/*65.33*/("""{"""),format.raw/*65.34*/("""
            """),format.raw/*66.13*/("""var d = new Date();
            $("#pic").removeAttr("src").attr("src", "/activations/data?timestamp=" + new Date().getTime());
        """),format.raw/*68.9*/("""}"""),format.raw/*68.10*/(""", 3000);
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
                  DATE: Sat Jan 19 12:31:33 AEDT 2019
                  SOURCE: c:/DL4J/Git/deeplearning4j/deeplearning4j/deeplearning4j-ui-parent/deeplearning4j-play/src/main/views/org/deeplearning4j/ui/views/convolutional/Activations.scala.html
                  HASH: 658f3b7fd366911400cee3a1d089e573588101f6
                  MATRIX: 657->0|1716->1090|1758->1104|2186->1504|2215->1505|2257->1519|2415->1650|2444->1651|2483->1663|2515->1667|2544->1668|2586->1682|2708->1777|2737->1778|2776->1790|2811->1797|2840->1798|2882->1812|3069->1972|3098->1973|3137->1985|3175->1995|3204->1996|3246->2010|3368->2105|3397->2106|3434->2116|3547->2201|3576->2202|3618->2216|3783->2354|3812->2355
                  LINES: 25->1|49->25|50->26|60->36|60->36|61->37|65->41|65->41|67->43|67->43|67->43|68->44|71->47|71->47|73->49|73->49|73->49|74->50|79->55|79->55|81->57|81->57|81->57|82->58|85->61|85->61|86->62|89->65|89->65|90->66|92->68|92->68
                  -- GENERATED --
              */
          