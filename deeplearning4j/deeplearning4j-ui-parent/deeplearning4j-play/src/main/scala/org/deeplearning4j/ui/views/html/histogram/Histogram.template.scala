
package org.deeplearning4j.ui.views.html.histogram

import play.twirl.api._
import play.twirl.api.TemplateMagic._


     object Histogram_Scope0 {
import models._
import controllers._
import play.api.i18n._
import views.html._
import play.api.templates.PlayMagic._
import play.api.mvc._
import play.api.data._

class Histogram extends BaseScalaTemplate[play.twirl.api.HtmlFormat.Appendable,Format[play.twirl.api.HtmlFormat.Appendable]](play.twirl.api.HtmlFormat) with play.twirl.api.Template0[play.twirl.api.HtmlFormat.Appendable] {

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

<html>
    <head>
        <meta charset="utf-8"/>
        <title>Weights/Gradients/Score</title>
        <style>
        .bar rect """),format.raw/*24.19*/("""{"""),format.raw/*24.20*/("""
            """),format.raw/*25.13*/("""fill: steelblue;
            shape-rendering: crispEdges;
        """),format.raw/*27.9*/("""}"""),format.raw/*27.10*/("""

        """),format.raw/*29.9*/(""".bar text """),format.raw/*29.19*/("""{"""),format.raw/*29.20*/("""
            """),format.raw/*30.13*/("""fill: #EFEFEF;
        """),format.raw/*31.9*/("""}"""),format.raw/*31.10*/("""

        """),format.raw/*33.9*/(""".area """),format.raw/*33.15*/("""{"""),format.raw/*33.16*/("""
            """),format.raw/*34.13*/("""fill: steelblue;
        """),format.raw/*35.9*/("""}"""),format.raw/*35.10*/("""

        """),format.raw/*37.9*/(""".axis path, .axis line """),format.raw/*37.32*/("""{"""),format.raw/*37.33*/("""
            """),format.raw/*38.13*/("""fill: none;
            stroke: #000;
            shape-rendering: crispEdges;
        """),format.raw/*41.9*/("""}"""),format.raw/*41.10*/("""

        """),format.raw/*43.9*/(""".tick line """),format.raw/*43.20*/("""{"""),format.raw/*43.21*/("""
            """),format.raw/*44.13*/("""opacity: 0.2;
            shape-rendering: crispEdges;
        """),format.raw/*46.9*/("""}"""),format.raw/*46.10*/("""

        """),format.raw/*48.9*/("""path """),format.raw/*48.14*/("""{"""),format.raw/*48.15*/("""
            """),format.raw/*49.13*/("""stroke: steelblue;
            stroke-width: 2;
            fill: none;
        """),format.raw/*52.9*/("""}"""),format.raw/*52.10*/("""

        """),format.raw/*54.9*/(""".legend """),format.raw/*54.17*/("""{"""),format.raw/*54.18*/("""
            """),format.raw/*55.13*/("""font-size: 12px;
            text-anchor: middle;
        """),format.raw/*57.9*/("""}"""),format.raw/*57.10*/("""

        """),format.raw/*59.9*/(""".brush .extent """),format.raw/*59.24*/("""{"""),format.raw/*59.25*/("""
            """),format.raw/*60.13*/("""stroke: #fff;
            fill-opacity: .125;
            shape-rendering: crispEdges;
        """),format.raw/*63.9*/("""}"""),format.raw/*63.10*/("""

        """),format.raw/*65.9*/("""</style>

            <!-- jQuery -->
        <script src="/assets/legacy/jquery-2.2.0.min.js"></script>

        <link href='/assets/legacy/roboto.css' rel='stylesheet' type='text/css'>

            <!-- Latest compiled and minified CSS -->
        <link rel="stylesheet" href="/assets/legacy/bootstrap.min.css" />

            <!-- Optional theme -->
        <link rel="stylesheet" href="/assets/legacy/bootstrap-theme.min.css" />

            <!-- Latest compiled and minified JavaScript -->
        <script src="/assets/legacy/bootstrap.min.js" ></script>

            <!-- d3 -->
        <script src="/assets/legacy/d3.v3.min.js" charset="utf-8"></script>

        <script src="/assets/legacy/jquery-fileupload.js"></script>

            <!-- Booststrap Notify plugin-->
        <script src="/assets/legacy/bootstrap-notify.min.js"></script>

            <!-- Latest compiled and minified CSS -->
        <link rel="stylesheet" href="/assets/legacy/bootstrap-select.min.css" />

            <!-- Latest compiled and minified JavaScript -->
        <script src="/assets/legacy/bootstrap-select.min.js"></script>

            <!-- DateTime formatter-->
        <script src="/assets/legacy/DateTimeFormat.js"></script>

        <script src="/assets/legacy/renderWeightsProper.js"></script>

        <script src="/assets/legacy/common.js"></script>

        <style>
        body """),format.raw/*103.14*/("""{"""),format.raw/*103.15*/("""
            """),format.raw/*104.13*/("""font-family: 'Roboto', sans-serif;
            color: #333;
            font-weight: 300;
            font-size: 16px;
        """),format.raw/*108.9*/("""}"""),format.raw/*108.10*/("""

        """),format.raw/*110.9*/(""".hd """),format.raw/*110.13*/("""{"""),format.raw/*110.14*/("""
            """),format.raw/*111.13*/("""background-color: #000000;
            font-size: 18px;
            color: #FFFFFF;
        """),format.raw/*114.9*/("""}"""),format.raw/*114.10*/("""

        """),format.raw/*116.9*/(""".block """),format.raw/*116.16*/("""{"""),format.raw/*116.17*/("""
            """),format.raw/*117.13*/("""width: 250px;
            height: 350px;
            display: inline-block;
            border: 1px solid #DEDEDE;
            margin-right: 64px;
        """),format.raw/*122.9*/("""}"""),format.raw/*122.10*/("""

        """),format.raw/*124.9*/(""".hd-small """),format.raw/*124.19*/("""{"""),format.raw/*124.20*/("""
            """),format.raw/*125.13*/("""background-color: #000000;
            font-size: 14px;
            color: #FFFFFF;
        """),format.raw/*128.9*/("""}"""),format.raw/*128.10*/("""

        """),format.raw/*130.9*/(""".charts, .chart """),format.raw/*130.25*/("""{"""),format.raw/*130.26*/("""
            """),format.raw/*131.13*/("""font-size: 10px;
            font-color: #000000;
            position: relative;
        """),format.raw/*134.9*/("""}"""),format.raw/*134.10*/("""

        """),format.raw/*136.9*/(""".scoreboard """),format.raw/*136.21*/("""{"""),format.raw/*136.22*/("""
            """),format.raw/*137.13*/("""position: absolute;
            top: 20px;
            right: 10px;
            z-index: 1000;
            font-size: 11px;
        """),format.raw/*142.9*/("""}"""),format.raw/*142.10*/("""

        """),format.raw/*144.9*/(""".score """),format.raw/*144.16*/("""{"""),format.raw/*144.17*/("""
            """),format.raw/*145.13*/("""font-size: 11px;
        """),format.raw/*146.9*/("""}"""),format.raw/*146.10*/("""
        """),format.raw/*147.9*/("""</style>

    </head>
    <body>
        <table style="width: 100%;
            padding: 5px;" class="hd">
            <tbody>
                <tr>
                    <td style="width: 48px;"><a href="/"><img src="/assets/legacy/deeplearning4j.img" border="0"/></a></td>
                    <td>DeepLearning4j UI</td>
                    <td style="width: 512px;
                        text-align: right;" class="hd-small">&nbsp; Available sessions: <select class="selectpicker" id="sessionSelector" onchange="window.location.href = 'weights?sid=' + this.options[this.selectedIndex].value ;" style="color: #000000;
                        display: inline-block;
                        width: 256px;">
                        <option value="0" selected="selected">Pick a session to track</option>
                    </select> &nbsp;&nbsp;
                        <script>
            buildSessionSelector("HISTOGRAM");
          </script>
                    </td>
                    <td style="width: 256px;" class="hd-small">&nbsp;Updated at: <b><span id="updatetime">No updates so far</span></b> &nbsp;</td>
                </tr>
            </tbody>
        </table>

        <div style="width: 100%;
            text-align: center;">
            <div id="display" style="width: 1540px;
                height: 900px;
                text-align: left;
                background-color: #FFFFFF;
                display: inline-block;
                overflow: hidden; ">
                <div id="scores" style="background-color: #EFEFEF;
                    display: block;
                    float: left;
                    width: 750px;
                    height: 400px;
                    border: 1px solid #CECECE;
                    margin: 10px;">
                    <h5>&nbsp;&nbsp;Score vs. Iteration #</h5>
                    <div class="chart" id="schart">
        </div>
                </div>
                <div id="model" style="position: relative;
                    background-color: #EFEFEF;
                    display: block;
                    float: left;
                    width: 750px;
                    height: 400px;
                    border: 1px solid #CECECE;
                    margin: 10px;">
                    <h5>&nbsp;&nbsp;Model</h5>
                    <div class="charts"></div>
                    <div style="position: absolute;
                        top: 5px;
                        right: 5px;">
                        <select id="modelSelector" onchange="selectModel();">
          </select>
                    </div>
                </div>
                <div id="gradient" style="position: relative;
                    background-color: #EFEFEF;
                    display: block;
                    float: left;
                    width: 750px;
                    height: 400px;
                    border: 1px solid #CECECE;
                    margin: 10px;">
                    <h5>&nbsp;&nbsp;Gradient</h5>
                    <div class="charts"></div>
                    <div style="position: absolute;
                        top: 5px;
                        right: 5px;">
                        <select id="gradientSelector" onchange="selectGradient();">
          </select>
                    </div>
                </div>
                <div id="magnitudes" style="position: relative;
                    background-color: #EFEFEF;
                    display: block;
                    float: left;
                    width: 750px;
                    height: 400px;
                    border: 1px solid #CECECE;
                    margin: 10px;">
                    <h5>&nbsp;&nbsp;Mean Magnitudes: Parameters and Updates</h5>
                    <div class="charts"></div>
                    <div style="position: absolute;
                        top: 5px;
                        right: 5px;">
                        <select id="magnitudeSelector" onchange="selectMagnitude();">
          </select>
                    </div>
                </div>
                    <!--<div id="lastupdate">
            <div class="updatetime">-1</div>
        </div>-->
            </div>

                <!--
    <div style="display: block;">
        nav bar
    </div> -->
        </div>
            <!--
<div id="score" style="display: inline-block; width: 650px; height: 400px; border: 1px solid #CECECE;">
    <h4>Score</h4>
    <div class="score"></div>
</div>-->

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
object Histogram extends Histogram_Scope0.Histogram
              /*
                  -- GENERATED --
                  DATE: Sat Jan 19 12:31:33 AEDT 2019
                  SOURCE: c:/DL4J/Git/deeplearning4j/deeplearning4j/deeplearning4j-ui-parent/deeplearning4j-play/src/main/views/org/deeplearning4j/ui/views/histogram/Histogram.scala.html
                  HASH: 85e90b78e2ac617a501aec7122117d1826790d73
                  MATRIX: 649->0|1624->947|1653->948|1695->962|1790->1030|1819->1031|1858->1043|1896->1053|1925->1054|1967->1068|2018->1092|2047->1093|2086->1105|2120->1111|2149->1112|2191->1126|2244->1152|2273->1153|2312->1165|2363->1188|2392->1189|2434->1203|2551->1293|2580->1294|2619->1306|2658->1317|2687->1318|2729->1332|2821->1397|2850->1398|2889->1410|2922->1415|2951->1416|2993->1430|3103->1513|3132->1514|3171->1526|3207->1534|3236->1535|3278->1549|3365->1609|3394->1610|3433->1622|3476->1637|3505->1638|3547->1652|3672->1750|3701->1751|3740->1763|5187->3181|5217->3182|5260->3196|5419->3327|5449->3328|5489->3340|5522->3344|5552->3345|5595->3359|5718->3454|5748->3455|5788->3467|5824->3474|5854->3475|5897->3489|6085->3649|6115->3650|6155->3662|6194->3672|6224->3673|6267->3687|6390->3782|6420->3783|6460->3795|6505->3811|6535->3812|6578->3826|6699->3919|6729->3920|6769->3932|6810->3944|6840->3945|6883->3959|7048->4096|7078->4097|7118->4109|7154->4116|7184->4117|7227->4131|7281->4157|7311->4158|7349->4168
                  LINES: 25->1|48->24|48->24|49->25|51->27|51->27|53->29|53->29|53->29|54->30|55->31|55->31|57->33|57->33|57->33|58->34|59->35|59->35|61->37|61->37|61->37|62->38|65->41|65->41|67->43|67->43|67->43|68->44|70->46|70->46|72->48|72->48|72->48|73->49|76->52|76->52|78->54|78->54|78->54|79->55|81->57|81->57|83->59|83->59|83->59|84->60|87->63|87->63|89->65|127->103|127->103|128->104|132->108|132->108|134->110|134->110|134->110|135->111|138->114|138->114|140->116|140->116|140->116|141->117|146->122|146->122|148->124|148->124|148->124|149->125|152->128|152->128|154->130|154->130|154->130|155->131|158->134|158->134|160->136|160->136|160->136|161->137|166->142|166->142|168->144|168->144|168->144|169->145|170->146|170->146|171->147
                  -- GENERATED --
              */
          