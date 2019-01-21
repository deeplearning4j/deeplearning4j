
package org.deeplearning4j.ui.views.html.flow

import play.twirl.api._
import play.twirl.api.TemplateMagic._


     object Flow_Scope0 {
import models._
import controllers._
import play.api.i18n._
import views.html._
import play.api.templates.PlayMagic._
import play.api.mvc._
import play.api.data._

class Flow extends BaseScalaTemplate[play.twirl.api.HtmlFormat.Appendable,Format[play.twirl.api.HtmlFormat.Appendable]](play.twirl.api.HtmlFormat) with play.twirl.api.Template0[play.twirl.api.HtmlFormat.Appendable] {

  /**/
  def apply():play.twirl.api.HtmlFormat.Appendable = {
    _display_ {
      {


Seq[Any](format.raw/*1.1*/("""<!--~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
        <meta charset="utf-8" />

        <title>Flow overview</title>


            <!-- jQuery -->
        <script src="/assets/legacy/jquery-2.2.0.min.js"></script>

        <link href='/assets/legacy/roboto.css' rel='stylesheet' type='text/css'>

            <!-- Latest compiled and minified CSS -->
        <link rel="stylesheet" href="/assets/legacy/bootstrap.min.css" />

            <!-- Optional theme -->
        <link rel="stylesheet" href="/assets/legacy/bootstrap-theme.min.css" />

            <!-- Latest compiled and minified JavaScript -->
        <script src="/assets/legacy/bootstrap.min.js" ></script>


            <!-- Booststrap Notify plugin-->
        <script src="/assets/legacy/bootstrap-notify.min.js"></script>

            <!-- DateTime formatter-->
        <script src="/assets/legacy/DateTimeFormat.js"></script>

            <!-- d3 -->
        <script src="/assets/legacy/d3.v3.min.js" charset="utf-8"></script>

        <script src="/assets/legacy/Connection.js"></script>
        <script src="/assets/legacy/Layer.js"></script>
        <script src="/assets/legacy/Layers.js"></script>

        <script src="/assets/legacy/common.js"></script>

        <script src="/assets/legacy/renderFlow.js"></script>
        <style>
        body """),format.raw/*56.14*/("""{"""),format.raw/*56.15*/("""
            """),format.raw/*57.13*/("""font-family: 'Roboto', sans-serif;
            color: #333;
            font-weight: 300;
            font-size: 16px;
        """),format.raw/*61.9*/("""}"""),format.raw/*61.10*/("""
        """),format.raw/*62.9*/(""".hd """),format.raw/*62.13*/("""{"""),format.raw/*62.14*/("""
            """),format.raw/*63.13*/("""background-color: #000000;
            font-size: 18px;
            color: #FFFFFF;
        """),format.raw/*66.9*/("""}"""),format.raw/*66.10*/("""
        """),format.raw/*67.9*/(""".block """),format.raw/*67.16*/("""{"""),format.raw/*67.17*/("""
            """),format.raw/*68.13*/("""width: 250px;
            height: 350px;
            display: inline-block;
            border: 1px solid #DEDEDE;
            margin-right: 64px;
        """),format.raw/*73.9*/("""}"""),format.raw/*73.10*/("""
        """),format.raw/*74.9*/(""".hd-small """),format.raw/*74.19*/("""{"""),format.raw/*74.20*/("""
            """),format.raw/*75.13*/("""background-color: #000000;
            font-size: 14px;
            color: #FFFFFF;
        """),format.raw/*78.9*/("""}"""),format.raw/*78.10*/("""
        """),format.raw/*79.9*/(""".charts, .chart """),format.raw/*79.25*/("""{"""),format.raw/*79.26*/("""
            """),format.raw/*80.13*/("""font-size: 10px;
            font-color: #000000;
        """),format.raw/*82.9*/("""}"""),format.raw/*82.10*/("""
        """),format.raw/*83.9*/(""".tooltip """),format.raw/*83.18*/("""{"""),format.raw/*83.19*/("""
            """),format.raw/*84.13*/("""position: absolute;
            top: 140px;
            left: 0px;
            right: 0;
            width: 220px;
            padding: 2px 0;
            background-color: #000;
            background-color: rgba(0, 0, 0, 0.65);
            color: #fff;
            opacity: 0;
            transition: opacity .5s ease-in-out;
            text-align: center;
            font-family: Arial;
            font-size: 14px;
            z-index: 100;
        """),format.raw/*99.9*/("""}"""),format.raw/*99.10*/("""
        """),format.raw/*100.9*/(""".viewpanel """),format.raw/*100.20*/("""{"""),format.raw/*100.21*/("""
            """),format.raw/*101.13*/("""position: absolute;
            background-color: #FFF;
            top: 60px;
            bottom: 0px;
        """),format.raw/*105.9*/("""}"""),format.raw/*105.10*/("""

        """),format.raw/*107.9*/(""".perftd """),format.raw/*107.17*/("""{"""),format.raw/*107.18*/("""
            """),format.raw/*108.13*/("""padding-right: 10px;
            padding-bottom: 1px;
            font-family: Arial;
            font-size: 14px;
        """),format.raw/*112.9*/("""}"""),format.raw/*112.10*/("""

        """),format.raw/*114.9*/(""".bar rect """),format.raw/*114.19*/("""{"""),format.raw/*114.20*/("""
            """),format.raw/*115.13*/("""fill: steelblue;
            shape-rendering: crispEdges;
        """),format.raw/*117.9*/("""}"""),format.raw/*117.10*/("""

        """),format.raw/*119.9*/(""".bar text """),format.raw/*119.19*/("""{"""),format.raw/*119.20*/("""
            """),format.raw/*120.13*/("""fill: #EFEFEF;
        """),format.raw/*121.9*/("""}"""),format.raw/*121.10*/("""

        """),format.raw/*123.9*/(""".area """),format.raw/*123.15*/("""{"""),format.raw/*123.16*/("""
            """),format.raw/*124.13*/("""fill: steelblue;
        """),format.raw/*125.9*/("""}"""),format.raw/*125.10*/("""

        """),format.raw/*127.9*/(""".axis path, .axis line """),format.raw/*127.32*/("""{"""),format.raw/*127.33*/("""
            """),format.raw/*128.13*/("""fill: none;
            stroke: #000;
            stroke-width: 1.5;
            shape-rendering: crispEdges;
        """),format.raw/*132.9*/("""}"""),format.raw/*132.10*/("""

        """),format.raw/*134.9*/(""".tick line """),format.raw/*134.20*/("""{"""),format.raw/*134.21*/("""
            """),format.raw/*135.13*/("""opacity: 0.2;
            stroke-width: 1.5;
            shape-rendering: crispEdges;
        """),format.raw/*138.9*/("""}"""),format.raw/*138.10*/("""

        """),format.raw/*140.9*/(""".tick """),format.raw/*140.15*/("""{"""),format.raw/*140.16*/("""
            """),format.raw/*141.13*/("""font-size: 9px;
        """),format.raw/*142.9*/("""}"""),format.raw/*142.10*/("""

        """),format.raw/*144.9*/("""path """),format.raw/*144.14*/("""{"""),format.raw/*144.15*/("""
            """),format.raw/*145.13*/("""stroke: steelblue;
            stroke-width: 1.5;
            fill: none;
        """),format.raw/*148.9*/("""}"""),format.raw/*148.10*/("""

        """),format.raw/*150.9*/(""".legend """),format.raw/*150.17*/("""{"""),format.raw/*150.18*/("""
            """),format.raw/*151.13*/("""font-size: 12px;
            text-anchor: middle;
        """),format.raw/*153.9*/("""}"""),format.raw/*153.10*/("""

        """),format.raw/*155.9*/(""".layerDesc """),format.raw/*155.20*/("""{"""),format.raw/*155.21*/("""
            """),format.raw/*156.13*/("""font-family: Arial;
            font-size: 12px;
        """),format.raw/*158.9*/("""}"""),format.raw/*158.10*/("""

        """),format.raw/*160.9*/(""".brush .extent """),format.raw/*160.24*/("""{"""),format.raw/*160.25*/("""
            """),format.raw/*161.13*/("""stroke: #fff;
            stroke-width: 1.5;
            fill-opacity: .125;
            shape-rendering: crispEdges;
        """),format.raw/*165.9*/("""}"""),format.raw/*165.10*/("""
        """),format.raw/*166.9*/("""</style>
    </head>
    <body>
        <table style="width: 100%; padding: 5px;" class="hd">
            <tbody>
                <tr>
                    <td style="width: 48px;"><a href="/"><img src="/assets/legacy/deeplearning4j.img" border="0"/></a></td>
                    <td>DeepLearning4j UI</td>
                    <td style="width: 512px; text-align: right;" class="hd-small">&nbsp; Available sessions: <select class="selectpicker" id="sessionSelector" onchange="window.location.href = 'flow?sid='+ this.options[this.selectedIndex].value ;" style="color: #000000; display: inline-block; width: 256px;">
                        <option value="0" selected="selected">Pick a session to track</option>
                    </select>&nbsp;&nbsp;
                        <script>
                            buildSessionSelector2("flow/listSessions","");
                        </script>
                    </td>
                    <td style="width: 256px;" class="hd-small">&nbsp;Updated at: <b><span id="updatetime">No updates so far</span></b>&nbsp;</td>
                </tr>
            </tbody>
        </table>
        <br /> <br />
        <div style="width: 100%; text-align: center;">
            <div id="display" style="display: inline-block; width: 900px;">
                    <!-- NN rendering pane -->
            </div>
        </div>
        <div id="tooltip" class="tooltip">
                &nbsp;
        </div>

            <!-- Left view panel -->
        <div style="left: 10px; width: 400px;" class="viewpanel">
            <center>
                <table style="margin: 10px; width: 200px;">
                    <tr>
                        <td><b>Score vs iteration:</b></td>
                    </tr>
                </table>
            </center>
            <div id="scoreChart" style="background-color: #FFF; height: 250px;">
                    &nbsp;
            </div>
            <br/>
            <div style="width: 100%; background-color: #FFF; text-align:center; display: block; ">
                <center>
                    <table style="margin: 10px; width: 200px;">
                        <thead style="width: 200px;">
                            <td colspan="2"><b>Model training status:</b></td>
                        </thead>
                        <tbody>
                            <tr>
                                <td class="perftd">Current score:</td>
                                <td class="perftd" id="ss">0.0</td>
                            </tr>
                            <tr>
                                <td class="perftd">Time spent so far:</td>
                                <td class="perftd" id="st">00:00:00</td>
                            </tr>
                        </tbody>
                    </table>
                </center>
            </div>
            <br/>
            <div style="width: 100%; background-color: #FFF; text-align:center; display: block; ">
                <center>
                    <table style="margin: 10px; width: 200px;">
                        <thead style="width: 200px;">
                            <td colspan="2"><b>Performance status:</b></td>
                        </thead>
                        <tbody>
                            <tr>
                                <td class="perftd">Sampes per sec:</td>
                                <td class="perftd" id="ps">0.0/sec</td>
                            </tr>
                            <tr>
                                <td class="perftd">Batches per sec:</td>
                                <td class="perftd" id="pb">0.0/sec</td>
                            </tr>
                            <tr>
                                <td class="perftd">Iteration time:</td>
                                <td class="perftd" id="pt">0 ms</td>
                            </tr>
                        </tbody>
                    </table>
                </center>
            </div>
        </div>

            <!-- Right view panel -->
        <div style="right: 10px; width: 400px; position: absolute;" class="viewpanel" id="viewport">
            <div style='position: relative; top: 45%; height: 40px; margin: 0 auto;' id='hint'><b>&lt; Click on any node for detailed report</b></div>
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
object Flow extends Flow_Scope0.Flow
              /*
                  -- GENERATED --
                  DATE: Mon Jan 21 12:53:55 AEDT 2019
                  SOURCE: c:/DL4J/Git/deeplearning4j/deeplearning4j/deeplearning4j-ui-parent/deeplearning4j-play/src/main/views/org/deeplearning4j/ui/views/flow/Flow.scala.html
                  HASH: ed4f1818d8f93bb340d9244b6eba24bf2631f297
                  MATRIX: 634->0|2782->2120|2811->2121|2853->2135|3011->2266|3040->2267|3077->2277|3109->2281|3138->2282|3180->2296|3302->2391|3331->2392|3368->2402|3403->2409|3432->2410|3474->2424|3661->2584|3690->2585|3727->2595|3765->2605|3794->2606|3836->2620|3958->2715|3987->2716|4024->2726|4068->2742|4097->2743|4139->2757|4226->2817|4255->2818|4292->2828|4329->2837|4358->2838|4400->2852|4897->3322|4926->3323|4964->3333|5004->3344|5034->3345|5077->3359|5221->3475|5251->3476|5291->3488|5328->3496|5358->3497|5401->3511|5556->3638|5586->3639|5626->3651|5665->3661|5695->3662|5738->3676|5834->3744|5864->3745|5904->3757|5943->3767|5973->3768|6016->3782|6068->3806|6098->3807|6138->3819|6173->3825|6203->3826|6246->3840|6300->3866|6330->3867|6370->3879|6422->3902|6452->3903|6495->3917|6645->4039|6675->4040|6715->4052|6755->4063|6785->4064|6828->4078|6953->4175|6983->4176|7023->4188|7058->4194|7088->4195|7131->4209|7184->4234|7214->4235|7254->4247|7288->4252|7318->4253|7361->4267|7474->4352|7504->4353|7544->4365|7581->4373|7611->4374|7654->4388|7742->4448|7772->4449|7812->4461|7852->4472|7882->4473|7925->4487|8012->4546|8042->4547|8082->4559|8126->4574|8156->4575|8199->4589|8357->4719|8387->4720|8425->4730
                  LINES: 25->1|80->56|80->56|81->57|85->61|85->61|86->62|86->62|86->62|87->63|90->66|90->66|91->67|91->67|91->67|92->68|97->73|97->73|98->74|98->74|98->74|99->75|102->78|102->78|103->79|103->79|103->79|104->80|106->82|106->82|107->83|107->83|107->83|108->84|123->99|123->99|124->100|124->100|124->100|125->101|129->105|129->105|131->107|131->107|131->107|132->108|136->112|136->112|138->114|138->114|138->114|139->115|141->117|141->117|143->119|143->119|143->119|144->120|145->121|145->121|147->123|147->123|147->123|148->124|149->125|149->125|151->127|151->127|151->127|152->128|156->132|156->132|158->134|158->134|158->134|159->135|162->138|162->138|164->140|164->140|164->140|165->141|166->142|166->142|168->144|168->144|168->144|169->145|172->148|172->148|174->150|174->150|174->150|175->151|177->153|177->153|179->155|179->155|179->155|180->156|182->158|182->158|184->160|184->160|184->160|185->161|189->165|189->165|190->166
                  -- GENERATED --
              */
          