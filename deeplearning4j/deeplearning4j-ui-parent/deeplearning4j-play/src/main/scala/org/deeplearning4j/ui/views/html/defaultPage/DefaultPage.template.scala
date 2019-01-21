
package org.deeplearning4j.ui.views.html.defaultPage

import play.twirl.api._
import play.twirl.api.TemplateMagic._


     object DefaultPage_Scope0 {
import models._
import controllers._
import play.api.i18n._
import views.html._
import play.api.templates.PlayMagic._
import play.api.mvc._
import play.api.data._

class DefaultPage extends BaseScalaTemplate[play.twirl.api.HtmlFormat.Appendable,Format[play.twirl.api.HtmlFormat.Appendable]](play.twirl.api.HtmlFormat) with play.twirl.api.Template0[play.twirl.api.HtmlFormat.Appendable] {

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
        <title>DeepLearning4j UI</title>
        <meta charset="utf-8" />

            <!-- jQuery -->
        <script src="https://code.jquery.com/jquery-2.2.0.min.js"></script>

        <link href='http://fonts.googleapis.com/css?family=Roboto:400,300' rel='stylesheet' type='text/css'>

            <!-- Latest compiled and minified CSS -->
        <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.6/css/bootstrap.min.css" integrity="sha384-1q8mTJOASx8j1Au+a5WDVnPi2lkFfwwEAa8hDDdjZlpLegxhjVME1fgjWPGmkzs7" crossorigin="anonymous" />

            <!-- Optional theme -->
        <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.6/css/bootstrap-theme.min.css" integrity="sha384-fLW2N01lMqjakBkx3l/M9EahuwpSfeNvV63J5ezn3uZzapT0u7EYsXMjQV+0En5r" crossorigin="anonymous" />

            <!-- Latest compiled and minified JavaScript -->
        <script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.6/js/bootstrap.min.js" integrity="sha384-0mSbJDEHialfmuBBQP6A4Qrprq5OVfW37PRR3j5ELqxss1yVqOtnepnHVP9aJ7xS" crossorigin="anonymous"></script>

        <style>
        body """),format.raw/*37.14*/("""{"""),format.raw/*37.15*/("""
            """),format.raw/*38.13*/("""font-family: 'Roboto', sans-serif;
            color: #333;
            font-weight: 300;
            font-size: 16px;
        """),format.raw/*42.9*/("""}"""),format.raw/*42.10*/("""
        """),format.raw/*43.9*/(""".hd """),format.raw/*43.13*/("""{"""),format.raw/*43.14*/("""
            """),format.raw/*44.13*/("""background-color: #000000;
            font-size: 18px;
            color: #FFFFFF;
        """),format.raw/*47.9*/("""}"""),format.raw/*47.10*/("""
        """),format.raw/*48.9*/(""".block """),format.raw/*48.16*/("""{"""),format.raw/*48.17*/("""
            """),format.raw/*49.13*/("""width: 250px;
            height: 350px;
            display: inline-block;

            margin-right: 64px;
        """),format.raw/*54.9*/("""}"""),format.raw/*54.10*/("""
        """),format.raw/*55.9*/(""".hd-small """),format.raw/*55.19*/("""{"""),format.raw/*55.20*/("""
            """),format.raw/*56.13*/("""background-color: #000000;
            font-size: 14px;
            color: #FFFFFF;
        """),format.raw/*59.9*/("""}"""),format.raw/*59.10*/("""
        """),format.raw/*60.9*/("""</style>

            <!-- Booststrap Notify plugin-->
        <script src="./assets/bootstrap-notify.min.js"></script>

        """),format.raw/*65.56*/("""
    """),format.raw/*66.5*/("""</head>
    <body>
        <table style="width: 100%; padding: 5px;" class="hd">
            <tbody>
                <tr>
                    <td style="width: 48px;"><img src="./assets/deeplearning4j.img"  border="0"/></td>
                    <td>DeepLearning4j UI</td>
                    <td style="width: 128px;">&nbsp; <!-- placeholder for future use --></td>
                </tr>
            </tbody>
        </table>

        <br />
        <br />
        <br />
            <!--
    Here we should provide nav to available modules:
    T-SNE visualization
    NN activations
    HistogramListener renderer
 -->
        <div style="width: 100%; text-align: center">
            <div class="block">
                    <!-- TSNE block. It's session-dependant. -->
                <b>T-SNE</b><br/><br/>
                """),format.raw/*91.181*/("""
                """),format.raw/*92.17*/("""<a href="#"><img src="./assets/i_plot.img" border="0" style="opacity: 1.0" id="TSNE"/></a><br/><br/>
                <div style="text-align: left; margin: 5px;">
                        &nbsp;Plot T-SNE data uploaded by user or retrieved from DL4j.
                </div>
            </div>

            <div class="block">
                    <!-- W2V block -->
                <b>WordVectors</b><br/><br/>
                <a href="./word2vec"><img src="./assets/i_nearest.img" border="0" /></a><br/><br/>
                <div style="text-align: left; margin: 5px;">
                        &nbsp;wordsNearest UI for WordVectors (GloVe/Word2Vec compatible)
                </div>
            </div>

            <div class="block">
                <b>Activations</b><br/><br/>
                """),format.raw/*109.206*/("""
                """),format.raw/*110.17*/("""<a href="#"><img src="./assets/i_ladder.img" border="0"  style="opacity: 0.2" id="ACTIVATIONS" /></a><br/><br/>
                <div style="text-align: left; margin: 5px;">
                        &nbsp;Activations retrieved from Convolution Neural network.
                </div>
            </div>

            <div class="block">
                    <!-- Histogram block. It's session-dependant block -->
                <b>Histograms &amp; Score</b><br/><br/>
                """),format.raw/*119.194*/("""
                """),format.raw/*120.17*/("""<a href="#"><img id="HISTOGRAM" src="./assets/i_histo.img" border="0" style="opacity: 0.2"/></a><br/><br/>
                <div style="text-align: left; margin: 5px;">
                        &nbsp;Neural network scores retrieved from DL4j during training.
                </div>
            </div>

            <div class="block">
                    <!-- Flow  block. It's session-dependant block -->
                <b>Model flow</b><br/><br/>
                """),format.raw/*129.180*/("""
                """),format.raw/*130.17*/("""<a href="#"><img id="FLOW" src="./assets/i_flow.img" border="0" style="opacity: 0.2"/></a><br/><br/>
                <div style="text-align: left; margin: 5px;">
                        &nbsp;MultiLayerNetwork/ComputationalGraph model state rendered.
                </div>
            </div>

            <div class="block">
                    <!-- Arbiter block. It's session-dependant block -->
                <b>Arbiter </b><br/><br/>
                """),format.raw/*139.192*/("""
                """),format.raw/*140.17*/("""<a href="#"><img id="ARBITER" src="./assets/i_arbiter.img" border="0" style="opacity: 0.2"/></a><br/><br/>
                <div style="text-align: left; margin: 5px;">
                        &nbsp;State &amp; management for Arbiter optimization processes.
                </div>
            </div>
        </div>

        <div  id="sessionSelector" style="position: fixed; top: 0px; bottom: 0px; left: 0px; right: 0px; z-index: 95; display: none;">
            <div style="position: fixed; top: 50%; left: 50%; -webkit-transform: translate(-50%, -50%); transform: translate(-50%, -50%); z-index: 100;   background-color: rgba(255, 255, 255,255); border: 1px solid #CECECE; height: 400px; width: 300px; -moz-box-shadow: 0 0 3px #ccc; -webkit-box-shadow: 0 0 3px #ccc; box-shadow: 0 0 3px #ccc;">

                <table class="table table-hover" style="margin-left: 10px; margin-right: 10px; margin-top: 5px; margin-bottom: 5px;">
                    <thead style="display: block; margin-bottom: 3px; width: 100%;">
                        <tr style="width: 100%">
                            <th style="width: 100%">Available sessions</th>
                        </tr>
                    </thead>
                    <tbody id="sessionList" style="display: block; width: 95%; height: 300px; overflow-y: auto; overflow-x: hidden;">

                    </tbody>
                </table>

                <div style="display: inline-block; position: fixed; bottom: 3px; left: 50%;  -webkit-transform: translate(-50%); transform: translate(-50%); ">
                    <input type="button" class="btn btn-default" style="" value=" Cancel " onclick="$('#sessionSelector').css('display','none');"/>
                </div>
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
object DefaultPage extends DefaultPage_Scope0.DefaultPage
              /*
                  -- GENERATED --
                  DATE: Mon Jan 21 11:34:12 AEDT 2019
                  SOURCE: c:/DL4J/Git/deeplearning4j/deeplearning4j/deeplearning4j-ui-parent/deeplearning4j-play/src/main/views/org/deeplearning4j/ui/views/defaultPage/DefaultPage.scala.html
                  HASH: da6a311236bb425835c99277e9483be69b05e8b4
                  MATRIX: 655->0|2638->1955|2667->1956|2709->1970|2867->2101|2896->2102|2933->2112|2965->2116|2994->2117|3036->2131|3158->2226|3187->2227|3224->2237|3259->2244|3288->2245|3330->2259|3479->2381|3508->2382|3545->2392|3583->2402|3612->2403|3654->2417|3776->2512|3805->2513|3842->2523|4004->2704|4037->2710|4918->3726|4964->3744|5805->4744|5852->4762|6371->5428|6418->5446|6920->6081|6967->6099|7463->6740|7510->6758
                  LINES: 25->1|61->37|61->37|62->38|66->42|66->42|67->43|67->43|67->43|68->44|71->47|71->47|72->48|72->48|72->48|73->49|78->54|78->54|79->55|79->55|79->55|80->56|83->59|83->59|84->60|89->65|90->66|115->91|116->92|133->109|134->110|143->119|144->120|153->129|154->130|163->139|164->140
                  -- GENERATED --
              */
          