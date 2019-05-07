
package org.deeplearning4j.ui.views.html.samediff

import play.twirl.api._
import play.twirl.api.TemplateMagic._


     object SameDiffUI_Scope0 {
import models._
import controllers._
import play.api.i18n._
import views.html._
import play.api.templates.PlayMagic._
import play.api.mvc._
import play.api.data._

class SameDiffUI extends BaseScalaTemplate[play.twirl.api.HtmlFormat.Appendable,Format[play.twirl.api.HtmlFormat.Appendable]](play.twirl.api.HtmlFormat) with play.twirl.api.Template0[play.twirl.api.HtmlFormat.Appendable] {

  /**/
  def apply/*1.2*/():play.twirl.api.HtmlFormat.Appendable = {
    _display_ {
      {


Seq[Any](format.raw/*1.4*/("""
"""),format.raw/*2.1*/("""<!DOCTYPE html>

<!--~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  ~ Copyright (c) 2015-2019 Skymind, Inc.
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

<html lang="en" style="height: 100%">
    <head>

        <meta charset="utf-8">
        <title>SameDiff Graph Visualization</title>
            <!-- start: Mobile Specific -->
        <meta name="viewport" content="width=device-width, initial-scale=1">
            <!-- end: Mobile Specific -->

        <link id="bootstrap-style" href="/assets/webjars/bootstrap/4.2.1/dist/css/bootstrap.min.css" rel="stylesheet">

        <link href="/assets/css/samediff/samediff.css" rel="stylesheet">

            <!-- The HTML5 shim, for IE6-8 support of HTML5 elements -->
            <!--[if lt IE 9]>
	  	<script src="http://html5shim.googlecode.com/svn/trunk/html5.js"></script>
		<link id="ie-style" href="/assets/css/ie.css" rel="stylesheet"/>
	<![endif]-->

            <!--[if IE 9]>
		<link id="ie9style" href="/assets/css/ie9.css" rel="stylesheet"/>
	<![endif]-->
    </head>

    <body>
            <!-- Start JavaScript-->
        <script src="/assets/webjars/jquery/2.2.0/jquery.min.js"></script>
        <script src="/assets/webjars/jquery-ui/1.10.2/ui/minified/jquery-ui.min.js"></script>
        <script src="/assets/webjars/bootstrap/4.2.1/dist/js/bootstrap.min.js"></script>
        <script src="/assets/webjars/jquery-cookie/1.4.1-1/jquery.cookie.js"></script>
        <script src="/assets/webjars/flatbuffers/1.9.0/js/flatbuffers.js"></script>

        <script src="/assets/webjars/cytoscape/3.3.3/dist/cytoscape.min.js"></script>
        <script src="/assets/webjars/dagre/0.8.4/dist/dagre.min.js"></script>
        <script src="/assets/webjars/cytoscape-dagre/2.1.0/cytoscape-dagre.js"></script>
        <script src="/assets/webjars/cytoscape-cose-bilkent/4.0.0/cytoscape-cose-bilkent.js"></script>
        <script src="/assets/webjars/webcola/3.1.3/WebCola/cola.js"></script>
        <script src="/assets/webjars/cytoscape-cola/2.3.0/cytoscape-cola.js"></script>
        <script src="/assets/webjars/cytoscape-euler/1.2.1/cytoscape-euler.js"></script>
        <script src="/assets/webjars/klayjs/0.4.1/klay.js"></script>
        <script src="/assets/webjars/cytoscape-klay/3.1.2/cytoscape-klay.js"></script>
        <script src="/assets/webjars/weaverjs/1.2.0/dist/weaver.js"></script>
        <script src="/assets/webjars/cytoscape-spread/3.0.0/cytoscape-spread.js"></script>
        <script src="/assets/webjars/flot/0.8.3/jquery.flot.js"></script>
        <script src="/assets/webjars/flot/0.8.3/jquery.flot.pie.js"></script>
        <script src="/assets/webjars/flot/0.8.3/jquery.flot.stack.js"></script>
        <script src="/assets/webjars/flot/0.8.3/jquery.flot.resize.min.js"></script>
        <script src="/assets/webjars/flot/0.8.3/jquery.flot.selection.js"></script>


        <script src="/assets/js/samediff/generated/uigraphevents_generated.js"></script>
        <script src="/assets/js/samediff/generated/uigraphstatic_generated.js"></script>
        <script src="/assets/js/samediff/generated/array_generated.js"></script>
        <script src="/assets/js/samediff/generated/utils_generated.js"></script>
        <script src="/assets/js/samediff/generated/variable_generated.js"></script>

        <script src="/assets/js/samediff/samediff-ui.js"></script>
        <script src="/assets/js/samediff/samediff-graph.js"></script>
        <script src="/assets/js/samediff/samediff-plots.js"></script>
        <script src="/assets/js/samediff/flatbuffers-utils.js"></script>
        """),format.raw/*80.34*/("""
        """),format.raw/*81.9*/("""<div class="container-fluid" style="min-height: 100%">
            <div class="row">
                    <!-- NavBar - Bootstrap classes -->
                <nav class="navbar navbar-expand navbar-dark bg-dark" style="width: 100pc">
                    <a class="navbar-brand" href="#">SameDiff</a>
                    <button class="navbar-toggler" type="button" data-toggle="collapse" data-target="#navbarNavAltMarkup" aria-controls="navbarNavAltMarkup" aria-expanded="false" aria-label="Toggle navigation">
                        <span class="navbar-toggler-icon"></span>
                    </button>
                    <div class="collapse navbar-collapse" id="navbarNavAltMarkup">
                        <div class="navbar-nav">
                            <a id="sdnavgraph" class="nav-item nav-link active" href="#" onclick="samediffSetPage('graph')">
                                Graph</a>
                            <a id="sdnavplots" class="nav-item nav-link" href="#" onclick="samediffSetPage('plots')">
                                Plots</a>
                            <a id="sdnaveval" class="nav-item nav-link" href="#" onclick="samediffSetPage('evaluation')">
                                Evaluation</a>
                            <a id="sdnavperf" class="nav-item nav-link" href="#" onclick="samediffSetPage('performance')">
                                Performance</a>
                            <a class="nav-item nav-link" href="#" onclick="toggleSidebar()">Toggle Sidebar</a>
                        </div>
                    </div>
                </nav>
            </div>

            <div class="row" style="min-height: 100%">
                    <!-- Sidebar -->
                """),format.raw/*107.47*/("""
                """),format.raw/*108.17*/("""<div class="col-md-4 col-12" style="min-width: 300px; max-width: 300px; background-color: #e6e6e6; height:100%; min-height:100vh">
                    """),format.raw/*109.81*/("""
                    """),format.raw/*110.21*/("""<div id="sidebartop" class="row p-2">
                        <div style="width:auto">
                            <label class="input-group-btn">
                                <span class="btn btn-secondary btn-sm">
                                    Select File<input type="file" id="fileselect" style="display: none;" multiple>
                                </span>
                            </label>
                        </div>
                        <div id="selectedfile" class="w-100">[No File Loaded]</div>
                    </div>

                    <div class="sidebarline"></div>

                    """),format.raw/*123.85*/("""
                    """),format.raw/*124.21*/("""<div id="sidebarmid" class="row p-2">
                        <div class="w-100"><b>Selected Node:</b></div>
                        <div id="sidebarmid-content" class="w-100">(None)</div>
                    </div>

                    <div class="sidebarline"></div>

                    <div id="sidebarmid2" class="row p-2">
                        <div style="width:100%">
                            <b>Find Node:</b><br>
                        </div>
                        <input id="findnodetxt" type="text" oninput="onGraphNodeSearch()">
                        <div id="findnoderesults">

                        </div>
                    </div>

                    <div class="sidebarline"></div>

                    """),format.raw/*143.87*/("""
                    """),format.raw/*144.21*/("""<div id="sidebarbottom" class="row p-2">
                        <br><br>
                        <strong>Graph Layout:</strong>
                        <div class="btn-group btn-group-toggle w-100" data-toggle="buttons" style="height: 40px">
                            <label class="btn btn-secondary active" onclick="setLayout('klay_down')">
                                <input type="radio" name="options" id="option1" autocomplete="off" checked>Down</label>
                            <label class="btn btn-secondary" onclick="setLayout('klay_lr')">
                                <input type="radio" name="options" id="option2" autocomplete="off">Right</label>
                            <label class="btn btn-secondary" onclick="setLayout('dagre')">
                                <input type="radio" name="options" id="option3" autocomplete="off">Alt</label>
                            <label class="btn btn-secondary" onclick="setLayout('cose-bilkent')">
                                <input type="radio" name="options" id="option3" autocomplete="off">Spread</label>
                        </div>
                        <br>
                        <br>
                        <br>
                    </div>
                </div>

                    <!-- Page Content -->
                <div id="samediffcontent" class="col-md col-12 main pa-1">
                    <div id="graphdiv" style="height: 100%;
                        width: 100%;
                        display: table"></div>
                </div>
            </div>
        </div>


            <!-- Execute once on page load -->
        <script>
                document.getElementById('fileselect').addEventListener('change', fileSelect, false);
                $(document).ready(function () """),format.raw/*176.47*/("""{"""),format.raw/*176.48*/("""
                    """),format.raw/*177.21*/("""renderSameDiffGraph();
                """),format.raw/*178.17*/("""}"""),format.raw/*178.18*/(""");
        </script>
    </body>
</html>
"""))
      }
    }
  }

  def render(): play.twirl.api.HtmlFormat.Appendable = apply()

  def f:(() => play.twirl.api.HtmlFormat.Appendable) = () => apply()

  def ref: this.type = this

}


}

/**/
object SameDiffUI extends SameDiffUI_Scope0.SameDiffUI
              /*
                  -- GENERATED --
                  DATE: Tue May 07 13:23:16 AEST 2019
                  SOURCE: c:/DL4J/Git/deeplearning4j/deeplearning4j/deeplearning4j-ui-parent/deeplearning4j-play/src/main/views/org/deeplearning4j/ui/views/samediff/SameDiffUI.scala.html
                  HASH: fb990759266c42da63c2679f551c6d2ddd026f17
                  MATRIX: 561->1|657->3|685->5|4990->4307|5027->4317|6807->6098|6854->6116|7035->6328|7086->6350|7755->7054|7806->7076|8588->7895|8639->7917|10485->9734|10515->9735|10566->9757|10635->9797|10665->9798
                  LINES: 20->1|25->1|26->2|104->80|105->81|131->107|132->108|133->109|134->110|147->123|148->124|167->143|168->144|200->176|200->176|201->177|202->178|202->178
                  -- GENERATED --
              */
          