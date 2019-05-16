
/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.ui.views.html.samediff

import play.twirl.api._
import play.twirl.api.TemplateMagic._


     object SameDiffUIBackup_Scope0 {
import models._
import controllers._
import play.api.i18n._
import views.html._
import play.api.templates.PlayMagic._
import play.api.mvc._
import play.api.data._

class SameDiffUIBackup extends BaseScalaTemplate[play.twirl.api.HtmlFormat.Appendable,Format[play.twirl.api.HtmlFormat.Appendable]](play.twirl.api.HtmlFormat) with play.twirl.api.Template0[play.twirl.api.HtmlFormat.Appendable] {

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

        <link id="bootstrap-style" href="/assets/webjars/bootstrap/2.3.1/css/bootstrap.min.css" rel="stylesheet">

            <!-- The HTML5 shim, for IE6-8 support of HTML5 elements -->
            <!--[if lt IE 9]>
	  	<script src="http://html5shim.googlecode.com/svn/trunk/html5.js"></script>
		<link id="ie-style" href="/assets/css/ie.css" rel="stylesheet"/>
	<![endif]-->

            <!--[if IE 9]>
		<link id="ie9style" href="/assets/css/ie9.css" rel="stylesheet"/>
	<![endif]-->
    </head>

    <body style="height: 100%;
        margin: 0;">
            <!-- Start JavaScript-->
        <script src="/assets/webjars/jquery/2.2.0/jquery.min.js"></script>
        <script src="/assets/webjars/jquery-ui/1.10.2/ui/minified/jquery-ui.min.js"></script>
        <script src="/assets/webjars/bootstrap/2.3.1/js/bootstrap.min.js"></script>
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


        <script src="/assets/js/samediff/generated/uigraphevents_generated.js"></script>
        <script src="/assets/js/samediff/generated/uigraphstatic_generated.js"></script>
        <script src="/assets/js/samediff/generated/array_generated.js"></script>
        <script src="/assets/js/samediff/generated/utils_generated.js"></script>
        <script src="/assets/js/samediff/generated/variable_generated.js"></script>

        <script src="/assets/js/samediff/samediff-ui.js"></script>
        <script src="/assets/js/samediff/flatbuffers-utils.js"></script>

        <div class="container-fluid-full">
            <div class="row-fluid">
                <input type="file" id="file" name="file" />
                <output id="list"></output>
                """),format.raw/*77.78*/("""
                """),format.raw/*78.17*/("""<div class="row-fluid">
                    Layout:
                    <button onclick="setLayout('dagre')">Dagre</button>
                    <button onclick="setLayout('klay_down')">Klay (Down)</button>
                    <button onclick="setLayout('klay_lr')">Klay (Right)</button>
                    """),format.raw/*83.76*/("""
                    """),format.raw/*84.78*/("""
                    """),format.raw/*85.74*/("""
                    """),format.raw/*86.21*/("""<button onclick="setLayout('cose-bilkent')">CoSE Bilkent</button>
                    <button onclick="setLayout('breadthfirst')">Breadth First</button>
                    """),format.raw/*88.74*/("""
                """),format.raw/*89.17*/("""</div>
            </div>
        </div>
        <div id="graphdiv" style="height: calc(100% - 60px);
            width: 100%;
            display: table">

        </div>


            <!-- Execute once on page load -->
        <script>
                document.getElementById('file').addEventListener('change', fileSelect, false);
                $(document).ready(function () """),format.raw/*102.47*/("""{"""),format.raw/*102.48*/("""
                    """),format.raw/*103.21*/("""renderSameDiffGraph();
                """),format.raw/*104.17*/("""}"""),format.raw/*104.18*/(""");
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
object SameDiffUIBackup extends SameDiffUIBackup_Scope0.SameDiffUIBackup
              /*
                  -- GENERATED --
                  DATE: Sat Jan 26 18:11:00 AEDT 2019
                  SOURCE: c:/DL4J/Git/deeplearning4j/deeplearning4j/deeplearning4j-ui-parent/deeplearning4j-play/src/main/views/org/deeplearning4j/ui/views/samediff/SameDiffUIBackup.scala.html
                  HASH: 5bbca606352004bcd370a7bff06c376141f0d082
                  MATRIX: 573->1|669->3|697->5|4607->3948|4653->3966|4993->4333|5043->4412|5093->4487|5143->4509|5346->4737|5392->4755|5813->5147|5843->5148|5894->5170|5963->5210|5993->5211
                  LINES: 20->1|25->1|26->2|101->77|102->78|107->83|108->84|109->85|110->86|112->88|113->89|126->102|126->102|127->103|128->104|128->104
                  -- GENERATED --
              */
          