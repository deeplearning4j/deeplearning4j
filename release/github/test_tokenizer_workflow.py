"""Tokenizer-only route contracts; no Maven/native execution."""
import re
import subprocess
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def workflow(name):
    return yaml.safe_load((ROOT / '.github/workflows' / name).read_text())


class TokenizerWorkflowTests(unittest.TestCase):
    def test_registered_dispatcher_is_exclusive(self):
        dispatcher = workflow('build-deploy-cross-platform.yml')
        inputs = dispatcher.get('on', dispatcher.get(True))['workflow_dispatch']['inputs']
        self.assertLessEqual(len(inputs), 25)
        self.assertEqual(dispatcher['jobs']['tokenizers-only']['if'],
                         "inputs.workflow == 'tokenizers-only'")
        for name, job in dispatcher['jobs'].items():
            if name != 'tokenizers-only':
                self.assertIn("inputs.workflow != 'tokenizers-only'", job['if'])

    def test_four_hosted_producers_install_only_tokenizers(self):
        doc = workflow('tokenizer-only-release.yml')
        producer = doc['jobs']['produce']
        rows = producer['strategy']['matrix']['include']
        self.assertEqual({r['classifier'] for r in rows},
                         {'linux-x86_64', 'linux-arm64', 'windows-x86_64', 'macosx-arm64'})
        self.assertEqual(len(rows), 4)
        self.assertFalse(producer['strategy']['fail-fast'])
        self.assertEqual(producer['needs'], 'preflight')
        steps = producer['steps']
        build = next(s for s in steps if s.get('name') == 'Install only tokenizer reactor')
        self.assertEqual(build['env']['DL4J_MAVEN_GOAL'], 'install')
        self.assertEqual(build['env']['DL4J_BUILD_SDX'], '0')
        self.assertEqual(build['env']['DL4J_TOKENIZERS_JAVA'], '0')
        self.assertEqual(build['env']['DL4J_RELEASE_METADATA'], '1')
        self.assertIn('-Dproject.build.outputTimestamp=', build['env']['MAVEN_OPTS'])
        self.assertIn('-Dnotimestamp=true', build['env']['MAVEN_OPTS'])
        self.assertIn('--run-tokenizers', build['run'])
        self.assertNotIn('--run-java', build['run'])
        self.assertIn('--run-attempt', build['run'])
        self.assertFalse(any('_release-worker' in s.get('uses', '') or
                             'run-release-worker' in s.get('uses', '') for s in steps))

    def test_windows_path_cannot_switch_the_invoking_shell(self):
        producer = workflow('tokenizer-only-release.yml')['jobs']['produce']
        build = next(s for s in producer['steps'] if s.get('name') == 'Install only tokenizer reactor')
        self.assertEqual(build['shell'], 'bash')
        self.assertIn('/c/msys64/usr/bin:', build['run'])
        self.assertIn('"$BASH" build-scripts/release/cross-platform.sh --run-tokenizers', build['run'])
        self.assertNotRegex(build['run'], r'(?m)^\s*bash build-scripts/release/cross-platform.sh')

    def test_retention_precedes_serialized_non_canceling_upload(self):
        doc = workflow('tokenizer-only-release.yml')
        retain = doc['jobs']['merge-retain']
        publish = doc['jobs']['publish']
        self.assertEqual(retain['needs'], 'produce')
        self.assertEqual(publish['needs'], 'merge-retain')
        self.assertNotIn('concurrency', retain)
        self.assertEqual(publish['concurrency'],
                         {'group': 'ossrh-upload-snapshots', 'cancel-in-progress': False})
        self.assertEqual(publish['if'], '${{ !inputs.dryRun }}')
        script = next(s['run'] for s in publish['steps'] if 'run' in s)
        self.assertLess(script.index(' verify '), script.index(' deploy '))
        self.assertGreater(script.index(' verify --remote '), script.index(' deploy '))
        retention_script = next(s['run'] for s in retain['steps']
                                if s.get('name') == 'Retain and read-back verify before publication queue')
        self.assertIn('checksum="$RUNNER_TEMP/payload.tar.sha256"', retention_script)
        self.assertNotIn('merged/payload.tar.sha256', retention_script)
        self.assertIn('--run-id', script)
        self.assertIn('--run-attempt', script)
        for job in doc['jobs'].values():
            for step in job['steps']:
                if step.get('uses', '').startswith('actions/upload-artifact@'):
                    self.assertEqual(step['with']['retention-days'], 90)
                    self.assertEqual(step['with']['if-no-files-found'], 'error')

    def test_all_embedded_shell_steps_parse(self):
        for name in ('tokenizer-only-release.yml', 'build-deploy-cross-platform.yml'):
            for job in workflow(name)['jobs'].values():
                for step in job.get('steps', []):
                    if step.get('shell') == 'bash' and 'run' in step:
                        script = re.sub(r'\$\{\{.*?\}\}', 'fixture', step['run'])
                        checked = subprocess.run(['bash', '-n'], input=script, text=True,
                                                 capture_output=True)
                        self.assertEqual(checked.returncode, 0, checked.stderr)


if __name__ == '__main__':
    unittest.main()
