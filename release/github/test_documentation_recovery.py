"""Remote-only contracts for fail-closed, heading-only release recovery."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from release.github import documentation_recovery as docs


class DocumentationRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source, self.fix, self.output = [self.root / name for name in ('source', 'fix', 'output')]
        self.output.mkdir()
        self.fix.mkdir()
        self.originals, self.fixed = {}, {}
        for name, repairs in docs.REPAIRS.items():
            lines = [b'\n'] * (max(repairs) + 1)
            for number, heading in repairs.items():
                lines[number - 1] = ('     * <h3>' + heading + '</h3>\n').encode()
            original = b''.join(lines)
            path = self.source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(original)
            self.originals[name] = original
            self.fixed[name] = docs.repaired(original, repairs)

    def git(self, command, cwd, **kwargs):
        if command[1] == 'rev-parse':
            return docs.SOURCE_COMMIT if cwd == self.source else 'b' * 40
        name = command[2].split(':', 1)[1]
        return (self.originals if cwd == self.source else self.fixed)[name]

    def prepare(self):
        with patch.object(docs.subprocess, 'check_output', side_effect=self.git):
            return docs.prepare(self.source, self.fix, 'b' * 40, docs.SOURCE_COMMIT, self.output)

    def test_records_both_revisions_and_all_file_hashes(self):
        result = self.prepare()
        self.assertEqual(docs.SOURCE_COMMIT, result['sourceCommit'])
        self.assertEqual('b' * 40, result['documentationFixCommit'])
        self.assertEqual(6, sum(len(row['lines']) for row in result['files']))
        for name in docs.REPAIRS:
            self.assertEqual(self.fixed[name], (self.source / name).read_bytes())
        self.assertEqual(result, json.loads((self.output / 'documentation-recovery-provenance.json').read_text()))

    def test_code_change_rejected_before_any_write(self):
        name = list(docs.REPAIRS)[-1]
        self.fixed[name] += b'class Injected {}\n'
        with self.assertRaisesRegex(ValueError, 'compiled-code'):
            self.prepare()
        for name in docs.REPAIRS:
            self.assertEqual(self.originals[name], (self.source / name).read_bytes())

    def test_extra_documentation_or_line_number_changes_rejected(self):
        for suffix in (b'/** extra */', b'\n', b'\\u000a'):
            with self.subTest(suffix=suffix):
                name = next(iter(docs.REPAIRS))
                original = self.fixed[name]
                self.fixed[name] += suffix
                with self.assertRaises(ValueError):
                    self.prepare()
                self.fixed[name] = original

    def test_dirty_source_rejected(self):
        (self.source / next(iter(docs.REPAIRS))).write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'pristine'):
            self.prepare()

    def test_wrong_revision_and_unaudited_source_rejected(self):
        with patch.object(docs.subprocess, 'check_output', return_value='c' * 40):
            with self.assertRaisesRegex(ValueError, 'revision'):
                docs.prepare(self.source, self.fix, 'b' * 40, docs.SOURCE_COMMIT, self.output)
        for source, fix in (('a' * 40, 'b' * 40), (docs.SOURCE_COMMIT, 'main')):
            with self.assertRaises(ValueError):
                docs.prepare(self.source, self.fix, fix, source, self.output)

    def test_heading_context_must_match(self):
        with self.assertRaisesRegex(ValueError, 'audited Javadoc'):
            docs.repaired(b'wrong\n', {1: 'Example Usage:'})

    def test_real_pinned_source_matches_only_audited_edits(self):
        original_root = os.environ.get('DOCUMENTATION_CONTRACT_SOURCE')
        if not original_root:
            self.skipTest('remote immutable source checkout required')
        current = Path(__file__).resolve().parents[2]
        for name, repairs in docs.REPAIRS.items():
            original = (Path(original_root) / name).read_bytes()
            self.assertEqual(docs.repaired(original, repairs), (current / name).read_bytes(), name)
