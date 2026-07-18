import os
import tempfile
import unittest

from agent_platform.indexer.go_parser import index_go_file
from agent_platform.indexer.java_parser import index_java_file
from agent_platform.indexer.fallback import index_fallback_file, detect_language
from agent_platform.indexer.index import CodeIndexer


GO_SOURCE = """\
package audit

type IAuditAdapter interface {
\tCreateTask(ctx context.Context) error
\tComputeResult(ctx context.Context) error
}

func PushAudit(ctx context.Context, adapter IAuditAdapter) error {
\tconfig := getPushConfig(ctx)
\tlock := redis.Lock("push_audit")
\ttask, err := adapter.CreateTask(ctx)
\tif err != nil {
\t\treturn err
\t}
\treturn dorami.AddAuditData(ctx, task)
}

type AuditService struct {
\tadapter IAuditAdapter
}

func (s *AuditService) Handle(ctx context.Context) error {
\treturn s.adapter.CreateTask(ctx)
}
"""

JAVA_SOURCE = """\
package com.example;

public class UserService {
    public void createUser(String name) {
        validate(name);
        repo.save(name);
    }

    private void validate(String name) {
        if (name == null) {
            throw new IllegalArgumentException("name required");
        }
    }
}
"""

PYTHON_SOURCE = """\
def hello():
    print("hello")
"""


class GoParserTest(unittest.TestCase):
    def test_extracts_functions_methods_types(self):
        with tempfile.TemporaryDirectory() as tmp:
            rel = "audit/service.go"
            os.makedirs(os.path.join(tmp, "audit"))
            with open(os.path.join(tmp, rel), "w") as f:
                f.write(GO_SOURCE)
            symbols, chunks = index_go_file(tmp, rel, "onboard")
            sym_names = [s["symbol_name"] for s in symbols]
            self.assertIn("IAuditAdapter", sym_names)
            self.assertIn("PushAudit", sym_names)
            self.assertIn("AuditService", sym_names)
            self.assertIn("AuditService.Handle", sym_names)

            iface = next(s for s in symbols if s["symbol_name"] == "IAuditAdapter")
            self.assertEqual(iface["symbol_type"], "interface")

            struct = next(s for s in symbols if s["symbol_name"] == "AuditService")
            self.assertEqual(struct["symbol_type"], "struct")

            push = next(s for s in symbols if s["symbol_name"] == "PushAudit")
            callees = [c["callee"] for c in push["calls"]]
            self.assertIn("getPushConfig", callees)
            self.assertIn("redis.Lock", callees)
            self.assertIn("adapter.CreateTask", callees)
            self.assertIn("dorami.AddAuditData", callees)
            self.assertEqual(push["qualified_name"], "onboard.audit.PushAudit")

            handle = next(s for s in symbols if s["symbol_name"] == "AuditService.Handle")
            handle_callees = [c["callee"] for c in handle["calls"]]
            self.assertIn("s.adapter.CreateTask", handle_callees)

            self.assertTrue(all(c["repo"] == "onboard" for c in chunks))

    def test_skips_test_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            rel = "audit/service_test.go"
            os.makedirs(os.path.join(tmp, "audit"))
            with open(os.path.join(tmp, rel), "w") as f:
                f.write(GO_SOURCE)
            symbols, chunks = index_go_file(tmp, rel, "onboard")
            self.assertEqual(symbols, [])
            self.assertEqual(chunks, [])


class JavaParserTest(unittest.TestCase):
    def test_extracts_class_and_methods(self):
        with tempfile.TemporaryDirectory() as tmp:
            rel = "src/UserService.java"
            os.makedirs(os.path.join(tmp, "src"))
            with open(os.path.join(tmp, rel), "w") as f:
                f.write(JAVA_SOURCE)
            symbols, chunks = index_java_file(tmp, rel, "test-repo")
            sym_names = [s["symbol_name"] for s in symbols]
            self.assertIn("UserService", sym_names)
            self.assertIn("UserService.createUser", sym_names)
            self.assertIn("UserService.validate", sym_names)
            create_user = next(s for s in symbols if s["symbol_name"] == "UserService.createUser")
            callees = [c["callee"] for c in create_user["calls"]]
            self.assertIn("validate", callees)
            self.assertIn("repo.save", callees)
            self.assertTrue(all(c["repo"] == "test-repo" for c in chunks))


class FallbackTest(unittest.TestCase):
    def test_python_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            rel = "app.py"
            with open(os.path.join(tmp, rel), "w") as f:
                f.write(PYTHON_SOURCE)
            chunks = index_fallback_file(tmp, rel, "test-repo")
            self.assertEqual(len(chunks), 1)
            self.assertEqual(chunks[0]["language"], "python")
            self.assertEqual(chunks[0]["repo"], "test-repo")
            self.assertIn("def hello", chunks[0]["text"])

    def test_detect_language(self):
        self.assertEqual(detect_language("foo.py"), "python")
        self.assertEqual(detect_language("bar.ts"), "typescript")
        self.assertEqual(detect_language("baz.go"), "")


class CodeIndexerTest(unittest.TestCase):
    def test_indexes_mixed_repo(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "src"))
            with open(os.path.join(tmp, "src", "Svc.java"), "w") as f:
                f.write(JAVA_SOURCE)
            with open(os.path.join(tmp, "main.py"), "w") as f:
                f.write(PYTHON_SOURCE)
            os.makedirs(os.path.join(tmp, "pkg"))
            with open(os.path.join(tmp, "pkg", "handler.go"), "w") as f:
                f.write(GO_SOURCE)
            indexer = CodeIndexer()
            result = indexer.index_repo(tmp, "mixed")
            self.assertTrue(result.symbols)
            self.assertTrue(result.chunks)
            repos = {c.get("repo") for c in result.chunks}
            self.assertEqual(repos, {"mixed"})
            languages = {c.get("language") for c in result.chunks}
            self.assertTrue({"go", "java", "python"}.issubset(languages))

    def test_cache_hit(self):
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "a.py"), "w") as f:
                f.write("x = 1\n")
            indexer = CodeIndexer()
            r1 = indexer.index_repo(tmp, "cached")
            r2 = indexer.index_repo(tmp, "cached")
            self.assertIs(r1, r2)


if __name__ == "__main__":
    unittest.main()
