# backend/scanner/backend_parser.py
"""
Backend parser for detecting server-side routes in Node (Express) and Python (FastAPI/Flask).

Exports:
  - parse_backend_routes(root_path: str) -> List[Dict]

Each discovered route dict:
{
  "framework": "express" | "fastapi" | "flask" | "other",
  "method": "GET|POST|PUT|DELETE|PATCH|ALL|UNKNOWN",
  "path": "/some/path",
  "file": "<filename>",
  "line": <int>,
  "handler": "<function_name_or_lambda_or_inline>"
}

Implementation notes:
 - JS/TS detection uses regex heuristics for common Express patterns:
     app.get('/x', ...), router.post("/y", ...), router.route('/a').get(...)
 - Python detection uses the `ast` module to find function defs decorated with
     @app.get(...), @router.post(...), @app.route(...), @bp.route(...), etc.
 - This is intended for robust local demo usage and to provide inputs for the OpenAPI synthesizer.
"""

from pathlib import Path
from typing import List, Dict, Optional
import re
import ast

# JS regexes (similar to project_scanner but focused on backend patterns)
EXPRESS_SIMPLE_RE = re.compile(r"\b(app|router)\.(get|post|put|delete|patch|all)\s*\(\s*(['\"])([^'\"]+)\3", re.IGNORECASE)
EXPRESS_ROUTE_CHAIN_RE = re.compile(r"\b(router|app)\.route\(\s*(['\"])([^'\"]+)\2\)\s*\.\s*(get|post|put|delete|patch|all)", re.IGNORECASE)
EXPRESS_USE_RE = re.compile(r"\b(app|router)\.use\(\s*(['\"])([^'\"]+)\2", re.IGNORECASE)

# python decorator patterns to match: @app.get("/x"), @router.post("/y"), @bp.route("/z", methods=["GET","POST"])
PY_ROUTE_DEC_RE = re.compile(r"(get|post|put|delete|patch|options|head|route)", re.IGNORECASE)


def _safe_read(path: Path) -> Optional[str]:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        try:
            return path.read_text(encoding="latin-1")
        except Exception:
            return None


def _extract_express_from_source(source: str, filename: str) -> List[Dict]:
    routes: List[Dict] = []
    for m in EXPRESS_SIMPLE_RE.finditer(source):
        obj = m.group(1)
        method = m.group(2).upper()
        path = m.group(4)
        line = source.count("\n", 0, m.start()) + 1
        routes.append({
            "framework": "express",
            "method": method,
            "path": path,
            "file": filename,
            "line": line,
            "handler": None
        })

    # route chaining: router.route('/x').get(...)
    for m in EXPRESS_ROUTE_CHAIN_RE.finditer(source):
        path = m.group(3)
        method = m.group(4).upper()
        line = source.count("\n", 0, m.start()) + 1
        routes.append({
            "framework": "express",
            "method": method,
            "path": path,
            "file": filename,
            "line": line,
            "handler": None
        })

    # app.use('/prefix', router) -> note prefix
    for m in EXPRESS_USE_RE.finditer(source):
        prefix = m.group(3)
        line = source.count("\n", 0, m.start()) + 1
        routes.append({
            "framework": "express",
            "method": "USE",
            "path": prefix,
            "file": filename,
            "line": line,
            "handler": "use/mounted-router"
        })

    return routes


# ---------- Python AST parsing ----------
class RouteVisitor(ast.NodeVisitor):
    """
    Visits function defs and collects decorators that look like FastAPI/Flask routes.
    """
    def __init__(self, filename: str):
        self.filename = filename
        self.routes: List[Dict] = []

    def visit_FunctionDef(self, node: ast.FunctionDef):
        # examine decorators
        for dec in node.decorator_list:
            try:
                route_info = self._parse_decorator(dec)
                if route_info:
                    route_info.update({
                        "file": self.filename,
                        "line": node.lineno,
                        "handler": node.name
                    })
                    self.routes.append(route_info)
            except Exception:
                # ignore parse errors for individual decorators
                continue
        # continue visiting inner nodes
        self.generic_visit(node)

    def _parse_decorator(self, dec: ast.AST) -> Optional[Dict]:
        """
        Attempt to parse decorators of form:
          @app.get("/path")
          @router.post("/x")
          @app.route("/x", methods=["GET","POST"])
        """
        # Decorator could be ast.Call or ast.Attribute or ast.Name
        if isinstance(dec, ast.Call):
            func = dec.func
            # get full name like app.get or router.post or bp.route
            name = _get_full_name(func)
            if not name:
                return None
            # lower-case check
            lower = name.lower()
            # extract path argument if present as first positional arg and it's a string
            path = None
            if dec.args:
                first = dec.args[0]
                if isinstance(first, ast.Constant) and isinstance(first.value, str):
                    path = first.value
            # methods kwarg for Flask
            methods = None
            for kw in dec.keywords:
                if kw.arg and kw.arg.lower() == "methods":
                    # methods may be list of strings
                    if isinstance(kw.value, (ast.List, ast.Tuple)):
                        vals = []
                        for elt in kw.value.elts:
                            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                vals.append(elt.value.upper())
                        methods = vals
            # Decide framework and method
            if ".route" in lower:
                # likely Flask or blueprint
                method = "ALL"
                if methods:
                    method = ",".join(methods)
                return {"framework": "flask", "method": method, "path": path or "/",}
            # fastapi style decorators like app.get, router.post
            parts = lower.split(".")
            if len(parts) >= 2 and parts[-1] in {"get", "post", "put", "delete", "patch", "options", "head"}:
                return {"framework": "fastapi", "method": parts[-1].upper(), "path": path or "/"}
        else:
            # decorator without call e.g., @some_decorator
            # skip
            return None
        return None


def _get_full_name(node: ast.AST) -> Optional[str]:
    """
    Attempt to reconstruct full dotted name from ast.Attribute / ast.Name
    """
    if isinstance(node, ast.Attribute):
        value = _get_full_name(node.value)
        if value:
            return f"{value}.{node.attr}"
        else:
            return node.attr
    elif isinstance(node, ast.Name):
        return node.id
    elif isinstance(node, ast.Call):
        return _get_full_name(node.func)
    return None


def _extract_python_routes_from_source(source: str, filename: str) -> List[Dict]:
    routes: List[Dict] = []
    try:
        tree = ast.parse(source)
    except Exception:
        return routes
    visitor = RouteVisitor(filename)
    visitor.visit(tree)
    return visitor.routes


# ------------- Public API -------------
def _is_js_like(path: Path) -> bool:
    return path.suffix.lower() in {".js", ".jsx", ".ts", ".tsx"}

def _is_py_like(path: Path) -> bool:
    return path.suffix.lower() == ".py"


def parse_backend_routes(root_path: str) -> List[Dict]:
    """
    Walk root_path and return a list of discovered backend routes.
    """
    root = Path(root_path)
    if not root.exists():
        raise FileNotFoundError(f"Root path not found: {root_path}")

    results: List[Dict] = []
    # prioritize server/ backend folders
    prioritized = ["server", "backend", "api", "src"]

    file_list = [p for p in root.rglob("*") if p.is_file() and (_is_js_like(p) or _is_py_like(p))]

    # sort by priority (so backend directories are scanned earlier)
    def _prio(p: Path):
        for i, name in enumerate(prioritized):
            if name in p.parts:
                return i
        return len(prioritized)
    file_list = sorted(file_list, key=_prio)

    for path in file_list:
        try:
            content = _safe_read(path)
            if not content:
                continue
            if _is_js_like(path):
                extracted = _extract_express_from_source(content, str(path))
                if extracted:
                    results.extend(extracted)
            elif _is_py_like(path):
                extracted = _extract_python_routes_from_source(content, str(path))
                if extracted:
                    results.extend(extracted)
        except Exception:
            # non-fatal; continue scanning other files
            continue

    # Normalize results: ensure required keys exist
    normalized = []
    for r in results:
        normalized.append({
            "framework": r.get("framework", "other"),
            "method": r.get("method", "UNKNOWN"),
            "path": r.get("path", "/"),
            "file": r.get("file"),
            "line": r.get("line", 0),
            "handler": r.get("handler"),
        })

    return normalized
