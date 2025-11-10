# quick_runner_check.py (repo root)
import importlib, inspect, json, traceback, os, sys
print("Working dir:", os.getcwd())
print("sys.path first entries:", sys.path[:5])
module_name = "backend.test_executor.test_runner"

def run_check():
    try:
        m = importlib.import_module(module_name)
        print("Imported", module_name, "attrs:", [n for n in dir(m) if not n.startswith("_")][:80])
        print("Has execute_tests_for_job?:", hasattr(m, "execute_tests_for_job"))
        if hasattr(m, "execute_tests_for_job"):
            sig = inspect.signature(m.execute_tests_for_job)
            print("Signature:", sig)
            try:
                print("Calling execute_tests_for_job('nonexistent', 'dryrun') ...")
                res = m.execute_tests_for_job("nonexistent", "dryrun")
                print("Returned summary (head):", json.dumps(res, indent=2)[:1000])
            except Exception as e:
                print("Runner raised:", e)
                traceback.print_exc()
    except Exception as e:
        print("IMPORT ERROR for module", module_name, ":", repr(e))
        traceback.print_exc()

if __name__ == "__main__":
    run_check()
