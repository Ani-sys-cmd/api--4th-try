// frontend/src/components/ScanStatus.jsx
import React, { useEffect, useState, useRef } from "react";
import axios from "axios";

const POLL_INTERVAL_MS = 3000;

const ScanStatus = ({ jobId, onUpdate }) => {
  const [job, setJob] = useState(null);
  const [loadingAction, setLoadingAction] = useState(null);
  const pollRef = useRef(null);

  useEffect(() => {
    if (!jobId) return;

    // initial fetch + start polling
    fetchStatus();

    pollRef.current = setInterval(() => {
      fetchStatus();
    }, POLL_INTERVAL_MS);

    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [jobId]);

  const fetchStatus = async () => {
    try {
      const res = await axios.get(`http://localhost:8000/api/scan-status/${jobId}`);
      setJob(res.data.record || res.data);
      if (onUpdate) onUpdate(res.data.record || res.data);
    } catch (err) {
      // keep previous state; optionally show error
      console.error("Failed to fetch scan status:", err?.response?.data || err.message);
    }
  };

  const startScan = async () => {
    setLoadingAction("start_scan");
    try {
      await axios.post(`http://localhost:8000/api/start-scan/${jobId}`);
      // fetch immediately to update UI
      await fetchStatus();
    } catch (err) {
      console.error("start-scan failed:", err?.response?.data || err.message);
      alert("Failed to start scan. See console for details.");
    } finally {
      setLoadingAction(null);
    }
  };

  const generateTests = async () => {
    setLoadingAction("generate_tests");
    try {
      await axios.post(`http://localhost:8000/api/generate-tests/${jobId}`);
      await fetchStatus();
    } catch (err) {
      console.error("generate-tests failed:", err?.response?.data || err.message);
      alert("Failed to generate tests.");
    } finally {
      setLoadingAction(null);
    }
  };

  const runTests = async () => {
    setLoadingAction("run_tests");
    try {
      await axios.post(`http://localhost:8000/api/run-tests/${jobId}`);
      await fetchStatus();
    } catch (err) {
      console.error("run-tests failed:", err?.response?.data || err.message);
      alert("Failed to run tests.");
    } finally {
      setLoadingAction(null);
    }
  };

  const triggerHeal = async () => {
    setLoadingAction("heal");
    try {
      await axios.post(`http://localhost:8000/api/heal/${jobId}`);
      await fetchStatus();
    } catch (err) {
      console.error("heal failed:", err?.response?.data || err.message);
      alert("Failed to trigger healing.");
    } finally {
      setLoadingAction(null);
    }
  };

  if (!jobId) {
    return (
      <div className="p-4 text-center text-gray-300">
        Provide a job ID to view status.
      </div>
    );
  }

  return (
    <div className="p-6 bg-gray-900 rounded-2xl shadow-lg w-full max-w-2xl mx-auto text-gray-100">
      <h3 className="text-lg font-semibold mb-2">🛰️ Job Status</h3>
      <div className="text-sm text-gray-300 mb-4">
        <div><strong>Job ID:</strong> {jobId}</div>
        <div><strong>Status:</strong> {job?.status ?? "unknown"}</div>
        {job?.project_name && <div><strong>Project:</strong> {job.project_name}</div>}
        {job?.created_at && <div><strong>Created:</strong> {new Date(job.created_at).toLocaleString()}</div>}
      </div>

      <div className="grid grid-cols-2 gap-3 mb-4">
        <button
          onClick={startScan}
          disabled={loadingAction !== null}
          className={`py-2 rounded-lg font-semibold ${loadingAction === "start_scan" ? "bg-gray-600 cursor-not-allowed" : "bg-indigo-600 hover:bg-indigo-700"}`}
        >
          {loadingAction === "start_scan" ? "Starting..." : "Start Scan"}
        </button>

        <button
          onClick={generateTests}
          disabled={loadingAction !== null}
          className={`py-2 rounded-lg font-semibold ${loadingAction === "generate_tests" ? "bg-gray-600 cursor-not-allowed" : "bg-green-600 hover:bg-green-700"}`}
        >
          {loadingAction === "generate_tests" ? "Generating..." : "Generate Tests"}
        </button>

        <button
          onClick={runTests}
          disabled={loadingAction !== null}
          className={`py-2 rounded-lg font-semibold ${loadingAction === "run_tests" ? "bg-gray-600 cursor-not-allowed" : "bg-yellow-600 hover:bg-yellow-700"}`}
        >
          {loadingAction === "run_tests" ? "Running..." : "Run Tests"}
        </button>

        <button
          onClick={triggerHeal}
          disabled={loadingAction !== null}
          className={`py-2 rounded-lg font-semibold ${loadingAction === "heal" ? "bg-gray-600 cursor-not-allowed" : "bg-pink-600 hover:bg-pink-700"}`}
        >
          {loadingAction === "heal" ? "Healing..." : "Trigger Heal"}
        </button>
      </div>

      <div className="bg-gray-800 rounded-md p-3 text-sm text-gray-200">
        <div className="font-medium mb-2">Details</div>
        <pre className="text-xs max-h-44 overflow-auto bg-transparent p-2">
          {JSON.stringify(job ?? {}, null, 2)}
        </pre>
      </div>
    </div>
  );
};

export default ScanStatus;
