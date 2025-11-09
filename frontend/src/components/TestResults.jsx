// frontend/src/components/TestResults.jsx
import React, { useEffect, useState } from "react";
import axios from "axios";

const TestResults = ({ jobId }) => {
  const [summary, setSummary] = useState(null);
  const [healMeta, setHealMeta] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const fetchSummary = async () => {
    if (!jobId) return;
    setLoading(true);
    setError(null);
    try {
      const res = await axios.get(`http://localhost:8000/api/test-result/${jobId}`);
      setSummary(res.data.summary ?? res.data);
    } catch (err) {
      setError(err?.response?.data?.detail || err.message || "Failed to fetch test result");
    } finally {
      setLoading(false);
    }
  };

  const fetchHeal = async () => {
    if (!jobId) return;
    try {
      const res = await axios.get(`http://localhost:8000/api/heal-status/${jobId}`);
      setHealMeta(res.data);
    } catch (err) {
      // non-fatal
      console.warn("Failed to fetch heal status:", err?.message || err);
    }
  };

  useEffect(() => {
    if (!jobId) return;
    fetchSummary();
    fetchHeal();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [jobId]);

  const refresh = async () => {
    await fetchSummary();
    await fetchHeal();
  };

  const downloadArtifact = (url) => {
    // open artifact path in new tab assuming backend serves raw files (you may add an endpoint)
    window.open(url, "_blank");
  };

  return (
    <div className="p-6 bg-gray-900 rounded-2xl shadow-lg w-full max-w-2xl mx-auto text-gray-100">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold">🧪 Test Results</h3>
        <div>
          <button
            onClick={refresh}
            className="px-3 py-1 bg-blue-600 rounded-lg hover:bg-blue-700 text-sm"
            disabled={!jobId || loading}
          >
            {loading ? "Refreshing..." : "Refresh"}
          </button>
        </div>
      </div>

      {!jobId && <div className="text-sm text-gray-400">Select or create a job to view results.</div>}

      {error && (
        <div className="mb-3 text-sm text-red-400">
          Error: {error}
        </div>
      )}

      <div className="mb-4">
        <div className="text-sm text-gray-300 mb-2"><strong>Job ID:</strong> {jobId}</div>

        <div className="bg-gray-800 rounded-md p-3 text-sm text-gray-200">
          <div className="font-medium mb-2">Summary</div>
          {summary ? (
            <pre className="text-xs max-h-48 overflow-auto bg-transparent p-2">
              {JSON.stringify(summary, null, 2)}
            </pre>
          ) : (
            <div className="text-sm text-gray-400">No test summary available yet.</div>
          )}
        </div>
      </div>

      <div>
        <div className="font-medium mb-2">Self-Heal</div>
        {healMeta ? (
          <div className="bg-gray-800 rounded-md p-3 text-sm text-gray-200">
            <div className="mb-2"><strong>Status:</strong> {healMeta.status}</div>
            <div className="mb-2"><strong>Last heal result:</strong></div>
            <pre className="text-xs max-h-40 overflow-auto bg-transparent p-2">
              {JSON.stringify(healMeta.last_heal_result ?? healMeta.self_heal ?? {}, null, 2)}
            </pre>

            {/* show healed artifact link if available */}
            {healMeta.last_heal_result && healMeta.last_heal_result.healed_collection && (
              <div className="mt-2">
                <button
                  onClick={() => downloadArtifact(`http://localhost:8000/${healMeta.last_heal_result.healed_collection}`)}
                  className="px-3 py-1 bg-green-600 rounded-lg hover:bg-green-700 text-sm"
                >
                  Download Healed Collection
                </button>
              </div>
            )}

            {/* fallback: show healed artifact path from job */}
            {healMeta.self_heal && healMeta.self_heal.patches && (
              <div className="mt-3 text-xs text-gray-300">
                <div><strong>Patches:</strong></div>
                <pre className="max-h-36 overflow-auto">{JSON.stringify(healMeta.self_heal.patches, null, 2)}</pre>
              </div>
            )}
          </div>
        ) : (
          <div className="text-sm text-gray-400">No self-heal information available.</div>
        )}
      </div>
    </div>
  );
};

export default TestResults;
