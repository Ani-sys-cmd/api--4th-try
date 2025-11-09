// frontend/src/components/RLCharts.jsx
import React, { useEffect, useState } from "react";
import axios from "axios";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
} from "chart.js";
import { Line } from "react-chartjs-2";

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

const POLL_INTERVAL_MS = 3000;

const RLCharts = ({ jobId }) => {
  const [metrics, setMetrics] = useState({ timestamps: [], rewards: [], coverage: [] });
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    let timer = null;
    if (!jobId) return;

    const fetchMetrics = async () => {
      setLoading(true);
      try {
        const res = await axios.get(`http://localhost:8000/api/rl-metrics/${jobId}`);
        // expected shape: { timestamps: [...], rewards: [...], coverage: [...] }
        const payload = res.data || {};
        setMetrics({
          timestamps: payload.timestamps || [],
          rewards: payload.rewards || [],
          coverage: payload.coverage || []
        });
      } catch (err) {
        console.warn("Failed to fetch RL metrics:", err?.message || err);
      } finally {
        setLoading(false);
      }
    };

    fetchMetrics();
    timer = setInterval(fetchMetrics, POLL_INTERVAL_MS);
    return () => {
      if (timer) clearInterval(timer);
    };
  }, [jobId]);

  const labels = metrics.timestamps.length ? metrics.timestamps : metrics.rewards.map((_, i) => `t${i+1}`);

  const data = {
    labels,
    datasets: [
      {
        label: "Reward",
        data: metrics.rewards || [],
        tension: 0.4,
        borderWidth: 2,
        yAxisID: "y"
      },
      {
        label: "Coverage (%)",
        data: metrics.coverage || [],
        tension: 0.4,
        borderWidth: 2,
        yAxisID: "y1"
      }
    ]
  };

  const options = {
    responsive: true,
    interaction: {
      mode: "index",
      intersect: false
    },
    stacked: false,
    plugins: {
      title: {
        display: true,
        text: "RL Metrics (Reward & Coverage)"
      },
      legend: { position: "top" }
    },
    scales: {
      y: {
        type: "linear",
        display: true,
        position: "left",
        title: { display: true, text: "Reward" }
      },
      y1: {
        type: "linear",
        display: true,
        position: "right",
        title: { display: true, text: "Coverage (%)" },
        grid: { drawOnChartArea: false }
      }
    }
  };

  return (
    <div className="p-6 bg-gray-900 rounded-2xl shadow-lg w-full max-w-3xl mx-auto text-gray-100">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold">🤖 RL Insights</h3>
        <div className="text-sm text-gray-400">{loading ? "Loading..." : "Live"}</div>
      </div>

      <div>
        <Line options={options} data={data} />
      </div>
    </div>
  );
};

export default RLCharts;
