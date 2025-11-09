// frontend/src/components/UploadForm.jsx
import React, { useState } from "react";
import axios from "axios";

const UploadForm = ({ onJobCreated }) => {
  const [file, setFile] = useState(null);
  const [gitUrl, setGitUrl] = useState("");
  const [projectName, setProjectName] = useState("");
  const [uploading, setUploading] = useState(false);
  const [status, setStatus] = useState("");

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!file && !gitUrl.trim()) {
      alert("Please select a file or provide a Git URL.");
      return;
    }

    const formData = new FormData();
    if (file) formData.append("file", file);
    if (gitUrl) formData.append("git_url", gitUrl);
    if (projectName) formData.append("project_name", projectName);

    try {
      setUploading(true);
      setStatus("Uploading...");
      const res = await axios.post("http://localhost:8000/api/upload-project", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      const jobId = res.data.job_id;
      setStatus(`✅ Upload successful. Job ID: ${jobId}`);
      if (onJobCreated) onJobCreated(jobId);
    } catch (err) {
      console.error(err);
      setStatus("❌ Upload failed.");
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="p-6 bg-gray-900 rounded-2xl shadow-lg w-full max-w-lg mx-auto text-gray-100">
      <h2 className="text-xl font-bold mb-4 text-center">📦 Upload Project</h2>

      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label className="block text-sm font-medium mb-1">Project Name</label>
          <input
            type="text"
            className="w-full bg-gray-800 border border-gray-700 rounded-lg p-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
            placeholder="My Demo Project"
            value={projectName}
            onChange={(e) => setProjectName(e.target.value)}
          />
        </div>

        <div>
          <label className="block text-sm font-medium mb-1">Upload ZIP</label>
          <input
            type="file"
            accept=".zip,.tar,.gz"
            onChange={handleFileChange}
            className="w-full text-sm text-gray-300"
          />
        </div>

        <div className="text-center text-gray-400">— OR —</div>

        <div>
          <label className="block text-sm font-medium mb-1">Git Repository URL</label>
          <input
            type="url"
            className="w-full bg-gray-800 border border-gray-700 rounded-lg p-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
            placeholder="https://github.com/user/repo.git"
            value={gitUrl}
            onChange={(e) => setGitUrl(e.target.value)}
          />
        </div>

        <button
          type="submit"
          disabled={uploading}
          className={`w-full py-2 rounded-lg font-semibold transition ${
            uploading
              ? "bg-gray-600 cursor-not-allowed"
              : "bg-blue-600 hover:bg-blue-700"
          }`}
        >
          {uploading ? "Uploading..." : "Start Upload"}
        </button>
      </form>

      {status && (
        <div className="mt-4 text-sm text-center text-gray-300">{status}</div>
      )}
    </div>
  );
};

export default UploadForm;
