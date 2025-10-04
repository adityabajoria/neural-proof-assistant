import { useState } from "react";

export default function App() {
  const [proof, setProof] = useState("");
  const [tactic, setTactic] = useState("");
  const [subject, setSubject] = useState("");
  const [top3Tactic, setTop3Tactic] = useState([]); // fixed spelling
  const [top3Subject, setTop3Subject] = useState([]);

  const handlePredict = async () => {
    try {
      const response = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ goal: proof }),
      });

      const data = await response.json();
      setTactic(data.tactic ?? "");
      setSubject(data.subject ?? "");
      setTop3Tactic(Array.isArray(data?.top3?.tactic) ? data.top3.tactic : []); // fixed
      setTop3Subject(
        Array.isArray(data?.top3?.subject) ? data.top3.subject : []
      );
    } catch (error) {
      console.error("Error contacting backend:", error);
      setTactic("Error: Could not get prediction.");
    }
  };

  return (
    <div style={{ fontFamily: "sans-serif", padding: "20px" }}>
      <h1>Neural Proof Assistant</h1>

      <label>
        Enter Mathematics Proof:
        <input
          type="text"
          value={proof}
          onChange={(e) => setProof(e.target.value)}
          style={{ marginLeft: "10px" }}
        />
      </label>
      <button
        onClick={handlePredict}
        style={{
          marginLeft: "10px",
          padding: "6px 12px",
          cursor: "pointer",
          borderRadius: "5px",
        }}
      >
        Predict Tactic
      </button>

      {tactic && (
        <div>
          <h3>Predicted Tactic: {String(tactic)}</h3>
        </div>
      )}
      {subject && (
        <div>
          <h3>Predicted Subject: {String(subject)}</h3>
        </div>
      )}

      {top3Tactic.length > 0 && (
        <div>
          <h4>Top 3 Tactics (with probabilities)</h4>
          {top3Tactic.map((t, i) => (
            <div key={`tac-${i}`} style={{ marginBottom: "6px" }}>
              <strong>{t.label}</strong> ({(t.proba * 100).toFixed(1)}%)
              <div
                style={{
                  height: "10px",
                  width: "200px",
                  backgroundColor: "#eee",
                  borderRadius: "5px",
                  overflow: "hidden",
                }}
              >
                <div
                  style={{
                    width: `${t.proba * 100}%`,
                    height: "100%",
                    backgroundColor: "#007bff",
                  }}
                />
              </div>
            </div>
          ))}
        </div>
      )}

      {top3Subject.length > 0 && (
        <div>
          <h4>Top 3 Subjects (with confidence)</h4>
          {top3Subject.map((s, i) => (
            <div key={`subj-${i}`} style={{ marginBottom: "6px" }}>
              <strong>{s.label}</strong> ({(s.proba * 100).toFixed(1)}%)
              <div
                style={{
                  height: "10px",
                  width: "200px",
                  backgroundColor: "#eee",
                  borderRadius: "5px",
                  overflow: "hidden",
                }}
              >
                <div
                  style={{
                    width: `${s.proba * 100}%`,
                    height: "100%",
                    backgroundColor: "#28a745",
                  }}
                />
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
