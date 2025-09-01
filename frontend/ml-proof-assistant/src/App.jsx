import { useState } from "react";

export default function App() {
  const [proof, setProof] = useState("");
  const [tactic, setTactic] = useState("");
  const [subject, setSubject] = useState("");
  const [top3Tatic, setTop3Tactic] = useState([]);
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
      setTop3Tactic(Array.isArray(data?.top3?.tactic) ? data.top3.tactic : []);
      setTop3Subject(
        Array.isArray(data?.top3?.subject) ? data.top3.subject : []
      );
    } catch (error) {
      console.error("Error contacting backend:", error);
      setTactic("Error: Could not get prediction.");
    }
  };
  return (
    <div>
      <h1>Neural Proof Assistant</h1>
      <form
        onSubmit={(e) => {
          e.preventDefault();
          handlePredict();
        }}
      >
        <label>
          Enter Mathematics Proof pls:
          <input
            type="text"
            value={proof}
            onChange={(e) => setProof(e.target.value)}
          />
        </label>
        <button type="submit">Predict Tactic</button>
      </form>
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

      {top3Tatic.length > 0 && (
        <div>
          <h4>Top 3 Tactics</h4>
          <ol>
            {top3Tatic.map((t, i) => (
              <li key={`tac-${i}`}>
                {String(t.label)}{" "}
                {t.proba != null && `(${Number(t.proba).toFixed(3)})`}
              </li>
            ))}
          </ol>
        </div>
      )}
      {top3Subject.length > 0 && (
        <div>
          <h4>Top 3 Subjects</h4>
          <ol>
            {top3Subject.map((s, i) => (
              <li key={`subj-${i}`}>
                {String(s.label)}{" "}
                {s.proba != null && `(conf: ${Number(s.proba).toFixed(3)})`}
              </li>
            ))}
          </ol>
        </div>
      )}
    </div>
  );
}
