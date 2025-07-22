import { useState } from "react";

export default function App() {
  const [proof, setProof] = useState("");
  const [tactic, setTactic] = useState("");

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
      setTactic(data.tactic); // Show returned tactic
    } catch (error) {
      console.error("Error contacting backend:", error);
      setTactic("Error: Could not get prediction.");
    }
  };
  return (
    <div>
      <h1>Neural Proof Assistant</h1>
      <label>
        Enter mathematics proof:
        <input
          type="text"
          value={proof}
          onChange={(e) => setProof(e.target.value)}
        />
      </label>
      <button onClick={handlePredict}>Predict Tactic</button>
    </div>
  );
}
