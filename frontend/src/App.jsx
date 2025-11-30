import React, { useState, useEffect } from 'react';

// Inline styles for a sophisticated mathematical aesthetic
const styles = {
  // Color palette - inspired by mathematical notation and academic elegance
  colors: {
    bg: '#0a0a0f',
    bgSecondary: '#12121a',
    bgTertiary: '#1a1a25',
    accent: '#6366f1',
    accentLight: '#818cf8',
    text: '#e2e8f0',
    textMuted: '#94a3b8',
    border: '#2d2d3d',
    success: '#10b981',
    warning: '#f59e0b',
    error: '#ef4444',
  }
};

// Mathematical symbols for decoration
const MathSymbol = ({ children, className = '' }) => (
  <span style={{
    fontFamily: '"Cambria Math", "STIX Two Math", serif',
    opacity: 0.15,
    fontSize: '2rem',
    userSelect: 'none',
  }} className={className}>
    {children}
  </span>
);

// Floating mathematical background
const MathBackground = () => {
  const symbols = ['∫', '∑', '∏', '√', '∞', '∂', '∇', '∆', '∈', '∀', '∃', '⊂', '⊃', '∪', '∩', 'λ', 'π', 'θ', 'φ', '→', '⇒', '≡', '≈'];
  
  return (
    <div style={{
      position: 'fixed',
      inset: 0,
      overflow: 'hidden',
      pointerEvents: 'none',
      zIndex: 0,
    }}>
      {symbols.map((symbol, i) => (
        <span
          key={i}
          style={{
            position: 'absolute',
            left: `${(i * 37) % 100}%`,
            top: `${(i * 53) % 100}%`,
            fontFamily: '"Cambria Math", serif',
            fontSize: `${1 + (i % 3)}rem`,
            color: styles.colors.accent,
            opacity: 0.03 + (i % 5) * 0.01,
            transform: `rotate(${i * 15}deg)`,
            animation: `float ${10 + i % 5}s ease-in-out infinite`,
            animationDelay: `${i * 0.5}s`,
          }}
        >
          {symbol}
        </span>
      ))}
    </div>
  );
};

// Technique badge component
const TechniqueBadge = ({ technique, confidence, info, isSelected }) => {
  const opacity = Math.max(0.3, confidence);
  
  return (
    <div style={{
      padding: '1rem 1.5rem',
      borderRadius: '12px',
      background: isSelected 
        ? `linear-gradient(135deg, ${info.color}22, ${info.color}11)` 
        : styles.colors.bgTertiary,
      border: `1px solid ${isSelected ? info.color : styles.colors.border}`,
      cursor: 'pointer',
      transition: 'all 0.3s ease',
      opacity: opacity,
      transform: isSelected ? 'scale(1.02)' : 'scale(1)',
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '0.5rem' }}>
        <span style={{
          fontSize: '1.5rem',
          fontFamily: '"Cambria Math", serif',
          color: info.color,
        }}>
          {info.icon}
        </span>
        <span style={{ fontWeight: 600, color: styles.colors.text }}>
          {technique}
        </span>
      </div>
      <div style={{
        height: '4px',
        borderRadius: '2px',
        background: styles.colors.bgSecondary,
        overflow: 'hidden',
      }}>
        <div style={{
          height: '100%',
          width: `${confidence * 100}%`,
          background: `linear-gradient(90deg, ${info.color}, ${info.color}aa)`,
          transition: 'width 0.5s ease',
        }} />
      </div>
      <div style={{
        marginTop: '0.5rem',
        fontSize: '0.875rem',
        color: styles.colors.textMuted,
      }}>
        {(confidence * 100).toFixed(1)}% confidence
      </div>
    </div>
  );
};

// Labeling function visualization
const LabelingFunctionViz = ({ lfs }) => {
  if (!lfs || lfs.length === 0) {
    return (
      <div style={{
        padding: '2rem',
        textAlign: 'center',
        color: styles.colors.textMuted,
        fontStyle: 'italic',
      }}>
        No labeling functions triggered
      </div>
    );
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
      {lfs.map((lf, idx) => (
        <div
          key={idx}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '1rem',
            padding: '0.75rem 1rem',
            background: styles.colors.bgTertiary,
            borderRadius: '8px',
            borderLeft: `3px solid ${styles.colors.accent}`,
            animation: `slideIn 0.3s ease forwards`,
            animationDelay: `${idx * 0.1}s`,
            opacity: 0,
          }}
        >
          <code style={{
            fontSize: '0.8rem',
            color: styles.colors.accentLight,
            background: styles.colors.bgSecondary,
            padding: '0.25rem 0.5rem',
            borderRadius: '4px',
            fontFamily: '"JetBrains Mono", "Fira Code", monospace',
          }}>
            {lf.name}
          </code>
          <span style={{ color: styles.colors.textMuted }}>→</span>
          <span style={{ color: styles.colors.text, fontWeight: 500 }}>
            {lf.technique}
          </span>
          <span style={{
            marginLeft: 'auto',
            fontSize: '0.75rem',
            color: styles.colors.textMuted,
            background: styles.colors.bgSecondary,
            padding: '0.25rem 0.5rem',
            borderRadius: '4px',
          }}>
            weight: {lf.weight}
          </span>
        </div>
      ))}
    </div>
  );
};

// Demo proof card
const DemoProofCard = ({ demo, onSelect, isLoading }) => (
  <button
    onClick={() => onSelect(demo.text)}
    disabled={isLoading}
    style={{
      textAlign: 'left',
      padding: '1rem 1.25rem',
      background: styles.colors.bgTertiary,
      border: `1px solid ${styles.colors.border}`,
      borderRadius: '12px',
      cursor: isLoading ? 'not-allowed' : 'pointer',
      transition: 'all 0.2s ease',
      opacity: isLoading ? 0.5 : 1,
      color: styles.colors.text,
    }}
    onMouseEnter={(e) => {
      if (!isLoading) {
        e.currentTarget.style.borderColor = styles.colors.accent;
        e.currentTarget.style.transform = 'translateY(-2px)';
      }
    }}
    onMouseLeave={(e) => {
      e.currentTarget.style.borderColor = styles.colors.border;
      e.currentTarget.style.transform = 'translateY(0)';
    }}
  >
    <div style={{
      fontWeight: 600,
      marginBottom: '0.5rem',
      color: styles.colors.accentLight,
    }}>
      {demo.title}
    </div>
    <div style={{
      fontSize: '0.85rem',
      color: styles.colors.textMuted,
      lineHeight: 1.5,
      display: '-webkit-box',
      WebkitLineClamp: 2,
      WebkitBoxOrient: 'vertical',
      overflow: 'hidden',
    }}>
      {demo.text.slice(0, 120)}...
    </div>
  </button>
);

// Main App component
export default function App() {
  const [proofText, setProofText] = useState('');
  const [analysis, setAnalysis] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [demos, setDemos] = useState([]);
  const [techniques, setTechniques] = useState([]);
  const [error, setError] = useState(null);
  const [showLFs, setShowLFs] = useState(false);

  const API_BASE = 'http://localhost:8000';

  useEffect(() => {
    // Load demo proofs and techniques
    fetch(`${API_BASE}/demo-proofs`)
      .then(res => res.json())
      .then(data => setDemos(data.demos))
      .catch(err => console.error('Failed to load demos:', err));

    fetch(`${API_BASE}/techniques`)
      .then(res => res.json())
      .then(data => setTechniques(data.techniques))
      .catch(err => console.error('Failed to load techniques:', err));
  }, []);

  const analyzeProof = async () => {
    if (!proofText.trim()) {
      setError('Please enter a proof to analyze');
      return;
    }

    setIsLoading(true);
    setError(null);
    setAnalysis(null);

    try {
      const response = await fetch(`${API_BASE}/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: proofText }),
      });

      if (!response.ok) throw new Error('Analysis failed');

      const result = await response.json();
      setAnalysis(result);
    } catch (err) {
      setError('Failed to analyze proof. Make sure the backend is running.');
      console.error(err);
    } finally {
      setIsLoading(false);
    }
  };

  const selectDemo = (text) => {
    setProofText(text);
    setAnalysis(null);
    setError(null);
  };

  return (
    <>
      {/* Global styles */}
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Crimson+Pro:ital,wght@0,400;0,600;0,700;1,400&family=JetBrains+Mono:wght@400;500&display=swap');
        
        * {
          box-sizing: border-box;
          margin: 0;
          padding: 0;
        }
        
        body {
          font-family: 'Crimson Pro', Georgia, serif;
          background: ${styles.colors.bg};
          color: ${styles.colors.text};
          min-height: 100vh;
          line-height: 1.6;
        }
        
        @keyframes float {
          0%, 100% { transform: translateY(0) rotate(0deg); }
          50% { transform: translateY(-20px) rotate(5deg); }
        }
        
        @keyframes slideIn {
          from { opacity: 0; transform: translateX(-10px); }
          to { opacity: 1; transform: translateX(0); }
        }
        
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.5; }
        }
        
        @keyframes fadeIn {
          from { opacity: 0; transform: translateY(10px); }
          to { opacity: 1; transform: translateY(0); }
        }
        
        ::selection {
          background: ${styles.colors.accent}44;
        }
        
        ::-webkit-scrollbar {
          width: 8px;
        }
        
        ::-webkit-scrollbar-track {
          background: ${styles.colors.bgSecondary};
        }
        
        ::-webkit-scrollbar-thumb {
          background: ${styles.colors.border};
          border-radius: 4px;
        }
        
        textarea:focus, button:focus {
          outline: none;
        }
      `}</style>

      <MathBackground />

      <div style={{
        position: 'relative',
        zIndex: 1,
        maxWidth: '1400px',
        margin: '0 auto',
        padding: '2rem',
        minHeight: '100vh',
      }}>
        {/* Header */}
        <header style={{
          textAlign: 'center',
          marginBottom: '3rem',
          animation: 'fadeIn 0.6s ease',
        }}>
          <div style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: '1rem',
            marginBottom: '1rem',
          }}>
            <span style={{
              fontSize: '3rem',
              fontFamily: '"Cambria Math", serif',
              color: styles.colors.accent,
            }}>∴</span>
            <h1 style={{
              fontSize: '2.5rem',
              fontWeight: 700,
              background: `linear-gradient(135deg, ${styles.colors.text}, ${styles.colors.accentLight})`,
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
              letterSpacing: '-0.02em',
            }}>
              Neural Proof Assistant
            </h1>
          </div>
          <p style={{
            color: styles.colors.textMuted,
            fontSize: '1.125rem',
            maxWidth: '600px',
            margin: '0 auto',
          }}>
            Classify mathematical proof techniques using weak supervision and natural language processing
          </p>
          <div style={{
            display: 'flex',
            gap: '0.5rem',
            justifyContent: 'center',
            marginTop: '1rem',
          }}>
            <span style={{
              fontSize: '0.75rem',
              padding: '0.25rem 0.75rem',
              background: styles.colors.bgTertiary,
              borderRadius: '9999px',
              color: styles.colors.textMuted,
              border: `1px solid ${styles.colors.border}`,
            }}>
              Snorkel • Weak Supervision
            </span>
            <span style={{
              fontSize: '0.75rem',
              padding: '0.25rem 0.75rem',
              background: styles.colors.bgTertiary,
              borderRadius: '9999px',
              color: styles.colors.textMuted,
              border: `1px solid ${styles.colors.border}`,
            }}>
              FastAPI • React
            </span>
          </div>
        </header>

        {/* Main content */}
        <main style={{
          display: 'grid',
          gridTemplateColumns: '1fr 1fr',
          gap: '2rem',
          animation: 'fadeIn 0.6s ease 0.2s both',
        }}>
          {/* Left panel - Input */}
          <div style={{
            display: 'flex',
            flexDirection: 'column',
            gap: '1.5rem',
          }}>
            {/* Proof input */}
            <div style={{
              background: styles.colors.bgSecondary,
              borderRadius: '16px',
              border: `1px solid ${styles.colors.border}`,
              padding: '1.5rem',
            }}>
              <label style={{
                display: 'block',
                marginBottom: '0.75rem',
                fontWeight: 600,
                color: styles.colors.text,
              }}>
                <span style={{ marginRight: '0.5rem', fontFamily: '"Cambria Math", serif' }}>📝</span>
                Proof Text
              </label>
              <textarea
                value={proofText}
                onChange={(e) => setProofText(e.target.value)}
                placeholder="Enter a mathematical proof to analyze...

Example: Suppose √2 is rational. Then √2 = a/b where a, b are integers with no common factors..."
                style={{
                  width: '100%',
                  minHeight: '250px',
                  padding: '1rem',
                  background: styles.colors.bgTertiary,
                  border: `1px solid ${styles.colors.border}`,
                  borderRadius: '12px',
                  color: styles.colors.text,
                  fontSize: '1rem',
                  fontFamily: '"Crimson Pro", Georgia, serif',
                  lineHeight: 1.7,
                  resize: 'vertical',
                  transition: 'border-color 0.2s ease',
                }}
                onFocus={(e) => e.target.style.borderColor = styles.colors.accent}
                onBlur={(e) => e.target.style.borderColor = styles.colors.border}
              />
              
              {error && (
                <div style={{
                  marginTop: '0.75rem',
                  padding: '0.75rem 1rem',
                  background: `${styles.colors.error}22`,
                  borderRadius: '8px',
                  color: styles.colors.error,
                  fontSize: '0.9rem',
                }}>
                  {error}
                </div>
              )}
              
              <button
                onClick={analyzeProof}
                disabled={isLoading}
                style={{
                  marginTop: '1rem',
                  width: '100%',
                  padding: '1rem 2rem',
                  background: isLoading 
                    ? styles.colors.bgTertiary 
                    : `linear-gradient(135deg, ${styles.colors.accent}, ${styles.colors.accentLight})`,
                  border: 'none',
                  borderRadius: '12px',
                  color: styles.colors.text,
                  fontSize: '1.1rem',
                  fontWeight: 600,
                  cursor: isLoading ? 'not-allowed' : 'pointer',
                  transition: 'all 0.2s ease',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: '0.75rem',
                }}
              >
                {isLoading ? (
                  <>
                    <span style={{ animation: 'pulse 1s infinite' }}>⟳</span>
                    Analyzing...
                  </>
                ) : (
                  <>
                    <span style={{ fontFamily: '"Cambria Math", serif' }}>∴</span>
                    Analyze Proof Technique
                  </>
                )}
              </button>
            </div>

            {/* Demo proofs */}
            <div style={{
              background: styles.colors.bgSecondary,
              borderRadius: '16px',
              border: `1px solid ${styles.colors.border}`,
              padding: '1.5rem',
            }}>
              <h3 style={{
                marginBottom: '1rem',
                fontWeight: 600,
                color: styles.colors.text,
              }}>
                <span style={{ marginRight: '0.5rem' }}>📚</span>
                Sample Proofs
              </h3>
              <div style={{
                display: 'grid',
                gridTemplateColumns: '1fr 1fr',
                gap: '0.75rem',
              }}>
                {demos.map((demo, idx) => (
                  <DemoProofCard
                    key={idx}
                    demo={demo}
                    onSelect={selectDemo}
                    isLoading={isLoading}
                  />
                ))}
              </div>
            </div>
          </div>

          {/* Right panel - Results */}
          <div style={{
            display: 'flex',
            flexDirection: 'column',
            gap: '1.5rem',
          }}>
            {/* Primary result */}
            <div style={{
              background: styles.colors.bgSecondary,
              borderRadius: '16px',
              border: `1px solid ${styles.colors.border}`,
              padding: '1.5rem',
              minHeight: '200px',
            }}>
              <h3 style={{
                marginBottom: '1rem',
                fontWeight: 600,
                color: styles.colors.text,
              }}>
                <span style={{ marginRight: '0.5rem', fontFamily: '"Cambria Math", serif' }}>⊢</span>
                Classification Result
              </h3>
              
              {!analysis && !isLoading && (
                <div style={{
                  textAlign: 'center',
                  padding: '3rem 2rem',
                  color: styles.colors.textMuted,
                }}>
                  <div style={{
                    fontSize: '3rem',
                    marginBottom: '1rem',
                    opacity: 0.3,
                    fontFamily: '"Cambria Math", serif',
                  }}>
                    ?
                  </div>
                  <p>Enter a proof and click analyze to see the classification</p>
                </div>
              )}
              
              {isLoading && (
                <div style={{
                  textAlign: 'center',
                  padding: '3rem 2rem',
                  color: styles.colors.textMuted,
                }}>
                  <div style={{
                    fontSize: '2rem',
                    marginBottom: '1rem',
                    animation: 'pulse 1s infinite',
                  }}>
                    ⟳
                  </div>
                  <p>Applying labeling functions...</p>
                </div>
              )}
              
              {analysis && !isLoading && (
                <div style={{ animation: 'fadeIn 0.4s ease' }}>
                  <div style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '1.5rem',
                    padding: '1.5rem',
                    background: `linear-gradient(135deg, ${analysis.technique_info.color}22, ${analysis.technique_info.color}11)`,
                    borderRadius: '12px',
                    border: `1px solid ${analysis.technique_info.color}44`,
                    marginBottom: '1rem',
                  }}>
                    <div style={{
                      fontSize: '3rem',
                      fontFamily: '"Cambria Math", serif',
                      color: analysis.technique_info.color,
                    }}>
                      {analysis.technique_info.icon}
                    </div>
                    <div>
                      <div style={{
                        fontSize: '1.5rem',
                        fontWeight: 700,
                        color: styles.colors.text,
                        marginBottom: '0.25rem',
                      }}>
                        {analysis.predicted_technique}
                      </div>
                      <div style={{
                        fontSize: '0.9rem',
                        color: styles.colors.textMuted,
                      }}>
                        {analysis.technique_info.description}
                      </div>
                    </div>
                    <div style={{
                      marginLeft: 'auto',
                      textAlign: 'right',
                    }}>
                      <div style={{
                        fontSize: '2rem',
                        fontWeight: 700,
                        color: analysis.technique_info.color,
                      }}>
                        {(analysis.confidence * 100).toFixed(0)}%
                      </div>
                      <div style={{
                        fontSize: '0.75rem',
                        color: styles.colors.textMuted,
                        textTransform: 'uppercase',
                        letterSpacing: '0.05em',
                      }}>
                        confidence
                      </div>
                    </div>
                  </div>
                  
                  <div style={{
                    display: 'flex',
                    gap: '1rem',
                    fontSize: '0.85rem',
                  }}>
                    <div style={{
                      padding: '0.5rem 1rem',
                      background: styles.colors.bgTertiary,
                      borderRadius: '8px',
                      color: styles.colors.textMuted,
                    }}>
                      <span style={{ color: styles.colors.text, fontWeight: 600 }}>
                        {analysis.triggered_labeling_functions.length}
                      </span> LFs triggered
                    </div>
                    <div style={{
                      padding: '0.5rem 1rem',
                      background: styles.colors.bgTertiary,
                      borderRadius: '8px',
                      color: styles.colors.textMuted,
                    }}>
                      <span style={{ color: styles.colors.text, fontWeight: 600 }}>
                        {(analysis.coverage * 100).toFixed(0)}%
                      </span> coverage
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* Probability distribution */}
            {analysis && (
              <div style={{
                background: styles.colors.bgSecondary,
                borderRadius: '16px',
                border: `1px solid ${styles.colors.border}`,
                padding: '1.5rem',
                animation: 'fadeIn 0.4s ease 0.1s both',
              }}>
                <h3 style={{
                  marginBottom: '1rem',
                  fontWeight: 600,
                  color: styles.colors.text,
                }}>
                  <span style={{ marginRight: '0.5rem', fontFamily: '"Cambria Math", serif' }}>∑</span>
                  Probability Distribution
                </h3>
                <div style={{
                  display: 'grid',
                  gridTemplateColumns: '1fr 1fr',
                  gap: '0.75rem',
                }}>
                  {Object.entries(analysis.probability_distribution)
                    .sort((a, b) => b[1].probability - a[1].probability)
                    .map(([technique, data]) => (
                      <TechniqueBadge
                        key={technique}
                        technique={technique}
                        confidence={data.probability}
                        info={data}
                        isSelected={technique === analysis.predicted_technique}
                      />
                    ))}
                </div>
              </div>
            )}

            {/* Labeling functions */}
            {analysis && (
              <div style={{
                background: styles.colors.bgSecondary,
                borderRadius: '16px',
                border: `1px solid ${styles.colors.border}`,
                padding: '1.5rem',
                animation: 'fadeIn 0.4s ease 0.2s both',
              }}>
                <button
                  onClick={() => setShowLFs(!showLFs)}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    width: '100%',
                    background: 'none',
                    border: 'none',
                    cursor: 'pointer',
                    color: styles.colors.text,
                    fontWeight: 600,
                    fontSize: '1rem',
                    fontFamily: 'inherit',
                  }}
                >
                  <span>
                    <span style={{ marginRight: '0.5rem', fontFamily: '"Cambria Math", serif' }}>λ</span>
                    Triggered Labeling Functions
                  </span>
                  <span style={{
                    transform: showLFs ? 'rotate(180deg)' : 'rotate(0deg)',
                    transition: 'transform 0.2s ease',
                  }}>
                    ▼
                  </span>
                </button>
                
                {showLFs && (
                  <div style={{ marginTop: '1rem' }}>
                    <LabelingFunctionViz lfs={analysis.triggered_labeling_functions} />
                  </div>
                )}
              </div>
            )}
          </div>
        </main>

        {/* Footer */}
        <footer style={{
          marginTop: '4rem',
          textAlign: 'center',
          color: styles.colors.textMuted,
          fontSize: '0.875rem',
          animation: 'fadeIn 0.6s ease 0.4s both',
        }}>
          <div style={{
            display: 'flex',
            justifyContent: 'center',
            gap: '2rem',
            marginBottom: '1rem',
          }}>
            {techniques.slice(0, 7).map((t, idx) => (
              <span
                key={idx}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.5rem',
                  opacity: 0.6,
                }}
                title={t.description}
              >
                <span style={{ color: t.color, fontFamily: '"Cambria Math", serif' }}>
                  {t.icon}
                </span>
                <span style={{ fontSize: '0.75rem' }}>{t.name}</span>
              </span>
            ))}
          </div>
          <p>
            Built with weak supervision using Snorkel-style labeling functions
          </p>
          <p style={{ marginTop: '0.5rem', opacity: 0.6 }}>
            Neural Proof Assistant • Aditya Bajoria
          </p>
        </footer>
      </div>
    </>
  );
}
