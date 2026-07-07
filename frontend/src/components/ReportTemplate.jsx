import React from 'react';
import {
  Chart as ChartJS,
  RadialLinearScale,
  PointElement,
  LineElement,
  Filler,
  Tooltip,
  Legend,
  CategoryScale,
  LinearScale,
  BarElement,
  ArcElement,
  Title,
} from 'chart.js';
import { Bar, Doughnut } from 'react-chartjs-2';
import ChartDataLabels from 'chartjs-plugin-datalabels';
import { 
  AlertCircle, Target, CheckCircle2, AlertTriangle, Activity, Music, 
  TrendingUp, TrendingDown, Minus, Info, Settings2, BarChart2 
} from 'lucide-react';
import { 
  getBenchmarkData, getHealthSummary, generateRecommendations, 
  getQuickInsights, getAudioCharacterSummary, getPredictionContributors, 
  getViralityAlignment 
} from '../utils/reportAnalytics';
import './ReportTemplate.css';

ChartJS.register(
  RadialLinearScale, PointElement, LineElement, Filler, Tooltip, Legend,
  CategoryScale, LinearScale, BarElement, ArcElement, Title, ChartDataLabels
);

const formatAppxVal = (val, formatFn) => {
  if (val === null || val === undefined || isNaN(val)) return 'Not Available';
  return formatFn(val);
};

const getConfidenceLevel = (conf) => {
  if (conf >= 75) return 'High';
  if (conf >= 40) return 'Moderate';
  return 'Low';
};

const ReportTemplate = React.forwardRef(({ result }, ref) => {
  if (!result || !result.features) return null;

  const { features, viralScore, isViral, confidence, fileName } = result;
  
  const benchmarks = getBenchmarkData(features);
  const healthSummary = getHealthSummary(benchmarks);
  const recommendations = generateRecommendations(features);
  const insights = getQuickInsights(benchmarks);
  const characterSummary = getAudioCharacterSummary(features, isViral);
  const contributors = getPredictionContributors(benchmarks);
  const alignment = getViralityAlignment(benchmarks, viralScore);

  const VIRAL_THRESHOLD = 50;
  const gapToViral = viralScore < VIRAL_THRESHOLD ? VIRAL_THRESHOLD - viralScore : 0;
  const confLevel = getConfidenceLevel(confidence);

  // Gauge Chart Data - Add visual cutoff zone at 50%
  const gaugeData = {
    labels: ['Score', 'Gap'],
    datasets: [{
      data: [viralScore, 100 - viralScore],
      backgroundColor: [
        viralScore >= VIRAL_THRESHOLD ? '#10b981' : '#ef4444',
        '#f1f5f9'
      ],
      borderWidth: 0,
      circumference: 180,
      rotation: 270,
      cutout: '80%'
    }]
  };
  
  const barCompareData = {
    labels: benchmarks.filter(b => b.id !== 'tempo').map(b => b.feature),
    datasets: [
      {
        label: 'Current Value',
        data: benchmarks.filter(b => b.id !== 'tempo').map(b => b.current * 100),
        backgroundColor: '#4f46e5',
        barPercentage: 0.7,
        categoryPercentage: 0.8
      },
      {
        label: 'Target Value',
        data: benchmarks.filter(b => b.id !== 'tempo').map(b => b.ideal * 100),
        backgroundColor: '#cbd5e1',
        barPercentage: 0.7,
        categoryPercentage: 0.8
      }
    ]
  };

  const barCompareOptions = {
    indexAxis: 'y',
    responsive: true,
    maintainAspectRatio: false,
    plugins: { 
      legend: { position: 'bottom', labels: { font: { family: 'Inter', size: 13 } } },
      datalabels: {
        color: '#ffffff',
        font: { weight: 'bold', size: 11, family: 'Inter' },
        formatter: (value) => value.toFixed(1) + '%',
        anchor: 'end',
        align: 'start',
        offset: 4,
        display: function(context) { return context.dataset.data[context.dataIndex] > 10; }
      }
    },
    scales: {
      x: { min: 0, max: 100, grid: { color: '#f1f5f9' }, ticks: { callback: (val) => val + '%' } },
      y: { grid: { display: false } }
    }
  };

  const mfccValues = [...Array(20)].map((_, i) => features._all_features[`mfcc_${i+1}`] || 0);
  const mfccData = {
    labels: [...Array(20)].map((_, i) => `M${i+1}`),
    datasets: [{ label: 'MFCC', data: mfccValues, backgroundColor: '#4f46e5' }]
  };

  return (
    <div ref={ref} className="pdf-report-container">
      
      {/* 1. EXECUTIVE DASHBOARD */}
      <div className="pdf-page bg-slate-50">
        <div className="dashboard-header">
          <div className="header-logo"><Music size={28} /></div>
          <div>
            <h1>Executive Analytics Dashboard</h1>
            <p className="subtitle">Music Virality Prediction System • Powered by XGBoost</p>
          </div>
        </div>

        <div className="meta-card">
          <div className="meta-item">
            <span className="meta-label">Analyzed Track</span>
            <span className="meta-value">{fileName}</span>
          </div>
          <div className="meta-item">
            <span className="meta-label">Report Date</span>
            <span className="meta-value">{new Date().toLocaleDateString('en-US', { month: 'long', day: 'numeric', year: 'numeric' })}</span>
          </div>
        </div>

        <div className="dashboard-hero avoid-break">
          {/* Prediction Card */}
          <div className="hero-card gauge-card">
            <h2 className="card-title text-center">Prediction Result</h2>
            <div className="prediction-banner" style={{ background: isViral ? '#ecfdf5' : '#fef2f2', color: isViral ? '#065f46' : '#991b1b' }}>
              {isViral ? 'Viral Hit Detected' : 'Not Viral'}
            </div>
            
            <div className="gauge-wrapper mt-4">
              <Doughnut data={gaugeData} options={{ plugins: { datalabels: { display: false }, tooltip: { enabled: false }, legend: { display: false } }, maintainAspectRatio: false }} />
              <div className="gauge-center">
                <span className={`gauge-score ${isViral ? 'text-green' : 'text-red'}`}>{viralScore}%</span>
                <span className="gauge-label">Virality Score</span>
              </div>
            </div>

            <div className="gauge-context mt-2 text-center">
              <p className="m-0 text-muted">Viral Threshold: <strong>{VIRAL_THRESHOLD}%</strong></p>
              {!isViral && <p className="m-0 text-red fw-semibold">Gap To Viral: {gapToViral.toFixed(1)}%</p>}
            </div>

            <div className="prediction-summary-box mt-4">
               <div className="ps-row">
                 <span className="ps-label">Classification</span>
                 <span className={`ps-val fw-bold ${isViral ? 'text-green' : 'text-red'}`}>{isViral ? 'Viral' : 'Not Viral'}</span>
               </div>
               <div className="ps-row">
                 <span className="ps-label">Top Recommendation</span>
                 <span className="ps-val">{recommendations[0]?.feature || 'None Needed'}</span>
               </div>
            </div>

            <div className="prediction-pill mt-4">
              <Activity size={16} />
              <span>Confidence: <strong>{confidence}%</strong> ({confLevel})</span>
            </div>
          </div>
          
          {/* Quick Insights Panel */}
          <div className="hero-card insights-card d-flex flex-column">
            <h2 className="card-title">Quick Insights Panel</h2>
            <div className="insights-grid flex-grow-1">
              <div className="insight-row">
                <span className="i-label"><TrendingUp size={16} className="text-green" /> Top Strength</span>
                <span className="i-val text-indigo">{insights.topStrength}</span>
              </div>
              <div className="insight-row">
                <span className="i-label"><TrendingDown size={16} className="text-red" /> Top Weakness</span>
                <span className="i-val text-amber">{insights.topWeakness}</span>
              </div>
              <div className="insight-row">
                <span className="i-label"><Target size={16} className="text-indigo" /> Closest To Viral</span>
                <span className="i-val">{insights.closestToViral}</span>
              </div>
              <div className="insight-row">
                <span className="i-label"><AlertTriangle size={16} className="text-amber" /> Largest Gap</span>
                <span className="i-val">{insights.largestGap}</span>
              </div>
            </div>

            <div className="audio-char-box mt-4">
              <h3>Audio Character Summary</h3>
              <p>{characterSummary}</p>
            </div>
          </div>
        </div>
      </div>


      {/* 2. HOW CLOSE ARE YOU & SCORECARD */}
      <div className="pdf-page pt-4">
        <div className="avoid-break mb-6">
          <h2 className="section-title">How Close Are You To Viral?</h2>
          <div className="alignment-banner">
            <div className="align-stat">
              <span className="a-val">{alignment.score}%</span>
              <span className="a-lab">Current Score</span>
            </div>
            <div className="align-divider"></div>
            <div className="align-stat">
              <span className="a-val text-indigo">{alignment.alignment}%</span>
              <span className="a-lab">Benchmark Alignment</span>
            </div>
            <div className="align-divider"></div>
            <div className="align-stat">
              <span className="a-val text-green">{alignment.meeting}/{alignment.total}</span>
              <span className="a-lab">Features Meeting Target</span>
            </div>
            <div className="align-divider"></div>
            <div className="align-stat">
              <span className="a-val text-red">{alignment.requiring}/{alignment.total}</span>
              <span className="a-lab">Features Requiring Fix</span>
            </div>
          </div>
        </div>

        <div className="avoid-break">
          <h2 className="section-title">Feature Health Scorecard</h2>
          <p className="section-desc">Detailed evaluation of core acoustic features against standard viral pop thresholds.</p>
          
          <div className="scorecard-table-wrapper">
            <table className="dashboard-table">
              <thead>
                <tr>
                  <th>Audio Feature</th>
                  <th>Current Value</th>
                  <th>Ideal Viral Range</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {benchmarks.map((b, i) => (
                  <tr key={i}>
                    <td className="fw-semibold">{b.feature}</td>
                    <td><strong className="text-indigo">{b.id === 'tempo' ? b.current.toFixed(0) + ' BPM' : (b.current * 100).toFixed(1) + '%'}</strong></td>
                    <td className="text-muted">{b.targetLabel}</td>
                    <td>
                      <span className={`status-badge status-${b.status.replace(/ /g, '-').toLowerCase()}`}>
                        {b.status === 'Optimal' ? <CheckCircle2 size={14} /> : b.status === 'Close To Target' ? <Minus size={14} /> : <AlertCircle size={14} />}
                        {b.status}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>


      {/* 3. BENCHMARK COMPARISON & WHY IT HAPPENED */}
      <div className="pdf-page pt-4">
        <div className="avoid-break mb-6">
          <h2 className="section-title">Why This Prediction Happened</h2>
          <p className="section-desc">The primary acoustic contributors driving the XGBoost model's decision, ranked by distance from viral optimal bounds.</p>
          
          <div className="contributors-grid">
            <div className="contributor-col pos-col">
              <h3 className="text-green"><TrendingUp size={20} /> Top Positive Contributors</h3>
              {contributors.positive.length > 0 ? contributors.positive.map((c, i) => (
                <div key={i} className="c-card">
                  <div className="c-rank">{i+1}</div>
                  <div className="c-body">
                    <strong>{c.feature}</strong>
                    <p>{c.desc}</p>
                  </div>
                </div>
              )) : (
                <p className="text-muted p-3">No major positive contributors found.</p>
              )}
            </div>
            
            <div className="contributor-col neg-col">
              <h3 className="text-red"><TrendingDown size={20} /> Top Negative Contributors</h3>
              {contributors.negative.length > 0 ? contributors.negative.map((c, i) => (
                <div key={i} className="c-card">
                  <div className="c-rank">{i+1}</div>
                  <div className="c-body">
                    <strong>{c.feature}</strong>
                    <p>{c.desc}</p>
                  </div>
                </div>
              )) : (
                <p className="text-muted p-3">No major negative contributors found.</p>
              )}
            </div>
          </div>
        </div>

        <div className="avoid-break">
          <h2 className="section-title">Core Features vs. Viral Benchmark</h2>
          <p className="section-desc">A direct comparison of normalized DSP values to ideal viral targets with exact labels.</p>
          <div className="large-chart-card">
            <Bar data={barCompareData} options={barCompareOptions} />
          </div>
        </div>
      </div>


      {/* 4. ROAD TO VIRAL RECOMMENDATIONS */}
      <div className="pdf-page pt-4">
        {/* Wrap the title, description, and the FIRST recommendation in a single avoid-break so the title is never orphaned. */}
        <div className="avoid-break mb-4">
          <h2 className="section-title">The Road to Viral</h2>
          <p className="section-desc">Actionable production, mixing, and arrangement advice derived directly from the track's feature gaps.</p>
          
          {recommendations.length > 0 ? (
            <div className="rec-card mb-4">
              <div className="rec-header">
                <div className="rec-title-group">
                  <Settings2 size={20} className="text-indigo" />
                  <h3 className="fw-bold">{recommendations[0].feature}</h3>
                </div>
                <div className="rec-badges">
                  <span className="diff-badge">Difficulty: {recommendations[0].difficulty}</span>
                  <span className={`priority-badge priority-${recommendations[0].priority.toLowerCase()}`}>Priority: {recommendations[0].priority}</span>
                </div>
              </div>
              
              <div className="rec-metrics">
                <div className="metric">
                  <span className="m-label">Current Value</span>
                  <span className="m-val text-red">{recommendations[0].current}</span>
                </div>
                <div className="metric">
                  <span className="m-label">Target Value</span>
                  <span className="m-val text-green">{recommendations[0].target}</span>
                </div>
              </div>

              <div className="rec-body">
                <div className="rec-block">
                  <strong><Info size={16} /> Why it matters:</strong>
                  <p>{recommendations[0].why}</p>
                </div>
                <div className="rec-block action">
                  <strong><Target size={16} /> Specific Actions:</strong>
                  <ul>
                    {recommendations[0].actions.map((act, j) => <li key={j}>{act}</li>)}
                  </ul>
                </div>
              </div>
            </div>
          ) : (
            <div className="rec-card success-card mb-4">
              <CheckCircle2 size={32} className="text-green" />
              <h3 className="fw-bold">Excellent Profile</h3>
              <p>Your track meets or exceeds all core viral benchmarks. No major acoustic corrections are recommended.</p>
            </div>
          )}
        </div>
        
        {/* Render remaining recommendations as separate avoid-break blocks */}
        {recommendations.length > 1 && recommendations.slice(1).map((rec, i) => (
          <div key={i+1} className="rec-card avoid-break mb-4">
            <div className="rec-header">
              <div className="rec-title-group">
                <Settings2 size={20} className="text-indigo" />
                <h3 className="fw-bold">{rec.feature}</h3>
              </div>
              <div className="rec-badges">
                <span className="diff-badge">Difficulty: {rec.difficulty}</span>
                <span className={`priority-badge priority-${rec.priority.toLowerCase()}`}>Priority: {rec.priority}</span>
              </div>
            </div>
            
            <div className="rec-metrics">
              <div className="metric">
                <span className="m-label">Current Value</span>
                <span className="m-val text-red">{rec.current}</span>
              </div>
              <div className="metric">
                <span className="m-label">Target Value</span>
                <span className="m-val text-green">{rec.target}</span>
              </div>
            </div>

            <div className="rec-body">
              <div className="rec-block">
                <strong><Info size={16} /> Why it matters:</strong>
                <p>{rec.why}</p>
              </div>
              <div className="rec-block action">
                <strong><Target size={16} /> Specific Actions:</strong>
                <ul>
                  {rec.actions.map((act, j) => <li key={j}>{act}</li>)}
                </ul>
              </div>
            </div>
          </div>
        ))}
      </div>


      {/* 5. TECHNICAL APPENDIX & CONCLUSION */}
      <div className="pdf-page pt-4">
        {/* Wrap Final Conclusion Title and content together */}
        <div className="avoid-break mb-6">
          <h2 className="section-title">Final Report Conclusion</h2>
          <div className="conclusion-card">
            <div className="c-row">
               <span className="c-lab">Current Viral Potential:</span> 
               <span className="c-val text-indigo fs-large">{viralScore}%</span>
            </div>
            <div className="c-row">
               <span className="c-lab">Primary Limitation:</span> 
               <span className="c-val text-red">{insights.topWeakness || 'None'}</span>
            </div>
            <div className="c-row">
               <span className="c-lab">Secondary Limitation:</span> 
               <span className="c-val text-amber">{recommendations[1]?.feature || 'None'}</span>
            </div>
            <div className="c-actions">
              <strong className="c-lab d-block mb-3">Top Recommended Actions:</strong>
              <ol className="premium-list">
                {recommendations.slice(0, 3).map((r, i) => (
                  <li key={i}><strong>{r.feature}:</strong> {r.actions[0]}</li>
                ))}
                {recommendations.length === 0 && <li>Proceed with release and marketing strategy.</li>}
              </ol>
            </div>
          </div>
        </div>

        {/* Wrap Technical Appendix Title, Tables, and MFCC Chart together to force them on one page */}
        <div className="avoid-break mb-6">
          <h2 className="section-title">Technical Appendix</h2>
          <p className="section-desc">Complete extraction of {features._feature_count} DSP features parsed by the pipeline.</p>
          
          <div className="appendix-grid">
            <div className="appendix-col">
              <h4>Spectral Analysis</h4>
              <table className="dashboard-table small">
                <tbody>
                  <tr><td>Spectral Centroid</td><td>{formatAppxVal(features._all_features?.spectral_centroid_hz, v => v.toFixed(1) + ' Hz')}</td></tr>
                  <tr><td>Spectral Bandwidth</td><td>{formatAppxVal(features._all_features?.spectral_bandwidth_hz, v => v.toFixed(1) + ' Hz')}</td></tr>
                  <tr><td>Spectral Rolloff</td><td>{formatAppxVal(features._all_features?.spectral_rolloff_hz, v => v.toFixed(1) + ' Hz')}</td></tr>
                  <tr><td>Spectral Flatness</td><td>{formatAppxVal(features._all_features?.spectral_flatness_mean, v => (v * 100).toFixed(3) + '%')}</td></tr>
                </tbody>
              </table>
              
              <h4 className="mt-4">Rhythm Analysis</h4>
              <table className="dashboard-table small">
                <tbody>
                  <tr><td>Beat Regularity</td><td>{formatAppxVal(features._all_features?.beat_regularity, v => (v * 100).toFixed(1) + '%')}</td></tr>
                  <tr><td>Rhythm Strength</td><td>{formatAppxVal(features._all_features?.rhythm_strength, v => (v * 100).toFixed(1) + '%')}</td></tr>
                  <tr><td>Onset Strength</td><td>{formatAppxVal(features._all_features?.onset_strength_mean, v => v.toFixed(3))}</td></tr>
                </tbody>
              </table>
            </div>
            
            <div className="appendix-col">
              <h4>Harmonic Analysis</h4>
              <table className="dashboard-table small">
                <tbody>
                  <tr><td>Harmonic Ratio</td><td>{formatAppxVal(features._all_features?.harmonic_ratio, v => (v * 100).toFixed(1) + '%')}</td></tr>
                  <tr><td>Percussive Ratio</td><td>{formatAppxVal(features._all_features?.percussive_ratio, v => (v * 100).toFixed(1) + '%')}</td></tr>
                  <tr><td>Tonnetz Mean</td><td>{formatAppxVal(features._all_features?.tonnetz_mean, v => v.toFixed(5))}</td></tr>
                  <tr><td>Key Strength</td><td>{formatAppxVal(features._all_features?.key_strength, v => (v * 100).toFixed(1) + '%')}</td></tr>
                </tbody>
              </table>

              <h4 className="mt-4">Energy Analysis</h4>
              <table className="dashboard-table small">
                <tbody>
                  <tr><td>RMS Mean</td><td>{formatAppxVal(features._all_features?.rms_mean, v => v.toFixed(4))}</td></tr>
                  <tr><td>Dynamic Range</td><td>{formatAppxVal(features._all_features?.dynamic_range, v => v.toFixed(2) + ' dB')}</td></tr>
                  <tr><td>Zero Crossing</td><td>{formatAppxVal(features._all_features?.zero_crossing_rate, v => v.toFixed(4))}</td></tr>
                </tbody>
              </table>
            </div>
          </div>

          <div className="mfcc-section">
              <div className="mfcc-header-row">
                <h4><BarChart2 size={18} /> Timbral Texture (MFCC Profile)</h4>
              </div>
              <p className="mfcc-desc">Mel-frequency cepstral coefficients mapping the track's unique timbral signature.</p>
              
              <div className="mfcc-chart-wrapper mb-3">
                <Bar data={mfccData} options={{ maintainAspectRatio: false, plugins: { datalabels: { display: false }, legend: { display: false } } }} />
              </div>

              <div className="mfcc-interpretation">
                <strong><Info size={16} /> Interpretation:</strong>
                <p>MFCC analysis indicates a balanced timbral profile with {features._all_features?.mfcc_1 > 0 ? 'prominent' : 'moderate'} harmonic richness and stable spectral characteristics across the frequency bands. This suggests a solid, well-rounded core mix.</p>
              </div>
          </div>
        </div>
      </div>

    </div>
  );
});

export default ReportTemplate;
