// Utility functions for generating insights and recommendations for the Virality Report

export const VIRAL_THRESHOLDS = {
  danceability: { min: 0.65, ideal: 0.75, name: 'Danceability', type: 'higher_better', targetLabel: '> 65%' },
  energy: { min: 0.65, ideal: 0.80, name: 'Energy', type: 'higher_better', targetLabel: '> 65%' },
  valence: { min: 0.40, ideal: 0.60, name: 'Valence', type: 'higher_better', targetLabel: '40% - 80%' },
  speechiness: { max: 0.20, ideal: 0.10, name: 'Speechiness', type: 'lower_better', targetLabel: '< 20%' },
  acousticness: { max: 0.30, ideal: 0.15, name: 'Acousticness', type: 'lower_better', targetLabel: '< 30%' },
  liveness: { max: 0.25, ideal: 0.15, name: 'Liveness', type: 'lower_better', targetLabel: '< 25%' },
  tempo: { min: 90, max: 135, ideal: 120, name: 'Tempo', type: 'range', targetLabel: '90 - 135 BPM' }
};

export const getFeatureStatus = (key, value) => {
  const threshold = VIRAL_THRESHOLDS[key];
  if (!threshold) return { status: 'Unknown', gap: 0, health: 'Average', absGap: 0 };
  
  if (threshold.type === 'higher_better') {
    const gap = value - threshold.ideal;
    if (value >= threshold.min) return { status: 'Optimal', gap, absGap: Math.abs(gap), health: 'Strong' };
    if (value >= threshold.min - 0.15) return { status: 'Close To Target', gap, absGap: Math.abs(gap), health: 'Average' };
    return { status: 'Needs Improvement', gap, absGap: Math.abs(gap), health: 'Weak' };
  } 
  
  if (threshold.type === 'lower_better') {
    const gap = threshold.ideal - value;
    if (value <= threshold.max) return { status: 'Optimal', gap, absGap: Math.abs(gap), health: 'Strong' };
    if (value <= threshold.max + 0.15) return { status: 'Close To Target', gap, absGap: Math.abs(gap), health: 'Average' };
    return { status: 'Needs Improvement', gap, absGap: Math.abs(gap), health: 'Weak' };
  }
  
  if (threshold.type === 'range') {
    let gap = 0;
    if (value < threshold.min) gap = value - threshold.ideal;
    else if (value > threshold.max) gap = threshold.ideal - value;

    if (value >= threshold.min && value <= threshold.max) return { status: 'Optimal', gap, absGap: 0, health: 'Strong' };
    if (value >= threshold.min - 15 && value <= threshold.max + 15) return { status: 'Close To Target', gap, absGap: Math.abs(gap), health: 'Average' };
    return { status: 'Needs Improvement', gap, absGap: Math.abs(gap), health: 'Weak' };
  }
  
  return { status: 'Unknown', gap: 0, absGap: 0, health: 'Average' };
};

export const getBenchmarkData = (features) => {
  return Object.keys(VIRAL_THRESHOLDS).map(key => {
    const value = features[key];
    const { status, gap, absGap, health } = getFeatureStatus(key, value);
    
    return {
      id: key,
      feature: VIRAL_THRESHOLDS[key].name,
      current: value,
      ideal: VIRAL_THRESHOLDS[key].ideal, 
      targetLabel: VIRAL_THRESHOLDS[key].targetLabel,
      status: status,
      gap: gap,
      absGap: absGap,
      health: health
    };
  });
};

export const getHealthSummary = (benchmarks) => {
  const summary = { Strong: 0, Average: 0, Weak: 0 };
  benchmarks.forEach(b => {
    if (summary[b.health] !== undefined) summary[b.health]++;
  });
  return summary;
};

export const getQuickInsights = (benchmarks) => {
  const sortedByAbsGap = [...benchmarks].sort((a, b) => {
    // Normalize tempo gap to 0-1 scale for fair comparison
    const aGap = a.id === 'tempo' ? a.absGap / 200 : a.absGap;
    const bGap = b.id === 'tempo' ? b.absGap / 200 : b.absGap;
    return aGap - bGap;
  });

  const closest = sortedByAbsGap[0];
  const largestGap = sortedByAbsGap[sortedByAbsGap.length - 1];

  const sortedByGap = [...benchmarks].sort((a, b) => {
    const aGap = a.id === 'tempo' ? a.gap / 200 : a.gap;
    const bGap = b.id === 'tempo' ? b.gap / 200 : b.gap;
    return bGap - aGap; // Positive gaps first
  });

  const topStrength = sortedByGap[0];
  const topWeakness = sortedByGap[sortedByGap.length - 1];

  return {
    topStrength: topStrength.feature,
    topWeakness: topWeakness.feature,
    closestToViral: closest.feature,
    largestGap: largestGap.feature
  };
};

export const getPredictionContributors = (benchmarks) => {
  const sorted = [...benchmarks].sort((a, b) => {
    const aGap = a.id === 'tempo' ? a.gap / 200 : a.gap;
    const bGap = b.id === 'tempo' ? b.gap / 200 : b.gap;
    return bGap - aGap; // Most positive to most negative
  });

  const positive = sorted.filter(b => (b.id === 'tempo' ? b.gap > -10 : b.gap > -0.1)).slice(0, 3);
  const negative = sorted.filter(b => (b.id === 'tempo' ? b.gap <= -10 : b.gap <= -0.1)).reverse().slice(0, 3); // Reverse so most negative is first

  const descriptions = {
    'Danceability': {
      pos: 'Current value aligns well with viral trends, promoting rhythmic engagement.',
      neg: 'Current value is below viral benchmarks, reducing user retention and danceability.'
    },
    'Energy': {
      pos: 'High energy matches typical viral hits, preventing skips.',
      neg: 'Current value is significantly below the viral benchmark, reducing engagement potential.'
    },
    'Valence': {
      pos: 'Emotional resonance is well-balanced for mass appeal.',
      neg: 'Emotional tone diverges from optimal viral positivity.'
    },
    'Speechiness': {
      pos: 'Spoken-word elements are kept to an optimal minimum, favoring melodic hooks.',
      neg: 'Excessive spoken word may alienate listeners seeking melodic pop engagement.'
    },
    'Acousticness': {
      pos: 'Modern production style aligns well with algorithmic pop preferences.',
      neg: 'Overly acoustic characteristics might struggle in modern electronic-heavy viral trends.'
    },
    'Liveness': {
      pos: 'Clean studio production quality maximizes algorithmic distribution.',
      neg: 'Live or ambient noise characteristics reduce perceived production value.'
    },
    'Tempo': {
      pos: 'BPM is perfectly seated in the highly viral sweet spot.',
      neg: 'Tempo is outside the standard viral range, misaligning with common groove patterns.'
    }
  };

  return {
    positive: positive.map(b => ({ feature: b.feature, desc: descriptions[b.feature].pos })),
    negative: negative.map(b => ({ feature: b.feature, desc: descriptions[b.feature].neg }))
  };
};

export const getViralityAlignment = (benchmarks, viralScore) => {
  // Normalize gaps to calculate an overall alignment percentage
  let totalAlignment = 0;
  benchmarks.forEach(b => {
    const maxGap = b.id === 'tempo' ? 100 : 1.0;
    const normalizedAbsGap = Math.min((b.id === 'tempo' ? b.absGap : b.absGap) / maxGap, 1);
    totalAlignment += (1 - normalizedAbsGap);
  });
  const alignmentPercent = (totalAlignment / benchmarks.length) * 100;

  const meeting = benchmarks.filter(b => b.status === 'Optimal' || b.status === 'Close To Target').length;
  const requiring = benchmarks.length - meeting;

  return {
    score: viralScore,
    alignment: alignmentPercent.toFixed(0),
    meeting,
    requiring,
    total: benchmarks.length
  };
};

export const getAudioCharacterSummary = (features, isViral) => {
  const dLevel = features.danceability > 0.65 ? 'high' : 'moderate';
  const eLevel = features.energy > 0.65 ? 'high' : 'low';
  const vLevel = features.valence > 0.5 ? 'uplifting' : 'melancholic';
  
  let summary = `This track exhibits ${dLevel} danceability, ${eLevel} energy, and ${vLevel} emotional characteristics. `;
  
  if (isViral) {
    summary += "The overall sonic profile is highly optimized for mass engagement, aligning closely with prevailing trends in viral music.";
  } else {
    summary += "The primary factor limiting broader virality is its deviation from standard energetic and rhythmic benchmarks found in modern hits.";
  }
  
  return summary;
};

export const getPriorityAndDifficulty = (gap, id) => {
  if (id === 'tempo') {
    if (gap < -20) return { priority: 'Critical', difficulty: 'Moderate' };
    if (gap < -10) return { priority: 'Important', difficulty: 'Low' };
    return { priority: 'Minor', difficulty: 'Low' };
  } else {
    if (gap < -0.20) return { priority: 'Critical', difficulty: 'High' };
    if (gap < -0.10) return { priority: 'Important', difficulty: 'Moderate' };
    return { priority: 'Minor', difficulty: 'Low' };
  }
};

export const generateRecommendations = (features) => {
  const recommendations = [];
  const benchmarks = getBenchmarkData(features);
  
  benchmarks.forEach(b => {
    if (b.status === 'Needs Improvement' || b.status === 'Close To Target') {
      const { priority, difficulty } = getPriorityAndDifficulty(b.gap, b.id);
      let why = '';
      let actions = [];
      
      const formatVal = (v) => b.id === 'tempo' ? `${v.toFixed(0)} BPM` : `${(v * 100).toFixed(1)}%`;
      
      switch(b.id) {
        case 'danceability':
          why = 'Strong danceability is highly correlated with user retention, playlist placements, and viral loop engagement.';
          actions = [
            'Improve rhythmic consistency throughout the track',
            'Solidify the bassline and drum groove pattern',
            'Emphasize the downbeat to make the track more danceable'
          ];
          break;
        case 'energy':
          why = 'High-energy tracks generally perform better in short-form social media environments, preventing early skips.';
          actions = [
            'Increase kick and bass prominence in the mix',
            'Add stronger dynamic build-ups before the chorus',
            'Improve perceived loudness via mastering compression',
            'Reduce sparse or overly quiet sections'
          ];
          break;
        case 'valence':
          why = 'While melancholic tracks can succeed, viral pop tends to favor uplifting, positive emotional resonance.';
          actions = [
            'Consider introducing more uplifting melodic progressions',
            'Utilize major chords in the primary hook',
            'Brighten the upper harmonic frequencies in the mix'
          ];
          break;
        case 'speechiness':
          why = 'Excessive spoken word or rap-like cadence in pop/electronic genres can alienate listeners seeking melodic hooks.';
          actions = [
            'Reduce spoken-word dominance',
            'Elongate vocal phrasing to favor sustained notes',
            'Ensure the primary hook is melodically driven'
          ];
          break;
        case 'acousticness':
          why = 'Modern viral tracks lean heavily on electronic and synthesized production rather than purely acoustic elements.';
          actions = [
            'Blend acoustic instruments with synthesized textures',
            'Layer acoustic drums with punchy electronic samples',
            'Introduce modern synth basslines'
          ];
          break;
        case 'liveness':
          why = 'Studio-quality, clean mixes perform significantly better algorithmically than tracks sounding like live concert recordings.';
          actions = [
            'Reduce ambient room noise using gates or AI isolation',
            'Tighten vocal takes and apply precise studio EQ',
            'Reduce excessive concert-hall reverb on the master bus'
          ];
          break;
        case 'tempo':
          why = 'Tempos outside the 90-135 BPM sweet spot often struggle to align with viral dance trends or common walking paces.';
          actions = b.gap < 0 && b.current < 90
            ? ['Increase the BPM slightly to inject more urgency', 'Align with standard 110-125 pop/dance tempos']
            : ['Reduce the BPM slightly to anchor the groove', 'Make the track more accessible for short-form choreography'];
          break;
        default:
          break;
      }
      
      if (why && actions.length > 0) {
        recommendations.push({
          feature: b.feature,
          current: formatVal(b.current),
          target: b.targetLabel,
          why,
          actions,
          priority,
          difficulty
        });
      }
    }
  });
  
  const priorityScore = { 'Critical': 3, 'Important': 2, 'Minor': 1 };
  recommendations.sort((a, b) => priorityScore[b.priority] - priorityScore[a.priority]);
  
  return recommendations;
};
