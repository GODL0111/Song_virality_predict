import React from 'react'
import './Recommendations.css'

export default function Recommendations() {
  const recommendations = [
    {
      id: 1,
      title: 'Increase Danceability',
      description: 'Add more rhythmic patterns and upbeat tempo to make the song more danceable.',
      impact: 'High'
    },
    {
      id: 2,
      title: 'Boost Energy Levels',
      description: 'Use louder instruments, faster drums, and more intense vocals to increase energy.',
      impact: 'High'
    },
    {
      id: 3,
      title: 'Enhance Positive Vibes',
      description: 'Increase valence by using major keys and uplifting melodies.',
      impact: 'Medium'
    },
    {
      id: 4,
      title: 'Reduce Acoustic Elements',
      description: 'Mix in electronic or produced sounds to decrease acousticness and modernize the track.',
      impact: 'Medium'
    },
    {
      id: 5,
      title: 'Optimize Song Length',
      description: 'Keep songs between 2:30-4:00 minutes for maximum radio-friendly appeal.',
      impact: 'Medium'
    },
    {
      id: 6,
      title: 'Perfect Your Hook',
      description: 'Create a memorable chorus that listeners want to repeat and share.',
      impact: 'High'
    }
  ]

  const tips = [
    {
      category: 'Production',
      items: [
        'Use modern production techniques and EQ',
        'Layer multiple instruments for depth',
        'Add subtle effects like reverb and delay',
        'Keep bass punchy and clear'
      ]
    },
    {
      category: 'Composition',
      items: [
        'Start strong - hook in first 15 seconds',
        'Use chord progressions that feel fresh',
        'Build tension and release throughout',
        'Create a catchy and memorable melody'
      ]
    },
    {
      category: 'Vocals',
      items: [
        'Clear and well-mixed vocals',
        'Emotional delivery that connects with listeners',
        'Singalong-friendly chorus',
        'Good vocal timing and rhythm'
      ]
    },
    {
      category: 'Marketing',
      items: [
        'Engage with TikTok and short-form platforms',
        'Collaborate with popular creators',
        'Teaser clips and behind-the-scenes content',
        'Consistent release schedule'
      ]
    }
  ]

  return (
    <div className="recommendations">
      <div className="page-header">
        <h2>Recommendations</h2>
        <p>Get actionable insights to improve your song</p>
      </div>

      <div className="recommendations-container">
        {/* Quick Recommendations */}
        <section className="recommendations-section">
          <h3 className="section-title">Quick Improvements</h3>
          <div className="recommendations-grid">
            {recommendations.map((rec) => (
              <div key={rec.id} className="recommendation-card">
                <h4>{rec.title}</h4>
                <p>{rec.description}</p>
                <div className="impact-badge" data-impact={rec.impact.toLowerCase()}>
                  {rec.impact} Impact
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* Tips and Tricks */}
        <section className="tips-section">
          <h3 className="section-title">Tips & Tricks</h3>
          <div className="tips-grid">
            {tips.map((tip, idx) => (
              <div key={idx} className="tip-card">
                <h4 className="tip-category">{tip.category}</h4>
                <ul className="tip-list">
                  {tip.items.map((item, itemIdx) => (
                    <li key={itemIdx}>
                      <span className="checkmark">✓</span>
                      <span>{item}</span>
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
        </section>

        {/* Success Stories */}
        <section className="success-section">
          <h3 className="section-title">What Makes Hits?</h3>
          <div className="success-grid">
            <div className="success-card">
              <div className="success-number">87%</div>
              <p>of viral songs have high danceability</p>
            </div>
            <div className="success-card">
              <div className="success-number">91%</div>
              <p>of hits feature strong vocal performances</p>
            </div>
            <div className="success-card">
              <div className="success-number">76%</div>
              <p>of chart-toppers have catchy hooks</p>
            </div>
            <div className="success-card">
              <div className="success-number">85%</div>
              <p>of modern hits use electronic elements</p>
            </div>
          </div>
        </section>
      </div>
    </div>
  )
}
