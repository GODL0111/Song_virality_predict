import React from 'react'
import './Creators.css'

export default function Creators() {
  const creators = [
    {
      id: 1,
      name: 'Arin Karmakar',
      title: 'Full Stack Developer & ML Engineer',
      role: '👨‍💻 Lead Developer',
      bio: 'Built the complete Song Virality Prediction system with AI/ML integration, React frontend, and Flask API. Expertise in machine learning, full-stack development, and blockchain integration.',
      expertise: ['Python', 'React', 'Machine Learning', 'Flask', 'XGBoost', 'Blockchain'],
      social: {
        github: 'https://github.com/GODL0111',
        linkedin: '#',
        twitter: '#'
      },
      image: '👨‍💼'
    }
  ]

  const features = [
    {
      title: 'AI-Powered Predictions',
      description: 'XGBoost machine learning model trained on thousands of songs',
      icon: '🤖'
    },
    {
      title: 'Real-Time Analysis',
      description: 'Instant feedback on song viral potential with detailed metrics',
      icon: '⚡'
    },
    {
      title: 'Beautiful UI',
      description: '3D animations, responsive design, and smooth interactions',
      icon: '✨'
    },
    {
      title: 'Live Testing',
      description: 'Upload and analyze your own songs with visual feedback',
      icon: '🎧'
    },
    {
      title: 'Smart Recommendations',
      description: 'Get actionable insights to improve your music production',
      icon: '💡'
    },
    {
      title: 'Blockchain Ready',
      description: 'Avalanche smart contract integration for Web3 capabilities',
      icon: '⛓️'
    }
  ]

  const stack = [
    {
      category: 'Frontend',
      technologies: ['React 18', 'Vite', 'CSS3', 'JavaScript ES6+']
    },
    {
      category: 'Backend',
      technologies: ['Python 3.11', 'Flask', 'XGBoost', 'Scikit-learn']
    },
    {
      category: 'ML/AI',
      technologies: ['Machine Learning', 'Feature Engineering', 'Data Analysis', 'Model Training']
    },
    {
      category: 'Blockchain',
      technologies: ['Avalanche', 'Solidity', 'Web3.py', 'Smart Contracts']
    },
    {
      category: 'Deployment',
      technologies: ['Vercel', 'Git', 'Docker', 'CI/CD']
    },
    {
      category: 'Tools',
      technologies: ['VS Code', 'Hardhat', 'Pandas', 'Numpy']
    }
  ]

  return (
    <div className="creators">
      <div className="page-header">
        <h2>👨‍💻 Meet the Creator</h2>
        <p>Brilliant minds behind SoundViral</p>
      </div>

      <div className="creators-container">
        {/* Creator Cards */}
        <section className="creators-section">
          <div className="creators-grid">
            {creators.map((creator) => (
              <div key={creator.id} className="creator-card">
                <div className="creator-header">
                  <div className="creator-image">{creator.image}</div>
                  <div className="role-badge">{creator.role}</div>
                </div>

                <div className="creator-info">
                  <h3>{creator.name}</h3>
                  <p className="creator-title">{creator.title}</p>
                  <p className="creator-bio">{creator.bio}</p>

                  <div className="expertise-section">
                    <h4>Expertise</h4>
                    <div className="expertise-tags">
                      {creator.expertise.map((skill, idx) => (
                        <span key={idx} className="expertise-tag">
                          {skill}
                        </span>
                      ))}
                    </div>
                  </div>

                  <div className="social-links">
                    <a href={creator.social.github} className="social-link" title="GitHub">
                      <span>GitHub</span>
                    </a>
                    <a href={creator.social.linkedin} className="social-link" title="LinkedIn">
                      <span>LinkedIn</span>
                    </a>
                    <a href={creator.social.twitter} className="social-link" title="Twitter">
                      <span>Twitter</span>
                    </a>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* Key Features */}
        <section className="features-section">
          <h3 className="section-title">Project Highlights</h3>
          <div className="features-grid">
            {features.map((feature, idx) => (
              <div key={idx} className="feature-card">
                <div className="feature-icon">{feature.icon}</div>
                <h4>{feature.title}</h4>
                <p>{feature.description}</p>
              </div>
            ))}
          </div>
        </section>

        {/* Tech Stack */}
        <section className="stack-section">
          <h3 className="section-title">Technology Stack</h3>
          <div className="stack-grid">
            {stack.map((group, idx) => (
              <div key={idx} className="stack-card">
                <h4 className="stack-category">{group.category}</h4>
                <ul className="tech-list">
                  {group.technologies.map((tech, techIdx) => (
                    <li key={techIdx}>
                      <span className="tech-dot"></span>
                      {tech}
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
        </section>

        {/* About Project */}
        <section className="about-section">
          <h3 className="section-title">About SoundViral</h3>
          <div className="about-content">
            <p>
              <strong>SoundViral</strong> is an AI-powered platform that predicts whether a song will go viral
              using advanced machine learning algorithms and deep analysis of musical features.
            </p>
            <p>
              The system combines multiple technologies including XGBoost for predictions, React for an intuitive
              UI, Flask for the API backend, and blockchain integration for Web3 capabilities.
            </p>
            <p>
              Whether you're a music producer, artist, or music enthusiast, SoundViral helps you understand
              what makes a song successful and provides actionable recommendations for improvement.
            </p>
            <div className="project-stats">
              <div className="stat">
                <span className="stat-value">8000+</span>
                <span className="stat-label">Songs Analyzed</span>
              </div>
              <div className="stat">
                <span className="stat-value">87%</span>
                <span className="stat-label">Accuracy Rate</span>
              </div>
              <div className="stat">
                <span className="stat-value">6+</span>
                <span className="stat-label">Tech Stacks</span>
              </div>
              <div className="stat">
                <span className="stat-value">1</span>
                <span className="stat-label">Developer</span>
              </div>
            </div>
          </div>
        </section>

        {/* Contact CTA */}
        <section className="contact-section">
          <h3>Get In Touch</h3>
          <p>Have questions or feedback? We'd love to hear from you!</p>
          <div className="contact-buttons">
            <button className="btn primary large">
              💌 Send Feedback
            </button>
            <button className="btn large">
              ⭐ Star on GitHub
            </button>
          </div>
        </section>
      </div>
    </div>
  )
}
