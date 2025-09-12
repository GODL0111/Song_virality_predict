# 🔗 Avalanche Blockchain Integration for Song Virality Predictor

This document describes the Avalanche blockchain integration that extends the Song Virality Predictor to store ML predictions on-chain for transparency, verification, and immutable record-keeping.

## 🎯 Overview

The Avalanche integration adds the following capabilities to the existing Song Hit Predictor:

- **On-chain Prediction Storage**: Store ML predictions on Avalanche blockchain
- **Prediction Verification**: Later verify predictions against actual outcomes
- **Transparency**: All predictions are publicly verifiable on the blockchain
- **Accuracy Tracking**: Calculate predictor accuracy based on verified outcomes
- **Immutable Records**: Predictions cannot be modified once stored

## 🏗️ Architecture

### Components

1. **MusicPredictionOracle.sol** - Solidity smart contract for storing predictions
2. **AvalancheSongPredictor** - Python class extending the base predictor
3. **AvalancheConfig** - Configuration and network management
4. **Deployment Scripts** - Tools for contract deployment

### Workflow

```
Song Features → ML Prediction → Blockchain Storage → Verification
     ↓               ↓               ↓               ↓
  Audio Data    Hit Probability   Transaction     Actual Outcome
  (JSON)        (0-100%)          (AVAX)          (True/False)
```

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.7+
- Avalanche wallet with AVAX for gas fees
- Access to existing Song Hit Predictor

### 2. Installation

```bash
# Install blockchain dependencies
pip install -r requirements_blockchain.txt

# Verify existing dependencies are installed
pip install pandas numpy scikit-learn xgboost matplotlib seaborn joblib scipy
```

### 3. Environment Setup

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your wallet information
nano .env
```

Required environment variables:
- `AVALANCHE_PRIVATE_KEY` - Your wallet's private key (starts with 0x)
- `WALLET_ADDRESS` - Your wallet's public address
- `PREDICTION_ORACLE_ADDRESS` - Contract address (set after deployment)

### 4. Contract Deployment

```bash
# Run deployment helper
python deploy_contract.py

# Follow the instructions for Foundry or Hardhat deployment
```

#### Option A: Foundry Deployment
```bash
forge create \
    --rpc-url https://api.avax-test.network/ext/bc/C/rpc \
    --private-key $AVALANCHE_PRIVATE_KEY \
    contracts/MusicPredictionOracle.sol:MusicPredictionOracle \
    --legacy
```

#### Option B: Hardhat Deployment
```bash
# Add network to hardhat.config.js, then:
npx hardhat run scripts/deploy-oracle.js --network fuji
```

### 5. Configuration Update

After deployment, update your `.env` file:
```bash
PREDICTION_ORACLE_ADDRESS=0x1234...  # Your deployed contract address
```

### 6. Test the Integration

```bash
# Run the demo
python avalanche_song_predictor.py
```

## 📋 Usage Examples

### Basic Prediction with Blockchain Storage

```python
from avalanche_song_predictor import AvalancheSongPredictor

# Initialize predictor (connects to Fuji testnet by default)
predictor = AvalancheSongPredictor(network='fuji')

# Load your trained model
predictor.load_model("best_enhanced_song_model")

# Song features
song_features = {
    'danceability': 0.735,
    'energy': 0.578,
    'key': 1,
    'loudness': -11.840,
    'mode': 0,
    'speechiness': 0.0461,
    'acousticness': 0.514,
    'instrumentalness': 0.0902,
    'liveness': 0.159,
    'valence': 0.636,
    'tempo': 98.002,
    'duration_ms': 207959
}

# Make prediction and store on blockchain
result = predictor.predict_and_store_on_avalanche(song_features)

print(f"Hit Probability: {result['prediction']['hit_probability']:.1%}")
print(f"Transaction Hash: {result['transaction_hash']}")
print(f"Prediction ID: {result['prediction_id']}")
```

### Retrieve Stored Predictions

```python
# Get prediction from blockchain
prediction_id = 5
chain_prediction = predictor.get_prediction_from_chain(prediction_id)

print(f"Predictor: {chain_prediction['predictor']}")
print(f"Timestamp: {chain_prediction['datetime']}")
print(f"Hit Probability: {chain_prediction['hit_probability']:.1%}")
print(f"Verified: {chain_prediction['verified']}")
```

### Check Predictor Accuracy

```python
# Get accuracy statistics
accuracy_stats = predictor.get_predictor_accuracy()

print(f"Accuracy: {accuracy_stats['accuracy_percentage']:.1f}%")
print(f"Total Verified: {accuracy_stats['total_verified_predictions']}")
print(f"Total Predictions: {accuracy_stats['total_predictions']}")
```

### Batch Predictions

```python
songs = [
    {'danceability': 0.8, 'energy': 0.7, ...},
    {'danceability': 0.6, 'energy': 0.9, ...},
    # ... more songs
]

results = []
for song in songs:
    result = predictor.predict_and_store_on_avalanche(song)
    results.append(result)

# All predictions are now stored on blockchain
```

## 🔧 Configuration Options

### Network Configuration

```python
# Fuji testnet (default)
predictor = AvalancheSongPredictor(network='fuji')

# Avalanche mainnet
predictor = AvalancheSongPredictor(network='mainnet')
```

### Prediction Without Blockchain Storage

```python
# Make prediction without storing on blockchain
result = predictor.predict_and_store_on_avalanche(
    song_features, 
    store_on_chain=False
)
```

### Custom Model Directory

```python
predictor = AvalancheSongPredictor(
    model_dir="custom_models",
    data_dir="custom_data",
    network='fuji'
)
```

## 📊 Smart Contract Functions

### Core Functions

- `storePrediction()` - Store a new prediction
- `getPrediction()` - Retrieve prediction by ID
- `verifyPrediction()` - Verify prediction with actual outcome (owner only)
- `calculatePredictorAccuracy()` - Calculate accuracy for a predictor
- `getPredictorHistory()` - Get all predictions by a predictor

### Data Structure

```solidity
struct Prediction {
    address predictor;          // Who made the prediction
    uint256 timestamp;          // When it was made
    uint256 songId;            // Unique song identifier
    uint256 hitProbability;    // Hit probability (0-10000 = 0-100.00%)
    bool isPredictedHit;       // True if predicted as hit
    string modelVersion;       // ML model version used
    string songHash;           // Hash of song features
    bool verified;             // Whether outcome is verified
    bool actualOutcome;        // Actual outcome (if verified)
}
```

## 🛡️ Security Considerations

### Private Key Management
- **Never commit private keys to version control**
- Use dedicated wallets for testing
- Consider hardware wallets for mainnet
- Implement key rotation policies for production

### Smart Contract Security
- Contract has been designed with standard security practices
- Owner-only functions for verification
- Input validation and access controls
- Consider professional audit before mainnet deployment

### Network Security
- Use HTTPS RPC endpoints
- Validate all transaction receipts
- Monitor for unusual gas usage
- Implement transaction timeout handling

## 💰 Cost Estimation

### Fuji Testnet (Free)
- Gas: Free testnet AVAX from faucet
- Storage: ~50,000 gas per prediction (~$0.001 equivalent)

### Avalanche Mainnet
- Gas Price: ~25 Gwei typical
- Storage Cost: ~$0.01-0.05 per prediction
- Daily Usage: 100 predictions ≈ $1-5 in gas fees

## 🧪 Testing Instructions

### For Reviewers

1. **Setup Environment**
   ```bash
   pip install -r requirements_blockchain.txt
   cp .env.example .env
   # Edit .env with your wallet details
   ```

2. **Deploy Contract**
   ```bash
   forge create --rpc-url https://api.avax-test.network/ext/bc/C/rpc \
     --private-key $AVALANCHE_PRIVATE_KEY \
     contracts/MusicPredictionOracle.sol:MusicPredictionOracle
   ```

3. **Update Configuration**
   ```bash
   # Set PREDICTION_ORACLE_ADDRESS in .env to deployed contract address
   ```

4. **Run Demo**
   ```bash
   python avalanche_song_predictor.py
   ```

5. **Verify on Explorer**
   - Visit https://testnet.snowtrace.io
   - Search for your transaction hash
   - Verify contract interactions

### Integration Tests

```python
# Test basic functionality
def test_integration():
    predictor = AvalancheSongPredictor(network='fuji')
    
    # Test prediction without blockchain
    result = predictor.predict_and_store_on_avalanche(
        sample_song, store_on_chain=False
    )
    assert result['prediction']['hit_probability'] is not None
    
    # Test blockchain storage
    result = predictor.predict_and_store_on_avalanche(
        sample_song, store_on_chain=True
    )
    assert result['transaction_hash'] is not None
    assert result['prediction_id'] is not None
    
    # Test retrieval
    chain_prediction = predictor.get_prediction_from_chain(
        result['prediction_id']
    )
    assert chain_prediction['song_id'] == result['song_id']
```

## 🔍 Monitoring and Analytics

### Transaction Monitoring

```python
# Monitor prediction storage
config = AvalancheConfig('fuji')

# Get transaction details
tx_hash = "0x1234..."
receipt = config.get_transaction_receipt(tx_hash)
print(f"Gas Used: {receipt.gasUsed}")
print(f"Status: {'Success' if receipt.status == 1 else 'Failed'}")

# Get explorer URL
explorer_url = config.get_block_explorer_url(tx_hash=tx_hash)
print(f"Explorer: {explorer_url}")
```

### Accuracy Analytics

```python
# Track predictor performance over time
predictor_address = "0x1234..."
history = predictor.get_predictor_history(predictor_address)

for pred_id in history:
    prediction = predictor.get_prediction_from_chain(pred_id)
    if prediction['verified']:
        print(f"Prediction {pred_id}: {prediction['is_predicted_hit']} -> {prediction['actual_outcome']}")
```

## 🛠️ Development and Deployment

### Local Development

```bash
# Install development dependencies
pip install pytest brownie-eth

# Run local tests
pytest tests/test_avalanche_integration.py

# Use local blockchain for testing
brownie console --network development
```

### Production Deployment

1. **Security Review**
   - Audit smart contract code
   - Review private key management
   - Test all error conditions

2. **Mainnet Deployment**
   ```bash
   # Deploy to mainnet (use with caution)
   python deploy_contract.py
   # Choose 'mainnet' when prompted
   ```

3. **Monitoring Setup**
   - Set up transaction monitoring
   - Configure alerting for failed transactions
   - Monitor gas usage and costs

## 📚 Additional Resources

### Documentation
- [Avalanche Documentation](https://docs.avax.network/)
- [Web3.py Documentation](https://web3py.readthedocs.io/)
- [Foundry Documentation](https://book.getfoundry.sh/)

### Networks
- [Fuji Testnet Faucet](https://faucet.avax.network/)
- [Snowtrace Explorer](https://snowtrace.io/)
- [Avalanche Bridge](https://bridge.avax.network/)

### Tools
- [MetaMask Wallet](https://metamask.io/)
- [Remix IDE](https://remix.ethereum.org/)
- [Hardhat Framework](https://hardhat.org/)

## ❓ Troubleshooting

### Common Issues

**"Contract not initialized"**
- Ensure PREDICTION_ORACLE_ADDRESS is set in .env
- Verify contract was deployed successfully

**"Insufficient funds for gas"**
- Check wallet balance: `python -c "from avalanche_config import *; print(get_avalanche_config().get_balance())"`
- Get testnet AVAX from faucet

**"Transaction failed"**
- Check gas limit and price
- Verify network connectivity
- Ensure contract address is correct

**"Private key format invalid"**
- Private key must start with 0x and be 66 characters
- Export correctly from wallet

### Getting Help

1. Check error messages and logs
2. Verify environment variables
3. Test network connectivity
4. Review transaction details on explorer
5. Check contract deployment status

## 🤝 Contributing

### Adding Features

1. Extend AvalancheSongPredictor class
2. Add new smart contract functions if needed
3. Update tests and documentation
4. Follow existing code style

### Reporting Issues

- Provide detailed error messages
- Include environment details
- Share transaction hashes if applicable
- Test on Fuji testnet first

---

**⚠️ Important Security Note**: This integration handles private keys and blockchain transactions. Always use testnet for development and testing. Never commit private keys or secrets to version control. Consider professional security audit before production use.