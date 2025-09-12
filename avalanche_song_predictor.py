"""
Avalanche Song Predictor: Extends SongHitPredictor to store predictions on Avalanche blockchain
"""

import json
import hashlib
import time
from datetime import datetime
from web3 import Web3
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the SongHitPredictor class from predict_V0.2.py
import importlib.util
spec = importlib.util.spec_from_file_location("predict_V0_2", "predict_V0.2.py")
predict_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(predict_module)
SongHitPredictor = predict_module.SongHitPredictor
from avalanche_config import AvalancheConfig


class AvalancheSongPredictor(SongHitPredictor):
    """
    Extended Song Hit Predictor that stores predictions on Avalanche blockchain
    
    This class extends the existing SongHitPredictor to add blockchain functionality,
    allowing predictions to be stored on-chain for later verification and transparency.
    """
    
    # Contract ABI for MusicPredictionOracle
    CONTRACT_ABI = [
        {
            "inputs": [
                {"internalType": "uint256", "name": "_songId", "type": "uint256"},
                {"internalType": "uint256", "name": "_hitProbability", "type": "uint256"},
                {"internalType": "bool", "name": "_isPredictedHit", "type": "bool"},
                {"internalType": "string", "name": "_modelVersion", "type": "string"},
                {"internalType": "string", "name": "_songHash", "type": "string"}
            ],
            "name": "storePrediction",
            "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
            "stateMutability": "nonpayable",
            "type": "function"
        },
        {
            "inputs": [{"internalType": "uint256", "name": "_predictionId", "type": "uint256"}],
            "name": "getPrediction",
            "outputs": [
                {
                    "components": [
                        {"internalType": "address", "name": "predictor", "type": "address"},
                        {"internalType": "uint256", "name": "timestamp", "type": "uint256"},
                        {"internalType": "uint256", "name": "songId", "type": "uint256"},
                        {"internalType": "uint256", "name": "hitProbability", "type": "uint256"},
                        {"internalType": "bool", "name": "isPredictedHit", "type": "bool"},
                        {"internalType": "string", "name": "modelVersion", "type": "string"},
                        {"internalType": "string", "name": "songHash", "type": "string"},
                        {"internalType": "bool", "name": "verified", "type": "bool"},
                        {"internalType": "bool", "name": "actualOutcome", "type": "bool"}
                    ],
                    "internalType": "struct MusicPredictionOracle.Prediction",
                    "name": "",
                    "type": "tuple"
                }
            ],
            "stateMutability": "view",
            "type": "function"
        },
        {
            "inputs": [{"internalType": "address", "name": "_predictor", "type": "address"}],
            "name": "calculatePredictorAccuracy",
            "outputs": [
                {"internalType": "uint256", "name": "accuracy", "type": "uint256"},
                {"internalType": "uint256", "name": "totalVerified", "type": "uint256"}
            ],
            "stateMutability": "view",
            "type": "function"
        },
        {
            "inputs": [],
            "name": "getTotalPredictions",
            "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
            "stateMutability": "view",
            "type": "function"
        }
    ]
    
    def __init__(self, model_dir="models", data_dir="data", network='fuji'):
        """
        Initialize Avalanche Song Predictor
        
        Args:
            model_dir (str): Directory for saving models
            data_dir (str): Directory for data files
            network (str): Avalanche network ('fuji' for testnet, 'mainnet' for mainnet)
        """
        # Initialize parent class
        super().__init__(model_dir, data_dir)
        
        # Initialize Avalanche configuration
        self.avalanche_config = AvalancheConfig(network)
        self.web3 = self.avalanche_config.get_web3()
        self.account = self.avalanche_config.get_account()
        
        # Initialize contract
        self.contract = None
        self._init_contract()
        
        # Model version for tracking
        self.model_version = "v0.2-avalanche"
        
        print(f"🔗 AvalancheSongPredictor initialized on {network}")
        print(f"📍 Wallet Address: {self.account.address}")
        print(f"💰 Balance: {self.avalanche_config.get_balance():.4f} AVAX")
    
    def _init_contract(self):
        """Initialize the MusicPredictionOracle contract"""
        oracle_address = self.avalanche_config.oracle_address
        
        if not oracle_address:
            print("⚠️  Warning: PREDICTION_ORACLE_ADDRESS not set in environment")
            print("   Contract functions will not be available until address is set")
            return
        
        try:
            self.contract = self.web3.eth.contract(
                address=oracle_address,
                abi=self.CONTRACT_ABI
            )
            print(f"✅ Connected to MusicPredictionOracle at {oracle_address}")
        except Exception as e:
            print(f"❌ Failed to initialize contract: {str(e)}")
            self.contract = None
    
    def _generate_song_hash(self, song_features):
        """
        Generate a hash of song features for verification
        
        Args:
            song_features (dict): Song features dictionary
            
        Returns:
            str: SHA256 hash of normalized song features
        """
        # Normalize features for consistent hashing
        normalized_features = {}
        for key, value in song_features.items():
            if isinstance(value, (int, float)):
                # Round to 6 decimal places for consistency
                normalized_features[key] = round(float(value), 6)
            else:
                normalized_features[key] = str(value)
        
        # Sort keys for deterministic hashing
        sorted_features = dict(sorted(normalized_features.items()))
        
        # Generate hash
        features_json = json.dumps(sorted_features, sort_keys=True)
        song_hash = hashlib.sha256(features_json.encode()).hexdigest()
        
        return song_hash
    
    def _generate_song_id(self, song_features):
        """
        Generate a unique song ID based on features
        
        Args:
            song_features (dict): Song features dictionary
            
        Returns:
            int: Unique song ID (converted from hash)
        """
        song_hash = self._generate_song_hash(song_features)
        # Convert first 8 characters of hash to integer
        song_id = int(song_hash[:8], 16)
        return song_id
    
    def predict_and_store_on_avalanche(self, song_features, store_on_chain=True):
        """
        Make a prediction and optionally store it on Avalanche blockchain
        
        Args:
            song_features (dict): Song features for prediction
            store_on_chain (bool): Whether to store prediction on blockchain
            
        Returns:
            dict: Prediction results with blockchain transaction info
        """
        # Make the prediction using parent class method
        prediction_result = self.predict_song_hit_probability(song_features)
        
        if not prediction_result:
            raise ValueError("Failed to generate prediction")
        
        # Prepare blockchain storage data
        song_id = self._generate_song_id(song_features)
        song_hash = self._generate_song_hash(song_features)
        hit_probability = int(prediction_result['hit_probability'] * 10000)  # Convert to basis points
        is_predicted_hit = prediction_result['is_hit_prediction']
        
        result = {
            'prediction': prediction_result,
            'song_id': song_id,
            'song_hash': song_hash,
            'blockchain_storage': None,
            'transaction_hash': None,
            'prediction_id': None
        }
        
        # Store on blockchain if requested and contract is available
        if store_on_chain and self.contract:
            try:
                blockchain_result = self._store_prediction_on_chain(
                    song_id, hit_probability, is_predicted_hit, song_hash
                )
                result['blockchain_storage'] = blockchain_result
                result['transaction_hash'] = blockchain_result.get('transaction_hash')
                result['prediction_id'] = blockchain_result.get('prediction_id')
                
                print(f"✅ Prediction stored on blockchain!")
                print(f"   Transaction: {result['transaction_hash']}")
                print(f"   Prediction ID: {result['prediction_id']}")
                
            except Exception as e:
                print(f"❌ Failed to store on blockchain: {str(e)}")
                result['blockchain_storage'] = {'error': str(e)}
        
        elif store_on_chain and not self.contract:
            print("⚠️  Cannot store on blockchain: Contract not initialized")
            result['blockchain_storage'] = {'error': 'Contract not initialized'}
        
        return result
    
    def _store_prediction_on_chain(self, song_id, hit_probability, is_predicted_hit, song_hash):
        """
        Store prediction on Avalanche blockchain
        
        Args:
            song_id (int): Unique song identifier
            hit_probability (int): Hit probability in basis points (0-10000)
            is_predicted_hit (bool): Whether predicted as hit
            song_hash (str): Hash of song features
            
        Returns:
            dict: Transaction result with hash and prediction ID
        """
        try:
            # Build transaction
            function_call = self.contract.functions.storePrediction(
                song_id,
                hit_probability,
                is_predicted_hit,
                self.model_version,
                song_hash
            )
            
            # Estimate gas
            gas_estimate = function_call.estimate_gas({'from': self.account.address})
            gas_limit = int(gas_estimate * 1.2)  # Add 20% buffer
            
            # Get gas price
            gas_price = self.avalanche_config.estimate_gas_price()
            
            # Build transaction
            transaction = function_call.build_transaction({
                'from': self.account.address,
                'gas': gas_limit,
                'gasPrice': gas_price,
                'nonce': self.web3.eth.get_transaction_count(self.account.address),
            })
            
            # Sign transaction
            signed_txn = self.web3.eth.account.sign_transaction(transaction, self.account.key)
            
            # Send transaction
            tx_hash = self.web3.eth.send_raw_transaction(signed_txn.rawTransaction)
            
            print(f"📤 Transaction sent: {tx_hash.hex()}")
            print("⏳ Waiting for confirmation...")
            
            # Wait for transaction receipt
            receipt = self.avalanche_config.wait_for_transaction(tx_hash)
            
            # Extract prediction ID from logs
            prediction_id = None
            if receipt.logs:
                # Parse logs to get prediction ID
                try:
                    logs = self.contract.events.PredictionStored().process_receipt(receipt)
                    if logs:
                        prediction_id = logs[0]['args']['predictionId']
                except Exception as e:
                    print(f"⚠️  Could not parse prediction ID from logs: {str(e)}")
            
            # Calculate transaction cost
            gas_used = receipt.gasUsed
            tx_cost = self.web3.from_wei(gas_used * gas_price, 'ether')
            
            return {
                'transaction_hash': tx_hash.hex(),
                'block_number': receipt.blockNumber,
                'gas_used': gas_used,
                'transaction_cost': float(tx_cost),
                'prediction_id': prediction_id,
                'explorer_url': self.avalanche_config.get_block_explorer_url(tx_hash=tx_hash.hex())
            }
            
        except Exception as e:
            raise RuntimeError(f"Failed to store prediction on blockchain: {str(e)}")
    
    def get_prediction_from_chain(self, prediction_id):
        """
        Retrieve a prediction from the blockchain
        
        Args:
            prediction_id (int): Prediction ID to retrieve
            
        Returns:
            dict: Prediction data from blockchain
        """
        if not self.contract:
            raise RuntimeError("Contract not initialized")
        
        try:
            prediction = self.contract.functions.getPrediction(prediction_id).call()
            
            return {
                'prediction_id': prediction_id,
                'predictor': prediction[0],
                'timestamp': prediction[1],
                'song_id': prediction[2],
                'hit_probability': prediction[3] / 10000,  # Convert from basis points
                'is_predicted_hit': prediction[4],
                'model_version': prediction[5],
                'song_hash': prediction[6],
                'verified': prediction[7],
                'actual_outcome': prediction[8],
                'datetime': datetime.fromtimestamp(prediction[1]).isoformat()
            }
            
        except Exception as e:
            raise RuntimeError(f"Failed to retrieve prediction from blockchain: {str(e)}")
    
    def get_predictor_accuracy(self, predictor_address=None):
        """
        Get accuracy statistics for a predictor from blockchain
        
        Args:
            predictor_address (str): Address of predictor (defaults to current wallet)
            
        Returns:
            dict: Accuracy statistics
        """
        if not self.contract:
            raise RuntimeError("Contract not initialized")
        
        if not predictor_address:
            predictor_address = self.account.address
        
        try:
            accuracy, total_verified = self.contract.functions.calculatePredictorAccuracy(
                predictor_address
            ).call()
            
            return {
                'predictor_address': predictor_address,
                'accuracy_percentage': accuracy / 100,  # Convert from basis points to percentage
                'total_verified_predictions': total_verified,
                'total_predictions': len(self.get_predictor_history(predictor_address))
            }
            
        except Exception as e:
            raise RuntimeError(f"Failed to get predictor accuracy: {str(e)}")
    
    def get_predictor_history(self, predictor_address=None):
        """
        Get prediction history for a predictor
        
        Args:
            predictor_address (str): Address of predictor (defaults to current wallet)
            
        Returns:
            list: List of prediction IDs
        """
        if not self.contract:
            raise RuntimeError("Contract not initialized")
        
        if not predictor_address:
            predictor_address = self.account.address
        
        try:
            history = self.contract.functions.getPredictorHistory(predictor_address).call()
            return list(history)
            
        except Exception as e:
            raise RuntimeError(f"Failed to get predictor history: {str(e)}")
    
    def demo_prediction_workflow(self, song_features=None):
        """
        Demonstrate the complete prediction and blockchain storage workflow
        
        Args:
            song_features (dict): Optional song features (uses default if not provided)
            
        Returns:
            dict: Complete workflow result
        """
        print("\n🎵 Avalanche Song Prediction Demo")
        print("=" * 40)
        
        # Use sample song features if none provided
        if not song_features:
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
        
        print(f"🎼 Song Features: {json.dumps(song_features, indent=2)}")
        
        # Make prediction and store on blockchain
        result = self.predict_and_store_on_avalanche(song_features, store_on_chain=True)
        
        print(f"\n📊 Prediction Results:")
        print(f"   Hit Probability: {result['prediction']['hit_probability']:.1%}")
        print(f"   Predicted as Hit: {result['prediction']['is_hit_prediction']}")
        print(f"   Song ID: {result['song_id']}")
        
        if result['prediction_id'] is not None:
            print(f"\n🔗 Blockchain Storage:")
            print(f"   Prediction ID: {result['prediction_id']}")
            print(f"   Transaction: {result['transaction_hash']}")
            
            # Retrieve prediction from blockchain to verify
            chain_prediction = self.get_prediction_from_chain(result['prediction_id'])
            print(f"\n✅ Verification from blockchain:")
            print(f"   Stored Probability: {chain_prediction['hit_probability']:.1%}")
            print(f"   Stored Prediction: {chain_prediction['is_predicted_hit']}")
            print(f"   Model Version: {chain_prediction['model_version']}")
        
        return result


if __name__ == "__main__":
    """Test the AvalancheSongPredictor"""
    try:
        # Initialize predictor
        predictor = AvalancheSongPredictor(network='fuji')
        
        # Load model if available
        if predictor.load_model("best_enhanced_song_model"):
            print("✅ Model loaded successfully")
        else:
            print("⚠️  No saved model found, using default configuration")
        
        # Run demo
        result = predictor.demo_prediction_workflow()
        
        print("\n🎉 Demo completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {str(e)}")
        print("Please ensure your .env file is configured correctly.")